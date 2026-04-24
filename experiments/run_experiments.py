"""
Experiment runner for improving RetinaFace Tiled + MultiScale.

Runs a series of controlled experiments, each changing ONE or TWO
hyperparameters from the current best configuration, and saves
results to experiments/ for side-by-side comparison.

Current best baseline:
  RetinaFace Tiled+MultiScale: F1=0.802, P=0.921, R=0.711
  Config: tiles=640, overlap=0.25, scales=[0.75,1.0,1.5],
          confidence=0.5, det_size=640, no preprocessing, no TTA

Experiments designed from failure analysis:
  - 94% of missed faces are < 32x32 px → more upscaling, smaller tiles
  - Heavy blur recall = 0.56 → CLAHE preprocessing
  - Heavy occlusion recall = 0.41 → lower confidence
  - 51+ face images recall = 0.62 → more tile overlap
"""

import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config.settings import (
    PipelineConfig, WIDER_IMAGES_DIR, WIDER_ANNOT_FILE, ensure_dirs,
)
from src.data_loader import parse_wider_annotations
from src.detectors.retinaface_detector import RetinaFaceDetector
from src.enhanced_detection import (
    MultiScaleDetector, TiledDetector, TTADetector, soft_nms,
)
from src.preprocessing import preprocess_image, PreprocessConfig
from src.detectors.base import DetectionResult
from src.evaluation import match_detections, compute_aggregate_metrics, ImageEvalResult
from src.quality_analysis import analyze_image_quality, quality_records_to_dataframe
from src.group_analysis import groupwise_analysis
from src.attribute_analysis import attribute_analysis, event_category_analysis
from src.pr_curve import compute_pr_curve, threshold_sensitivity
from src.utils import load_image, get_device, logger

import numpy as np


# ──────────────────────────────────────────────────────────────
# Experiment definitions
# ──────────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    """One experiment's full configuration."""
    name: str
    description: str
    # Detector
    det_size: int = 640
    confidence: float = 0.5
    # Tiling
    tile_size: int = 640
    tile_overlap: float = 0.25
    # Multi-scale
    scales: List[float] = None
    multiscale_max_dim: int = 2048
    # Soft-NMS
    soft_nms_sigma: float = 0.5
    soft_nms_threshold: float = 0.3
    # TTA
    enable_tta: bool = False
    # Preprocessing
    enable_preprocess: bool = False
    clahe_clip: float = 2.0
    denoise_strength: int = 10
    adaptive_preprocess: bool = True

    def __post_init__(self):
        if self.scales is None:
            self.scales = [0.75, 1.0, 1.5]


EXPERIMENTS = [
    # ── Experiment 0: Current best (baseline for comparison) ──
    ExperimentConfig(
        name="E0_current_best",
        description="Current best: Tiled(640,0.25) + MS[0.75,1.0,1.5] + conf=0.5",
    ),

    # ── Experiment 1: More aggressive upscaling ──
    ExperimentConfig(
        name="E1_aggressive_scales",
        description="Wider scale range [0.5, 0.75, 1.0, 1.5, 2.0] to catch smaller faces via 2x upscale",
        scales=[0.5, 0.75, 1.0, 1.5, 2.0],
        multiscale_max_dim=2560,
    ),

    # ── Experiment 2: Smaller tiles + more overlap ──
    ExperimentConfig(
        name="E2_smaller_tiles",
        description="Smaller tiles (480px) with more overlap (35%) for denser coverage of crowded images",
        tile_size=480,
        tile_overlap=0.35,
    ),

    # ── Experiment 3: Lower confidence threshold ──
    ExperimentConfig(
        name="E3_low_confidence",
        description="Lower confidence to 0.3 to catch marginal detections (occluded/blurred faces)",
        confidence=0.3,
    ),

    # ── Experiment 4: CLAHE preprocessing ──
    ExperimentConfig(
        name="E4_clahe_preprocess",
        description="CLAHE preprocessing (adaptive) to improve contrast for dark/low-light faces",
        enable_preprocess=True,
        adaptive_preprocess=True,
    ),

    # ── Experiment 5: TTA (horizontal flip) ──
    ExperimentConfig(
        name="E5_tta_flip",
        description="Add Test-Time Augmentation (horizontal flip) to catch profile faces",
        enable_tta=True,
    ),

    # ── Experiment 6: Combined - smaller tiles + aggressive scales ──
    ExperimentConfig(
        name="E6_small_tiles_agg_scales",
        description="Smaller tiles (480, 0.35 overlap) + aggressive scales [0.5,0.75,1.0,1.5,2.0]",
        tile_size=480,
        tile_overlap=0.35,
        scales=[0.5, 0.75, 1.0, 1.5, 2.0],
        multiscale_max_dim=2560,
    ),

    # ── Experiment 7: Everything combined ──
    ExperimentConfig(
        name="E7_kitchen_sink",
        description="All improvements: small tiles + agg scales + low conf + CLAHE + TTA",
        tile_size=480,
        tile_overlap=0.35,
        scales=[0.5, 0.75, 1.0, 1.5, 2.0],
        multiscale_max_dim=2560,
        confidence=0.3,
        enable_preprocess=True,
        adaptive_preprocess=True,
        enable_tta=True,
    ),

    # ── Experiment 8: Best combo without TTA (faster) ──
    ExperimentConfig(
        name="E8_best_no_tta",
        description="Small tiles + agg scales + low conf + CLAHE (no TTA for speed)",
        tile_size=480,
        tile_overlap=0.35,
        scales=[0.5, 0.75, 1.0, 1.5, 2.0],
        multiscale_max_dim=2560,
        confidence=0.3,
        enable_preprocess=True,
        adaptive_preprocess=True,
    ),
]


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────
EXP_DIR = Path(__file__).resolve().parent


def build_detector(cfg: ExperimentConfig, device: str):
    """Build the enhanced detector stack from experiment config."""
    base = RetinaFaceDetector(
        device=device,
        confidence_threshold=cfg.confidence,
        det_size=cfg.det_size,
    )

    # Tiled → MultiScale → TTA (innermost to outermost)
    detector = TiledDetector(
        base_detector=base,
        tile_size=cfg.tile_size,
        overlap=cfg.tile_overlap,
        soft_nms_sigma=cfg.soft_nms_sigma,
        soft_nms_threshold=cfg.soft_nms_threshold,
    )

    detector = MultiScaleDetector(
        base_detector=detector,
        scales=cfg.scales,
        soft_nms_sigma=cfg.soft_nms_sigma,
        soft_nms_threshold=cfg.soft_nms_threshold,
        max_dim=cfg.multiscale_max_dim,
    )

    if cfg.enable_tta:
        detector = TTADetector(
            base_detector=detector,
            soft_nms_sigma=cfg.soft_nms_sigma,
            soft_nms_threshold=cfg.soft_nms_threshold,
        )

    return detector


def run_experiment(cfg: ExperimentConfig, records, device: str, images_root: Path):
    """Run a single experiment and return results dict."""
    logger.info(f"\n{'='*60}")
    logger.info(f"  EXPERIMENT: {cfg.name}")
    logger.info(f"  {cfg.description}")
    logger.info(f"{'='*60}\n")

    detector = build_detector(cfg, device)

    # Preprocessing config
    preprocess_cfg = None
    if cfg.enable_preprocess:
        preprocess_cfg = PreprocessConfig(
            enable_clahe=True,
            clahe_clip_limit=cfg.clahe_clip,
            enable_denoise=True,
            denoise_strength=cfg.denoise_strength,
            adaptive=cfg.adaptive_preprocess,
        )

    eval_results: List[ImageEvalResult] = []
    quality_records = []
    total = len(records)
    t0 = time.time()

    for i, record in enumerate(records):
        img_path = images_root / record.image_path
        image = load_image(img_path)
        if image is None:
            continue

        # Preprocess
        scale = 1.0
        detect_image = image
        if preprocess_cfg is not None:
            detect_image, scale = preprocess_image(image, preprocess_cfg)

        # Detect
        detection = detector.detect(detect_image)

        # Map back
        if scale != 1.0 and detection.num_faces > 0:
            detection = DetectionResult(
                boxes=detection.boxes / scale,
                scores=detection.scores,
            )

        # Evaluate
        gt_boxes = record.gt_boxes(include_invalid=False)
        eval_result = match_detections(
            gt_boxes=gt_boxes,
            pred_boxes=detection.boxes,
            pred_scores=detection.scores,
            iou_threshold=0.5,
            image_path=record.image_path,
        )
        eval_results.append(eval_result)

        qr = analyze_image_quality(eval_result, images_root)
        quality_records.extend(qr)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(f"  [{cfg.name}] {i+1}/{total} ({fps:.1f} img/s)")

    elapsed = time.time() - t0

    # Compute all metrics
    agg = compute_aggregate_metrics(eval_results)
    gw_df = groupwise_analysis(eval_results)
    quality_df = quality_records_to_dataframe(quality_records)
    attr_df = attribute_analysis(records, eval_results)
    event_df = event_category_analysis(records, eval_results)
    pr_df, ap = compute_pr_curve(eval_results)
    thresh_df = threshold_sensitivity(eval_results)

    return {
        "config": cfg,
        "aggregate_metrics": agg,
        "ap": ap,
        "elapsed": elapsed,
        "eval_results": eval_results,
        "groupwise_df": gw_df,
        "quality_df": quality_df,
        "attribute_df": attr_df,
        "event_df": event_df,
        "pr_df": pr_df,
        "threshold_df": thresh_df,
    }


def save_experiment(result: dict, output_dir: Path):
    """Save one experiment's results to CSV files."""
    cfg = result["config"]
    name = cfg.name
    out = output_dir / "csv"
    out.mkdir(parents=True, exist_ok=True)

    agg = result["aggregate_metrics"]
    metrics_row = {
        "experiment": name,
        "description": cfg.description,
        "total_gt": agg.total_gt,
        "tp": agg.total_tp,
        "fp": agg.total_fp,
        "fn": agg.total_fn,
        "precision": agg.precision,
        "recall": agg.recall,
        "f1": agg.f1,
        "mean_iou": agg.mean_iou,
        "ap": result["ap"],
        "elapsed_s": round(result["elapsed"], 1),
        "tile_size": cfg.tile_size,
        "tile_overlap": cfg.tile_overlap,
        "scales": str(cfg.scales),
        "confidence": cfg.confidence,
        "enable_tta": cfg.enable_tta,
        "enable_preprocess": cfg.enable_preprocess,
    }
    pd.DataFrame([metrics_row]).to_csv(out / f"exp_metrics_{name}.csv", index=False)

    result["groupwise_df"].to_csv(out / f"exp_groupwise_{name}.csv", index=False)
    result["quality_df"].to_csv(out / f"exp_quality_{name}.csv", index=False)
    result["attribute_df"].to_csv(out / f"exp_attribute_{name}.csv", index=False)
    result["event_df"].to_csv(out / f"exp_event_{name}.csv", index=False)
    result["pr_df"].to_csv(out / f"exp_pr_curve_{name}.csv", index=False)
    result["threshold_df"].to_csv(out / f"exp_threshold_{name}.csv", index=False)

    logger.info(f"  Saved results for {name} to {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run improvement experiments")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment names to run (default: all). E.g. E0_current_best E3_low_confidence")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit images for quick testing (0 = all)")
    args = parser.parse_args()

    ensure_dirs()

    device = get_device("auto")
    logger.info(f"Device: {device}")

    # Load data once
    logger.info("Loading annotations...")
    records = parse_wider_annotations(WIDER_ANNOT_FILE)
    if args.max_images > 0:
        records = records[:args.max_images]
    logger.info(f"Loaded {len(records)} images")

    # Select experiments
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e.name in args.experiments]
        if not exps:
            print(f"No matching experiments. Available: {[e.name for e in EXPERIMENTS]}")
            return
    else:
        exps = EXPERIMENTS

    output_dir = EXP_DIR
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    for cfg in exps:
        result = run_experiment(cfg, records, device, WIDER_IMAGES_DIR)

        agg = result["aggregate_metrics"]
        logger.info(f"\n  RESULT [{cfg.name}]: P={agg.precision:.4f} R={agg.recall:.4f} "
                     f"F1={agg.f1:.4f} AP={result['ap']:.4f} "
                     f"TP={agg.total_tp} FP={agg.total_fp} FN={agg.total_fn} "
                     f"Time={result['elapsed']:.0f}s")

        save_experiment(result, output_dir)
        all_results.append(result)

    # Summary comparison table
    logger.info(f"\n{'='*80}")
    logger.info("  EXPERIMENT COMPARISON SUMMARY")
    logger.info(f"{'='*80}")

    summary_rows = []
    for r in all_results:
        cfg = r["config"]
        agg = r["aggregate_metrics"]
        summary_rows.append({
            "Experiment": cfg.name,
            "Precision": f"{agg.precision:.4f}",
            "Recall": f"{agg.recall:.4f}",
            "F1": f"{agg.f1:.4f}",
            "AP": f"{r['ap']:.4f}",
            "TP": agg.total_tp,
            "FP": agg.total_fp,
            "Time(s)": f"{r['elapsed']:.0f}",
        })

    summary_df = pd.DataFrame(summary_rows)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save combined summary
    combined = []
    for r in all_results:
        cfg = r["config"]
        agg = r["aggregate_metrics"]
        combined.append({
            "experiment": cfg.name,
            "description": cfg.description,
            "precision": agg.precision,
            "recall": agg.recall,
            "f1": agg.f1,
            "ap": r["ap"],
            "mean_iou": agg.mean_iou,
            "tp": agg.total_tp,
            "fp": agg.total_fp,
            "fn": agg.total_fn,
            "total_gt": agg.total_gt,
            "elapsed_s": round(r["elapsed"], 1),
            "tile_size": cfg.tile_size,
            "tile_overlap": cfg.tile_overlap,
            "scales": str(cfg.scales),
            "confidence": cfg.confidence,
            "enable_tta": cfg.enable_tta,
            "enable_preprocess": cfg.enable_preprocess,
        })
    combined_df = pd.DataFrame(combined)
    summary_path = output_dir / "csv" / "experiment_summary.csv"
    combined_df.to_csv(summary_path, index=False)
    logger.info(f"\nSummary saved to: {summary_path}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
