"""
AdaSR-Face Ablation Runner

Runs a controlled set of experiments, each adding one component, to
produce a clean ablation table for the paper:

  E1: SCRFD baseline (native resolution)
  E2: SCRFD + Tiled + MultiScale (existing enhancements)
  E3: SCRFD + Bicubic 2x (control — shows SR benefit > naive upscale)
  E4: SCRFD + Blind SR (full image — upper bound for SR)
  E5: SCRFD + Adaptive SR (our method — selective SR)
  E6: SCRFD + Adaptive SR + Tiled + MS (full pipeline)
  E7: det_10g + Adaptive SR + Tiled + MS (old model + our method, for comparison)

Results are saved to experiments/csv/ and the log is updated incrementally.
"""

import sys
import json
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config.settings import WIDER_IMAGES_DIR, WIDER_ANNOT_FILE, ensure_dirs
from src.data_loader import parse_wider_annotations
from src.detectors.base import DetectionResult
from src.evaluation import match_detections, compute_aggregate_metrics, ImageEvalResult
from src.quality_analysis import analyze_image_quality, quality_records_to_dataframe
from src.group_analysis import groupwise_analysis
from src.attribute_analysis import attribute_analysis, event_category_analysis
from src.pr_curve import compute_pr_curve, threshold_sensitivity
from src.enhanced_detection import MultiScaleDetector, TiledDetector, soft_nms
from src.utils import load_image, get_device, logger


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
EXP_ROOT = Path(__file__).resolve().parent
CSV_DIR = EXP_ROOT / "csv"
LOG_FILE = EXP_ROOT / "EXPERIMENT_LOG.md"


# ──────────────────────────────────────────────────────────────
# Experiment configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class AblationConfig:
    name: str
    description: str
    # Detector
    detector_type: str = "scrfd"       # 'scrfd' or 'retinaface'
    det_size: int = 640
    confidence: float = 0.5
    # Enhancements
    enable_tiled: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.25
    enable_multiscale: bool = False
    scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.5])
    multiscale_max_dim: int = 2048
    # SR
    sr_mode: str = "none"              # 'none', 'bicubic', 'blind', 'adaptive'
    sr_scale: int = 2
    sr_model: str = "RealESRGAN_x2plus"
    # Adaptive SR params
    ada_low_conf: float = 0.5
    ada_high_conf: float = 0.8
    ada_min_face_for_sr: int = 64
    ada_max_sr_regions: int = 20
    # Soft-NMS
    soft_nms_sigma: float = 0.5
    soft_nms_threshold: float = 0.3


ABLATION_EXPERIMENTS = [
    # ── E1: SCRFD Baseline ──
    AblationConfig(
        name="E1_scrfd_baseline",
        description="SCRFD (det_10g) at native resolution, no enhancements",
    ),

    # ── E2: SCRFD + Tiled + MultiScale ──
    AblationConfig(
        name="E2_scrfd_tiled_ms",
        description="SCRFD + Tiled(640, 0.25) + MultiScale[0.75, 1.0, 1.5]",
        enable_tiled=True,
        enable_multiscale=True,
    ),

    # ── E3: SCRFD + Bicubic 2x (control) ──
    AblationConfig(
        name="E3_scrfd_bicubic",
        description="SCRFD + Bicubic 2x upscale (control, no learned SR)",
        sr_mode="bicubic",
        sr_scale=2,
    ),

    # ── E4: SCRFD + Blind SR (full image) ──
    AblationConfig(
        name="E4_scrfd_blind_sr",
        description="SCRFD + Real-ESRGAN blind SR on full image (upper bound)",
        sr_mode="blind",
        sr_scale=2,
    ),

    # ── E5: SCRFD + Adaptive SR (OUR METHOD) ──
    AblationConfig(
        name="E5_scrfd_adaptive_sr",
        description="SCRFD + AdaSR-Face (confidence-guided selective SR) — NOVEL",
        sr_mode="adaptive",
        sr_scale=2,
        ada_low_conf=0.5,
        ada_high_conf=0.8,
        ada_min_face_for_sr=64,
        ada_max_sr_regions=8,
    ),

    # ── E6: SCRFD + Adaptive SR + Tiled + MS (full pipeline) ──
    AblationConfig(
        name="E6_scrfd_adasr_tiled_ms",
        description="SCRFD + AdaSR-Face + Tiled + MultiScale (full pipeline)",
        enable_tiled=True,
        enable_multiscale=True,
        sr_mode="adaptive",
        sr_scale=2,
    ),

    # ── E7: det_10g (old) + Adaptive SR + Tiled + MS (comparison) ──
    AblationConfig(
        name="E7_retinaface_adasr_tiled_ms",
        description="det_10g + AdaSR-Face + Tiled + MS (old model + our SR method)",
        detector_type="retinaface",
        enable_tiled=True,
        enable_multiscale=True,
        sr_mode="adaptive",
        sr_scale=2,
    ),
]


# ──────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────
def build_pipeline(cfg: AblationConfig, device: str):
    """
    Build detector + SR pipeline from config.

    Returns (detector_or_pipeline, sr_preprocessor_or_None)
    """
    # ── Base detector ──
    if cfg.detector_type == "scrfd":
        from experiments.models.scrfd_detector import SCRFDDetector
        base_detector = SCRFDDetector(
            device=device,
            confidence_threshold=cfg.confidence,
            det_size=cfg.det_size,
        )
    else:
        from src.detectors.retinaface_detector import RetinaFaceDetector
        base_detector = RetinaFaceDetector(
            device=device,
            confidence_threshold=cfg.confidence,
            det_size=cfg.det_size,
        )

    # ── Enhancement wrappers (Tiled → MultiScale) ──
    detector = base_detector
    if cfg.enable_tiled:
        detector = TiledDetector(
            base_detector=detector,
            tile_size=cfg.tile_size,
            overlap=cfg.tile_overlap,
            soft_nms_sigma=cfg.soft_nms_sigma,
            soft_nms_threshold=cfg.soft_nms_threshold,
        )
    if cfg.enable_multiscale:
        detector = MultiScaleDetector(
            base_detector=detector,
            scales=cfg.scales,
            soft_nms_sigma=cfg.soft_nms_sigma,
            soft_nms_threshold=cfg.soft_nms_threshold,
            max_dim=cfg.multiscale_max_dim,
        )

    # ── SR preprocessor ──
    if cfg.sr_mode == "none":
        return detector, None, None

    if cfg.sr_mode == "bicubic":
        from experiments.models.sr_preprocessor import BicubicSRPreprocessor
        sr = BicubicSRPreprocessor(scale=cfg.sr_scale)
    else:
        # Try Real-ESRGAN first, fall back to OpenCV, then bicubic
        sr = _load_best_sr(cfg, device)

    # ── Wrap with SR pipeline ──
    if cfg.sr_mode == "blind":
        from experiments.adaptive_cascade import BlindSRDetector
        pipeline = BlindSRDetector(
            detector=detector,
            sr_preprocessor=sr,
            max_dim=3072,
        )
        return pipeline, sr, None

    if cfg.sr_mode == "adaptive":
        from experiments.adaptive_cascade import AdaptiveSRCascade
        pipeline = AdaptiveSRCascade(
            detector=detector,
            sr_preprocessor=sr,
            low_confidence_threshold=cfg.ada_low_conf,
            high_confidence_threshold=cfg.ada_high_conf,
            min_face_size_for_sr=cfg.ada_min_face_for_sr,
            max_sr_regions=cfg.ada_max_sr_regions,
            soft_nms_sigma=cfg.soft_nms_sigma,
            soft_nms_threshold=cfg.soft_nms_threshold,
            sr_scale=cfg.sr_scale,
        )
        return pipeline, sr, pipeline  # Return cascade for stats

    # Bicubic — apply as a preprocessing step
    from experiments.adaptive_cascade import BlindSRDetector
    pipeline = BlindSRDetector(
        detector=detector,
        sr_preprocessor=sr,
        max_dim=3072,
    )
    return pipeline, sr, None


def _load_best_sr(cfg: AblationConfig, device: str):
    """Try to load best available SR model."""
    # Real-ESRGAN uses PyTorch — force CPU if torch CUDA unavailable
    import torch
    sr_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try Real-ESRGAN
    try:
        from experiments.models.sr_preprocessor import SRPreprocessor
        sr = SRPreprocessor(
            scale=cfg.sr_scale,
            model_name=cfg.sr_model,
            device=sr_device,
            tile_size=512,
        )
        return sr
    except ImportError as e:
        logger.warning(f"Real-ESRGAN not available ({e}), trying OpenCV SR...")

    # Try OpenCV DNN SR
    try:
        from experiments.models.sr_preprocessor import OpenCVSRPreprocessor
        sr = OpenCVSRPreprocessor(scale=cfg.sr_scale, model="edsr")
        return sr
    except (ImportError, Exception) as e:
        logger.warning(f"OpenCV SR not available ({e}), falling back to bicubic")

    # Fallback: bicubic
    from experiments.models.sr_preprocessor import BicubicSRPreprocessor
    return BicubicSRPreprocessor(scale=cfg.sr_scale)


# ──────────────────────────────────────────────────────────────
# Evaluation runner
# ──────────────────────────────────────────────────────────────
def run_ablation(cfg: AblationConfig, records, device: str, images_root: Path) -> dict:
    """Run one ablation experiment."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  ABLATION: {cfg.name}")
    logger.info(f"  {cfg.description}")
    logger.info(f"{'='*70}\n")

    pipeline, sr, cascade = build_pipeline(cfg, device)

    eval_results: List[ImageEvalResult] = []
    quality_records = []
    cascade_stats = []
    total = len(records)
    t0 = time.time()

    for i, record in enumerate(records):
        img_path = images_root / record.image_path
        image = load_image(img_path)
        if image is None:
            continue

        # Detect
        if hasattr(pipeline, "detect"):
            detection = pipeline.detect(image)
        else:
            detection = pipeline.detect(image)

        # Collect cascade stats
        if cascade is not None and hasattr(cascade, "last_stats"):
            cascade_stats.append(cascade.last_stats)

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

    # Compute metrics
    agg = compute_aggregate_metrics(eval_results)
    gw_df = groupwise_analysis(eval_results)
    quality_df = quality_records_to_dataframe(quality_records)
    attr_df = attribute_analysis(records, eval_results)
    event_df = event_category_analysis(records, eval_results)
    pr_df, ap = compute_pr_curve(eval_results)
    thresh_df = threshold_sensitivity(eval_results)

    # Analyze by face size bucket
    size_analysis = _analyze_by_size(eval_results)

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
        "size_analysis": size_analysis,
        "cascade_stats": cascade_stats,
    }


def _analyze_by_size(eval_results: List[ImageEvalResult]) -> pd.DataFrame:
    """Compute recall by face size bucket — key for SR impact analysis."""
    buckets = [
        ("0-10px", 0, 100),
        ("10-16px", 100, 256),
        ("16-32px", 256, 1024),
        ("32-64px", 1024, 4096),
        ("64-128px", 4096, 16384),
        ("128+px", 16384, float("inf")),
    ]

    rows = []
    for name, min_area, max_area in buckets:
        total_gt = 0
        detected = 0

        for er in eval_results:
            if er.num_gt == 0:
                continue
            for gi in range(er.num_gt):
                box = er.gt_boxes[gi]
                area = (box[2] - box[0]) * (box[3] - box[1])
                if min_area <= area < max_area:
                    total_gt += 1
                    if gi in er.detected_gt_indices:
                        detected += 1

        recall = detected / total_gt if total_gt > 0 else 0
        rows.append({
            "size_bucket": name,
            "total_gt": total_gt,
            "detected": detected,
            "missed": total_gt - detected,
            "recall": recall,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────────────────────
def save_ablation(result: dict, output_dir: Path):
    """Save one experiment's results."""
    cfg = result["config"]
    name = cfg.name
    out = output_dir / "csv"
    out.mkdir(parents=True, exist_ok=True)

    agg = result["aggregate_metrics"]

    # Metrics summary
    metrics_row = {
        "experiment": name,
        "description": cfg.description,
        "detector": cfg.detector_type,
        "sr_mode": cfg.sr_mode,
        "sr_scale": cfg.sr_scale,
        "tiled": cfg.enable_tiled,
        "multiscale": cfg.enable_multiscale,
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
    }
    pd.DataFrame([metrics_row]).to_csv(out / f"ablation_{name}.csv", index=False)

    # Sub-analyses
    result["groupwise_df"].to_csv(out / f"ablation_gw_{name}.csv", index=False)
    result["quality_df"].to_csv(out / f"ablation_quality_{name}.csv", index=False)
    result["attribute_df"].to_csv(out / f"ablation_attr_{name}.csv", index=False)
    result["event_df"].to_csv(out / f"ablation_event_{name}.csv", index=False)
    result["pr_df"].to_csv(out / f"ablation_pr_{name}.csv", index=False)
    result["threshold_df"].to_csv(out / f"ablation_thresh_{name}.csv", index=False)
    result["size_analysis"].to_csv(out / f"ablation_size_{name}.csv", index=False)

    # Cascade stats
    if result["cascade_stats"]:
        stats_df = pd.DataFrame(result["cascade_stats"])
        stats_df.to_csv(out / f"ablation_cascade_stats_{name}.csv", index=False)

    logger.info(f"  Saved {name} results to {out}")


def update_log(result: dict):
    """Append result to the experiment log file."""
    cfg = result["config"]
    agg = result["aggregate_metrics"]

    # Read existing log
    if LOG_FILE.exists():
        content = LOG_FILE.read_text(encoding="utf-8")
    else:
        content = "# Experiment Log\n\n"

    # Build log entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n### {cfg.name} — {timestamp}\n"
        f"- **Description:** {cfg.description}\n"
        f"- **Detector:** {cfg.detector_type} | SR: {cfg.sr_mode} "
        f"| Tiled: {cfg.enable_tiled} | MS: {cfg.enable_multiscale}\n"
        f"- **Results:** P={agg.precision:.4f} R={agg.recall:.4f} "
        f"F1={agg.f1:.4f} AP={result['ap']:.4f}\n"
        f"- **TP={agg.total_tp} FP={agg.total_fp} FN={agg.total_fn}** "
        f"| Time={result['elapsed']:.0f}s\n"
    )

    # Append size analysis
    sa = result["size_analysis"]
    entry += "- **Recall by face size:**\n"
    for _, row in sa.iterrows():
        entry += f"  - {row['size_bucket']}: {row['recall']:.3f} ({row['detected']}/{row['total_gt']})\n"

    # Append cascade stats summary if available
    if result["cascade_stats"]:
        stats = result["cascade_stats"]
        avg_sr_regions = np.mean([s.get("sr_regions", 0) for s in stats])
        avg_new_faces = np.mean([s.get("stage2_new_faces", 0) for s in stats])
        entry += (
            f"- **Cascade stats:** avg SR regions/image={avg_sr_regions:.1f}, "
            f"avg new faces from SR={avg_new_faces:.1f}\n"
        )

    entry += "\n"

    # Update the results table in the log
    table_marker = "| # | Experiment |"
    if table_marker in content:
        # Find the row for this experiment and update it
        lines = content.split("\n")
        updated = False
        for i, line in enumerate(lines):
            if f"| {cfg.name} |" in line or f"| {cfg.name.split('_')[0]} |" in line:
                # Replace placeholder row
                exp_id = cfg.name.split("_")[0]
                lines[i] = (
                    f"| {exp_id} | {cfg.name} | {agg.precision:.3f} | "
                    f"{agg.recall:.3f} | {agg.f1:.3f} | {result['ap']:.3f} | "
                    f"{result['elapsed']:.0f} | {cfg.sr_mode} |"
                )
                updated = True
                break
        if updated:
            content = "\n".join(lines)

    # Append detailed entry before "## Key Decisions"
    key_decisions_marker = "## Key Decisions"
    if key_decisions_marker in content:
        content = content.replace(
            key_decisions_marker,
            entry + key_decisions_marker,
        )
    else:
        content += entry

    LOG_FILE.write_text(content, encoding="utf-8")
    logger.info(f"  Updated log: {LOG_FILE}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AdaSR-Face Ablation Study")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Experiment names to run (default: all)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit images for quick testing (0=all)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    args = parser.parse_args()

    ensure_dirs()

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    # Load data
    logger.info("Loading annotations...")
    records = parse_wider_annotations(WIDER_ANNOT_FILE)
    if args.max_images > 0:
        records = records[:args.max_images]
    logger.info(f"Loaded {len(records)} images")

    # Select experiments
    if args.experiments:
        exps = [e for e in ABLATION_EXPERIMENTS if e.name in args.experiments]
        if not exps:
            names = [e.name for e in ABLATION_EXPERIMENTS]
            print(f"No match. Available: {names}")
            return
    else:
        exps = ABLATION_EXPERIMENTS

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Run
    all_results = []
    for cfg in exps:
        try:
            result = run_ablation(cfg, records, device, WIDER_IMAGES_DIR)

            agg = result["aggregate_metrics"]
            logger.info(
                f"\n  RESULT [{cfg.name}]: P={agg.precision:.4f} R={agg.recall:.4f} "
                f"F1={agg.f1:.4f} AP={result['ap']:.4f} "
                f"TP={agg.total_tp} FP={agg.total_fp} FN={agg.total_fn} "
                f"Time={result['elapsed']:.0f}s"
            )

            save_ablation(result, EXP_ROOT)
            update_log(result)
            all_results.append(result)

        except Exception as e:
            logger.error(f"  FAILED [{cfg.name}]: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    if all_results:
        logger.info(f"\n{'='*80}")
        logger.info("  ABLATION SUMMARY")
        logger.info(f"{'='*80}")

        rows = []
        for r in all_results:
            c = r["config"]
            a = r["aggregate_metrics"]
            rows.append({
                "Experiment": c.name,
                "SR": c.sr_mode,
                "Tiled": "Y" if c.enable_tiled else "",
                "MS": "Y" if c.enable_multiscale else "",
                "P": f"{a.precision:.4f}",
                "R": f"{a.recall:.4f}",
                "F1": f"{a.f1:.4f}",
                "AP": f"{r['ap']:.4f}",
                "Time": f"{r['elapsed']:.0f}s",
            })

        summary_df = pd.DataFrame(rows)
        logger.info(f"\n{summary_df.to_string(index=False)}")

        # Save combined summary
        combined = []
        for r in all_results:
            c = r["config"]
            a = r["aggregate_metrics"]
            combined.append({
                "experiment": c.name,
                "description": c.description,
                "detector": c.detector_type,
                "sr_mode": c.sr_mode,
                "sr_scale": c.sr_scale,
                "tiled": c.enable_tiled,
                "multiscale": c.enable_multiscale,
                "precision": a.precision,
                "recall": a.recall,
                "f1": a.f1,
                "ap": r["ap"],
                "mean_iou": a.mean_iou,
                "tp": a.total_tp,
                "fp": a.total_fp,
                "fn": a.total_fn,
                "elapsed_s": round(r["elapsed"], 1),
            })
        combined_df = pd.DataFrame(combined)
        summary_path = CSV_DIR / "ablation_summary.csv"
        combined_df.to_csv(summary_path, index=False)
        logger.info(f"\nSummary: {summary_path}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
