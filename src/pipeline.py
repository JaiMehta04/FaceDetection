"""
ETL Pipeline — Extract, Transform, Load for face anonymization.

Architecture:
  Extract  → Load images from source (disk, S3, API, etc.)
  Transform → Detect faces + anonymize + compute quality metrics
  Load     → Save anonymized images + evaluation CSV to output

Design Decisions:
- Pipeline is a class with pluggable detector and config.
- Each stage is a separate method for testability.
- Results are accumulated in memory, then flushed to disk.
- The pipeline can process single images (API use) or batch (CLI use).
"""

from pathlib import Path
from typing import List, Optional
import time

import cv2
import numpy as np
import pandas as pd

from config.settings import (
    PipelineConfig, WIDER_IMAGES_DIR, WIDER_ANNOT_FILE,
    ANONYMIZED_DIR, CSV_DIR, RESULTS_DIR, ensure_dirs,
)
from src.data_loader import ImageRecord, parse_wider_annotations
from src.detectors.base import BaseDetector, DetectionResult
from src.detectors.factory import get_detector, wrap_detector_with_enhancements, get_ensemble_detector
from src.evaluation import (
    match_detections, ImageEvalResult,
    compute_aggregate_metrics, print_metrics,
)
from src.quality_analysis import (
    analyze_image_quality, quality_records_to_dataframe,
    summarize_quality_by_category, FaceQualityRecord,
)
from src.group_analysis import groupwise_analysis
from src.distance_analysis import distance_analysis, summarize_distances
from src.attribute_analysis import attribute_analysis, event_category_analysis
from src.pr_curve import compute_pr_curve, threshold_sensitivity
from src.anonymization import anonymize_image
from src.preprocessing import preprocess_image, PreprocessConfig
from src.utils import load_image, resize_if_needed, logger


class FaceAnonymizationPipeline:
    """
    End-to-end ETL pipeline for face detection, evaluation, and anonymization.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        ensure_dirs()

    # ──────────────────────────────────────────────────────────
    # EXTRACT
    # ──────────────────────────────────────────────────────────
    def extract(
        self,
        annot_path: Path = None,
        images_root: Path = None,
    ) -> List[ImageRecord]:
        """Load dataset annotations."""
        annot_path = annot_path or WIDER_ANNOT_FILE
        logger.info(f"[EXTRACT] Loading annotations from {annot_path}")
        records = parse_wider_annotations(annot_path)

        if self.config.max_images > 0:
            records = records[: self.config.max_images]
            logger.info(f"[EXTRACT] Limited to {len(records)} images")

        return records

    # ──────────────────────────────────────────────────────────
    # TRANSFORM
    # ──────────────────────────────────────────────────────────
    def transform(
        self,
        records: List[ImageRecord],
        detector: BaseDetector,
        images_root: Path = None,
        save_anonymized: bool = True,
    ) -> dict:
        """
        Run detection, evaluation, quality analysis, and anonymization.

        Returns
        -------
        dict with keys:
            eval_results, quality_records, aggregate_metrics,
            groupwise_df, detector_name
        """
        images_root = images_root or WIDER_IMAGES_DIR
        det_name = detector.name

        # Build preprocessing config from enhancement settings
        enh = self.config.enhancement
        preprocess_cfg = None
        if enh.enable_preprocessing:
            preprocess_cfg = PreprocessConfig(
                enable_clahe=enh.clahe,
                enable_denoise=enh.denoise,
                adaptive=enh.adaptive_preprocess,
            )

        eval_results: List[ImageEvalResult] = []
        quality_records: List[FaceQualityRecord] = []

        total = len(records)
        t0 = time.time()

        for i, record in enumerate(records):
            img_path = images_root / record.image_path
            image = load_image(img_path)
            if image is None:
                continue

            # Preprocess image if enabled
            scale = 1.0
            detect_image = image
            if preprocess_cfg is not None:
                detect_image, scale = preprocess_image(image, preprocess_cfg)

            # Detect faces
            detection = detector.detect(detect_image)

            # Map boxes back to original coordinates if image was scaled
            if scale != 1.0 and detection.num_faces > 0:
                detection = DetectionResult(
                    boxes=detection.boxes / scale,
                    scores=detection.scores,
                    landmarks=detection.landmarks,
                )

            # Match against ground truth
            gt_boxes = record.gt_boxes(include_invalid=False)
            eval_result = match_detections(
                gt_boxes=gt_boxes,
                pred_boxes=detection.boxes,
                pred_scores=detection.scores,
                iou_threshold=self.config.detection.iou_threshold,
                image_path=record.image_path,
            )
            eval_results.append(eval_result)

            # Quality analysis
            qr = analyze_image_quality(eval_result, images_root)
            quality_records.extend(qr)

            # Anonymize and save
            if save_anonymized and detection.num_faces > 0:
                anon_image = anonymize_image(
                    image.copy(),
                    detection.boxes,
                    self.config.anonymization,
                )
                self._save_anonymized(anon_image, record.image_path, det_name)

            # Progress logging — every image when enhanced, else at batch intervals
            log_interval = 1 if self.config.enhancement.enable_multiscale else self.config.batch_log_interval
            if (i + 1) % log_interval == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    f"[TRANSFORM] [{det_name}] {i+1}/{total} "
                    f"({fps:.1f} img/s)"
                )

        # Aggregate
        agg_metrics = compute_aggregate_metrics(eval_results)
        print_metrics(agg_metrics, det_name)

        # Group-wise
        gw_df = groupwise_analysis(eval_results, self.config.groups)

        return {
            "eval_results": eval_results,
            "quality_records": quality_records,
            "aggregate_metrics": agg_metrics,
            "groupwise_df": gw_df,
            "detector_name": det_name,
            "records": records,
        }

    # ──────────────────────────────────────────────────────────
    # LOAD
    # ──────────────────────────────────────────────────────────
    def load(self, transform_output: dict):
        """Save all results to CSV files."""
        det_name = transform_output["detector_name"].replace(" ", "_").replace("(", "").replace(")", "")

        # 1. Quality CSV
        quality_df = quality_records_to_dataframe(transform_output["quality_records"])
        quality_path = CSV_DIR / f"quality_{det_name}.csv"
        quality_df.to_csv(quality_path, index=False)
        logger.info(f"[LOAD] Quality CSV → {quality_path}")

        # 2. Quality summary
        if not quality_df.empty:
            summary = summarize_quality_by_category(quality_df)
            summary_path = CSV_DIR / f"quality_summary_{det_name}.csv"
            summary.to_csv(summary_path, index=False)
            logger.info(f"[LOAD] Quality summary → {summary_path}")

        # 3. Group-wise CSV
        gw_df = transform_output["groupwise_df"]
        gw_path = CSV_DIR / f"groupwise_{det_name}.csv"
        gw_df.to_csv(gw_path, index=False)
        logger.info(f"[LOAD] Group-wise CSV → {gw_path}")

        # 4. Distance analysis CSV
        eval_results = transform_output["eval_results"]
        dist_df = distance_analysis(eval_results, max_images=500)
        dist_path = CSV_DIR / f"distance_{det_name}.csv"
        dist_df.to_csv(dist_path, index=False)
        logger.info(f"[LOAD] Distance CSV → {dist_path}")

        if not dist_df.empty:
            dist_summary = summarize_distances(dist_df)
            dist_summary_path = CSV_DIR / f"distance_summary_{det_name}.csv"
            dist_summary.to_csv(dist_summary_path, index=False)
            logger.info(f"[LOAD] Distance summary → {dist_summary_path}")

        # 5. Aggregate metrics
        m = transform_output["aggregate_metrics"]
        metrics_row = {
            "detector": det_name,
            "total_gt": m.total_gt,
            "tp": m.total_tp,
            "fp": m.total_fp,
            "fn": m.total_fn,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1,
            "mean_iou": m.mean_iou,
        }
        metrics_df = pd.DataFrame([metrics_row])
        metrics_path = CSV_DIR / f"metrics_{det_name}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"[LOAD] Metrics CSV → {metrics_path}")

        # 6. Attribute analysis (blur, occlusion, illumination, expression, pose)
        records = transform_output.get("records", [])
        attr_df = pd.DataFrame()
        if records:
            attr_df = attribute_analysis(records, eval_results)
            attr_path = CSV_DIR / f"attribute_{det_name}.csv"
            attr_df.to_csv(attr_path, index=False)
            logger.info(f"[LOAD] Attribute CSV → {attr_path}")

        # 7. Event/scene category analysis
        event_df = pd.DataFrame()
        if records:
            event_df = event_category_analysis(records, eval_results)
            event_path = CSV_DIR / f"event_{det_name}.csv"
            event_df.to_csv(event_path, index=False)
            logger.info(f"[LOAD] Event CSV → {event_path}")

        # 8. PR curve and threshold sensitivity
        pr_df, ap = compute_pr_curve(eval_results)
        pr_path = CSV_DIR / f"pr_curve_{det_name}.csv"
        pr_df.to_csv(pr_path, index=False)
        logger.info(f"[LOAD] PR curve CSV → {pr_path}  (AP={ap:.4f})")

        thresh_df = threshold_sensitivity(eval_results)
        thresh_path = CSV_DIR / f"threshold_{det_name}.csv"
        thresh_df.to_csv(thresh_path, index=False)
        logger.info(f"[LOAD] Threshold CSV → {thresh_path}")

        # Save AP into metrics
        metrics_row["ap"] = ap
        metrics_df = pd.DataFrame([metrics_row])
        metrics_df.to_csv(metrics_path, index=False)

        return {
            "quality_df": quality_df,
            "groupwise_df": gw_df,
            "distance_df": dist_df,
            "metrics_df": metrics_df,
            "attribute_df": attr_df,
            "event_df": event_df,
            "pr_df": pr_df,
            "ap": ap,
            "threshold_df": thresh_df,
        }

    # ──────────────────────────────────────────────────────────
    # Run full pipeline
    # ──────────────────────────────────────────────────────────
    def run(
        self,
        detector_names: List[str] = None,
        annot_path: Path = None,
        images_root: Path = None,
        save_anonymized: bool = True,
    ) -> dict:
        """
        Execute the full ETL pipeline for one or more detectors.

        Returns
        -------
        dict keyed by detector name, each containing load outputs.
        """
        if detector_names is None:
            detector_names = ["retinaface", "mtcnn"]

        from src.utils import get_device
        device = get_device(self.config.detection.device)
        logger.info(f"Using device: {device}")

        # Extract (shared across detectors)
        records = self.extract(annot_path, images_root)
        enh = self.config.enhancement

        all_outputs = {}

        # Ensemble mode: run both detectors and merge
        if enh.enable_ensemble:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Running ENSEMBLE pipeline")
            logger.info(f"{'='*60}\n")

            ensemble = get_ensemble_detector(
                device=device,
                confidence=0.3,
                enhancement_config=enh,
            )
            # Apply additional wrappers on top of ensemble
            enhanced = wrap_detector_with_enhancements(ensemble, enh)

            transform_out = self.transform(
                records, enhanced, images_root, save_anonymized
            )
            load_out = self.load(transform_out)
            all_outputs["ensemble"] = {**transform_out, **load_out}

        # Individual detectors
        for det_name in detector_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"  Running pipeline with: {det_name.upper()}")
            logger.info(f"{'='*60}\n")

            detector = get_detector(
                det_name,
                device=device,
                confidence=getattr(
                    self.config.detection,
                    f"{det_name}_confidence",
                    self.config.detection.retinaface_confidence,
                ),
            )

            # Wrap with enhancements (multi-scale, TTA, tiled)
            enhanced = wrap_detector_with_enhancements(detector, enh)

            transform_out = self.transform(
                records, enhanced, images_root, save_anonymized
            )
            load_out = self.load(transform_out)

            all_outputs[det_name] = {
                **transform_out,
                **load_out,
            }

        return all_outputs

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────
    def _save_anonymized(
        self, image: np.ndarray, rel_path: str, detector_name: str
    ):
        """Save an anonymized image preserving directory structure."""
        clean_name = detector_name.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = ANONYMIZED_DIR / clean_name / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
