"""
Image quality analysis per face and per image.

For each detected / undetected / false-positive face, compute:
- Face brightness (mean V channel of face crop)
- Image brightness (mean V channel of full image)
- Image blur (variance of Laplacian on full image)
- Face size (bounding box area in pixels)

These features let us understand *why* a detector fails.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd

from src.evaluation import ImageEvalResult
from src.utils import (
    load_image, compute_blur, compute_brightness,
    crop_box, box_area, logger,
)
from config.settings import QualityConfig


@dataclass
class FaceQualityRecord:
    """Quality metrics for a single face instance."""
    image_path: str
    face_idx: int
    category: str          # "detected", "undetected", "false_positive"
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    face_area: float
    face_brightness: float
    image_brightness: float
    image_blur: float
    confidence: float      # -1 for GT-only faces


def analyze_image_quality(
    eval_result: ImageEvalResult,
    images_root: Path,
) -> List[FaceQualityRecord]:
    """
    Compute quality metrics for all faces (detected, undetected, FP)
    in a single image.
    """
    img_path = images_root / eval_result.image_path
    image = load_image(img_path)
    if image is None:
        return []

    img_brightness = compute_brightness(image)
    img_blur = compute_blur(image)

    records: List[FaceQualityRecord] = []

    # --- Detected GT faces ---
    for match_i, gt_idx in enumerate(eval_result.detected_gt_indices):
        box = eval_result.gt_boxes[gt_idx]
        face_crop = crop_box(image, box)
        face_br = compute_brightness(face_crop) if face_crop is not None else 0.0
        area = float((box[2] - box[0]) * (box[3] - box[1]))

        pred_idx = eval_result.detected_pred_indices[match_i]
        conf = float(eval_result.pred_scores[pred_idx])

        records.append(FaceQualityRecord(
            image_path=eval_result.image_path,
            face_idx=gt_idx,
            category="detected",
            box_x1=float(box[0]), box_y1=float(box[1]),
            box_x2=float(box[2]), box_y2=float(box[3]),
            face_area=area,
            face_brightness=face_br,
            image_brightness=img_brightness,
            image_blur=img_blur,
            confidence=conf,
        ))

    # --- Undetected GT faces ---
    for gt_idx in eval_result.undetected_gt_indices:
        box = eval_result.gt_boxes[gt_idx]
        face_crop = crop_box(image, box)
        face_br = compute_brightness(face_crop) if face_crop is not None else 0.0
        area = float((box[2] - box[0]) * (box[3] - box[1]))

        records.append(FaceQualityRecord(
            image_path=eval_result.image_path,
            face_idx=gt_idx,
            category="undetected",
            box_x1=float(box[0]), box_y1=float(box[1]),
            box_x2=float(box[2]), box_y2=float(box[3]),
            face_area=area,
            face_brightness=face_br,
            image_brightness=img_brightness,
            image_blur=img_blur,
            confidence=-1.0,
        ))

    # --- False positives ---
    for pred_idx in eval_result.false_positive_indices:
        box = eval_result.pred_boxes[pred_idx]
        face_crop = crop_box(image, box)
        face_br = compute_brightness(face_crop) if face_crop is not None else 0.0
        area = float((box[2] - box[0]) * (box[3] - box[1]))
        conf = float(eval_result.pred_scores[pred_idx])

        records.append(FaceQualityRecord(
            image_path=eval_result.image_path,
            face_idx=pred_idx,
            category="false_positive",
            box_x1=float(box[0]), box_y1=float(box[1]),
            box_x2=float(box[2]), box_y2=float(box[3]),
            face_area=area,
            face_brightness=face_br,
            image_brightness=img_brightness,
            image_blur=img_blur,
            confidence=conf,
        ))

    return records


def quality_records_to_dataframe(
    all_records: List[FaceQualityRecord],
) -> pd.DataFrame:
    """Convert a list of FaceQualityRecord to a pandas DataFrame."""
    rows = [
        {
            "image_path": r.image_path,
            "face_idx": r.face_idx,
            "category": r.category,
            "box_x1": r.box_x1, "box_y1": r.box_y1,
            "box_x2": r.box_x2, "box_y2": r.box_y2,
            "face_area": r.face_area,
            "face_brightness": r.face_brightness,
            "image_brightness": r.image_brightness,
            "image_blur": r.image_blur,
            "confidence": r.confidence,
        }
        for r in all_records
    ]
    return pd.DataFrame(rows)


def summarize_quality_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a summary table of mean quality metrics
    grouped by category (detected / undetected / false_positive).
    """
    return df.groupby("category").agg(
        count=("face_area", "count"),
        mean_face_area=("face_area", "mean"),
        mean_face_brightness=("face_brightness", "mean"),
        mean_image_brightness=("image_brightness", "mean"),
        mean_image_blur=("image_blur", "mean"),
        mean_confidence=("confidence", lambda x: x[x >= 0].mean()),
    ).reset_index()
