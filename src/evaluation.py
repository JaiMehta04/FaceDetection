"""
Evaluation engine — matches detections to ground truth and computes metrics.

Matching Algorithm (Hungarian-style greedy):
1. Compute IoU matrix between predicted boxes and GT boxes.
2. Greedily assign the highest-IoU pair (pred → GT) if IoU > threshold.
3. Unmatched GTs → Undetected faces.
4. Unmatched preds → False Positives.

This is the standard PASCAL VOC / WIDER FACE matching protocol.

Design Decisions:
- Greedy matching is O(N*M) and equivalent to the official WIDER FACE
  evaluation when confident scores are pre-sorted (which we do).
- We store per-image results so downstream analysis can drill into them.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from src.detectors.base import DetectionResult
from src.utils import compute_iou_matrix, logger


# ──────────────────────────────────────────────────────────────
# Per-image evaluation result
# ──────────────────────────────────────────────────────────────
@dataclass
class ImageEvalResult:
    """Evaluation results for one image."""
    image_path: str

    # Ground-truth info
    num_gt: int = 0

    # Matched detections
    detected_gt_indices: List[int] = field(default_factory=list)
    detected_pred_indices: List[int] = field(default_factory=list)
    detected_ious: List[float] = field(default_factory=list)

    # GT boxes with corresponding detection info
    gt_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    pred_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    pred_scores: np.ndarray = field(default_factory=lambda: np.empty((0,)))

    # Undetected GT indices
    undetected_gt_indices: List[int] = field(default_factory=list)

    # False positive pred indices
    false_positive_indices: List[int] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.detected_gt_indices)

    @property
    def fn(self) -> int:
        return len(self.undetected_gt_indices)

    @property
    def fp(self) -> int:
        return len(self.false_positive_indices)


# ──────────────────────────────────────────────────────────────
# Matching logic
# ──────────────────────────────────────────────────────────────
def match_detections(
    gt_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
    image_path: str = "",
) -> ImageEvalResult:
    """
    Match predicted boxes to ground-truth using greedy IoU matching.

    Parameters
    ----------
    gt_boxes : (G, 4) [x1, y1, x2, y2]
    pred_boxes : (P, 4) [x1, y1, x2, y2]
    pred_scores : (P,) confidence scores
    iou_threshold : minimum IoU for a valid match
    image_path : identifier for this image

    Returns
    -------
    ImageEvalResult
    """
    result = ImageEvalResult(
        image_path=image_path,
        num_gt=len(gt_boxes),
        gt_boxes=gt_boxes,
        pred_boxes=pred_boxes,
        pred_scores=pred_scores,
    )

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    # Edge cases
    if num_gt == 0 and num_pred == 0:
        return result
    if num_gt == 0:
        result.false_positive_indices = list(range(num_pred))
        return result
    if num_pred == 0:
        result.undetected_gt_indices = list(range(num_gt))
        return result

    # Sort predictions by confidence (descending) — higher confidence first
    sorted_idx = np.argsort(-pred_scores)

    # Compute IoU matrix: (P, G)
    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)

    matched_gt = set()
    matched_pred = set()
    detected_gt_idx = []
    detected_pred_idx = []
    detected_ious = []

    for pi in sorted_idx:
        if pi in matched_pred:
            continue
        # Find best GT match for this prediction
        ious = iou_matrix[pi]
        # Mask already-matched GTs
        for gi in matched_gt:
            ious[gi] = 0.0

        best_gi = int(np.argmax(ious))
        best_iou = float(ious[best_gi])

        if best_iou >= iou_threshold:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            detected_gt_idx.append(best_gi)
            detected_pred_idx.append(int(pi))
            detected_ious.append(best_iou)
        else:
            # This prediction doesn't match any GT
            pass

    result.detected_gt_indices = detected_gt_idx
    result.detected_pred_indices = detected_pred_idx
    result.detected_ious = detected_ious
    result.undetected_gt_indices = [
        i for i in range(num_gt) if i not in matched_gt
    ]
    result.false_positive_indices = [
        int(i) for i in range(num_pred) if i not in matched_pred
    ]

    return result


# ──────────────────────────────────────────────────────────────
# Aggregate metrics
# ──────────────────────────────────────────────────────────────
@dataclass
class AggregateMetrics:
    """Summary metrics over the entire dataset."""
    total_gt: int = 0
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    mean_iou: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


def compute_aggregate_metrics(results: List[ImageEvalResult]) -> AggregateMetrics:
    """Compute precision, recall, F1 over all images."""
    total_gt = sum(r.num_gt for r in results)
    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)

    all_ious = []
    for r in results:
        all_ious.extend(r.detected_ious)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return AggregateMetrics(
        total_gt=total_gt,
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        mean_iou=mean_iou,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def print_metrics(metrics: AggregateMetrics, detector_name: str = ""):
    """Pretty-print aggregate metrics."""
    header = f"  Metrics: {detector_name}  " if detector_name else "  Metrics  "
    logger.info("=" * 50)
    logger.info(header)
    logger.info("=" * 50)
    logger.info(f"  Total GT faces : {metrics.total_gt}")
    logger.info(f"  True Positives : {metrics.total_tp}")
    logger.info(f"  False Positives: {metrics.total_fp}")
    logger.info(f"  False Negatives: {metrics.total_fn}")
    logger.info(f"  Mean IoU       : {metrics.mean_iou:.4f}")
    logger.info(f"  Precision      : {metrics.precision:.4f}")
    logger.info(f"  Recall         : {metrics.recall:.4f}")
    logger.info(f"  F1 Score       : {metrics.f1:.4f}")
    logger.info("=" * 50)
