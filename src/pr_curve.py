"""
Precision-Recall curve and confidence threshold sensitivity analysis.

Computes:
1. PR curve — precision and recall at every unique confidence threshold
2. Threshold sensitivity — how P, R, F1 change with confidence threshold
3. Average Precision (AP) — area under the PR curve

These use the per-detection confidence scores and TP/FP labels
pooled across all images.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.evaluation import ImageEvalResult
from src.utils import compute_iou_matrix


def compute_pr_curve(
    eval_results: List[ImageEvalResult],
) -> Tuple[pd.DataFrame, float]:
    """
    Compute the Precision-Recall curve across all images.

    Pools all detections, sorts by confidence (descending), and
    computes cumulative precision/recall at each threshold.

    Returns
    -------
    pr_df : DataFrame with columns: confidence, precision, recall
    ap    : Average Precision (area under the PR curve)
    """
    # Collect all detections with TP/FP labels
    all_scores = []
    all_tp = []
    total_gt = 0

    for ev in eval_results:
        total_gt += ev.num_gt
        detected_pred_set = set(ev.detected_pred_indices)

        for pi in range(len(ev.pred_scores)):
            score = float(ev.pred_scores[pi])
            is_tp = 1 if pi in detected_pred_set else 0
            all_scores.append(score)
            all_tp.append(is_tp)

    if not all_scores or total_gt == 0:
        return pd.DataFrame(columns=["confidence", "precision", "recall"]), 0.0

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)

    # Sort by confidence descending
    sorted_idx = np.argsort(-all_scores)
    all_scores = all_scores[sorted_idx]
    all_tp = all_tp[sorted_idx]

    # Cumulative TP and FP
    cum_tp = np.cumsum(all_tp)
    cum_fp = np.cumsum(1 - all_tp)

    precision = cum_tp / (cum_tp + cum_fp)
    recall = cum_tp / total_gt

    # Build dataframe (sample at most 2000 points for plotting efficiency)
    n = len(precision)
    if n > 2000:
        indices = np.linspace(0, n - 1, 2000, dtype=int)
    else:
        indices = np.arange(n)

    pr_df = pd.DataFrame({
        "confidence": all_scores[indices],
        "precision": precision[indices],
        "recall": recall[indices],
    })

    # Average Precision (AP) using all-points interpolation
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Monotonically decreasing precision (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))

    return pr_df, ap


def threshold_sensitivity(
    eval_results: List[ImageEvalResult],
    thresholds: List[float] = None,
) -> pd.DataFrame:
    """
    Compute precision, recall, F1 at varying confidence thresholds.

    Re-evaluates TP/FP/FN at each threshold by discarding detections
    below the threshold then re-matching.

    Parameters
    ----------
    eval_results : list of per-image evaluation results
    thresholds : list of confidence values to evaluate

    Returns
    -------
    DataFrame: threshold, tp, fp, fn, precision, recall, f1
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for thresh in thresholds:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0

        for ev in eval_results:
            total_gt += ev.num_gt

            # Filter predictions above threshold
            if len(ev.pred_scores) == 0:
                total_fn += ev.num_gt
                continue

            mask = ev.pred_scores >= thresh
            if not mask.any():
                total_fn += ev.num_gt
                continue

            pred_boxes = ev.pred_boxes[mask]
            pred_scores = ev.pred_scores[mask]

            # Re-match at this threshold
            gt_boxes = ev.gt_boxes
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue

            iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
            sorted_idx = np.argsort(-pred_scores)

            matched_gt = set()
            tp = 0
            fp = 0

            for pi in sorted_idx:
                ious = iou_matrix[pi].copy()
                for gi in matched_gt:
                    ious[gi] = 0.0
                best_gi = int(np.argmax(ious))
                if ious[best_gi] >= 0.5:
                    matched_gt.add(best_gi)
                    tp += 1
                else:
                    fp += 1

            fn = ev.num_gt - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "threshold": thresh,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    return pd.DataFrame(rows)
