"""
Group-wise analysis — performance degradation as face density increases.

Groups images by number of GT faces into bins:
  0–10, 11–20, 21–30, 31–40, 41–50, 51+

For each bin, compute precision, recall, F1 to show how
crowded images degrade detection performance.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

from src.evaluation import ImageEvalResult, compute_aggregate_metrics
from config.settings import GroupBins


def assign_bin(num_faces: int, bins: GroupBins) -> str:
    """Return the bin label for a given face count."""
    for i in range(len(bins.edges) - 1):
        if bins.edges[i] <= num_faces < bins.edges[i + 1]:
            return bins.bins[i]
    return bins.bins[-1]


def groupwise_analysis(
    results: List[ImageEvalResult],
    bins: GroupBins = None,
) -> pd.DataFrame:
    """
    Group images by face-count bin and compute metrics per group.

    Returns
    -------
    DataFrame with columns:
        bin, num_images, total_gt, tp, fp, fn, precision, recall, f1, mean_iou
    """
    if bins is None:
        bins = GroupBins()

    # Bucket images
    buckets = {label: [] for label in bins.bins}
    for r in results:
        label = assign_bin(r.num_gt, bins)
        buckets[label].append(r)

    rows = []
    for label in bins.bins:
        group = buckets[label]
        if not group:
            rows.append({
                "bin": label,
                "num_images": 0,
                "total_gt": 0,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "mean_iou": 0.0,
            })
            continue

        m = compute_aggregate_metrics(group)
        rows.append({
            "bin": label,
            "num_images": len(group),
            "total_gt": m.total_gt,
            "tp": m.total_tp,
            "fp": m.total_fp,
            "fn": m.total_fn,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "mean_iou": round(m.mean_iou, 4),
        })

    return pd.DataFrame(rows)
