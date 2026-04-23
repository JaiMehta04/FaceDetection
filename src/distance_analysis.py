"""
Distance-based failure analysis.

For each image, compute pairwise Euclidean distances between face centers:
  - Detected ↔ Detected
  - Detected ↔ Undetected
  - Undetected ↔ Undetected

Purpose:
- If Undetected faces are close to Detected faces, it suggests
  occlusion or NMS suppression caused the miss.
- If Undetected faces cluster together, it suggests crowded regions
  where the detector loses resolution.

Output: CSV with (image_name, pair_type, idx_a, idx_b, distance)
"""

from typing import List
from itertools import combinations

import numpy as np
import pandas as pd

from src.evaluation import ImageEvalResult
from src.utils import box_center


def compute_distance_pairs(eval_result: ImageEvalResult) -> List[dict]:
    """
    Compute pairwise Euclidean distances for all face center pairs
    in one image, labelled by pair type.
    """
    rows = []
    gt_boxes = eval_result.gt_boxes
    if len(gt_boxes) == 0:
        return rows

    gt_centers = box_center(gt_boxes)

    detected_set = set(eval_result.detected_gt_indices)
    undetected_set = set(eval_result.undetected_gt_indices)

    all_gt_indices = list(range(len(gt_boxes)))

    def _pair_type(i: int, j: int) -> str:
        i_det = i in detected_set
        j_det = j in detected_set
        if i_det and j_det:
            return "detected-detected"
        elif i_det or j_det:
            return "detected-undetected"
        else:
            return "undetected-undetected"

    for i, j in combinations(all_gt_indices, 2):
        ci = gt_centers[i]
        cj = gt_centers[j]
        dist = float(np.linalg.norm(ci - cj))
        rows.append({
            "image_name": eval_result.image_path,
            "pair_type": _pair_type(i, j),
            "idx_a": i,
            "idx_b": j,
            "distance": round(dist, 2),
        })

    return rows


def distance_analysis(
    results: List[ImageEvalResult],
    max_images: int = 0,
) -> pd.DataFrame:
    """
    Run distance analysis across all evaluated images.

    Parameters
    ----------
    results : list of ImageEvalResult
    max_images : limit processing for speed (0 = all)

    Returns
    -------
    DataFrame with columns:
        image_name, pair_type, idx_a, idx_b, distance
    """
    all_rows = []
    subset = results[:max_images] if max_images > 0 else results

    for r in subset:
        all_rows.extend(compute_distance_pairs(r))

    df = pd.DataFrame(all_rows)
    return df


def summarize_distances(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and median distance by pair type."""
    if df.empty:
        return pd.DataFrame()
    return df.groupby("pair_type").agg(
        count=("distance", "count"),
        mean_distance=("distance", "mean"),
        median_distance=("distance", "median"),
        std_distance=("distance", "std"),
    ).reset_index()
