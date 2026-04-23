"""
Attribute-based analysis using WIDER FACE annotation metadata.

WIDER FACE provides per-face attributes:
  - blur:         0=clear, 1=normal blur, 2=heavy blur
  - expression:   0=typical, 1=exaggerated
  - illumination: 0=normal, 1=extreme
  - occlusion:    0=none, 1=partial, 2=heavy
  - pose:         0=typical, 1=atypical

This module computes recall broken down by each attribute level,
revealing which conditions cause the most detection failures.
"""

from typing import List, Dict
import numpy as np
import pandas as pd

from src.data_loader import ImageRecord
from src.evaluation import ImageEvalResult


ATTRIBUTE_LABELS = {
    "blur": {0: "Clear", 1: "Normal Blur", 2: "Heavy Blur"},
    "expression": {0: "Typical", 1: "Exaggerated"},
    "illumination": {0: "Normal", 1: "Extreme"},
    "occlusion": {0: "No Occlusion", 1: "Partial", 2: "Heavy"},
    "pose": {0: "Typical", 1: "Atypical"},
}


def attribute_analysis(
    records: List[ImageRecord],
    eval_results: List[ImageEvalResult],
) -> pd.DataFrame:
    """
    Compute detection recall by WIDER FACE annotation attribute.

    For each attribute (blur, occlusion, etc.) and each level (0, 1, 2),
    count how many GT faces have that attribute and how many were detected.

    Parameters
    ----------
    records : list of ImageRecord (with FaceAnnotation metadata)
    eval_results : list of ImageEvalResult (must be aligned 1:1 with records)

    Returns
    -------
    DataFrame with columns: attribute, level, label, total, detected, recall
    """
    # Build lookup: image_path → eval_result
    eval_lookup: Dict[str, ImageEvalResult] = {
        r.image_path: r for r in eval_results
    }

    attributes = ["blur", "expression", "illumination", "occlusion", "pose"]

    # Counters: {(attr, level): {"total": int, "detected": int}}
    counters: Dict[tuple, Dict[str, int]] = {}
    for attr in attributes:
        for level in ATTRIBUTE_LABELS[attr]:
            counters[(attr, level)] = {"total": 0, "detected": 0}

    for record in records:
        ev = eval_lookup.get(record.image_path)
        if ev is None:
            continue

        detected_set = set(ev.detected_gt_indices)

        # Iterate only valid faces (same filter as gt_boxes(include_invalid=False))
        valid_idx = 0
        for face in record.faces:
            if face.invalid == 1:
                continue

            for attr in attributes:
                level = getattr(face, attr, 0)
                key = (attr, level)
                if key in counters:
                    counters[key]["total"] += 1
                    if valid_idx in detected_set:
                        counters[key]["detected"] += 1

            valid_idx += 1

    rows = []
    for (attr, level), counts in counters.items():
        total = counts["total"]
        detected = counts["detected"]
        recall = detected / total if total > 0 else 0.0
        rows.append({
            "attribute": attr,
            "level": level,
            "label": ATTRIBUTE_LABELS[attr].get(level, str(level)),
            "total": total,
            "detected": detected,
            "missed": total - detected,
            "recall": round(recall, 4),
        })

    return pd.DataFrame(rows)


def event_category_analysis(
    records: List[ImageRecord],
    eval_results: List[ImageEvalResult],
) -> pd.DataFrame:
    """
    Compute metrics grouped by WIDER FACE event category.

    WIDER FACE image paths follow: '<N>--<EventName>/image.jpg'
    E.g. '0--Parade/0_Parade_marchingband_1_849.jpg'

    Returns
    -------
    DataFrame: event, num_images, total_gt, tp, fn, fp, precision, recall, f1
    """
    eval_lookup = {r.image_path: r for r in eval_results}

    # Group by event
    event_results: Dict[str, List[ImageEvalResult]] = {}
    for record in records:
        ev = eval_lookup.get(record.image_path)
        if ev is None:
            continue

        # Extract event from path: "0--Parade/filename.jpg" → "Parade"
        parts = record.image_path.split("/")
        if len(parts) >= 2:
            event_raw = parts[0]  # e.g. "0--Parade"
            event = event_raw.split("--", 1)[1] if "--" in event_raw else event_raw
        else:
            event = "Unknown"

        if event not in event_results:
            event_results[event] = []
        event_results[event].append(ev)

    from src.evaluation import compute_aggregate_metrics

    rows = []
    for event, results in sorted(event_results.items()):
        m = compute_aggregate_metrics(results)
        rows.append({
            "event": event,
            "num_images": len(results),
            "total_gt": m.total_gt,
            "tp": m.total_tp,
            "fp": m.total_fp,
            "fn": m.total_fn,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("recall", ascending=True).reset_index(drop=True)
    return df
