"""
Visualization and plotting utilities.

All axes carry proper units and human-readable labels so every plot
is self-explanatory in a report, thesis, or presentation.

Unit reference (used throughout):
  - Face area          → pixels² (width × height of bounding box)
  - Face brightness    → Mean pixel value in HSV V-channel (0 = black, 255 = white)
  - Image brightness   → Mean pixel value in HSV V-channel (0 = black, 255 = white)
  - Image blur         → Variance of Laplacian (higher = sharper; lower = blurrier)
  - Confidence         → Model score (0.0 – 1.0, higher = more confident)
  - Distance           → Euclidean distance between face-box centres (pixels)
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter

from config.settings import PLOTS_DIR, ensure_dirs
from src.evaluation import ImageEvalResult


# ──────────────────────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

COLORS = {
    "retinaface": "#2196F3",
    "mtcnn": "#FF9800",
    "detected": "#4CAF50",
    "undetected": "#F44336",
    "false_positive": "#9C27B0",
}

CAT_LABELS = {
    "detected": "Detected (TP)",
    "undetected": "Undetected (FN)",
    "false_positive": "False Positive (FP)",
}

# Human-readable axis labels with units
METRIC_LABELS = {
    "face_area": "Face Bounding-Box Area (pixels²)",
    "face_brightness": "Face Brightness — mean HSV V-channel (0 = black, 255 = white)",
    "image_brightness": "Image Brightness — mean HSV V-channel (0 = black, 255 = white)",
    "image_blur": "Image Sharpness — Variance of Laplacian (higher = sharper)",
    "confidence": "Detection Confidence Score (0.0 – 1.0)",
    "distance": "Euclidean Distance Between Face Centres (pixels)",
}


def _clean_name(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def _thousands_fmt(x, _):
    """Format large numbers with comma separators."""
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.0f}"


# ──────────────────────────────────────────────────────────────
# 1. Metrics comparison bar chart
# ──────────────────────────────────────────────────────────────
def plot_metrics_comparison(all_outputs: dict, save_dir: Path = None):
    """
    Grouped bar chart: Precision / Recall / F1 / Mean IoU for each detector.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    detectors = list(all_outputs.keys())
    metric_keys = ["precision", "recall", "f1", "mean_iou"]
    metric_display = ["Precision", "Recall", "F1 Score", "Mean IoU"]

    data = {}
    for det in detectors:
        m = all_outputs[det]["aggregate_metrics"]
        data[det] = [m.precision, m.recall, m.f1, m.mean_iou]

    x = np.arange(len(metric_keys))
    n = len(detectors)
    width = 0.7 / max(n, 1)

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, det in enumerate(detectors):
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, data[det], width,
            label=det.upper(),
            color=COLORS.get(det, f"C{i}"),
            edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, data[det]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title("Face Detection — Performance Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_display)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Annotation
    ax.annotate(
        "Higher is better for all metrics",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=8, color="gray", style="italic",
    )

    fig.tight_layout()
    fig.savefig(save_dir / "metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 2. Group-wise performance degradation
# ──────────────────────────────────────────────────────────────
def plot_groupwise(all_outputs: dict, save_dir: Path = None):
    """
    Three sub-plots: Recall, Precision, F1 by face-count bin.
    Shows how crowded images degrade performance.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    metrics = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1 Score")]

    for ax, (col, label) in zip(axes, metrics):
        for det_name, output in all_outputs.items():
            gw = output["groupwise_df"]
            color = COLORS.get(det_name, "gray")
            ax.plot(
                gw["bin"], gw[col], "o-",
                label=det_name.upper(), color=color, linewidth=2, markersize=7,
            )
            # Value annotations on each point
            for xi, yi in zip(gw["bin"], gw[col]):
                ax.annotate(
                    f"{yi:.2f}", (xi, yi),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=color,
                )

        ax.set_title(f"{label} by Face Density")
        ax.set_xlabel("Number of Ground-Truth Faces per Image")
        ax.set_ylabel(f"{label} (0.0 – 1.0)")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "Detection Performance vs. Image Crowdedness\n"
        "(more faces per image → harder detection)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_dir / "groupwise_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 3. Quality distributions (box plots with units)
# ──────────────────────────────────────────────────────────────
def plot_quality_distributions(quality_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Box plots of face_area, face_brightness, image_brightness, image_blur
    split by category (Detected / Undetected / False Positive).
    Each axis has proper units and explanatory labels.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    metrics = ["face_area", "face_brightness", "image_brightness", "image_blur"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    categories = ["detected", "undetected", "false_positive"]

    for ax, metric in zip(axes.flat, metrics):
        data_to_plot = []
        labels_used = []
        colors_used = []
        medians = []

        for cat in categories:
            subset = quality_df[quality_df["category"] == cat][metric].dropna()
            if len(subset) > 0:
                data_to_plot.append(subset.values)
                labels_used.append(CAT_LABELS.get(cat, cat))
                colors_used.append(COLORS.get(cat, "gray"))
                medians.append(subset.median())

        if data_to_plot:
            bp = ax.boxplot(
                data_to_plot,
                labels=labels_used,
                patch_artist=True,
                showfliers=False,  # hide outliers for cleaner look
                widths=0.5,
                medianprops=dict(color="black", linewidth=2),
            )
            for patch, col in zip(bp["boxes"], colors_used):
                patch.set_facecolor(col)
                patch.set_alpha(0.65)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.8)

            # Annotate medians
            for j, med in enumerate(medians):
                fmt = f"{med:,.0f}" if med >= 100 else f"{med:.1f}"
                ax.text(
                    j + 1, med, f"  median={fmt}",
                    va="center", ha="left", fontsize=8,
                    color="black", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
                )

        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(METRIC_LABELS.get(metric, metric).split("—")[0].strip(), fontsize=12)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.yaxis.set_major_formatter(FuncFormatter(_thousands_fmt))

    fig.suptitle(
        f"Image Quality Analysis by Detection Outcome — {detector_name}\n"
        "(comparing quality features of detected vs. missed vs. false-positive faces)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(save_dir / f"quality_distributions_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 4. Face size histogram — detected vs missed
# ──────────────────────────────────────────────────────────────
def plot_face_size_histogram(quality_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Overlapping histograms showing the face-area distribution for
    detected vs undetected faces. Reveals the small-face blind spot.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in ["detected", "undetected"]:
        subset = quality_df[quality_df["category"] == cat]["face_area"]
        if subset.empty:
            continue
        # Clip to reasonable range for visibility
        clipped = subset.clip(upper=10000)
        ax.hist(
            clipped, bins=80, alpha=0.55,
            label=f"{CAT_LABELS[cat]}  (n={len(subset):,})",
            color=COLORS[cat], edgecolor="white", linewidth=0.3,
        )

    # Reference lines for intuitive size understanding
    size_refs = {256: "16×16 px", 1024: "32×32 px", 4096: "64×64 px"}
    for area, label in size_refs.items():
        ax.axvline(area, color="gray", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(area + 50, ax.get_ylim()[1] * 0.92, label,
                fontsize=8, color="gray", rotation=0)

    ax.set_xlabel("Face Bounding-Box Area (pixels²)\n"
                  "[e.g. 256 px² = 16×16 pixel face; 4096 px² = 64×64 pixel face]")
    ax.set_ylabel("Number of Faces")
    ax.set_title(f"Face Size Distribution: Detected vs. Missed — {detector_name}")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.xaxis.set_major_formatter(FuncFormatter(_thousands_fmt))

    fig.tight_layout()
    fig.savefig(save_dir / f"face_size_histogram_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 5. Brightness vs detection outcome scatter
# ──────────────────────────────────────────────────────────────
def plot_brightness_vs_size(quality_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Scatter plot: face brightness (x) vs face area (y), coloured by outcome.
    Reveals if dark + small = highest miss rate.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(12, 7))

    for cat in ["detected", "undetected", "false_positive"]:
        subset = quality_df[quality_df["category"] == cat]
        if subset.empty:
            continue
        ax.scatter(
            subset["face_brightness"],
            subset["face_area"].clip(upper=15000),
            alpha=0.25, s=12,
            color=COLORS[cat],
            label=f"{CAT_LABELS[cat]}  (n={len(subset):,})",
        )

    ax.set_xlabel(METRIC_LABELS["face_brightness"])
    ax.set_ylabel("Face Bounding-Box Area (pixels²)\n[clipped at 15,000 for visibility]")
    ax.set_title(f"Face Brightness vs. Face Size by Detection Outcome — {detector_name}")
    ax.legend(loc="upper right", markerscale=3, fontsize=9)
    ax.grid(alpha=0.2, linestyle="--")
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands_fmt))

    # Annotate danger zone
    ax.axhspan(0, 1024, alpha=0.06, color="red")
    ax.text(5, 800, "← Small-face danger zone (< 32×32 px)",
            fontsize=8, color="red", style="italic")

    fig.tight_layout()
    fig.savefig(save_dir / f"brightness_vs_size_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 6. Distance analysis histograms
# ──────────────────────────────────────────────────────────────
def plot_distance_histograms(distance_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Stacked histograms of pairwise face-centre distances by pair type.
    Short detected↔undetected distance → NMS / occlusion caused the miss.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    if distance_df.empty:
        return

    PAIR_COLORS = {
        "detected-detected": "#4CAF50",
        "detected-undetected": "#FF9800",
        "undetected-undetected": "#F44336",
    }
    PAIR_LABELS = {
        "detected-detected": "Detected ↔ Detected",
        "detected-undetected": "Detected ↔ Undetected  (possible NMS suppression)",
        "undetected-undetected": "Undetected ↔ Undetected  (clustered misses)",
    }

    pair_types = sorted(distance_df["pair_type"].unique())
    fig, ax = plt.subplots(figsize=(13, 6))

    for pt in pair_types:
        subset = distance_df[distance_df["pair_type"] == pt]["distance"]
        med = subset.median()
        ax.hist(
            subset, bins=60, alpha=0.45,
            label=f"{PAIR_LABELS.get(pt, pt)}  (n={len(subset):,}, median={med:.0f} px)",
            color=PAIR_COLORS.get(pt, "gray"),
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xlabel(METRIC_LABELS["distance"])
    ax.set_ylabel("Number of Face Pairs")
    ax.set_title(
        f"Pairwise Face-Centre Distance Distribution — {detector_name}\n"
        "(short Det↔Undet distance suggests NMS or occlusion caused the miss)"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_dir / f"distance_histogram_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 7. Confidence distribution
# ──────────────────────────────────────────────────────────────
def plot_confidence_distribution(quality_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Histogram of detection confidence for TP vs FP.
    Helps decide if the confidence threshold should be adjusted.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(12, 6))

    for cat in ["detected", "false_positive"]:
        subset = quality_df[quality_df["category"] == cat]["confidence"]
        subset = subset[subset >= 0]
        if subset.empty:
            continue
        ax.hist(
            subset, bins=50, alpha=0.55,
            label=f"{CAT_LABELS[cat]}  (n={len(subset):,}, mean={subset.mean():.3f})",
            color=COLORS[cat], edgecolor="white", linewidth=0.3,
        )

    ax.set_xlabel(METRIC_LABELS["confidence"])
    ax.set_ylabel("Number of Detections")
    ax.set_title(f"Detection Confidence Distribution (TP vs FP) — {detector_name}")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_dir / f"confidence_distribution_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 8. Detection overlay on sample images
# ──────────────────────────────────────────────────────────────
def visualize_detections(
    image: np.ndarray,
    eval_result: ImageEvalResult,
    save_path: Path = None,
):
    """
    Draw GT and predicted boxes on an image with color coding.
    Green = Detected (TP), Red = Undetected (FN), Purple = False Positive (FP).
    """
    vis = image.copy()

    # Detected GT boxes (green)
    for gt_idx in eval_result.detected_gt_indices:
        box = eval_result.gt_boxes[gt_idx].astype(int)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(vis, "DET", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Undetected GT boxes (red)
    for gt_idx in eval_result.undetected_gt_indices:
        box = eval_result.gt_boxes[gt_idx].astype(int)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(vis, "MISS", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # False positive boxes (purple)
    for pred_idx in eval_result.false_positive_indices:
        box = eval_result.pred_boxes[pred_idx].astype(int)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
        cv2.putText(vis, "FP", (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), vis)

    return vis


# ──────────────────────────────────────────────────────────────
# 9. Attribute-based recall analysis
# ──────────────────────────────────────────────────────────────
def plot_attribute_recall(attribute_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Grouped bar chart showing recall by WIDER FACE attribute level.
    One subplot per attribute (blur, occlusion, illumination, expression, pose).
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    attributes = attribute_df["attribute"].unique()
    n_attrs = len(attributes)
    fig, axes = plt.subplots(1, n_attrs, figsize=(4 * n_attrs, 6), squeeze=False)
    axes = axes[0]

    attr_colors = {
        "blur": ["#4CAF50", "#FF9800", "#F44336"],
        "occlusion": ["#4CAF50", "#FF9800", "#F44336"],
        "illumination": ["#4CAF50", "#F44336"],
        "expression": ["#4CAF50", "#FF9800"],
        "pose": ["#4CAF50", "#FF9800"],
    }

    for ax, attr in zip(axes, attributes):
        subset = attribute_df[attribute_df["attribute"] == attr]
        colors = attr_colors.get(attr, ["#2196F3"] * len(subset))
        bars = ax.bar(
            subset["label"], subset["recall"],
            color=colors[:len(subset)], edgecolor="white", linewidth=0.8,
        )
        for bar, val, total in zip(bars, subset["recall"], subset["total"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}\n(n={total:,})",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
        ax.set_title(attr.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel("Recall (0.0 – 1.0)")
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"Detection Recall by Face Attribute — {detector_name}\n"
        "(WIDER FACE annotation attributes: blur, occlusion, illumination, expression, pose)",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(save_dir / f"attribute_recall_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 10. Event/scene category analysis
# ──────────────────────────────────────────────────────────────
def plot_event_analysis(event_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Horizontal bar chart of recall by WIDER FACE event category.
    Shows which scene types are hardest/easiest for the detector.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    if event_df.empty:
        return

    # Sort by recall for visual clarity (already sorted, but ensure)
    df = event_df.sort_values("recall", ascending=True).tail(40)  # top 40 if many

    fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.35)))

    colors = plt.cm.RdYlGn(df["recall"].values)
    bars = ax.barh(df["event"], df["recall"], color=colors, edgecolor="white", linewidth=0.5)

    for bar, recall, n_img, total_gt in zip(bars, df["recall"], df["num_images"], df["total_gt"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{recall:.3f}  ({n_img} imgs, {total_gt:,} faces)",
            va="center", fontsize=8,
        )

    ax.set_xlabel("Recall (0.0 – 1.0)")
    ax.set_title(
        f"Detection Recall by Scene/Event Category — {detector_name}\n"
        "(WIDER FACE event types — sorted worst to best)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_dir / f"event_analysis_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 11. Precision-Recall curve
# ──────────────────────────────────────────────────────────────
def plot_pr_curve(pr_df: pd.DataFrame, ap: float, detector_name: str, save_dir: Path = None):
    """
    Plot the Precision-Recall curve with AP annotation.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(pr_df["recall"], pr_df["precision"],
            color="#2196F3", linewidth=2, label=f"{detector_name} (AP={ap:.4f})")
    ax.fill_between(pr_df["recall"], pr_df["precision"], alpha=0.15, color="#2196F3")

    ax.set_xlabel("Recall (0.0 – 1.0)\n[fraction of real faces found]")
    ax.set_ylabel("Precision (0.0 – 1.0)\n[fraction of detections that are correct]")
    ax.set_title(
        f"Precision-Recall Curve — {detector_name}\n"
        f"Average Precision (AP) = {ap:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")

    # F1 iso-lines
    for f1_val in [0.2, 0.4, 0.6, 0.8]:
        r = np.linspace(0.01, 1, 100)
        p = f1_val * r / (2 * r - f1_val)
        p[p < 0] = np.nan
        p[p > 1] = np.nan
        ax.plot(r, p, "--", color="gray", alpha=0.3, linewidth=0.8)
        valid = ~np.isnan(p)
        if valid.any():
            idx = np.where(valid)[0][-1]
            ax.annotate(f"F1={f1_val}", (r[idx], p[idx]),
                        fontsize=7, color="gray", alpha=0.6)

    fig.tight_layout()
    fig.savefig(save_dir / f"pr_curve_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 12. Confidence threshold sensitivity
# ──────────────────────────────────────────────────────────────
def plot_threshold_sensitivity(threshold_df: pd.DataFrame, detector_name: str, save_dir: Path = None):
    """
    Line chart: Precision, Recall, F1 as functions of confidence threshold.
    Helps choose the optimal operating point.
    """
    save_dir = save_dir or PLOTS_DIR
    ensure_dirs()

    fig, ax = plt.subplots(figsize=(10, 6))

    for col, color, marker in [
        ("precision", "#2196F3", "o"),
        ("recall", "#F44336", "s"),
        ("f1", "#4CAF50", "D"),
    ]:
        ax.plot(
            threshold_df["threshold"], threshold_df[col],
            f"-{marker}", color=color, linewidth=2, markersize=7,
            label=col.capitalize(),
        )
        # Annotate each point
        for x, y in zip(threshold_df["threshold"], threshold_df[col]):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color=color)

    # Mark the F1-optimal threshold
    best_idx = threshold_df["f1"].idxmax()
    best_t = threshold_df.loc[best_idx, "threshold"]
    best_f1 = threshold_df.loc[best_idx, "f1"]
    ax.axvline(best_t, color="green", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Best F1={best_f1:.3f}\nat thresh={best_t}",
        (best_t, best_f1), textcoords="offset points",
        xytext=(15, -20), fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="green"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score (0.0 – 1.0)")
    ax.set_title(
        f"Confidence Threshold Sensitivity — {detector_name}\n"
        "(how performance changes as we vary the detection confidence cutoff)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="center left", fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_dir / f"threshold_sensitivity_{_clean_name(detector_name)}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Master function
# ──────────────────────────────────────────────────────────────
def generate_all_plots(all_outputs: dict):
    """Generate every plot from pipeline outputs and save to PLOTS_DIR."""
    ensure_dirs()

    # 1. Metrics comparison
    plot_metrics_comparison(all_outputs)

    # 2. Group-wise degradation
    plot_groupwise(all_outputs)

    # 3. Per-detector plots
    for det_name, output in all_outputs.items():
        quality_df = output.get("quality_df")
        if quality_df is not None and not quality_df.empty:
            plot_quality_distributions(quality_df, det_name)
            plot_face_size_histogram(quality_df, det_name)
            plot_brightness_vs_size(quality_df, det_name)
            plot_confidence_distribution(quality_df, det_name)

        distance_df = output.get("distance_df")
        if distance_df is not None and not distance_df.empty:
            plot_distance_histograms(distance_df, det_name)

        # New analysis plots
        attribute_df = output.get("attribute_df")
        if attribute_df is not None and not attribute_df.empty:
            plot_attribute_recall(attribute_df, det_name)

        event_df = output.get("event_df")
        if event_df is not None and not event_df.empty:
            plot_event_analysis(event_df, det_name)

        pr_df = output.get("pr_df")
        ap = output.get("ap", 0.0)
        if pr_df is not None and not pr_df.empty:
            plot_pr_curve(pr_df, ap, det_name)

        threshold_df = output.get("threshold_df")
        if threshold_df is not None and not threshold_df.empty:
            plot_threshold_sensitivity(threshold_df, det_name)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
