"""
Shared utility functions used across the pipeline.

Responsibilities:
- Image I/O with validation
- IoU computation (vectorized)
- Bounding-box helpers
- Logging setup
- Device detection
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────
def setup_logger(name: str = "face_pipeline", level: int = logging.INFO) -> logging.Logger:
    """Configure a console + file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


logger = setup_logger()


# ──────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────
def get_device(preference: str = "auto") -> str:
    """Return 'cuda' if available (via PyTorch or ONNX Runtime), else 'cpu'."""
    if preference == "auto":
        # Check PyTorch CUDA first
        if torch.cuda.is_available():
            return "cuda"
        # Fallback: check ONNX Runtime CUDA provider (works even with CPU-only PyTorch)
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return preference


# ──────────────────────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────────────────────
def load_image(path: Path) -> Optional[np.ndarray]:
    """Load an image as BGR numpy array. Returns None on failure."""
    img = cv2.imread(str(path))
    if img is None:
        logger.warning(f"Failed to load image: {path}")
    return img


def resize_if_needed(image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """
    Resize image so that its longest side <= max_dim.
    Returns (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


# ──────────────────────────────────────────────────────────────
# Bounding-box utilities
# ──────────────────────────────────────────────────────────────
def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorized IoU between two sets of boxes.

    Parameters
    ----------
    boxes_a : (N, 4) array  [x1, y1, x2, y2]
    boxes_b : (M, 4) array  [x1, y1, x2, y2]

    Returns
    -------
    iou_matrix : (N, M) array
    """
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    # Broadcast: (N,1,4) vs (1,M,4)
    a = boxes_a[:, None, :]  # (N, 1, 4)
    b = boxes_b[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter_area

    iou = np.where(union > 0, inter_area / union, 0.0)
    return iou.astype(np.float32)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] → [x1, y1, x2, y2]."""
    out = boxes.copy().astype(np.float32)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Compute area of [x1, y1, x2, y2] boxes."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_center(boxes: np.ndarray) -> np.ndarray:
    """Return (cx, cy) for each [x1, y1, x2, y2] box."""
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    return np.stack([cx, cy], axis=1)


# ──────────────────────────────────────────────────────────────
# Image quality metrics
# ──────────────────────────────────────────────────────────────
def compute_blur(image: np.ndarray) -> float:
    """Variance of Laplacian — lower means blurrier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """Mean brightness in the V channel of HSV."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def crop_box(image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
    """
    Safely crop a bounding box [x1, y1, x2, y2] from an image.
    Clamps to image boundaries. Returns None if resulting crop is empty.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]
