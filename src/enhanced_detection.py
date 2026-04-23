"""
Advanced detection strategies for improved accuracy.

Implements:
1. Multi-scale inference — run detection at multiple image scales and merge
2. Soft-NMS — suppress overlapping boxes with Gaussian decay instead of hard removal
3. Test-Time Augmentation (TTA) — horizontal flip + merge
4. Ensemble detector — combine multiple detectors and merge results

These are wrappers that take any BaseDetector and augment it.

Key Insight from baseline analysis:
  Missed faces have mean area ~300px² while detected faces average ~1800px².
  Multi-scale inference directly addresses this by upscaling the image
  so small faces become large enough for the detector's receptive field.
"""

from typing import List, Tuple, Optional

import cv2
import numpy as np

from src.detectors.base import BaseDetector, DetectionResult
from src.utils import compute_iou_matrix, logger


# ──────────────────────────────────────────────────────────────
# Soft-NMS
# ──────────────────────────────────────────────────────────────
def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.3,
    method: str = "gaussian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Soft-NMS: instead of removing overlapping boxes entirely, decay their
    scores by a Gaussian function of IoU overlap.

    Why this helps:
    - Hard NMS with IoU > 0.3 often kills valid detections in crowded scenes
      where faces genuinely overlap.
    - Soft-NMS preserves overlapping faces by reducing (not zeroing) their
      confidence, then filtering by a lower threshold.

    Parameters
    ----------
    boxes   : (N, 4) [x1, y1, x2, y2]
    scores  : (N,) confidence scores
    sigma   : Gaussian decay parameter (lower = more aggressive suppression)
    score_threshold : minimum score to keep after decay
    method  : 'gaussian' or 'linear'

    Returns
    -------
    (filtered_boxes, filtered_scores)

    Reference: Bodla et al., "Soft-NMS", ICCV 2017
    """
    if len(boxes) == 0:
        return boxes, scores

    N = len(boxes)
    indices = np.arange(N)
    # Work with copies
    _boxes = boxes.copy()
    _scores = scores.copy().astype(np.float64)

    kept_boxes = []
    kept_scores = []

    while len(indices) > 0:
        # Pick the highest scoring box
        max_idx = np.argmax(_scores[indices])
        max_pos = indices[max_idx]

        kept_boxes.append(_boxes[max_pos])
        kept_scores.append(_scores[max_pos])

        # Remove this box from candidates
        remaining = np.delete(indices, max_idx)

        if len(remaining) == 0:
            break

        # Compute IoU between the selected box and remaining boxes
        ious = compute_iou_matrix(
            _boxes[max_pos:max_pos+1], _boxes[remaining]
        )[0]  # shape (len(remaining),)

        # Decay scores
        if method == "gaussian":
            decay = np.exp(-(ious ** 2) / sigma)
        else:  # linear
            decay = np.where(ious > 0.3, 1.0 - ious, 1.0)

        _scores[remaining] *= decay

        # Keep only above threshold
        keep_mask = _scores[remaining] >= score_threshold
        indices = remaining[keep_mask]

    if not kept_boxes:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    return (
        np.stack(kept_boxes).astype(np.float32),
        np.array(kept_scores, dtype=np.float32),
    )


# ──────────────────────────────────────────────────────────────
# Multi-Scale Inference
# ──────────────────────────────────────────────────────────────
class MultiScaleDetector(BaseDetector):
    """
    Run a base detector at multiple image scales and merge results.

    Why this helps:
    - Face detectors have a limited receptive field size.
    - A face that is 20x20 pixels at the original scale becomes 40x40
      when the image is upscaled 2x — now within the detector's range.
    - Conversely, downscaling helps detect very large faces faster.

    Scale strategy:
    - scales=[0.5, 1.0, 1.5, 2.0] covers a wide range
    - Each scale's detections are mapped back to original coordinates
    - Merged with Soft-NMS to remove duplicates
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        scales: List[float] = None,
        soft_nms_sigma: float = 0.5,
        soft_nms_threshold: float = 0.3,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
        max_dim: int = 2048,
    ):
        self._base = base_detector
        self._scales = scales or [0.75, 1.0, 1.5]
        self._sigma = soft_nms_sigma
        self._threshold = soft_nms_threshold
        self._max_dim = max_dim

    def detect(self, image: np.ndarray) -> DetectionResult:
        all_boxes = []
        all_scores = []

        h, w = image.shape[:2]

        for scale in self._scales:
            # Resize image
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w < 32 or new_h < 32:
                continue
            # Cap max dimension to avoid OOM / extreme slowness on CPU
            if max(new_w, new_h) > self._max_dim:
                cap_scale = self._max_dim / max(new_w, new_h)
                new_w = int(new_w * cap_scale)
                new_h = int(new_h * cap_scale)
                scale = scale * cap_scale  # adjust effective scale

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Detect on resized image
            result = self._base.detect(resized)

            if result.num_faces > 0:
                # Map boxes back to original coordinates
                boxes_orig = result.boxes / scale
                all_boxes.append(boxes_orig)
                all_scores.append(result.scores)

        if not all_boxes:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)

        # Merge with Soft-NMS
        final_boxes, final_scores = soft_nms(
            merged_boxes, merged_scores,
            sigma=self._sigma,
            score_threshold=self._threshold,
        )

        return DetectionResult(boxes=final_boxes, scores=final_scores)

    @property
    def name(self) -> str:
        return f"{self._base.name} (MultiScale)"


# ──────────────────────────────────────────────────────────────
# Test-Time Augmentation (Horizontal Flip)
# ──────────────────────────────────────────────────────────────
class TTADetector(BaseDetector):
    """
    Test-Time Augmentation: detect on original + horizontally flipped image,
    then merge results with Soft-NMS.

    Why this helps:
    - Some faces in profile are only detected from one direction.
    - Asymmetric occlusion (e.g., hand covering left side) becomes
      visible from the other side after flipping.
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        soft_nms_sigma: float = 0.5,
        soft_nms_threshold: float = 0.3,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ):
        self._base = base_detector
        self._sigma = soft_nms_sigma
        self._threshold = soft_nms_threshold

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]

        # Original detection
        result_orig = self._base.detect(image)

        # Flipped detection
        flipped = cv2.flip(image, 1)  # horizontal flip
        result_flip = self._base.detect(flipped)

        # Mirror flip boxes back to original coordinate space
        if result_flip.num_faces > 0:
            flip_boxes = result_flip.boxes.copy()
            # x1_new = w - x2_old, x2_new = w - x1_old
            new_x1 = w - flip_boxes[:, 2]
            new_x2 = w - flip_boxes[:, 0]
            flip_boxes[:, 0] = new_x1
            flip_boxes[:, 2] = new_x2

            all_boxes = [result_orig.boxes, flip_boxes] if result_orig.num_faces > 0 else [flip_boxes]
            all_scores = [result_orig.scores, result_flip.scores] if result_orig.num_faces > 0 else [result_flip.scores]
        elif result_orig.num_faces > 0:
            return result_orig
        else:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)

        final_boxes, final_scores = soft_nms(
            merged_boxes, merged_scores,
            sigma=self._sigma,
            score_threshold=self._threshold,
        )

        return DetectionResult(boxes=final_boxes, scores=final_scores)

    @property
    def name(self) -> str:
        return f"{self._base.name} (TTA)"


# ──────────────────────────────────────────────────────────────
# Ensemble Detector
# ──────────────────────────────────────────────────────────────
class EnsembleDetector(BaseDetector):
    """
    Run multiple detectors and merge their outputs with Soft-NMS.

    Why this helps:
    - RetinaFace (single-stage + FPN) excels at small/medium faces.
    - MTCNN (cascade) is better at large faces with landmarks.
    - Their error patterns are partially uncorrelated, so the union
      catches faces that either alone would miss.
    """

    def __init__(
        self,
        detectors: List[BaseDetector],
        soft_nms_sigma: float = 0.5,
        soft_nms_threshold: float = 0.3,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ):
        self._detectors = detectors
        self._sigma = soft_nms_sigma
        self._threshold = soft_nms_threshold

    def detect(self, image: np.ndarray) -> DetectionResult:
        all_boxes = []
        all_scores = []

        for det in self._detectors:
            result = det.detect(image)
            if result.num_faces > 0:
                all_boxes.append(result.boxes)
                all_scores.append(result.scores)

        if not all_boxes:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)

        final_boxes, final_scores = soft_nms(
            merged_boxes, merged_scores,
            sigma=self._sigma,
            score_threshold=self._threshold,
        )

        return DetectionResult(boxes=final_boxes, scores=final_scores)

    @property
    def name(self) -> str:
        names = " + ".join(d.name for d in self._detectors)
        return f"Ensemble ({names})"


# ──────────────────────────────────────────────────────────────
# Tiled Inference
# ──────────────────────────────────────────────────────────────
class TiledDetector(BaseDetector):
    """
    Split large images into overlapping tiles, detect on each tile,
    then merge results.

    Why this helps:
    - In high-resolution images (e.g., 4000x3000), small faces
      disappear after resize-to-640.
    - Tiling preserves face resolution in each crop.
    - Overlap ensures faces on tile boundaries are still detected.
    """

    def __init__(
        self,
        base_detector: BaseDetector,
        tile_size: int = 640,
        overlap: float = 0.25,
        soft_nms_sigma: float = 0.5,
        soft_nms_threshold: float = 0.3,
        device: str = "cpu",
        confidence_threshold: float = 0.3,
    ):
        self._base = base_detector
        self._tile_size = tile_size
        self._overlap = overlap
        self._sigma = soft_nms_sigma
        self._threshold = soft_nms_threshold

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        tile = self._tile_size
        stride = int(tile * (1 - self._overlap))

        all_boxes = []
        all_scores = []

        # Also run on the full image (catches large faces)
        full_result = self._base.detect(image)
        if full_result.num_faces > 0:
            all_boxes.append(full_result.boxes)
            all_scores.append(full_result.scores)

        # Only tile if image is significantly larger than tile size
        if max(h, w) > tile * 1.5:
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x2 = min(x + tile, w)
                    y2 = min(y + tile, h)
                    # Skip tiny edge tiles
                    if (x2 - x) < tile * 0.5 or (y2 - y) < tile * 0.5:
                        continue

                    crop = image[y:y2, x:x2]
                    result = self._base.detect(crop)

                    if result.num_faces > 0:
                        # Offset boxes to global coordinates
                        offset_boxes = result.boxes.copy()
                        offset_boxes[:, 0] += x
                        offset_boxes[:, 1] += y
                        offset_boxes[:, 2] += x
                        offset_boxes[:, 3] += y
                        all_boxes.append(offset_boxes)
                        all_scores.append(result.scores)

        if not all_boxes:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        merged_boxes = np.concatenate(all_boxes, axis=0)
        merged_scores = np.concatenate(all_scores, axis=0)

        final_boxes, final_scores = soft_nms(
            merged_boxes, merged_scores,
            sigma=self._sigma,
            score_threshold=self._threshold,
        )

        return DetectionResult(boxes=final_boxes, scores=final_scores)

    @property
    def name(self) -> str:
        return f"{self._base.name} (Tiled)"
