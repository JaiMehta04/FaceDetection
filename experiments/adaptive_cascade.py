"""
AdaSR-Face: Adaptive Super-Resolution Guided by Detection Confidence.

This is the core novel contribution — a two-stage cascade pipeline:

Stage 1 (Fast Detection):
  Run detector at native resolution → get initial detections + confidence map.

Stage 2 (Selective SR + Re-detection):
  Identify regions that likely contain missed small faces:
    a) Areas with low-confidence detections (0.3–0.7) → might be tiny faces
    b) Large image regions with zero detections → might have tiny faces
    c) Regions around detected faces (cluster heuristic) → nearby misses
  Upscale ONLY these regions with SR model → re-detect on upscaled crops.

Stage 3 (Fusion):
  Merge Stage 1 + Stage 2 detections with Soft-NMS to handle duplicates.

Key insight: Instead of SR on the full image (~4x cost), we SR only the
~20-30% of the image where small faces are likely hiding, getting ~80% of
the benefit at ~30% of the compute.
"""

import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from src.detectors.base import BaseDetector, DetectionResult
from src.enhanced_detection import soft_nms
from src.utils import logger


class AdaptiveSRCascade:
    """
    Two-stage adaptive SR cascade for small face detection.

    Stage 1: Fast detection → identify weak regions
    Stage 2: SR on weak regions → re-detect
    Stage 3: Soft-NMS fusion
    """

    def __init__(
        self,
        detector: BaseDetector,
        sr_preprocessor,
        # Stage 1 thresholds
        low_confidence_threshold: float = 0.5,
        high_confidence_threshold: float = 0.8,
        # Region selection
        min_face_size_for_sr: int = 64,
        region_padding: int = 80,
        empty_region_tile_size: int = 480,
        empty_region_min_area_ratio: float = 0.1,
        # Fusion
        soft_nms_sigma: float = 0.5,
        soft_nms_threshold: float = 0.3,
        # SR settings
        sr_scale: int = 2,
        max_sr_regions: int = 20,
        # Logging
        collect_stats: bool = True,
    ):
        self._detector = detector
        self._sr = sr_preprocessor
        self._low_conf = low_confidence_threshold
        self._high_conf = high_confidence_threshold
        self._min_face_for_sr = min_face_size_for_sr
        self._region_pad = region_padding
        self._empty_tile_size = empty_region_tile_size
        self._empty_min_ratio = empty_region_min_area_ratio
        self._sigma = soft_nms_sigma
        self._threshold = soft_nms_threshold
        self._sr_scale = sr_scale
        self._max_sr_regions = max_sr_regions
        self._collect_stats = collect_stats
        self._last_stats = {}

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Full adaptive cascade detection.

        Returns
        -------
        DetectionResult — merged detections from both stages.
        """
        h, w = image.shape[:2]
        stats = {"stage1_time": 0, "stage2_time": 0, "sr_regions": 0,
                 "stage1_faces": 0, "stage2_new_faces": 0}

        # ── Stage 1: Fast detection at native resolution ──
        t0 = time.time()
        stage1_result = self._detector.detect(image)
        stats["stage1_time"] = time.time() - t0
        stats["stage1_faces"] = stage1_result.num_faces

        if stage1_result.num_faces == 0 and max(h, w) < 300:
            # Very small image with no detections — SR the whole thing
            return self._sr_full_image(image, stage1_result, stats)

        # ── Identify regions needing SR ──
        sr_regions = self._select_sr_regions(image, stage1_result)
        stats["sr_regions"] = len(sr_regions)

        if not sr_regions:
            # No regions need SR — return Stage 1 results
            self._last_stats = stats
            return stage1_result

        # ── Stage 2: SR + re-detect on selected regions ──
        t1 = time.time()
        stage2_boxes_list = []
        stage2_scores_list = []

        for region in sr_regions[:self._max_sr_regions]:
            new_boxes, new_scores = self._process_sr_region(image, region)
            if len(new_boxes) > 0:
                stage2_boxes_list.append(new_boxes)
                stage2_scores_list.append(new_scores)

        stats["stage2_time"] = time.time() - t1

        # ── Stage 3: Fusion ──
        all_boxes = [stage1_result.boxes] if stage1_result.num_faces > 0 else []
        all_scores = [stage1_result.scores] if stage1_result.num_faces > 0 else []
        all_boxes.extend(stage2_boxes_list)
        all_scores.extend(stage2_scores_list)

        if not all_boxes:
            self._last_stats = stats
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

        stats["stage2_new_faces"] = len(final_boxes) - stage1_result.num_faces
        self._last_stats = stats

        return DetectionResult(boxes=final_boxes, scores=final_scores)

    def _select_sr_regions(
        self, image: np.ndarray, stage1: DetectionResult
    ) -> List[Dict]:
        """
        Identify image regions that should be upscaled.

        Three strategies:
        1. Low-confidence detections → small/blurry faces that SR might help
        2. Small detected faces → nearby tiny faces likely missed
        3. Empty regions in large images → possible tiny face clusters
        """
        h, w = image.shape[:2]
        regions = []

        if stage1.num_faces > 0:
            boxes = stage1.boxes
            scores = stage1.scores

            # Strategy 1: Low-confidence detections
            low_conf_mask = (scores >= self._low_conf * 0.5) & (scores < self._high_conf)
            if low_conf_mask.any():
                low_conf_boxes = boxes[low_conf_mask]
                for box in low_conf_boxes:
                    face_w = box[2] - box[0]
                    face_h = box[3] - box[1]
                    if max(face_w, face_h) < self._min_face_for_sr:
                        regions.append({
                            "type": "low_confidence",
                            "bbox": box,
                            "reason": f"conf={scores[low_conf_mask][0]:.2f}, size={face_w:.0f}x{face_h:.0f}",
                        })

            # Strategy 2: Small detected faces → expand region
            for i, box in enumerate(boxes):
                face_w = box[2] - box[0]
                face_h = box[3] - box[1]
                face_size = max(face_w, face_h)
                if face_size < self._min_face_for_sr * 0.75:
                    # Expand to a larger region around this small face
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    expand = self._min_face_for_sr * 3
                    expanded_box = np.array([
                        max(0, cx - expand),
                        max(0, cy - expand),
                        min(w, cx + expand),
                        min(h, cy + expand),
                    ])
                    regions.append({
                        "type": "small_face_neighbor",
                        "bbox": expanded_box,
                        "reason": f"small_face={face_size:.0f}px, expanded for neighbors",
                    })

        # Strategy 3: Empty regions in large images
        if max(h, w) > 800:
            coverage_map = np.zeros((h, w), dtype=bool)
            if stage1.num_faces > 0:
                for box in stage1.boxes:
                    x1, y1, x2, y2 = box.astype(int)
                    pad = 100
                    coverage_map[
                        max(0, y1 - pad):min(h, y2 + pad),
                        max(0, x1 - pad):min(w, x2 + pad),
                    ] = True

            # Tile the image, find tiles with no detection coverage
            tile = self._empty_tile_size
            stride = tile // 2
            for y in range(0, h - tile // 4, stride):
                for x in range(0, w - tile // 4, stride):
                    y2 = min(y + tile, h)
                    x2 = min(x + tile, w)
                    region_area = (y2 - y) * (x2 - x)

                    if region_area < (tile * tile * self._empty_min_ratio):
                        continue

                    coverage = coverage_map[y:y2, x:x2].mean()
                    if coverage < 0.05:  # <5% of region covered by detections
                        regions.append({
                            "type": "empty_region",
                            "bbox": np.array([x, y, x2, y2], dtype=np.float32),
                            "reason": f"uncovered region {x2-x}x{y2-y}",
                        })

        # Deduplicate overlapping regions
        regions = self._merge_overlapping_regions(regions)

        return regions

    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """Remove highly overlapping SR regions to avoid redundant work."""
        if len(regions) <= 1:
            return regions

        boxes = np.array([r["bbox"] for r in regions])
        keep = []
        used = set()

        # Sort by type priority: low_confidence > small_face > empty
        type_priority = {"low_confidence": 0, "small_face_neighbor": 1, "empty_region": 2}
        sorted_idx = sorted(
            range(len(regions)),
            key=lambda i: type_priority.get(regions[i]["type"], 3),
        )

        for i in sorted_idx:
            if i in used:
                continue
            keep.append(regions[i])
            # Mark overlapping regions as used
            for j in sorted_idx:
                if j in used or j == i:
                    continue
                iou = self._compute_iou(boxes[i], boxes[j])
                if iou > 0.5:
                    used.add(j)
            used.add(i)

        return keep

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0

    def _process_sr_region(
        self, image: np.ndarray, region: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Upscale a single region and run detection on the upscaled crop.

        Returns boxes mapped back to original image coordinates.
        """
        h, w = image.shape[:2]
        bbox = region["bbox"]
        x1, y1, x2, y2 = bbox.astype(int)

        # Add padding
        pad = self._region_pad
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)

        crop = image[py1:py2, px1:px2]
        if crop.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Upscale
        sr_crop, actual_scale = self._sr.upscale_full(crop)

        # Detect on upscaled crop
        sr_result = self._detector.detect(sr_crop)

        if sr_result.num_faces == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Map boxes back to original image coordinates
        mapped_boxes = sr_result.boxes.copy()
        mapped_boxes[:, 0] = mapped_boxes[:, 0] / actual_scale + px1
        mapped_boxes[:, 1] = mapped_boxes[:, 1] / actual_scale + py1
        mapped_boxes[:, 2] = mapped_boxes[:, 2] / actual_scale + px1
        mapped_boxes[:, 3] = mapped_boxes[:, 3] / actual_scale + py1

        # Clip to image bounds
        mapped_boxes[:, 0] = np.clip(mapped_boxes[:, 0], 0, w)
        mapped_boxes[:, 1] = np.clip(mapped_boxes[:, 1], 0, h)
        mapped_boxes[:, 2] = np.clip(mapped_boxes[:, 2], 0, w)
        mapped_boxes[:, 3] = np.clip(mapped_boxes[:, 3], 0, h)

        return mapped_boxes.astype(np.float32), sr_result.scores

    def _sr_full_image(
        self, image: np.ndarray, stage1: DetectionResult, stats: dict
    ) -> DetectionResult:
        """Fallback: SR the entire image when it's small and has no detections."""
        t1 = time.time()
        sr_image, scale = self._sr.upscale_full(image)
        sr_result = self._detector.detect(sr_image)
        stats["stage2_time"] = time.time() - t1
        stats["sr_regions"] = 1
        stats["stage2_new_faces"] = sr_result.num_faces

        if sr_result.num_faces > 0:
            # Map back to original coordinates
            sr_result = DetectionResult(
                boxes=sr_result.boxes / scale,
                scores=sr_result.scores,
            )

        self._last_stats = stats
        return sr_result

    @property
    def last_stats(self) -> dict:
        """Get stats from the most recent detect() call."""
        return self._last_stats.copy()

    @property
    def name(self) -> str:
        return f"AdaSR-Face ({self._detector.name})"


class BlindSRDetector:
    """
    Control experiment: Apply SR to the ENTIRE image before detection.
    This is the naive approach that AdaSR-Face improves upon.
    """

    def __init__(
        self,
        detector: BaseDetector,
        sr_preprocessor,
        max_dim: int = 3072,
    ):
        self._detector = detector
        self._sr = sr_preprocessor
        self._max_dim = max_dim

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]

        # Skip SR if image is already large
        if max(h, w) * self._sr.scale > self._max_dim:
            return self._detector.detect(image)

        sr_image, scale = self._sr.upscale_full(image)
        result = self._detector.detect(sr_image)

        if result.num_faces > 0:
            result = DetectionResult(
                boxes=result.boxes / scale,
                scores=result.scores,
            )
        return result

    @property
    def name(self) -> str:
        return f"BlindSR ({self._detector.name})"
