"""
SCRFD Detector — InsightFace's newer, more accurate face detection model.

SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
redistributes computation towards harder detection scales (small/medium faces),
achieving better accuracy than RetinaFace on WIDER FACE Hard subset.

Published benchmarks (WIDER FACE Val AP):
  SCRFD-34GF: Easy=96.06, Medium=94.92, Hard=89.29
  SCRFD-10GF: Easy=95.16, Medium=93.87, Hard=83.05
  SCRFD-2.5GF: Easy=93.78, Medium=92.16, Hard=77.87

This wrapper uses the same insightface model_zoo API as the existing
RetinaFace detector, making it a drop-in replacement.
"""

import os
import numpy as np

from src.detectors.base import BaseDetector, DetectionResult
from src.utils import logger


class SCRFDDetector(BaseDetector):
    """SCRFD face detector via insightface model_zoo."""

    # Available SCRFD models in insightface (ordered by compute)
    AVAILABLE_MODELS = {
        "scrfd_500m": "buffalo_sc",        # 500M FLOPs — ultra-light
        "scrfd_2.5g": "buffalo_s",         # 2.5G FLOPs — light
        "scrfd_10g": "buffalo_l",          # 10G FLOPs — balanced (default)
        "scrfd_34g": None,                 # 34G FLOPs — highest accuracy (separate download)
    }

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        det_size: int = 640,
        model_name: str = "scrfd_10g",
    ):
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._det_size = det_size
        self._model_name = model_name
        self._det_model = None
        self._load_model()

    def _load_model(self):
        """Load SCRFD model from insightface model zoo."""
        try:
            from insightface.model_zoo import get_model
        except ImportError:
            raise ImportError("insightface is required: pip install insightface")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )

        home = os.path.expanduser("~")

        # Strategy 1: Try loading specific SCRFD ONNX file
        scrfd_paths = [
            os.path.join(home, ".insightface", "models", "buffalo_l", "det_10g.onnx"),
            os.path.join(home, ".insightface", "models", "buffalo_sc", "det_500m.onnx"),
        ]

        # Try to find a SCRFD-specific model first
        scrfd_specific = os.path.join(
            home, ".insightface", "models", f"{self._model_name}.onnx"
        )
        if os.path.exists(scrfd_specific):
            scrfd_paths.insert(0, scrfd_specific)

        # Also check the buffalo_l directory for SCRFD models
        buffalo_dir = os.path.join(home, ".insightface", "models", "buffalo_l")
        if os.path.isdir(buffalo_dir):
            for f in os.listdir(buffalo_dir):
                if f.startswith("det_") and f.endswith(".onnx"):
                    path = os.path.join(buffalo_dir, f)
                    if path not in scrfd_paths:
                        scrfd_paths.insert(0, path)

        for model_path in scrfd_paths:
            if os.path.exists(model_path):
                try:
                    det_model = get_model(model_path, providers=providers)
                    det_model.prepare(
                        ctx_id=0 if self._device == "cuda" else -1,
                        input_size=(self._det_size, self._det_size),
                        det_thresh=self._confidence_threshold,
                    )
                    self._det_model = det_model
                    self._actual_model_path = model_path
                    logger.info(
                        f"SCRFD loaded: {os.path.basename(model_path)} "
                        f"(det_size={self._det_size}, device={self._device})"
                    )
                    return
                except Exception as e:
                    logger.debug(f"Failed to load {model_path}: {e}")
                    continue

        # Fallback: download via FaceAnalysis
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(
                ctx_id=0 if self._device == "cuda" else -1,
                det_size=(self._det_size, self._det_size),
            )
            self._det_model = app.det_model
            self._det_model.det_thresh = self._confidence_threshold
            self._actual_model_path = "buffalo_l (FaceAnalysis)"
            logger.info(f"SCRFD loaded via FaceAnalysis fallback (det_size={self._det_size})")
            return
        except Exception as e:
            raise RuntimeError(f"Could not load any SCRFD/detection model: {e}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run SCRFD detection on a BGR image."""
        bboxes, kpss = self._det_model.detect(image)

        if bboxes is None or len(bboxes) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        # bboxes shape: (N, 5) — [x1, y1, x2, y2, score]
        scores = bboxes[:, 4].astype(np.float32)
        boxes = bboxes[:, :4].astype(np.float32)

        landmarks = None
        if kpss is not None and len(kpss) > 0:
            landmarks = kpss.astype(np.float32)

        return DetectionResult(boxes=boxes, scores=scores, landmarks=landmarks)

    @property
    def name(self) -> str:
        return f"SCRFD ({self._model_name})"

    @property
    def model_info(self) -> dict:
        """Return model metadata for logging."""
        return {
            "model_name": self._model_name,
            "model_path": getattr(self, "_actual_model_path", "unknown"),
            "det_size": self._det_size,
            "confidence_threshold": self._confidence_threshold,
            "device": self._device,
        }
