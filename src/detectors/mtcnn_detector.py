"""
MTCNN detector wrapper.

Model:  Multi-task Cascaded Convolutional Networks
Stages: P-Net → R-Net → O-Net
Paper:  Zhang et al., 2016

Design Decisions:
- We use the `facenet-pytorch` MTCNN implementation (PyTorch).
  It's well-maintained, GPU-accelerated, and returns boxes + landmarks.
- The three cascade stages provide built-in hard-negative mining:
  P-Net proposes candidates → R-Net refines → O-Net outputs.
- We convert outputs to the standard [x1, y1, x2, y2] format.

Install: pip install facenet-pytorch
"""

import numpy as np
import cv2

from src.detectors.base import BaseDetector, DetectionResult
from src.utils import logger


class MTCNNDetector(BaseDetector):
    """Wrapper around facenet-pytorch MTCNN."""

    def __init__(self, device: str = "cpu", confidence_threshold: float = 0.5):
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._model = None
        self._load_model()

    def _load_model(self):
        try:
            from facenet_pytorch import MTCNN
        except ImportError:
            raise ImportError(
                "facenet-pytorch is not installed.\n"
                "Install via: pip install facenet-pytorch\n"
            )

        # MTCNN uses PyTorch — fall back to CPU if torch lacks CUDA
        import torch
        device = self._device
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("MTCNN: PyTorch has no CUDA support — falling back to CPU")
            device = "cpu"
            self._device = device

        self._model = MTCNN(
            keep_all=True,
            device=device,
            min_face_size=12,               # lowered from 20 for small faces
            thresholds=[0.5, 0.6, 0.6],     # relaxed from [0.6, 0.7, 0.7]
            post_process=False,
            factor=0.707,                   # finer pyramid scale for small faces
        )
        logger.info(f"MTCNN loaded on {self._device}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run MTCNN on a BGR numpy image.
        MTCNN (facenet-pytorch) expects RGB PIL or numpy.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, probs, landmarks = self._model.detect(rgb, landmarks=True)

        # Handle no-detection case
        if boxes is None or len(boxes) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        # Filter by confidence
        mask = probs >= self._confidence_threshold
        boxes = boxes[mask].astype(np.float32)
        probs = probs[mask].astype(np.float32)

        lm = None
        if landmarks is not None:
            lm = landmarks[mask].astype(np.float32) if mask.any() else None

        return DetectionResult(
            boxes=boxes,
            scores=probs,
            landmarks=lm,
        )

    @property
    def name(self) -> str:
        return "MTCNN"
