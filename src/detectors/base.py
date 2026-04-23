"""
Abstract base class for all face detectors.

Design Decision:
- A common interface lets the evaluation engine, ETL pipeline, and analysis
  code work with *any* detector without caring about internal details.
- Each detector returns a standardized DetectionResult.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class DetectionResult:
    """Standardized output from any face detector."""
    # (N, 4) array of [x1, y1, x2, y2]
    boxes: np.ndarray
    # (N,) confidence scores in [0, 1]
    scores: np.ndarray
    # (N, 5, 2) facial landmarks (optional; not all detectors provide them)
    landmarks: Optional[np.ndarray] = None

    @property
    def num_faces(self) -> int:
        return len(self.boxes) if self.boxes.size > 0 else 0


class BaseDetector(ABC):
    """Interface that all detector wrappers must implement."""

    @abstractmethod
    def __init__(self, device: str = "cpu", confidence_threshold: float = 0.5):
        ...

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Run face detection on a BGR numpy image.

        Parameters
        ----------
        image : np.ndarray of shape (H, W, 3), dtype uint8, BGR.

        Returns
        -------
        DetectionResult with boxes in image-coordinate space.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name for logging and reports."""
        ...
