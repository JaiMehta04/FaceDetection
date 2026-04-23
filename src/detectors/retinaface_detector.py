"""
RetinaFace detector wrapper.

Model:  RetinaFace (InsightFace implementation)
Backbone: ResNet-50 + FPN
Tasks:  Face classification, bbox regression, landmark detection

Design Decisions:
- We use the `retinaface` PyPI package (or insightface) which provides
  pre-trained weights. No custom training needed for transfer learning
  evaluation — the model is already trained on WIDER FACE train split.
- Outputs are normalized to [x1, y1, x2, y2] format.
- Confidence filtering is applied at inference time to reduce noise.

Install: pip install retinaface  (uses tensorflow backend)
    OR:  pip install insightface onnxruntime-gpu
"""

import numpy as np

from src.detectors.base import BaseDetector, DetectionResult
from src.utils import logger


class RetinaFaceDetector(BaseDetector):
    """Wrapper around the retinaface PyPI package."""

    def __init__(self, device: str = "cpu", confidence_threshold: float = 0.5, det_size: int = 640):
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._det_size = det_size
        self._model = None
        self._det_model = None          # detection-only model for speed
        self._backend = None
        self._load_model()

    def _load_model(self):
        """
        Try insightface first (ONNX, faster); fall back to retinaface (TF).

        IMPORTANT: We load only the detection model, not the full FaceAnalysis
        pipeline (which also runs landmark, recognition, gender/age models on
        every face — 5x slower for no detection benefit).
        """
        # Attempt 1: insightface detection-only (fast)
        try:
            from insightface.model_zoo import get_model
            import os
            import glob

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda"
                else ["CPUExecutionProvider"]
            )

            # Find the detection model in insightface's model zoo
            home = os.path.expanduser("~")
            model_dir = os.path.join(home, ".insightface", "models", "buffalo_l")
            det_path = os.path.join(model_dir, "det_10g.onnx")

            if os.path.exists(det_path):
                det_model = get_model(det_path, providers=providers)
                det_model.prepare(
                    ctx_id=0 if self._device == "cuda" else -1,
                    input_size=(self._det_size, self._det_size),
                )
                self._det_model = det_model
                self._backend = "insightface_det"
                logger.info("RetinaFace loaded via insightface detection-only (ONNX, fast)")
                return
            else:
                # Model not downloaded yet — fall through to FaceAnalysis
                # which downloads automatically, then we can use det-only next time
                logger.info("Detection model not found, downloading via FaceAnalysis...")
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name="buffalo_l", providers=providers)
                app.prepare(ctx_id=0 if self._device == "cuda" else -1,
                            det_size=(self._det_size, self._det_size))
                # Extract just the detection model
                self._det_model = app.det_model
                self._backend = "insightface_det"
                logger.info("RetinaFace loaded via insightface detection-only (ONNX, fast)")
                return
        except (ImportError, Exception) as e:
            logger.debug(f"insightface det-only load failed: {e}")

        # Attempt 2: insightface full FaceAnalysis (fallback)
        try:
            from insightface.app import FaceAnalysis

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self._device == "cuda"
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name="buffalo_l", providers=providers)
            app.prepare(ctx_id=0 if self._device == "cuda" else -1,
                        det_size=(self._det_size, self._det_size))
            self._model = app
            self._backend = "insightface"
            logger.info("RetinaFace loaded via insightface FaceAnalysis (ONNX)")
            return
        except ImportError:
            pass

        # Attempt 3: retinaface PyPI (TensorFlow)
        try:
            from retinaface import RetinaFace as RF
            self._model = RF
            self._backend = "retinaface"
            logger.info("RetinaFace loaded via retinaface (TensorFlow)")
            return
        except ImportError:
            pass

        raise ImportError(
            "Neither 'insightface' nor 'retinaface' is installed.\n"
            "Install one of:\n"
            "  pip install insightface onnxruntime-gpu\n"
            "  pip install retinaface\n"
        )

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run RetinaFace detection on a BGR image."""
        if self._backend == "insightface_det":
            return self._detect_insightface_det_only(image)
        elif self._backend == "insightface":
            return self._detect_insightface(image)
        else:
            return self._detect_retinaface_tf(image)

    def _detect_insightface_det_only(self, image: np.ndarray) -> DetectionResult:
        """
        Fast path: use only the detection model (det_10g.onnx).
        Skips landmark, recognition, gender/age — 5x faster than FaceAnalysis.get().
        """
        bboxes, kpss = self._det_model.detect(image)

        if bboxes is None or len(bboxes) == 0:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        # bboxes shape: (N, 5) where last column is confidence
        scores = bboxes[:, 4].astype(np.float32)
        boxes = bboxes[:, :4].astype(np.float32)

        # Filter by confidence
        mask = scores >= self._confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]

        landmarks = None
        if kpss is not None and len(kpss) > 0:
            landmarks = kpss[mask].astype(np.float32) if mask.any() else None

        return DetectionResult(boxes=boxes, scores=scores, landmarks=landmarks)

    def _detect_insightface(self, image: np.ndarray) -> DetectionResult:
        faces = self._model.get(image)
        if not faces:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        boxes, scores, landmarks_list = [], [], []
        for face in faces:
            score = float(face.det_score)
            if score < self._confidence_threshold:
                continue
            boxes.append(face.bbox.astype(np.float32))
            scores.append(score)
            if face.kps is not None:
                landmarks_list.append(face.kps)

        if not boxes:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        return DetectionResult(
            boxes=np.stack(boxes),
            scores=np.array(scores, dtype=np.float32),
            landmarks=np.stack(landmarks_list) if landmarks_list else None,
        )

    def _detect_retinaface_tf(self, image: np.ndarray) -> DetectionResult:
        import cv2

        # retinaface package expects RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resp = self._model.detect_faces(rgb)

        if not resp:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        boxes, scores = [], []
        for key, face_info in resp.items():
            score = face_info["score"]
            if score < self._confidence_threshold:
                continue
            area = face_info["facial_area"]
            # facial_area is [x1, y1, x2, y2]
            boxes.append(np.array(area, dtype=np.float32))
            scores.append(float(score))

        if not boxes:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=np.float32),
                scores=np.empty((0,), dtype=np.float32),
            )

        return DetectionResult(
            boxes=np.stack(boxes),
            scores=np.array(scores, dtype=np.float32),
        )

    @property
    def name(self) -> str:
        return f"RetinaFace ({self._backend})"
