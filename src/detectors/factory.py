"""
Factory function to instantiate detectors by name, with optional enhancements.
"""

from src.detectors.base import BaseDetector
from config.settings import EnhancementConfig


def get_detector(name: str, device: str = "cpu", confidence: float = 0.5) -> BaseDetector:
    """
    Create a detector instance by name.

    Parameters
    ----------
    name : 'retinaface' or 'mtcnn'
    device : 'cpu' or 'cuda'
    confidence : minimum confidence threshold
    """
    name_lower = name.lower().strip()

    if name_lower == "retinaface":
        from src.detectors.retinaface_detector import RetinaFaceDetector
        return RetinaFaceDetector(device=device, confidence_threshold=confidence)

    elif name_lower == "mtcnn":
        from src.detectors.mtcnn_detector import MTCNNDetector
        return MTCNNDetector(device=device, confidence_threshold=confidence)

    else:
        raise ValueError(
            f"Unknown detector '{name}'. Choose from: retinaface, mtcnn"
        )


def wrap_detector_with_enhancements(
    detector: BaseDetector,
    config: EnhancementConfig,
) -> BaseDetector:
    """
    Wrap a base detector with enhancement layers (multi-scale, TTA, tiled).
    Wrappers are applied from innermost to outermost.
    """
    from src.enhanced_detection import (
        MultiScaleDetector, TTADetector, TiledDetector,
    )

    enhanced = detector

    # Order matters: tiled → multi-scale → TTA
    # Innermost wraps are applied first during detection

    if config.enable_tiled:
        enhanced = TiledDetector(
            base_detector=enhanced,
            tile_size=config.tile_size,
            overlap=config.tile_overlap,
            soft_nms_sigma=config.soft_nms_sigma,
            soft_nms_threshold=config.soft_nms_threshold,
        )

    if config.enable_multiscale:
        enhanced = MultiScaleDetector(
            base_detector=enhanced,
            scales=config.scales,
            soft_nms_sigma=config.soft_nms_sigma,
            soft_nms_threshold=config.soft_nms_threshold,
            max_dim=config.multiscale_max_dim,
        )

    if config.enable_tta:
        enhanced = TTADetector(
            base_detector=enhanced,
            soft_nms_sigma=config.soft_nms_sigma,
            soft_nms_threshold=config.soft_nms_threshold,
        )

    return enhanced


def get_ensemble_detector(
    device: str = "cpu",
    confidence: float = 0.3,
    enhancement_config: EnhancementConfig = None,
) -> BaseDetector:
    """
    Create an ensemble of all available detectors.
    """
    from src.enhanced_detection import EnsembleDetector

    config = enhancement_config or EnhancementConfig()

    detectors = []
    for name in ["retinaface", "mtcnn"]:
        try:
            det = get_detector(name, device=device, confidence=confidence)
            detectors.append(det)
        except ImportError:
            pass

    if not detectors:
        raise RuntimeError("No detectors available for ensemble")

    return EnsembleDetector(
        detectors=detectors,
        soft_nms_sigma=config.soft_nms_sigma,
        soft_nms_threshold=config.soft_nms_threshold,
    )
