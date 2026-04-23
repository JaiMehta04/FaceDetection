"""
Centralized configuration for the Face Detection & Anonymization pipeline.

Design Decision:
- All paths, thresholds, and hyperparameters live here so experiments
  are reproducible and nothing is hard-coded in business logic.
- dataclass provides type safety and IDE autocompletion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# ──────────────────────────────────────────────────────────────
# Path configuration
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
PLOTS_DIR = OUTPUT_DIR / "plots"
ANONYMIZED_DIR = OUTPUT_DIR / "anonymized"
CSV_DIR = OUTPUT_DIR / "csv"

# WIDER FACE specific
WIDER_IMAGES_DIR = RAW_DIR / "WIDER_val" / "images"
WIDER_ANNOT_FILE = RAW_DIR / "wider_face_split" / "wider_face_val_bbx_gt.txt"


@dataclass
class DetectionConfig:
    """Thresholds and parameters for face detectors."""
    iou_threshold: float = 0.5
    retinaface_confidence: float = 0.5
    mtcnn_confidence: float = 0.5
    # Maximum image dimension for inference (resize if larger)
    max_image_dim: int = 1920
    # Device: 'cuda' or 'cpu' (auto-detected at runtime)
    device: str = "auto"


@dataclass
class EnhancementConfig:
    """Accuracy enhancement settings (all disabled by default for fair baseline)."""
    # Preprocessing
    enable_preprocessing: bool = False
    clahe: bool = True
    denoise: bool = True
    adaptive_preprocess: bool = True     # only preprocess poor-quality images

    # Multi-scale inference
    enable_multiscale: bool = False
    scales: List[float] = field(default_factory=lambda: [0.75, 1.0, 1.5])
    # Max pixels per side when upscaling (prevents OOM / slowness on CPU)
    multiscale_max_dim: int = 2048

    # Test-time augmentation (horizontal flip)
    enable_tta: bool = False

    # Tiled inference for high-res images
    enable_tiled: bool = False
    tile_size: int = 640
    tile_overlap: float = 0.25

    # Ensemble (run both detectors and merge)
    enable_ensemble: bool = False

    # Soft-NMS parameters (used by all merging strategies)
    soft_nms_sigma: float = 0.5
    soft_nms_threshold: float = 0.3


@dataclass
class QualityConfig:
    """Image-quality buckets for analysis."""
    blur_threshold_sharp: float = 100.0
    blur_threshold_moderate: float = 50.0
    brightness_low: float = 60.0
    brightness_high: float = 180.0
    small_face_area: int = 1024        # 32x32
    medium_face_area: int = 4096       # 64x64


@dataclass
class GroupBins:
    """Face-count bins for group-wise analysis."""
    bins: List[str] = field(default_factory=lambda: [
        "0-10", "11-20", "21-30", "31-40", "41-50", "51+"
    ])
    edges: List[int] = field(default_factory=lambda: [
        0, 11, 21, 31, 41, 51, 999999
    ])


@dataclass
class AnonymizationConfig:
    """Anonymization method settings."""
    method: str = "gaussian"          # gaussian | pixelate | solid
    gaussian_kernel: int = 51
    pixelate_block_size: int = 10
    solid_color: tuple = (0, 0, 0)    # black


@dataclass
class PipelineConfig:
    """Top-level config aggregating all sub-configs."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    enhancement: EnhancementConfig = field(default_factory=EnhancementConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    groups: GroupBins = field(default_factory=GroupBins)
    anonymization: AnonymizationConfig = field(default_factory=AnonymizationConfig)
    # Limit number of images for quick testing (0 = no limit)
    max_images: int = 0
    # Number of worker threads for I/O-bound loading
    num_workers: int = 4
    batch_log_interval: int = 50


def get_config() -> PipelineConfig:
    """Factory that returns the default pipeline config."""
    return PipelineConfig()


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [RESULTS_DIR, PLOTS_DIR, ANONYMIZED_DIR, CSV_DIR, PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)
