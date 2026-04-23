"""
Image preprocessing to improve detection accuracy.

Techniques:
1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Improves face visibility in low-light / uneven illumination
   - Applied to the L channel of LAB color space (preserves color)

2. Denoising (Non-local Means)
   - Reduces noise that confuses detectors in low-light images
   - Applied selectively based on estimated noise level

3. Super-resolution upscaling
   - Doubles image resolution for small-face recovery
   - Uses bicubic interpolation (fast) or INTER_LANCZOS4 (sharper)

Design Decisions:
- Preprocessing is applied BEFORE detection, not to GT annotations.
- Each transform returns (processed_image, scale_factor) so bounding
  boxes can be mapped back to original coordinates.
- Transforms are composable via the `preprocess_image` function.
- Controlled by PreprocessConfig — disabled by default for fair
  baseline comparison, enabled via CLI flags.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.utils import compute_brightness, compute_blur, logger


@dataclass
class PreprocessConfig:
    """Toggle individual preprocessing steps."""
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    enable_denoise: bool = True
    denoise_strength: int = 10          # h parameter for fastNlMeansDenoisingColored

    enable_upscale: bool = False        # upscale small images
    upscale_threshold: int = 720        # min height to trigger upscale
    upscale_factor: float = 2.0

    # Adaptive: only apply if image quality is poor
    adaptive: bool = True
    brightness_low_threshold: float = 80.0
    blur_low_threshold: float = 50.0


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE on the luminance channel of LAB color space.
    This enhances local contrast without distorting color.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(grid_size, grid_size),
    )
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def apply_denoise(image: np.ndarray, h: int = 10) -> np.ndarray:
    """
    Non-local means denoising for color images.
    Reduces sensor noise that creates false edges / false faces.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)


def apply_upscale(
    image: np.ndarray,
    factor: float = 2.0,
) -> Tuple[np.ndarray, float]:
    """
    Upscale image by a factor using Lanczos interpolation.
    Returns (upscaled_image, scale_factor).
    """
    h, w = image.shape[:2]
    new_w = int(w * factor)
    new_h = int(h * factor)
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return upscaled, factor


def preprocess_image(
    image: np.ndarray,
    config: PreprocessConfig = None,
) -> Tuple[np.ndarray, float]:
    """
    Apply the full preprocessing pipeline to an image.

    Parameters
    ----------
    image : BGR numpy array
    config : preprocessing settings

    Returns
    -------
    (processed_image, scale_factor)
    scale_factor is used to map predicted boxes back to original coords:
        original_box = predicted_box / scale_factor
    """
    if config is None:
        config = PreprocessConfig()

    scale = 1.0
    processed = image.copy()

    # Compute quality metrics for adaptive decisions
    brightness = compute_brightness(processed)
    blur_score = compute_blur(processed)

    # 1. CLAHE — enhance contrast in dark images
    if config.enable_clahe:
        if not config.adaptive or brightness < config.brightness_low_threshold:
            processed = apply_clahe(
                processed,
                clip_limit=config.clahe_clip_limit,
                grid_size=config.clahe_grid_size,
            )

    # 2. Denoise — reduce noise in blurry/noisy images
    if config.enable_denoise:
        if not config.adaptive or blur_score < config.blur_low_threshold:
            processed = apply_denoise(processed, h=config.denoise_strength)

    # 3. Upscale — enlarge small images to improve small-face detection
    if config.enable_upscale:
        h, w = processed.shape[:2]
        if min(h, w) < config.upscale_threshold:
            processed, scale = apply_upscale(processed, config.upscale_factor)

    return processed, scale
