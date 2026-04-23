"""
Anonymization module — apply privacy-preserving transformations to detected faces.

Supported methods:
  1. Gaussian blur — fast, retains image structure
  2. Pixelation   — classic CCTV-style anonymization
  3. Solid mask   — strongest privacy, replaces face with solid color

Design Decisions:
- Each method operates in-place on the image for memory efficiency.
- The `anonymize_image` function accepts a list of boxes so it can be
  called once per image (not once per face).
- This module is stateless — it plugs directly into the ETL pipeline.
"""

from typing import List
import numpy as np
import cv2

from config.settings import AnonymizationConfig


def _apply_gaussian_blur(
    image: np.ndarray,
    box: np.ndarray,
    kernel_size: int = 51,
) -> np.ndarray:
    """Apply heavy Gaussian blur to the face region."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(box[0])), max(0, int(box[1])),
        min(w, int(box[2])), min(h, int(box[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return image

    # Ensure kernel size is odd
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    face_region = image[y1:y2, x1:x2]
    blurred = cv2.GaussianBlur(face_region, (k, k), 30)
    image[y1:y2, x1:x2] = blurred
    return image


def _apply_pixelation(
    image: np.ndarray,
    box: np.ndarray,
    block_size: int = 10,
) -> np.ndarray:
    """Pixelate the face region by downscaling and upscaling."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(box[0])), max(0, int(box[1])),
        min(w, int(box[2])), min(h, int(box[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return image

    face = image[y1:y2, x1:x2]
    fh, fw = face.shape[:2]
    # Downscale
    small = cv2.resize(
        face,
        (max(1, fw // block_size), max(1, fh // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    # Upscale back
    pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
    image[y1:y2, x1:x2] = pixelated
    return image


def _apply_solid_mask(
    image: np.ndarray,
    box: np.ndarray,
    color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Replace the face region with a solid color."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(box[0])), max(0, int(box[1])),
        min(w, int(box[2])), min(h, int(box[3])),
    )
    if x2 <= x1 or y2 <= y1:
        return image

    image[y1:y2, x1:x2] = color
    return image


def anonymize_image(
    image: np.ndarray,
    boxes: np.ndarray,
    config: AnonymizationConfig = None,
) -> np.ndarray:
    """
    Anonymize all detected faces in an image.

    Parameters
    ----------
    image : BGR numpy array (modified in-place and returned)
    boxes : (N, 4) array of [x1, y1, x2, y2]
    config : anonymization settings

    Returns
    -------
    Anonymized image (same object as input).
    """
    if config is None:
        config = AnonymizationConfig()

    if boxes.size == 0:
        return image

    for box in boxes:
        if config.method == "gaussian":
            image = _apply_gaussian_blur(image, box, config.gaussian_kernel)
        elif config.method == "pixelate":
            image = _apply_pixelation(image, box, config.pixelate_block_size)
        elif config.method == "solid":
            image = _apply_solid_mask(image, box, config.solid_color)
        else:
            raise ValueError(f"Unknown anonymization method: {config.method}")

    return image
