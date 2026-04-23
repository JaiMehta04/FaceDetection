"""
Quick test script — run pipeline on a small subset to verify everything works.

Usage:
    python scripts/quick_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_config, ensure_dirs
from src.pipeline import FaceAnonymizationPipeline
from src.visualization import generate_all_plots
from src.utils import logger


def main():
    logger.info("Running quick test with 10 images...")

    config = get_config()
    config.max_images = 10
    config.detection.device = "auto"

    pipeline = FaceAnonymizationPipeline(config)

    # Test with just MTCNN first (faster to load)
    try:
        all_outputs = pipeline.run(
            detector_names=["mtcnn"],
            save_anonymized=True,
        )
        generate_all_plots(all_outputs)
        logger.info("Quick test PASSED — MTCNN working.")
    except Exception as e:
        logger.error(f"MTCNN test failed: {e}")

    # Test RetinaFace
    try:
        all_outputs = pipeline.run(
            detector_names=["retinaface"],
            save_anonymized=True,
        )
        generate_all_plots(all_outputs)
        logger.info("Quick test PASSED — RetinaFace working.")
    except Exception as e:
        logger.error(f"RetinaFace test failed: {e}")


if __name__ == "__main__":
    main()
