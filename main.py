"""
Main entry point for the Face Detection & Anonymization ETL Pipeline.

Usage:
    # Run full pipeline with both detectors (baseline)
    python main.py

    # Run with specific detector
    python main.py --detectors retinaface

    # Quick test with limited images
    python main.py --max-images 50

    # Enable accuracy enhancements
    python main.py --enhance multiscale tta preprocess

    # Enable all enhancements
    python main.py --enhance all

    # Ensemble mode (merge RetinaFace + MTCNN)
    python main.py --enhance ensemble

    # Custom anonymization method
    python main.py --anon-method pixelate

    # Skip anonymization (evaluation only)
    python main.py --no-anonymize

    # Custom data paths
    python main.py --annot-path /path/to/annotations.txt --images-root /path/to/images
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_config, ensure_dirs
from src.pipeline import FaceAnonymizationPipeline
from src.visualization import generate_all_plots
from src.utils import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Detection & Anonymization ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["retinaface", "mtcnn"],
        choices=["retinaface", "mtcnn"],
        help="Detector(s) to evaluate (default: both)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit number of images (0 = all). Useful for quick testing.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching (default: 0.5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence (default: 0.5)",
    )
    parser.add_argument(
        "--anon-method",
        type=str,
        default="gaussian",
        choices=["gaussian", "pixelate", "solid"],
        help="Anonymization method (default: gaussian)",
    )
    parser.add_argument(
        "--no-anonymize",
        action="store_true",
        help="Skip anonymization (evaluation only)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--annot-path",
        type=str,
        default=None,
        help="Path to WIDER FACE annotation file",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=None,
        help="Path to WIDER FACE images root directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto)",
    )

    # ── Enhancement flags ──
    parser.add_argument(
        "--enhance",
        nargs="+",
        default=[],
        choices=["preprocess", "multiscale", "tta", "tiled", "ensemble", "all"],
        help=(
            "Accuracy enhancements to enable. "
            "Options: preprocess, multiscale, tta, tiled, ensemble, all"
        ),
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0],
        help="Scales for multi-scale inference (default: 0.5 1.0 1.5 2.0)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Face Detection & Anonymization ETL Pipeline")
    logger.info("=" * 60)

    # Build config from CLI args
    config = get_config()
    config.max_images = args.max_images
    config.detection.iou_threshold = args.iou_threshold
    config.detection.retinaface_confidence = args.confidence
    config.detection.mtcnn_confidence = args.confidence
    config.detection.device = args.device
    config.anonymization.method = args.anon_method

    # Apply enhancement flags
    enhancements = set(args.enhance)
    if "all" in enhancements:
        enhancements = {"preprocess", "multiscale", "tta", "tiled", "ensemble"}

    enh = config.enhancement
    enh.enable_preprocessing = "preprocess" in enhancements
    enh.enable_multiscale = "multiscale" in enhancements
    enh.enable_tta = "tta" in enhancements
    enh.enable_tiled = "tiled" in enhancements
    enh.enable_ensemble = "ensemble" in enhancements
    enh.scales = args.scales

    active_enhancements = [e for e in ["preprocess", "multiscale", "tta", "tiled", "ensemble"] if getattr(enh, f"enable_{e}", False) or (e == "preprocess" and enh.enable_preprocessing)]

    logger.info(f"Detectors      : {args.detectors}")
    logger.info(f"Max images     : {args.max_images or 'ALL'}")
    logger.info(f"IoU threshold  : {args.iou_threshold}")
    logger.info(f"Confidence     : {args.confidence}")
    logger.info(f"Anonymization  : {args.anon_method}")
    logger.info(f"Device         : {args.device}")
    logger.info(f"Enhancements   : {active_enhancements or 'None (baseline)'}")
    if enh.enable_multiscale:
        logger.info(f"Scales         : {enh.scales}")

    # Initialize pipeline
    pipeline = FaceAnonymizationPipeline(config)

    # Optional path overrides
    annot_path = Path(args.annot_path) if args.annot_path else None
    images_root = Path(args.images_root) if args.images_root else None

    # Run ETL
    t_start = time.time()
    all_outputs = pipeline.run(
        detector_names=args.detectors,
        annot_path=annot_path,
        images_root=images_root,
        save_anonymized=not args.no_anonymize,
    )
    elapsed = time.time() - t_start

    logger.info(f"\nTotal pipeline time: {elapsed:.1f}s")

    # Generate plots
    if not args.no_plots:
        logger.info("\nGenerating plots...")
        generate_all_plots(all_outputs)

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    main()
