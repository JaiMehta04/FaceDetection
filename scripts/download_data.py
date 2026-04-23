"""
Download and prepare the WIDER FACE validation dataset.

Usage:
    python scripts/download_data.py

This will download:
1. WIDER FACE validation images (~1.9 GB)
2. WIDER FACE annotations

Files are saved to data/raw/
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import RAW_DIR


WIDER_VAL_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
WIDER_ANNOT_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"


def download_file(url: str, dest: Path):
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  Already exists: {dest}")
        return

    print(f"  Downloading: {url}")
    print(f"  Destination: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(count, block_size, total_size):
        percent = count * block_size * 100 / total_size if total_size > 0 else 0
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print()


def extract_zip(zip_path: Path, dest_dir: Path):
    """Extract a zip file."""
    print(f"  Extracting: {zip_path}")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    print(f"  Extracted to: {dest_dir}")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  WIDER FACE Dataset Download")
    print("=" * 50)

    # 1. Validation images
    val_zip = RAW_DIR / "WIDER_val.zip"
    print("\n[1/2] Validation images")
    download_file(WIDER_VAL_URL, val_zip)
    if not (RAW_DIR / "WIDER_val").exists():
        extract_zip(val_zip, RAW_DIR)

    # 2. Annotations
    annot_zip = RAW_DIR / "wider_face_split.zip"
    print("\n[2/2] Annotations")
    download_file(WIDER_ANNOT_URL, annot_zip)
    if not (RAW_DIR / "wider_face_split").exists():
        extract_zip(annot_zip, RAW_DIR)

    print("\nDone! Dataset is ready at:", RAW_DIR)
    print("\nExpected structure:")
    print("  data/raw/WIDER_val/images/         (validation images)")
    print("  data/raw/wider_face_split/         (annotation files)")


if __name__ == "__main__":
    main()
