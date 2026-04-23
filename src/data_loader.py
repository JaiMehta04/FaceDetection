"""
WIDER FACE dataset loader.

The WIDER FACE annotation format (validation / test):
    Line 1:  image path (relative to images root)
    Line 2:  number of faces N
    Lines 3..N+2:  x1 y1 w h blur expression illumination invalid occlusion pose

Design Decisions:
- We parse into a list of lightweight dataclass records for O(1) access.
- Boxes with `invalid == 1` are kept but flagged — the evaluator can decide
  whether to include them.
- All boxes are stored as [x1, y1, x2, y2] (xyxy) for consistency with
  detector outputs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from src.utils import logger, xywh_to_xyxy


@dataclass
class FaceAnnotation:
    """A single ground-truth face."""
    bbox_xyxy: np.ndarray       # [x1, y1, x2, y2]
    blur: int = 0               # 0=clear, 1=normal, 2=heavy
    expression: int = 0         # 0=typical, 1=exaggerated
    illumination: int = 0       # 0=normal, 1=extreme
    invalid: int = 0            # 0=valid, 1=invalid
    occlusion: int = 0          # 0=none, 1=partial, 2=heavy
    pose: int = 0               # 0=typical, 1=atypical


@dataclass
class ImageRecord:
    """All annotations for one image."""
    image_path: str             # relative path from images root
    faces: List[FaceAnnotation] = field(default_factory=list)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def gt_boxes(self, include_invalid: bool = False) -> np.ndarray:
        """Return (N, 4) array of ground-truth boxes [x1, y1, x2, y2]."""
        boxes = [
            f.bbox_xyxy for f in self.faces
            if include_invalid or f.invalid == 0
        ]
        if not boxes:
            return np.empty((0, 4), dtype=np.float32)
        return np.stack(boxes).astype(np.float32)


def parse_wider_annotations(annot_path: Path) -> List[ImageRecord]:
    """
    Parse the WIDER FACE ground-truth annotation file.

    Parameters
    ----------
    annot_path : Path to wider_face_val_bbx_gt.txt (or train).

    Returns
    -------
    records : List[ImageRecord]
    """
    if not annot_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    records: List[ImageRecord] = []
    lines = annot_path.read_text(encoding="utf-8").strip().splitlines()

    idx = 0
    total_faces = 0
    while idx < len(lines):
        # Image path
        image_path = lines[idx].strip()
        idx += 1

        # Number of faces
        num_faces = int(lines[idx].strip())
        idx += 1

        record = ImageRecord(image_path=image_path)

        # Handle the edge case where num_faces == 0
        # WIDER FACE still has one line with "0 0 0 0 0 0 0 0 0 0"
        if num_faces == 0:
            idx += 1  # skip the dummy line
            records.append(record)
            continue

        for _ in range(num_faces):
            parts = lines[idx].strip().split()
            idx += 1

            x1, y1, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

            # Convert xywh → xyxy; guard against zero/negative dimensions
            w = max(w, 1.0)
            h = max(h, 1.0)
            bbox = np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

            face = FaceAnnotation(
                bbox_xyxy=bbox,
                blur=int(parts[4]) if len(parts) > 4 else 0,
                expression=int(parts[5]) if len(parts) > 5 else 0,
                illumination=int(parts[6]) if len(parts) > 6 else 0,
                invalid=int(parts[7]) if len(parts) > 7 else 0,
                occlusion=int(parts[8]) if len(parts) > 8 else 0,
                pose=int(parts[9]) if len(parts) > 9 else 0,
            )
            record.faces.append(face)

        total_faces += record.num_faces
        records.append(record)

    logger.info(
        f"Parsed {len(records)} images with {total_faces} faces "
        f"from {annot_path.name}"
    )
    return records


def build_annotation_index(records: List[ImageRecord]) -> Dict[str, ImageRecord]:
    """Return a dict keyed by image_path for O(1) lookup."""
    return {r.image_path: r for r in records}
