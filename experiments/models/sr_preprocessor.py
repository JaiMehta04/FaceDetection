"""
Super-Resolution Preprocessor — Real-ESRGAN for face image upscaling.

Provides three modes:
  1. Blind SR   — upscale the entire image (expensive, upper bound)
  2. Selective SR — upscale only crops around small/low-confidence detections
  3. Tiled SR   — split into tiles, SR each (memory-efficient for large images)

Real-ESRGAN is a practical image restoration model trained on diverse
degradation types. It significantly improves detection of <32px faces
by providing the detector with higher-resolution input.

Memory: ~300MB VRAM for Real-ESRGAN x4 model.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List

from src.utils import logger


class SRPreprocessor:
    """Real-ESRGAN super-resolution preprocessor."""

    def __init__(
        self,
        scale: int = 2,
        model_name: str = "RealESRGAN_x2plus",
        device: str = "cpu",
        tile_size: int = 512,
        tile_pad: int = 10,
        half: bool = False,
    ):
        """
        Parameters
        ----------
        scale : int
            Upscaling factor (2 or 4).
        model_name : str
            Model name: 'RealESRGAN_x2plus', 'RealESRGAN_x4plus',
                        'RealESRNet_x4plus' (faster, less detail).
        device : str
            'cuda' or 'cpu'.
        tile_size : int
            Process image in tiles to save VRAM. 0 = no tiling.
        tile_pad : int
            Padding between tiles to avoid seam artifacts.
        half : bool
            Use FP16 inference (faster on GPU, may lose quality).
        """
        self._scale = scale
        self._model_name = model_name
        self._device = device
        self._tile_size = tile_size
        self._tile_pad = tile_pad
        self._half = half and device == "cuda"
        self._upsampler = None
        self._load_model()

    def _load_model(self):
        """Load Real-ESRGAN model."""
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError:
            raise ImportError(
                "Real-ESRGAN dependencies required:\n"
                "  pip install realesrgan basicsr\n"
                "  (basicsr provides the RRDBNet architecture)"
            )

        # Configure architecture based on model name
        if "x2plus" in self._model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            netscale = 2
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
        elif "x4plus" in self._model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        else:
            # Default to x2
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=2,
            )
            netscale = 2
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"

        # Download model weights if needed
        weights_dir = Path(__file__).resolve().parent.parent / "model_weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_filename = model_url.split("/")[-1]
        model_path = weights_dir / model_filename

        if not model_path.exists():
            logger.info(f"Downloading {model_filename}...")
            import urllib.request
            urllib.request.urlretrieve(model_url, str(model_path))
            logger.info(f"Downloaded to {model_path}")

        self._upsampler = RealESRGANer(
            scale=netscale,
            model_path=str(model_path),
            model=model,
            tile=self._tile_size,
            tile_pad=self._tile_pad,
            pre_pad=0,
            half=self._half,
            device=self._device,
        )

        logger.info(
            f"Real-ESRGAN loaded: {self._model_name} "
            f"(scale={self._scale}, tile={self._tile_size}, device={self._device})"
        )

    def upscale_full(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Upscale entire image (blind SR).

        Returns
        -------
        (upscaled_image, scale_factor)
        """
        try:
            output, _ = self._upsampler.enhance(image, outscale=self._scale)
            actual_scale = output.shape[0] / image.shape[0]
            return output, actual_scale
        except Exception as e:
            logger.warning(f"SR failed, returning original: {e}")
            return image, 1.0

    def upscale_regions(
        self,
        image: np.ndarray,
        regions: List[np.ndarray],
        padding: int = 50,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Selective SR — upscale only specific regions of the image.

        This is the novel contribution: instead of upscaling the entire image,
        we only SR the regions where the detector found low-confidence faces
        or where we suspect small faces exist.

        Parameters
        ----------
        image : (H, W, 3) BGR image
        regions : list of [x1, y1, x2, y2] bounding boxes to upscale
        padding : extra pixels around each region to include context

        Returns
        -------
        (composite_image, region_info_list)
        composite_image: original image with SR regions pasted back (at original res)
        region_info_list: metadata about each SR region for re-detection
        """
        h, w = image.shape[:2]
        sr_regions = []

        for bbox in regions:
            x1, y1, x2, y2 = bbox.astype(int)

            # Add padding for context
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding)
            py2 = min(h, y2 + padding)

            crop = image[py1:py2, px1:px2]

            if crop.size == 0:
                continue

            # Upscale this crop
            try:
                sr_crop, actual_scale = self._upsampler.enhance(
                    crop, outscale=self._scale
                )
            except Exception as e:
                logger.debug(f"SR failed on region {bbox}: {e}")
                sr_crop = crop
                actual_scale = 1.0

            sr_regions.append({
                "original_bbox": [px1, py1, px2, py2],
                "sr_crop": sr_crop,
                "scale": actual_scale,
                "original_crop_shape": crop.shape[:2],
            })

        return image, sr_regions

    def upscale_tiles(
        self,
        image: np.ndarray,
        tile_size: int = 640,
        overlap: float = 0.25,
        max_dim: int = 4096,
    ) -> Tuple[np.ndarray, float]:
        """
        Tiled SR — for large images, process in tiles to limit VRAM.

        If the resulting image would exceed max_dim, use a lower scale.
        """
        h, w = image.shape[:2]

        # Check if upscaled size would be too large
        if max(h, w) * self._scale > max_dim:
            # Use a reduced scale
            effective_scale = max_dim / max(h, w)
            if effective_scale < 1.2:
                logger.debug("Image already large enough, skipping SR")
                return image, 1.0
        else:
            effective_scale = self._scale

        return self.upscale_full(image)

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def model_info(self) -> dict:
        return {
            "model_name": self._model_name,
            "scale": self._scale,
            "tile_size": self._tile_size,
            "device": self._device,
            "half": self._half,
        }


class OpenCVSRPreprocessor:
    """
    Fallback SR using OpenCV's DNN super-resolution.

    Uses EDSR/ESPCN/FSRCNN/LapSRN models — lighter than Real-ESRGAN,
    works without GPU, good enough for ablation comparison.
    """

    def __init__(self, scale: int = 2, model: str = "edsr"):
        """
        Parameters
        ----------
        scale : 2, 3, or 4
        model : 'edsr', 'espcn', 'fsrcnn', 'lapsrn'
        """
        self._scale = scale
        self._model_name = model
        self._sr = None
        self._load_model()

    def _load_model(self):
        """Load OpenCV DNN SR model."""
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
        except AttributeError:
            raise ImportError(
                "OpenCV contrib required for DNN super-resolution:\n"
                "  pip install opencv-contrib-python"
            )

        # Model file paths (auto-download)
        model_dir = Path(__file__).resolve().parent.parent / "model_weights"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_dir / f"{self._model_name}_x{self._scale}.pb"

        if not model_file.exists():
            # Download the model
            urls = {
                "edsr": f"https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/EDSR_x{self._scale}.pb",
                "espcn": f"https://raw.githubusercontent.com/fannypackz/ESPCN-TensorFlow/master/export/ESPCN_x{self._scale}.pb",
                "fsrcnn": f"https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN_x{self._scale}.pb",
                "lapsrn": f"https://raw.githubusercontent.com/fannypackz/LapSRN-tensorflow/master/export/LapSRN_x{self._scale}.pb",
            }
            url = urls.get(self._model_name)
            if url:
                logger.info(f"Downloading {self._model_name} x{self._scale} model...")
                import urllib.request
                urllib.request.urlretrieve(url, str(model_file))
            else:
                raise ValueError(f"Unknown model: {self._model_name}")

        sr.readModel(str(model_file))
        sr.setModel(self._model_name, self._scale)
        self._sr = sr
        logger.info(f"OpenCV SR loaded: {self._model_name} x{self._scale}")

    def upscale_full(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Upscale entire image."""
        result = self._sr.upsample(image)
        actual_scale = result.shape[0] / image.shape[0]
        return result, actual_scale

    def upscale_regions(
        self,
        image: np.ndarray,
        regions: List[np.ndarray],
        padding: int = 50,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Selective SR on regions."""
        h, w = image.shape[:2]
        sr_regions = []

        for bbox in regions:
            x1, y1, x2, y2 = bbox.astype(int)
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding)
            py2 = min(h, y2 + padding)

            crop = image[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            sr_crop = self._sr.upsample(crop)
            actual_scale = sr_crop.shape[0] / crop.shape[0]

            sr_regions.append({
                "original_bbox": [px1, py1, px2, py2],
                "sr_crop": sr_crop,
                "scale": actual_scale,
                "original_crop_shape": crop.shape[:2],
            })

        return image, sr_regions

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def model_info(self) -> dict:
        return {
            "model_name": f"opencv_{self._model_name}",
            "scale": self._scale,
        }


class BicubicSRPreprocessor:
    """
    Simplest baseline: bicubic interpolation upscaling.
    No learned SR — serves as a control experiment to show that
    learned SR actually contributes beyond naive upscaling.
    """

    def __init__(self, scale: int = 2):
        self._scale = scale

    def upscale_full(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        new_h, new_w = int(h * self._scale), int(w * self._scale)
        result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return result, self._scale

    def upscale_regions(
        self,
        image: np.ndarray,
        regions: List[np.ndarray],
        padding: int = 50,
    ) -> Tuple[np.ndarray, List[dict]]:
        h, w = image.shape[:2]
        sr_regions = []

        for bbox in regions:
            x1, y1, x2, y2 = bbox.astype(int)
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding)
            py2 = min(h, y2 + padding)

            crop = image[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            sr_crop = cv2.resize(
                crop,
                (int(cw * self._scale), int(ch * self._scale)),
                interpolation=cv2.INTER_CUBIC,
            )

            sr_regions.append({
                "original_bbox": [px1, py1, px2, py2],
                "sr_crop": sr_crop,
                "scale": self._scale,
                "original_crop_shape": crop.shape[:2],
            })

        return image, sr_regions

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def model_info(self) -> dict:
        return {"model_name": "bicubic", "scale": self._scale}
