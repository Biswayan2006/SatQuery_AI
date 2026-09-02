"""
SatQuery AI — Input Validator
Validates uploaded satellite images and detects modality.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("satquery.validator")


@dataclass
class ValidationResult:
    valid: bool
    modality: str  # "optical" | "sar" | "multispectral" | "unknown"
    shape: list  # [H, W, C]
    bands: int
    is_geotiff: bool
    crs: Optional[str]
    message: str
    extra: dict = field(default_factory=dict)


class InputValidator:
    """
    Validates image files and detects remote sensing modality.

    Modality heuristics:
      - 1-band float32/float64 with negative values → SAR (dB scale)
      - 1-band uint16 with high dynamic range → SAR (linear scale) or panchromatic
      - 2-band → SAR (VV + VH)
      - 3-band uint8 → optical RGB
      - 3-band uint16 → optical (high bit-depth)
      - 4+ bands → multispectral
    """

    GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
    MAX_BANDS_DISPLAY = 13  # Sentinel-2 max

    # ── Public API ────────────────────────────────────────────────────────────

    def validate_image(self, file_path: str) -> ValidationResult:
        """Full validation pipeline."""
        if not os.path.exists(file_path):
            return ValidationResult(
                valid=False, modality="unknown", shape=[], bands=0,
                is_geotiff=False, crs=None, message=f"File not found: {file_path}"
            )

        ext = os.path.splitext(file_path)[1].lower()

        if ext in self.GEOTIFF_EXTENSIONS:
            return self._validate_geotiff(file_path)
        else:
            return self._validate_standard_image(file_path)

    def detect_modality(self, image_array: np.ndarray, dtype=None) -> str:
        """
        Detect image modality from numpy array.
        image_array shape: (H, W) or (H, W, C) or (C, H, W)
        """
        if image_array.ndim == 2:
            bands = 1
            arr = image_array
        elif image_array.shape[0] <= self.MAX_BANDS_DISPLAY and image_array.ndim == 3:
            # Could be (C, H, W)
            if image_array.shape[0] < image_array.shape[1]:
                bands = image_array.shape[0]
                arr = image_array[0]
            else:
                bands = image_array.shape[2]
                arr = image_array[:, :, 0]
        else:
            bands = image_array.shape[2] if image_array.ndim == 3 else 1
            arr = image_array[:, :, 0] if image_array.ndim == 3 else image_array

        return self._infer_modality(bands, arr, dtype or image_array.dtype)

    def check_compatibility(
        self,
        meta1: ValidationResult,
        meta2: ValidationResult,
    ) -> Tuple[bool, str]:
        """
        Check whether two images are compatible for pairwise analysis
        (change detection or SAR-optical fusion).
        """
        if not meta1.valid or not meta2.valid:
            return False, "One or both images failed validation"

        # Fusion: SAR + optical is explicitly compatible
        modalities = {meta1.modality, meta2.modality}
        if modalities == {"sar", "optical"} or modalities == {"sar", "multispectral"}:
            return True, "SAR-optical pair — suitable for fusion analysis"

        # Change detection: same modality
        if meta1.modality == meta2.modality:
            # Optionally check spatial resolution similarity
            h1, w1 = meta1.shape[:2]
            h2, w2 = meta2.shape[:2]
            ratio_h = max(h1, h2) / max(min(h1, h2), 1)
            ratio_w = max(w1, w2) / max(min(w1, w2), 1)
            if ratio_h > 4 or ratio_w > 4:
                return (
                    False,
                    f"Image sizes differ too much ({h1}×{w1} vs {h2}×{w2}). "
                    "Consider resampling before change detection.",
                )
            return True, f"Matching modality ({meta1.modality}) — suitable for change detection"

        return (
            True,
            f"Mixed modalities ({meta1.modality}, {meta2.modality}) — will attempt analysis",
        )

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate_geotiff(self, file_path: str) -> ValidationResult:
        try:
            import rasterio

            with rasterio.open(file_path) as ds:
                bands = ds.count
                height = ds.height
                width = ds.width
                dtype = ds.dtypes[0]
                crs = str(ds.crs) if ds.crs else None

                # Read a small sample for modality detection
                sample = ds.read(
                    out_shape=(bands, min(height, 64), min(width, 64)),
                    resampling=rasterio.enums.Resampling.nearest,
                ).astype(np.float32)

            modality = self._infer_modality(bands, sample[0], np.dtype(dtype))
            shape = [height, width, bands]

            return ValidationResult(
                valid=True,
                modality=modality,
                shape=shape,
                bands=bands,
                is_geotiff=True,
                crs=crs,
                message="GeoTIFF validated successfully",
                extra={"dtype": dtype},
            )
        except Exception as exc:
            logger.warning("GeoTIFF validation failed: %s", exc)
            return ValidationResult(
                valid=False, modality="unknown", shape=[], bands=0,
                is_geotiff=True, crs=None, message=f"GeoTIFF read error: {exc}"
            )

    def _validate_standard_image(self, file_path: str) -> ValidationResult:
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                width, height = img.size
                mode = img.mode
                bands = len(img.getbands())
                arr = np.array(img)

            modality = self._infer_modality(bands, arr[:, :, 0] if arr.ndim == 3 else arr, arr.dtype)
            shape = [height, width, bands]

            return ValidationResult(
                valid=True,
                modality=modality,
                shape=shape,
                bands=bands,
                is_geotiff=False,
                crs=None,
                message=f"Image validated ({mode}, {width}×{height})",
                extra={"mode": mode},
            )
        except Exception as exc:
            logger.warning("Image validation failed: %s", exc)
            return ValidationResult(
                valid=False, modality="unknown", shape=[], bands=0,
                is_geotiff=False, crs=None, message=f"Image read error: {exc}"
            )

    @staticmethod
    def _infer_modality(bands: int, sample: np.ndarray, dtype) -> str:
        """Infer modality from band count, sample values, and dtype."""
        if bands == 1:
            # SAR: float with negative values (dB), or high uint16 dynamic range
            if np.issubdtype(dtype, np.floating):
                if sample.min() < -5:
                    return "sar"
            if np.issubdtype(dtype, np.uint16):
                dynamic_range = float(sample.max()) - float(sample.min())
                if dynamic_range > 3000:
                    return "sar"
            return "optical"
        elif bands == 2:
            return "sar"  # VV + VH Sentinel-1
        elif bands == 3:
            if np.issubdtype(dtype, np.uint8):
                return "optical"
            if np.issubdtype(dtype, np.uint16):
                return "optical"
            return "optical"
        elif bands >= 4:
            return "multispectral"
        return "unknown"
