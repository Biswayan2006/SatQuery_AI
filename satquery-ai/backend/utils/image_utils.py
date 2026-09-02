"""
SatQuery AI — Image Utilities
Handles loading, normalising, and converting satellite images.
"""
from __future__ import annotations

import base64
import io
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.image_utils")


# ── Primary loader ────────────────────────────────────────────────────────────

def load_image(path: str) -> Dict:
    """
    Load an image from disk, handling GeoTIFF and standard formats.

    Returns a dict with keys:
      pil_image:    PIL.Image (RGB)
      numpy_array:  np.ndarray [H, W, C]
      bands:        int
      shape:        list [H, W, C]
      is_geotiff:   bool
      metadata:     dict (CRS, transform if GeoTIFF)
    """
    ext = os.path.splitext(path)[1].lower()
    is_geotiff = ext in (".tif", ".tiff")

    if is_geotiff:
        return _load_geotiff(path)
    else:
        return _load_standard(path)


def _load_geotiff(path: str) -> Dict:
    try:
        import rasterio

        with rasterio.open(path) as src:
            bands = src.count
            height = src.height
            width = src.width
            arr = src.read()  # [C, H, W]
            metadata = {
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform)[:6] if src.transform else None,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
            }

        arr_hwc = arr.transpose(1, 2, 0)  # [H, W, C]
        pil_image = normalize_to_rgb(arr_hwc, bands)

        return {
            "pil_image": pil_image,
            "numpy_array": arr_hwc,
            "bands": bands,
            "shape": [height, width, bands],
            "is_geotiff": True,
            "metadata": metadata,
        }
    except Exception as exc:
        logger.warning("GeoTIFF load failed (%s): %s — falling back to PIL", path, exc)
        return _load_standard(path)


def _load_standard(path: str) -> Dict:
    with Image.open(path) as img:
        img.load()
        bands = len(img.getbands())
        arr = np.array(img)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        height, width = arr.shape[:2]
        pil_image = img.convert("RGB")

    return {
        "pil_image": pil_image,
        "numpy_array": arr,
        "bands": bands,
        "shape": [height, width, bands],
        "is_geotiff": False,
        "metadata": {},
    }


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize_to_rgb(image_array: np.ndarray, bands: int) -> Image.Image:
    """
    Convert any-band image array to an 8-bit RGB PIL image.

    Handles:
      - 1-band SAR: stretch to [0,255], replicate to 3 channels
      - 2-band SAR: use band 0 as gray, skip band 1
      - 3-band: treat as RGB, normalize
      - 4+ band: create false-color composite using bands 3,2,1 (0-indexed)
    """
    arr = image_array.copy().astype(np.float32)

    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    _, _, c = arr.shape

    if c == 1:
        rgb = _stretch_to_uint8(arr[:, :, 0])
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif c == 2:
        gray = _stretch_to_uint8(arr[:, :, 0])
        rgb = np.stack([gray, gray, gray], axis=-1)
    elif c == 3:
        rgb = np.stack([
            _stretch_to_uint8(arr[:, :, 0]),
            _stretch_to_uint8(arr[:, :, 1]),
            _stretch_to_uint8(arr[:, :, 2]),
        ], axis=-1)
    else:
        # False colour: use indices 3, 2, 1 (Sentinel-2 style)
        idx_r = min(3, c - 1)
        idx_g = min(2, c - 1)
        idx_b = min(1, c - 1)
        rgb = np.stack([
            _stretch_to_uint8(arr[:, :, idx_r]),
            _stretch_to_uint8(arr[:, :, idx_g]),
            _stretch_to_uint8(arr[:, :, idx_b]),
        ], axis=-1)

    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def create_rgb_composite(
    multi_band_array: np.ndarray,
    rgb_indices: List[int] = (3, 2, 1),
) -> Image.Image:
    """
    Create an RGB composite from a multi-band array.
    multi_band_array: [H, W, C] or [C, H, W]
    rgb_indices: band indices for R, G, B channels (0-indexed)
    """
    arr = multi_band_array.astype(np.float32)

    # Handle [C, H, W] input
    if arr.ndim == 3 and arr.shape[0] < arr.shape[1]:
        arr = arr.transpose(1, 2, 0)

    bands = arr.shape[2] if arr.ndim == 3 else 1
    idx_r, idx_g, idx_b = [min(i, bands - 1) for i in rgb_indices]

    rgb = np.stack([
        _stretch_to_uint8(arr[:, :, idx_r]),
        _stretch_to_uint8(arr[:, :, idx_g]),
        _stretch_to_uint8(arr[:, :, idx_b]),
    ], axis=-1)

    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _stretch_to_uint8(band: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Percentile stretch and convert band to uint8."""
    band = band.copy().astype(np.float64)
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo = np.percentile(valid, low_pct)
    hi = np.percentile(valid, high_pct)
    if hi <= lo:
        return np.zeros_like(band, dtype=np.uint8)
    stretched = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
    return stretched.astype(np.uint8)


# ── GeoTIFF export ────────────────────────────────────────────────────────────

def save_geotiff(
    array: np.ndarray,
    output_path: str,
    reference_geotiff_path: Optional[str] = None,
) -> None:
    """
    Save a numpy array as GeoTIFF, optionally inheriting spatial metadata.
    array: [H, W] or [H, W, C]
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds

        if array.ndim == 2:
            array = array[:, :, np.newaxis]

        height, width, channels = array.shape
        arr_chw = array.transpose(2, 0, 1)  # [C, H, W]

        meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": channels,
            "dtype": str(arr_chw.dtype),
        }

        if reference_geotiff_path and os.path.exists(reference_geotiff_path):
            with rasterio.open(reference_geotiff_path) as ref:
                meta["crs"] = ref.crs
                meta["transform"] = ref.transform

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(arr_chw)

        logger.info("GeoTIFF saved: %s", output_path)
    except Exception as exc:
        logger.error("GeoTIFF save failed: %s", exc)
        raise


# ── Base64 helpers ────────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL image."""
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes))
