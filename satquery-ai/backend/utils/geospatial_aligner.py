"""
SatQuery AI — Geospatial Alignment Pipeline
Ensures two satellite images are spatially co-registered before pairwise
analysis (change detection, SAR-optical fusion).

Pipeline per image pair:
  1. Read geospatial metadata (CRS, bounds, resolution, transform)
  2. Validate CRS compatibility
  3. Validate geographic overlap
  4. Reproject image B to image A's CRS if needed
  5. Resample to a common pixel resolution (min GSD of the two)
  6. Crop/pad to the common spatial extent (intersection)
  7. Return aligned PIL images + alignment metadata

For plain PNG/JPEG pairs (no CRS):
  - Fall back to pixel-level size alignment via geo_utils.align_images_by_size
  - Mark geographic metadata as unavailable

All operations are read-only on the original files; intermediate outputs are
written to temporary files and cleaned up after use.
"""
from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger("satquery.geospatial_aligner")

# Minimum overlap fraction required to accept a pair for change detection.
# If the two images share less than this fraction of the smaller footprint,
# we consider them geographically incompatible.
MIN_OVERLAP_FRACTION = 0.10


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ImageGeoMeta:
    """
    Geospatial metadata for a single image.
    Populated from rasterio; None values indicate non-GeoTIFF or missing info.
    """
    path: str
    has_geo: bool                          # True if CRS + transform available
    crs_wkt: Optional[str] = None          # CRS as WKT string
    crs_epsg: Optional[int] = None         # EPSG code if parseable
    transform: Optional[List[float]] = None  # 6-element affine [a,b,c,d,e,f]
    bounds_native: Optional[List[float]] = None   # [west, south, east, north] in native CRS
    bounds_wgs84: Optional[List[float]] = None    # [west, south, east, north] in WGS84
    width: int = 0
    height: int = 0
    bands: int = 0
    res_x: Optional[float] = None         # pixel width in CRS units
    res_y: Optional[float] = None         # pixel height (absolute) in CRS units
    dtype: str = "uint8"
    nodata: Optional[float] = None


@dataclass
class AlignmentResult:
    """
    Output of the alignment pipeline.

    pil_a / pil_b  : aligned RGB PIL images ready for model inference
    meta_a / meta_b: updated geospatial metadata after alignment
    info           : dict suitable for insertion into ExecutionSummary.alignment
    error          : non-None if alignment failed (incompatible pair)
    """
    pil_a: Optional[Image.Image]
    pil_b: Optional[Image.Image]
    meta_a: ImageGeoMeta
    meta_b: ImageGeoMeta
    array_a: Optional[np.ndarray] = None
    array_b: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


# ── Public API ────────────────────────────────────────────────────────────────

def align_image_pair(
    image_data_a: dict,
    image_data_b: dict,
) -> AlignmentResult:
    """
    Align two images for pairwise analysis.

    Parameters
    ----------
    image_data_a, image_data_b :
        Dicts as returned by ``image_utils.load_image()``.
        Must contain at minimum: ``pil_image``, ``is_geotiff``,
        ``metadata`` (with optional ``crs``, ``transform``).

    Returns
    -------
    AlignmentResult
        On success: aligned PIL images + alignment metadata dict.
        On failure: ``result.error`` contains a human-readable explanation.
    """
    path_a = image_data_a.get("_path", "")
    path_b = image_data_b.get("_path", "")

    meta_a = _read_geo_meta(path_a, image_data_a)
    meta_b = _read_geo_meta(path_b, image_data_b)

    logger.info(
        "Alignment: A=%s (geo=%s, %dx%d) | B=%s (geo=%s, %dx%d)",
        os.path.basename(path_a), meta_a.has_geo, meta_a.width, meta_a.height,
        os.path.basename(path_b), meta_b.has_geo, meta_b.width, meta_b.height,
    )

    # ── Case 1: Neither image has geospatial info ─────────────────────────────
    if not meta_a.has_geo and not meta_b.has_geo:
        return _pixel_level_alignment(image_data_a, image_data_b, meta_a, meta_b)

    # ── Case 2: Only one image has geo info ───────────────────────────────────
    if meta_a.has_geo != meta_b.has_geo:
        logger.warning(
            "Mixed geo/non-geo pair — falling back to pixel-level alignment; "
            "geographic validation skipped"
        )
        return _pixel_level_alignment(
            image_data_a, image_data_b, meta_a, meta_b,
            note="One image lacks CRS; geographic validation skipped",
        )

    # ── Case 3: Both images have geo info ─────────────────────────────────────
    return _geo_alignment(path_a, path_b, image_data_a, image_data_b, meta_a, meta_b)


def read_image_geo_meta(path: str, image_data: dict) -> ImageGeoMeta:
    """
    Public helper: return ImageGeoMeta for an image.
    Exposed so controller / tests can call it directly.
    """
    return _read_geo_meta(path, image_data)


# ── Geo metadata reader ───────────────────────────────────────────────────────

def _read_geo_meta(path: str, image_data: dict) -> ImageGeoMeta:
    """
    Extract full geospatial metadata from a loaded image dict.
    Falls back gracefully when rasterio is unavailable or CRS is absent.
    """
    metadata = image_data.get("metadata", {})
    is_geotiff = image_data.get("is_geotiff", False)
    pil: Optional[Image.Image] = image_data.get("pil_image")
    arr = image_data.get("numpy_array")

    # Pixel dimensions from array or PIL
    if arr is not None:
        h, w = arr.shape[:2]
        bands = image_data.get("bands", arr.shape[2] if arr.ndim == 3 else 1)
    elif pil is not None:
        w, h = pil.size
        bands = image_data.get("bands", len(pil.getbands()))
    else:
        h, w, bands = 0, 0, 0

    crs_str = metadata.get("crs")
    transform_list = metadata.get("transform")  # 6-element list from load_image()
    nodata = metadata.get("nodata")
    dtype = metadata.get("dtype", "uint8")

    if not is_geotiff or not crs_str or not transform_list:
        return ImageGeoMeta(
            path=path, has_geo=False,
            width=w, height=h, bands=bands, dtype=dtype, nodata=nodata,
        )

    # Parse CRS + bounds from rasterio if file exists on disk
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        from affine import Affine

        a, b, c, d, e, f = transform_list[:6]
        aff = Affine(a, b, c, d, e, f)

        # Recompute bounds from transform + dimensions (more reliable than stored)
        if path and os.path.exists(path):
            with rasterio.open(path) as src:
                crs = src.crs
                native_bounds = list(src.bounds)   # [left, bottom, right, top]
                w_actual = src.width
                h_actual = src.height
                res_x = abs(src.transform.a)
                res_y = abs(src.transform.e)
                crs_wkt = crs.to_wkt() if crs else None
                try:
                    crs_epsg = crs.to_epsg()
                except Exception:
                    crs_epsg = None
                try:
                    wgs84_bounds = list(transform_bounds(crs, "EPSG:4326", *native_bounds))
                except Exception:
                    wgs84_bounds = None
        else:
            # Reconstruct from stored transform — less reliable but OK
            left = c
            top = f
            right = c + a * w
            bottom = f + e * h
            native_bounds = [min(left, right), min(top, bottom),
                             max(left, right), max(top, bottom)]
            res_x = abs(a)
            res_y = abs(e)
            w_actual, h_actual = w, h
            crs_wkt = crs_str
            crs_epsg = None
            wgs84_bounds = None

        return ImageGeoMeta(
            path=path,
            has_geo=True,
            crs_wkt=crs_wkt,
            crs_epsg=crs_epsg,
            transform=transform_list,
            bounds_native=native_bounds,
            bounds_wgs84=wgs84_bounds,
            width=w_actual,
            height=h_actual,
            bands=bands,
            res_x=res_x,
            res_y=res_y,
            dtype=dtype,
            nodata=nodata,
        )

    except Exception as exc:
        logger.debug("Geo meta extraction failed for %s: %s", path, exc)
        return ImageGeoMeta(
            path=path, has_geo=False,
            width=w, height=h, bands=bands, dtype=dtype, nodata=nodata,
        )


# ── Pixel-level alignment (no CRS) ───────────────────────────────────────────

def _pixel_level_alignment(
    image_data_a: dict,
    image_data_b: dict,
    meta_a: ImageGeoMeta,
    meta_b: ImageGeoMeta,
    note: str = "",
) -> AlignmentResult:
    """
    Align two images without CRS using CV registration, with a resize fallback.
    """
    from utils.geo_utils import align_arrays_by_size, align_images_by_cv, align_images_by_size
    from utils.image_utils import normalize_to_rgb

    pil_a = image_data_a.get("pil_image") or normalize_to_rgb(
        image_data_a["numpy_array"], image_data_a.get("bands", 3)
    )
    pil_b = image_data_b.get("pil_image") or normalize_to_rgb(
        image_data_b["numpy_array"], image_data_b.get("bands", 3)
    )

    orig_size_a = pil_a.size
    orig_size_b = pil_b.size

    array_a = image_data_a.get("numpy_array")
    array_b = image_data_b.get("numpy_array")

    try:
        aligned_a, aligned_b, array_a, array_b, cv_method, cv_score = align_images_by_cv(
            pil_a, pil_b, array_a, array_b
        )
        method = cv_method
        note_text = note or "No CRS available; applied CV feature alignment fallback"
    except Exception as exc:
        logger.warning("CV alignment failed: %s; using resize fallback", exc)
        aligned_a, aligned_b = align_images_by_size(pil_a, pil_b, method="resize_to_smaller")
        if array_a is not None and array_b is not None:
            array_a, array_b = align_arrays_by_size(array_a, array_b)
        method = "pixel_resize_to_smaller"
        cv_score = None
        note_text = note or f"CV alignment unavailable ({exc}); pixel resize applied"

    method = "pixel_resize_to_smaller"
    info: dict = {
        "performed": orig_size_a != aligned_a.size or orig_size_b != aligned_b.size,
        "method": method,
        "source_crs": None,
        "target_crs": None,
        "target_resolution": None,
        "common_bounds": None,
        "original_sizes": {
            "image_a": list(orig_size_a),
            "image_b": list(orig_size_b),
        },
        "aligned_size": list(aligned_a.size),
        "geographic_coordinates_available": False,
        "cv_alignment_score": round(cv_score, 6) if cv_score is not None else None,
        "note": note_text,
    }

    logger.info(
        "Pixel-level alignment: A %s→%s, B %s→%s",
        orig_size_a, aligned_a.size, orig_size_b, aligned_b.size,
    )
    return AlignmentResult(
        pil_a=aligned_a, pil_b=aligned_b,
        array_a=array_a,
        array_b=array_b,
        meta_a=meta_a, meta_b=meta_b,
        info=info,
    )


# ── Full geospatial alignment ─────────────────────────────────────────────────

def _geo_alignment(
    path_a: str,
    path_b: str,
    image_data_a: dict,
    image_data_b: dict,
    meta_a: ImageGeoMeta,
    meta_b: ImageGeoMeta,
) -> AlignmentResult:
    """
    Full geospatial alignment when both images have CRS information.

    Steps:
      1. Validate CRS (reproject B → A's CRS if needed)
      2. Validate geographic overlap
      3. Compute common extent (intersection in target CRS)
      4. Resample both images to common resolution + common extent
      5. Return PIL images from aligned arrays
    """
    try:
        import rasterio
        from rasterio.warp import (
            calculate_default_transform,
            reproject,
            Resampling,
            transform_bounds,
        )
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError as exc:
        logger.error("rasterio not available: %s", exc)
        return _pixel_level_alignment(
            image_data_a, image_data_b, meta_a, meta_b,
            note="rasterio unavailable; fell back to pixel alignment",
        )

    # ── 1. CRS validation / reprojection ──────────────────────────────────────
    target_crs_str = meta_a.crs_wkt  # A defines the target CRS
    needs_reproject = False

    if meta_a.crs_wkt != meta_b.crs_wkt:
        # Check EPSG shortcut first (avoids full WKT comparison)
        if meta_a.crs_epsg and meta_b.crs_epsg and meta_a.crs_epsg == meta_b.crs_epsg:
            needs_reproject = False
        else:
            needs_reproject = True
            logger.info(
                "CRS mismatch: A=%s, B=%s — will reproject B to A's CRS",
                _short_crs(meta_a.crs_wkt), _short_crs(meta_b.crs_wkt),
            )

    # ── 2. Geographic overlap validation ──────────────────────────────────────
    # Convert both bounds to WGS84 for a universal overlap check
    try:
        bounds_a_wgs84 = _to_wgs84(meta_a.bounds_native, meta_a.crs_wkt)
        bounds_b_wgs84 = _to_wgs84(meta_b.bounds_native, meta_b.crs_wkt)
    except Exception as exc:
        logger.warning("Bounds reprojection to WGS84 failed: %s — skipping overlap check", exc)
        bounds_a_wgs84 = meta_a.bounds_wgs84
        bounds_b_wgs84 = meta_b.bounds_wgs84

    if bounds_a_wgs84 and bounds_b_wgs84:
        overlap_ok, overlap_fraction, overlap_bounds_wgs84 = _check_overlap(
            bounds_a_wgs84, bounds_b_wgs84
        )
        if not overlap_ok:
            msg = (
                f"Images have insufficient geographic overlap "
                f"({overlap_fraction * 100:.1f}% < {MIN_OVERLAP_FRACTION * 100:.0f}% minimum). "
                f"Image A covers {_fmt_bounds(bounds_a_wgs84)}, "
                f"Image B covers {_fmt_bounds(bounds_b_wgs84)}. "
                "Cannot perform reliable change detection on non-overlapping scenes."
            )
            logger.warning("Alignment failed: %s", msg)
            return AlignmentResult(
                pil_a=None, pil_b=None,
                meta_a=meta_a, meta_b=meta_b,
                info={"performed": False, "geographic_coordinates_available": True},
                error=msg,
            )
    else:
        overlap_fraction = None
        overlap_bounds_wgs84 = None

    # ── 3. Target resolution = minimum GSD (finest available) ─────────────────
    res_a = meta_a.res_x or 1.0
    res_b_in_a_crs = _approx_res_in_crs(meta_b, meta_a.crs_wkt) if needs_reproject else (meta_b.res_x or 1.0)
    target_res = min(res_a, res_b_in_a_crs)

    # ── 4. Compute common extent in target CRS ────────────────────────────────
    if needs_reproject:
        from rasterio.warp import transform_bounds as tb
        bounds_b_in_a_crs = tb(meta_b.crs_wkt, target_crs_str, *meta_b.bounds_native)
    else:
        bounds_b_in_a_crs = meta_b.bounds_native

    bounds_a = meta_a.bounds_native  # [left, bottom, right, top]

    common_left   = max(bounds_a[0], bounds_b_in_a_crs[0])
    common_bottom = max(bounds_a[1], bounds_b_in_a_crs[1])
    common_right  = min(bounds_a[2], bounds_b_in_a_crs[2])
    common_top    = min(bounds_a[3], bounds_b_in_a_crs[3])

    if common_right <= common_left or common_top <= common_bottom:
        msg = (
            "Images have no spatial intersection in the target CRS. "
            "Change detection requires overlapping geographic extents."
        )
        logger.warning("Alignment failed: %s", msg)
        return AlignmentResult(
            pil_a=None, pil_b=None,
            meta_a=meta_a, meta_b=meta_b,
            info={"performed": False, "geographic_coordinates_available": True},
            error=msg,
        )

    # Output grid dimensions at target_res
    out_w = max(1, int(round((common_right - common_left) / target_res)))
    out_h = max(1, int(round((common_top - common_bottom) / target_res)))
    out_transform = from_bounds(common_left, common_bottom, common_right, common_top, out_w, out_h)

    logger.info(
        "Common extent: L=%.4f B=%.4f R=%.4f T=%.4f | res=%.4f | grid=%dx%d",
        common_left, common_bottom, common_right, common_top,
        target_res, out_w, out_h,
    )

    # ── 5. Resample both images to the common grid ────────────────────────────
    try:
        arr_a = _resample_to_grid(
            path_a, target_crs_str, out_transform, out_w, out_h,
            src_nodata=meta_a.nodata,
        )
        arr_b = _resample_to_grid(
            path_b, target_crs_str, out_transform, out_w, out_h,
            src_nodata=meta_b.nodata,
            src_override_crs=meta_b.crs_wkt if needs_reproject else None,
        )
    except Exception as exc:
        logger.error("Resampling failed: %s", exc)
        # Degrade gracefully to pixel-level alignment
        return _pixel_level_alignment(
            image_data_a, image_data_b, meta_a, meta_b,
            note=f"Rasterio resampling failed ({exc}); fell back to pixel alignment",
        )

    # ── 6. Convert arrays to PIL RGB ──────────────────────────────────────────
    from utils.image_utils import normalize_to_rgb

    bands_a = arr_a.shape[0] if arr_a.ndim == 3 else 1
    bands_b = arr_b.shape[0] if arr_b.ndim == 3 else 1

    # arr_a is [C, H, W]; normalize_to_rgb expects [H, W, C]
    arr_a_hwc = arr_a.transpose(1, 2, 0) if arr_a.ndim == 3 else arr_a[:, :, np.newaxis]
    arr_b_hwc = arr_b.transpose(1, 2, 0) if arr_b.ndim == 3 else arr_b[:, :, np.newaxis]

    pil_a_aligned = normalize_to_rgb(arr_a_hwc, bands_a)
    pil_b_aligned = normalize_to_rgb(arr_b_hwc, bands_b)

    # ── 7. Build updated metadata ─────────────────────────────────────────────
    new_transform = list(out_transform)[:6]
    common_bounds_native = [common_left, common_bottom, common_right, common_top]

    try:
        from rasterio.warp import transform_bounds as tb
        common_bounds_wgs84 = list(tb(target_crs_str, "EPSG:4326", *common_bounds_native))
    except Exception:
        common_bounds_wgs84 = None

    meta_a_out = ImageGeoMeta(
        path=meta_a.path, has_geo=True,
        crs_wkt=target_crs_str, crs_epsg=meta_a.crs_epsg,
        transform=new_transform,
        bounds_native=common_bounds_native,
        bounds_wgs84=common_bounds_wgs84,
        width=out_w, height=out_h,
        bands=bands_a, res_x=target_res, res_y=target_res,
        dtype=str(arr_a.dtype), nodata=meta_a.nodata,
    )
    meta_b_out = ImageGeoMeta(
        path=meta_b.path, has_geo=True,
        crs_wkt=target_crs_str, crs_epsg=meta_a.crs_epsg,
        transform=new_transform,
        bounds_native=common_bounds_native,
        bounds_wgs84=common_bounds_wgs84,
        width=out_w, height=out_h,
        bands=bands_b, res_x=target_res, res_y=target_res,
        dtype=str(arr_b.dtype), nodata=meta_b.nodata,
    )

    info: dict = {
        "performed": True,
        "method": "reproject_and_resample" if needs_reproject else "resample_to_common_extent",
        "source_crs": _short_crs(meta_b.crs_wkt) if needs_reproject else _short_crs(meta_a.crs_wkt),
        "target_crs": _short_crs(target_crs_str),
        "target_resolution": round(target_res, 6),
        "common_bounds": common_bounds_native,
        "common_bounds_wgs84": common_bounds_wgs84,
        "output_grid": {"width": out_w, "height": out_h},
        "overlap_fraction": round(overlap_fraction, 4) if overlap_fraction is not None else None,
        "geographic_coordinates_available": True,
    }

    logger.info(
        "Geo alignment complete: method=%s, grid=%dx%d, res=%.4f",
        info["method"], out_w, out_h, target_res,
    )
    return AlignmentResult(
        pil_a=pil_a_aligned, pil_b=pil_b_aligned,
        array_a=arr_a.transpose(1, 2, 0) if arr_a.ndim == 3 else arr_a,
        array_b=arr_b.transpose(1, 2, 0) if arr_b.ndim == 3 else arr_b,
        meta_a=meta_a_out, meta_b=meta_b_out,
        info=info,
    )


# ── Rasterio resampling helper ────────────────────────────────────────────────

def _resample_to_grid(
    src_path: str,
    dst_crs_str: str,
    dst_transform,
    dst_width: int,
    dst_height: int,
    src_nodata: Optional[float] = None,
    src_override_crs: Optional[str] = None,
) -> np.ndarray:
    """
    Open src_path with rasterio, reproject + resample to the given grid.
    Returns [C, H, W] float32 array.

    src_override_crs: if the file's embedded CRS should be overridden
    (used when the stored CRS differs from what we detected).
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS

    dst_crs = CRS.from_user_input(dst_crs_str)

    with rasterio.open(src_path) as src:
        src_crs = CRS.from_user_input(src_override_crs) if src_override_crs else src.crs
        bands = src.count
        out = np.zeros((bands, dst_height, dst_width), dtype=np.float32)

        for band_idx in range(1, bands + 1):
            src_data = src.read(band_idx).astype(np.float32)
            reproject(
                source=src_data,
                destination=out[band_idx - 1],
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                src_nodata=src_nodata if src_nodata is not None else src.nodata,
                dst_nodata=0.0,
            )

    return out


# ── Overlap helpers ───────────────────────────────────────────────────────────

def _check_overlap(
    bounds_a: List[float],
    bounds_b: List[float],
) -> Tuple[bool, float, Optional[List[float]]]:
    """
    Check geographic overlap between two bounding boxes (both in the same CRS).
    bounds format: [west, south, east, north]

    Returns (is_sufficient, overlap_fraction, overlap_bounds_or_None).
    overlap_fraction is relative to the smaller image's area.
    """
    inter_w = max(0.0, min(bounds_a[2], bounds_b[2]) - max(bounds_a[0], bounds_b[0]))
    inter_h = max(0.0, min(bounds_a[3], bounds_b[3]) - max(bounds_a[1], bounds_b[1]))
    inter_area = inter_w * inter_h

    area_a = (bounds_a[2] - bounds_a[0]) * (bounds_a[3] - bounds_a[1])
    area_b = (bounds_b[2] - bounds_b[0]) * (bounds_b[3] - bounds_b[1])
    smaller_area = min(area_a, area_b)

    if smaller_area <= 0:
        return False, 0.0, None

    overlap_fraction = inter_area / smaller_area

    if inter_area <= 0:
        overlap_bounds = None
    else:
        overlap_bounds = [
            max(bounds_a[0], bounds_b[0]),
            max(bounds_a[1], bounds_b[1]),
            min(bounds_a[2], bounds_b[2]),
            min(bounds_a[3], bounds_b[3]),
        ]

    return overlap_fraction >= MIN_OVERLAP_FRACTION, overlap_fraction, overlap_bounds


def _to_wgs84(bounds_native: List[float], crs_wkt: str) -> Optional[List[float]]:
    """Reproject bounds [L,B,R,T] from native CRS to WGS84."""
    try:
        from rasterio.warp import transform_bounds
        return list(transform_bounds(crs_wkt, "EPSG:4326", *bounds_native))
    except Exception:
        return None


def _approx_res_in_crs(meta: ImageGeoMeta, target_crs_wkt: str) -> float:
    """
    Approximate the pixel resolution of meta in the target CRS units.
    Used to decide the output resolution when reprojecting.
    Falls back to meta.res_x if transform is not possible.
    """
    if not meta.bounds_wgs84 or not meta.res_x:
        return meta.res_x or 1.0
    try:
        from rasterio.warp import transform_bounds
        bounds_in_target = transform_bounds("EPSG:4326", target_crs_wkt, *meta.bounds_wgs84)
        extent_x = abs(bounds_in_target[2] - bounds_in_target[0])
        if meta.width > 0:
            return extent_x / meta.width
    except Exception:
        pass
    return meta.res_x or 1.0


# ── String helpers ────────────────────────────────────────────────────────────

def _short_crs(crs_wkt: Optional[str]) -> Optional[str]:
    """Return a short CRS identifier (EPSG:XXXX if parseable, else first 60 chars)."""
    if not crs_wkt:
        return None
    try:
        from rasterio.crs import CRS
        epsg = CRS.from_user_input(crs_wkt).to_epsg()
        if epsg:
            return f"EPSG:{epsg}"
    except Exception:
        pass
    return crs_wkt[:60]


def _fmt_bounds(bounds: List[float]) -> str:
    return f"[{bounds[0]:.4f},{bounds[1]:.4f},{bounds[2]:.4f},{bounds[3]:.4f}]"
