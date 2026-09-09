"""
SatQuery AI — Geospatial Utilities
Helper functions for coordinate transforms and spatial operations.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("satquery.geo_utils")


# ── Coordinate conversion ─────────────────────────────────────────────────────

def pixel_to_latlon(
    pixel_x: float,
    pixel_y: float,
    transform,
    crs_str: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to (latitude, longitude) using a rasterio-style
    affine transform.

    transform: list of 6 affine coefficients [a, b, c, d, e, f]
    Returns (lat, lon) in WGS84 if CRS is provided, else (x, y) in native CRS.
    """
    try:
        from affine import Affine

        if len(transform) >= 6:
            a, b, c, d, e, f = transform[:6]
            aff = Affine(a, b, c, d, e, f)
        else:
            raise ValueError("Transform must have at least 6 coefficients")

        # Pixel centre → native coordinates
        native_x, native_y = aff * (pixel_x + 0.5, pixel_y + 0.5)

        if crs_str and "4326" not in crs_str:
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(native_x, native_y)
                return lat, lon
            except Exception:
                pass

        return native_y, native_x  # return as (lat, lon) ≈ (y, x) for geographic CRS

    except Exception as exc:
        logger.debug("pixel_to_latlon failed: %s", exc)
        return float(pixel_y), float(pixel_x)


def normalize_bbox_to_pixel(
    bbox_norm: List[float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert normalised [0,1] bounding box to absolute pixel coordinates.

    bbox_norm: [x1, y1, x2, y2] in [0, 1]
    Returns: (x1_px, y1_px, x2_px, y2_px) in pixels
    """
    x1, y1, x2, y2 = bbox_norm
    return (
        int(x1 * image_width),
        int(y1 * image_height),
        int(x2 * image_width),
        int(y2 * image_height),
    )


def compute_overlap(
    box1: List[float],
    box2: List[float],
) -> float:
    """
    Compute Intersection-over-Union (IoU) for two bounding boxes.
    Boxes in [x1, y1, x2, y2] format (normalised or pixel).
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


# ── Image alignment ───────────────────────────────────────────────────────────

def align_images_by_size(
    img1: "PIL.Image.Image",
    img2: "PIL.Image.Image",
    method: str = "resize_to_smaller",
) -> Tuple["PIL.Image.Image", "PIL.Image.Image"]:
    """
    Ensure two images have the same spatial dimensions for pairwise analysis.

    method:
      - "resize_to_smaller": downsample the larger image
      - "resize_to_larger": upsample the smaller image
      - "resize_to_fixed": resize both to 512x512
    """
    from PIL import Image

    w1, h1 = img1.size
    w2, h2 = img2.size

    if w1 == w2 and h1 == h2:
        return img1, img2

    if method == "resize_to_smaller":
        target_w = min(w1, w2)
        target_h = min(h1, h2)
    elif method == "resize_to_larger":
        target_w = max(w1, w2)
        target_h = max(h1, h2)
    else:  # fixed
        target_w = target_h = 512

    return (
        img1.resize((target_w, target_h), Image.BICUBIC),
        img2.resize((target_w, target_h), Image.BICUBIC),
    )


def align_arrays_by_size(
    array1: np.ndarray,
    array2: np.ndarray,
    method: str = "resize_to_smaller",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize numeric image arrays to the same spatial dimensions.

    Channels are resized independently so SAR values and polarization
    structure are preserved for downstream model preprocessing.
    """
    from PIL import Image

    if array1.ndim not in (2, 3) or array2.ndim not in (2, 3):
        raise ValueError("Image arrays must have shape [H, W] or [H, W, C]")

    h1, w1 = array1.shape[:2]
    h2, w2 = array2.shape[:2]
    if method == "resize_to_larger":
        target_w, target_h = max(w1, w2), max(h1, h2)
    elif method == "resize_to_fixed":
        target_w = target_h = 512
    else:
        target_w, target_h = min(w1, w2), min(h1, h2)

    def resize(array: np.ndarray) -> np.ndarray:
        if array.shape[:2] == (target_h, target_w):
            return array
        channels = array[:, :, np.newaxis] if array.ndim == 2 else array
        resized = [
            np.asarray(
                Image.fromarray(channel.astype(np.float32), mode="F").resize(
                    (target_w, target_h), Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            for channel in np.moveaxis(channels, 2, 0)
        ]
        output = np.stack(resized, axis=2)
        return output[:, :, 0] if array.ndim == 2 else output

    return resize(array1), resize(array2)


def align_images_by_cv(
    image1: "PIL.Image.Image",
    image2: "PIL.Image.Image",
    array1: Optional[np.ndarray] = None,
    array2: Optional[np.ndarray] = None,
) -> Tuple["PIL.Image.Image", "PIL.Image.Image", Optional[np.ndarray], Optional[np.ndarray], str, float]:
    """Register image2 to image1 using ECC, with phase correlation fallback."""
    import cv2
    from PIL import Image

    target_size = image1.size
    gray1 = np.asarray(image1.convert("L"), dtype=np.float32)
    gray2 = np.asarray(image2.convert("L").resize(target_size, Image.Resampling.BILINEAR), dtype=np.float32)
    gray1 = cv2.normalize(gray1, None, 0.0, 1.0, cv2.NORM_MINMAX)
    gray2 = cv2.normalize(gray2, None, 0.0, 1.0, cv2.NORM_MINMAX)

    warp = np.eye(2, 3, dtype=np.float32)
    method = "cv_ecc_affine"
    score = 0.0
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        score, warp = cv2.findTransformECC(
            gray1, gray2, warp, cv2.MOTION_AFFINE, criteria, None, 1
        )
    except cv2.error:
        shift, score = cv2.phaseCorrelate(gray1, gray2)
        warp = np.array([[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]]], dtype=np.float32)
        method = "cv_phase_correlation"

    def warp_array(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None:
            return None
        source = array.astype(np.float32, copy=False)
        channels = source[:, :, np.newaxis] if source.ndim == 2 else source
        warped = [
            cv2.warpAffine(
                cv2.resize(channel, target_size, interpolation=cv2.INTER_LINEAR)
                if channel.shape[:2] != (target_size[1], target_size[0]) else channel,
                warp, (target_size[0], target_size[1]),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for channel in np.moveaxis(channels, 2, 0)
        ]
        result = np.stack(warped, axis=2)
        return result[:, :, 0] if source.ndim == 2 else result

    aligned2 = cv2.warpAffine(
        np.asarray(image2.convert("RGB")), warp,
        (target_size[0], target_size[1]),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    aligned1_array = None
    if array1 is not None:
        aligned1_array = array1.astype(np.float32, copy=False)
        if aligned1_array.shape[:2] != (target_size[1], target_size[0]):
            channels = aligned1_array[:, :, np.newaxis] if aligned1_array.ndim == 2 else aligned1_array
            resized = [
                cv2.resize(channel, target_size, interpolation=cv2.INTER_LINEAR)
                for channel in np.moveaxis(channels, 2, 0)
            ]
            aligned1_array = np.stack(resized, axis=2)
            if array1.ndim == 2:
                aligned1_array = aligned1_array[:, :, 0]
    return (
        image1.convert("RGB"),
        Image.fromarray(aligned2.astype(np.uint8), mode="RGB"),
        aligned1_array,
        warp_array(array2),
        method,
        float(score),
    )


# ── Geospatial metadata ───────────────────────────────────────────────────────

def get_image_bounds(geotiff_path: str) -> Optional[Dict]:
    """
    Return the geographic bounding box of a GeoTIFF in WGS84 (lon/lat).
    Returns dict with keys: west, south, east, north.
    """
    try:
        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(geotiff_path) as src:
            if src.crs is None:
                return None
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
            west, south, east, north = bounds
            return {
                "west": round(west, 6),
                "south": round(south, 6),
                "east": round(east, 6),
                "north": round(north, 6),
                "center_lon": round((west + east) / 2, 6),
                "center_lat": round((south + north) / 2, 6),
            }
    except Exception as exc:
        logger.debug("get_image_bounds failed: %s", exc)
        return None


def estimate_ground_resolution(geotiff_path: str) -> Optional[float]:
    """Estimate ground sampling distance in metres."""
    try:
        import rasterio
        from rasterio.warp import transform

        with rasterio.open(geotiff_path) as src:
            if src.crs is None:
                return None
            # Transform two adjacent pixel centres to WGS84
            xs = [src.transform.c, src.transform.c + src.transform.a]
            ys = [src.transform.f, src.transform.f]
            lons, lats = transform(src.crs, "EPSG:4326", xs, ys)
            # Haversine
            dlat = np.radians(lats[1] - lats[0])
            dlon = np.radians(lons[1] - lons[0])
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(np.radians(lats[0])) * np.cos(np.radians(lats[1])) * np.sin(dlon / 2) ** 2
            )
            dist_m = 6371000 * 2 * np.arcsin(np.sqrt(a))
            return round(float(dist_m), 2)
    except Exception:
        return None


# ── Geo bounding-box helpers ──────────────────────────────────────────────────

def pixel_to_geo_bbox(
    pixel_bbox: dict,
    transform: list,
    crs_str: str,
    image_width: int,
    image_height: int,
) -> Optional[Dict]:
    """
    Convert a normalised pixel bounding-box dict to geographic coordinates.

    The pixel_bbox is expected to use normalised [0, 1] coordinates:
        {"x1": float, "y1": float, "x2": float, "y2": float, ...}

    transform : list of 6 affine coefficients [a, b, c, d, e, f]
        As stored in ImageGeoMeta.transform / metadata["transform"].
    crs_str   : CRS string (WKT, PROJ, or EPSG:XXXX).
    image_width, image_height : pixel dimensions of the *aligned* image.

    Returns a dict::

        {
            "min_lat": float, "max_lat": float,
            "min_lon": float, "max_lon": float,
            "center_lat": float, "center_lon": float,
        }

    or None if conversion fails (missing transform, projection error, etc.).
    """
    if not transform or not crs_str:
        return None

    try:
        x1_px = pixel_bbox["x1"] * image_width
        y1_px = pixel_bbox["y1"] * image_height
        x2_px = pixel_bbox["x2"] * image_width
        y2_px = pixel_bbox["y2"] * image_height

        # Four corners → convert each to lat/lon, then take envelope
        corners_px = [
            (x1_px, y1_px),
            (x2_px, y1_px),
            (x2_px, y2_px),
            (x1_px, y2_px),
        ]

        lats, lons = [], []
        for cx, cy in corners_px:
            lat, lon = pixel_to_latlon(cx, cy, transform, crs_str)
            lats.append(lat)
            lons.append(lon)

        min_lat = round(min(lats), 7)
        max_lat = round(max(lats), 7)
        min_lon = round(min(lons), 7)
        max_lon = round(max(lons), 7)

        return {
            "min_lat":    min_lat,
            "max_lat":    max_lat,
            "min_lon":    min_lon,
            "max_lon":    max_lon,
            "center_lat": round((min_lat + max_lat) / 2, 7),
            "center_lon": round((min_lon + max_lon) / 2, 7),
        }

    except Exception as exc:
        logger.debug("pixel_to_geo_bbox failed: %s", exc)
        return None


def convert_regions_to_geo_bbox(
    regions: list,
    transform: list,
    crs_str: str,
    image_width: int,
    image_height: int,
) -> list:
    """
    Enrich a list of change region dicts with geographic bounding boxes.

    Each region dict must already contain normalised pixel coordinates::

        {"x1": 0.1, "y1": 0.05, "x2": 0.4, "y2": 0.3, "area_pct": 12.5}

    Returns a new list where each item gains an optional ``geo_bbox`` key::

        {
            ...,
            "geo_bbox": {
                "min_lat": ..., "max_lat": ...,
                "min_lon": ..., "max_lon": ...,
                "center_lat": ..., "center_lon": ...,
            }
        }

    Regions where conversion fails receive ``"geo_bbox": None``.

    Parameters are the same as pixel_to_geo_bbox.
    """
    enriched = []
    for region in regions:
        geo = pixel_to_geo_bbox(
            region, transform, crs_str, image_width, image_height
        )
        enriched.append({**region, "geo_bbox": geo})
    return enriched
