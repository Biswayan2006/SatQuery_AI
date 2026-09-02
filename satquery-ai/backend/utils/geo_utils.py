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
