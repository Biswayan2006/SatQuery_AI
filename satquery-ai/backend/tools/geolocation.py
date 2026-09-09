"""
SatQuery AI — Deterministic Tool Layer: Geolocation Tools
=========================================================
Coordinate / extent computations that are only possible when the raster carries
a CRS + affine transform.  Without geolocation metadata every one of these tools
returns a structured ``unsupported`` result — we never invent coordinates.

  * :class:`PixelToLatLonTool`      — a detected pixel → (lat, lon) [+ bbox]
  * :class:`ImageBoundsTool`         — geographic bounding box of the whole scene
  * :class:`GroundResolutionTool`    — ground sampling distance (metres/pixel)

All reprojection to WGS84 reuses the affine transform stored on the raster and,
where the CRS is projected, ``pyproj``.  The math is deterministic; the AI layer
only interprets the resulting numbers.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from .base import Tool, ToolResult
from .raster import RasterInput


def _require_geo(tool_name: str, raster: RasterInput) -> Optional[ToolResult]:
    """Return an ``unsupported`` ToolResult if the raster lacks CRS+transform."""
    if not raster.has_geo:
        return ToolResult.unsupported(
            tool_name,
            reason="raster has no CRS + affine transform; geographic coordinates "
                   "cannot be computed",
            data={"has_crs": bool(raster.crs),
                  "has_transform": raster.transform is not None},
        )
    return None


def _affine_apply(transform: List[float], px: float, py: float) -> tuple:
    """Apply a 6-element affine [a,b,c,d,e,f] to a pixel *centre* (px+0.5, py+0.5)."""
    a, b, c, d, e, f = transform[:6]
    x = a * (px + 0.5) + b * (py + 0.5) + c
    y = d * (px + 0.5) + e * (py + 0.5) + f
    return x, y


def _to_wgs84(crs_str: str, x: float, y: float) -> tuple:
    """Reproject a native (x, y) to (lat, lon) in WGS84; identity for geographic CRS."""
    if crs_str and "4326" not in str(crs_str):
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
            lon, lat = tr.transform(x, y)
            return float(lat), float(lon)
        except Exception:
            pass
    # Geographic CRS (or reprojection unavailable): native y≈lat, x≈lon.
    return float(y), float(x)


class PixelToLatLonTool(Tool):
    name = "pixel_to_latlon"
    description = "Convert a detected pixel coordinate to geographic (lat, lon)"

    def run(  # type: ignore[override]
        self,
        raster: RasterInput,
        pixel_x: float,
        pixel_y: float,
        pixel_bbox: Optional[Dict[str, float]] = None,
    ) -> ToolResult:
        blocked = _require_geo(self.name, raster)
        if blocked:
            return blocked

        px = float(pixel_x)
        py = float(pixel_y)
        nx, ny = _affine_apply(raster.transform, px, py)
        lat, lon = _to_wgs84(raster.crs, nx, ny)

        data: Dict[str, Any] = {
            "pixel": {"x": round(px, 3), "y": round(py, 3)},
            "geographic": {"lat": round(lat, 7), "lon": round(lon, 7)},
        }

        # Optional: convert a normalised or pixel bbox to a geographic envelope.
        if pixel_bbox:
            data["geographic_bbox"] = self._bbox_to_geo(raster, pixel_bbox)

        return ToolResult.success(
            self.name, data=data,
            provenance={"crs": raster.crs, "transform": list(raster.transform)},
        )

    def _bbox_to_geo(self, raster: RasterInput, bbox: Dict[str, float]) -> Dict[str, float]:
        # bbox may be normalised [0,1] or absolute pixels; detect by magnitude.
        keys = ("x1", "y1", "x2", "y2")
        if not all(k in bbox for k in keys):
            return {}
        normalized = all(0.0 <= float(bbox[k]) <= 1.0 for k in keys)
        w, h = raster.width, raster.height
        x1 = bbox["x1"] * w if normalized else bbox["x1"]
        y1 = bbox["y1"] * h if normalized else bbox["y1"]
        x2 = bbox["x2"] * w if normalized else bbox["x2"]
        y2 = bbox["y2"] * h if normalized else bbox["y2"]

        lats, lons = [], []
        for cx, cy in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
            nx, ny = _affine_apply(raster.transform, cx, cy)
            lat, lon = _to_wgs84(raster.crs, nx, ny)
            lats.append(lat)
            lons.append(lon)
        return {
            "min_lat": round(min(lats), 7), "max_lat": round(max(lats), 7),
            "min_lon": round(min(lons), 7), "max_lon": round(max(lons), 7),
            "center_lat": round((min(lats) + max(lats)) / 2, 7),
            "center_lon": round((min(lons) + max(lons)) / 2, 7),
        }


class ImageBoundsTool(Tool):
    name = "image_bounds"
    description = "Geographic bounding box (WGS84) of the whole raster scene"

    def run(self, raster: RasterInput) -> ToolResult:  # type: ignore[override]
        blocked = _require_geo(self.name, raster)
        if blocked:
            return blocked

        w, h = raster.width, raster.height
        # Four scene corners in pixel space (top-left origin).
        corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
        lats, lons = [], []
        for cx, cy in corners_px:
            # Corners, not centres: apply affine directly without the +0.5 shift.
            a, b, c, d, e, f = raster.transform[:6]
            nx = a * cx + b * cy + c
            ny = d * cx + e * cy + f
            lat, lon = _to_wgs84(raster.crs, nx, ny)
            lats.append(lat)
            lons.append(lon)

        west, east = min(lons), max(lons)
        south, north = min(lats), max(lats)
        return ToolResult.success(
            self.name,
            data={
                "west": round(west, 7), "south": round(south, 7),
                "east": round(east, 7), "north": round(north, 7),
                "center_lon": round((west + east) / 2, 7),
                "center_lat": round((south + north) / 2, 7),
                "width_px": w, "height_px": h,
            },
            provenance={"crs": raster.crs, "transform": list(raster.transform)},
        )


class GroundResolutionTool(Tool):
    name = "ground_resolution"
    description = "Estimate ground sampling distance (metres per pixel) in x and y"

    def run(self, raster: RasterInput) -> ToolResult:  # type: ignore[override]
        blocked = _require_geo(self.name, raster)
        if blocked:
            return blocked

        # Native pixel size from the affine transform.
        a, b, c, d, e, f = raster.transform[:6]
        native_res_x = math.hypot(a, d)
        native_res_y = math.hypot(b, e)

        crs = str(raster.crs)
        is_geographic = "4326" in crs or "longlat" in crs.lower()

        if is_geographic:
            # Degrees → metres via haversine along one pixel step at the scene centre.
            lat0, lon0 = _to_wgs84(crs, c, f)
            # One pixel east and one pixel south.
            nx_e = a * 1 + b * 0 + c
            ny_e = d * 1 + e * 0 + f
            lat_e, lon_e = _to_wgs84(crs, nx_e, ny_e)
            nx_s = a * 0 + b * 1 + c
            ny_s = d * 0 + e * 1 + f
            lat_s, lon_s = _to_wgs84(crs, nx_s, ny_s)
            res_x_m = _haversine_m(lat0, lon0, lat_e, lon_e)
            res_y_m = _haversine_m(lat0, lon0, lat_s, lon_s)
            unit = "degrees_native"
        else:
            # Projected CRS: linear units are (almost always) metres.
            res_x_m = native_res_x
            res_y_m = native_res_y
            unit = "projected_linear"

        return ToolResult.success(
            self.name,
            data={
                "resolution_x_m": round(float(res_x_m), 4),
                "resolution_y_m": round(float(res_y_m), 4),
                "mean_resolution_m": round(float((res_x_m + res_y_m) / 2), 4),
                "native_resolution_x": round(float(native_res_x), 8),
                "native_resolution_y": round(float(native_res_y), 8),
                "native_unit": unit,
            },
            provenance={"crs": raster.crs, "transform": list(raster.transform)},
        )


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return r * 2 * math.asin(math.sqrt(h))
