"""
SatQuery AI — Deterministic Tool Layer: Change-Area & Connected-Region Tools
============================================================================
  * :class:`ChangeAreaTool`          — turn a binary change mask + pixel
    resolution into changed-pixel count, approximate changed area (m²/km²) and
    the percentage of the valid scene that changed.
  * :class:`ConnectedRegionStatisticsTool` — connected-component analysis of a
    binary mask: region count, per-region pixel/area/bbox/centroid, size stats.

Both operate on a boolean/0-1 mask that some upstream stage produced (a change
detector, a threshold on an index, …).  They are pure geometry + counting; no
model is asked to estimate an area it can compute exactly.

Area honesty
-----------
Area is only reported in ground units (m² / km²) when a positive per-pixel
ground resolution is supplied.  Without it the tools still report pixel counts
and percentages, but ``area_m2`` is ``None`` and a reason is attached — we do
not invent a pixel size.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .base import Tool, ToolResult

# OpenCV is used for connected components; degrade gracefully if unavailable.
try:  # pragma: no cover - import guard
    import cv2
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


class ChangeAreaTool(Tool):
    name = "change_area"
    description = (
        "Compute changed-pixel count, approximate changed area and percentage of "
        "the valid scene from a binary change mask + pixel ground resolution"
    )

    def run(  # type: ignore[override]
        self,
        change_mask: np.ndarray,
        pixel_resolution_m: Optional[float] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> ToolResult:
        m = np.asarray(change_mask)
        if m.ndim == 3:
            m = np.any(m != 0, axis=2)
        binary = m.astype(bool)
        total_pixels = int(binary.size)
        if total_pixels == 0:
            return ToolResult.unsupported(self.name, reason="empty change mask")

        # Valid-scene denominator: default is the full frame unless a valid mask
        # (e.g. non-nodata footprint) is given.
        if valid_mask is not None:
            vm = np.asarray(valid_mask).astype(bool)
            if vm.shape != binary.shape:
                return ToolResult.unsupported(
                    self.name,
                    reason=f"valid_mask shape {vm.shape} != change_mask shape {binary.shape}",
                )
            valid_pixels = int(vm.sum())
            changed_pixels = int((binary & vm).sum())
        else:
            valid_pixels = total_pixels
            changed_pixels = int(binary.sum())

        changed_pct = (100.0 * changed_pixels / valid_pixels) if valid_pixels else 0.0

        data: Dict[str, Any] = {
            "changed_pixels": changed_pixels,
            "valid_pixels": valid_pixels,
            "total_pixels": total_pixels,
            "changed_percentage": round(changed_pct, 4),
        }

        if pixel_resolution_m is not None and pixel_resolution_m > 0:
            px_area_m2 = float(pixel_resolution_m) ** 2
            area_m2 = changed_pixels * px_area_m2
            data["pixel_resolution_m"] = round(float(pixel_resolution_m), 4)
            data["pixel_area_m2"] = round(px_area_m2, 6)
            data["area_m2"] = round(area_m2, 4)
            data["area_km2"] = round(area_m2 / 1e6, 8)
            data["area_is_approximate"] = True
        else:
            data["area_m2"] = None
            data["area_km2"] = None
            data["area_reason"] = "no positive pixel_resolution_m supplied; area in ground units unavailable"

        return ToolResult.success(
            self.name, data=data,
            provenance={"area_formula": "changed_pixels * pixel_resolution_m^2"},
        )


class ConnectedRegionStatisticsTool(Tool):
    name = "connected_region_statistics"
    description = (
        "Connected-component analysis of a binary mask: region count, per-region "
        "pixel area / bbox / centroid and size statistics"
    )

    def run(  # type: ignore[override]
        self,
        mask: np.ndarray,
        pixel_resolution_m: Optional[float] = None,
        connectivity: int = 8,
        min_area_pixels: int = 1,
        max_regions: int = 50,
    ) -> ToolResult:
        m = np.asarray(mask)
        if m.ndim == 3:
            m = np.any(m != 0, axis=2)
        binary = m.astype(np.uint8)
        if binary.size == 0:
            return ToolResult.unsupported(self.name, reason="empty mask")

        h, w = binary.shape[:2]
        labels, stats, centroids = self._label(binary, connectivity)
        # Label 0 is background; iterate over foreground components.
        regions: List[Dict[str, Any]] = []
        px_area_m2 = (float(pixel_resolution_m) ** 2
                      if pixel_resolution_m and pixel_resolution_m > 0 else None)

        for lbl in range(1, len(stats)):
            x, y, bw, bh, area = stats[lbl]
            if area < min_area_pixels:
                continue
            cx, cy = centroids[lbl]
            region: Dict[str, Any] = {
                "area_pixels": int(area),
                "bbox_pixels": {"x1": int(x), "y1": int(y),
                                "x2": int(x + bw), "y2": int(y + bh)},
                "bbox_norm": {
                    "x1": round(x / w, 6), "y1": round(y / h, 6),
                    "x2": round((x + bw) / w, 6), "y2": round((y + bh) / h, 6),
                },
                "centroid_pixels": {"x": round(float(cx), 3), "y": round(float(cy), 3)},
                "area_fraction": round(float(area) / (h * w), 6),
            }
            if px_area_m2 is not None:
                region["area_m2"] = round(int(area) * px_area_m2, 4)
            regions.append(region)

        # Largest regions first; cap the reported list.
        regions.sort(key=lambda r: r["area_pixels"], reverse=True)
        truncated = len(regions) > max_regions
        reported = regions[:max_regions]

        areas = np.array([r["area_pixels"] for r in regions], dtype=np.float64)
        size_stats = {
            "region_count": int(len(regions)),
            "total_region_pixels": int(areas.sum()) if areas.size else 0,
            "largest_region_pixels": int(areas.max()) if areas.size else 0,
            "mean_region_pixels": round(float(areas.mean()), 4) if areas.size else 0.0,
            "median_region_pixels": round(float(np.median(areas)), 4) if areas.size else 0.0,
        }

        data: Dict[str, Any] = {
            **size_stats,
            "regions": reported,
            "regions_truncated": truncated,
            "connectivity": connectivity,
        }
        if px_area_m2 is not None:
            data["pixel_resolution_m"] = round(float(pixel_resolution_m), 4)
            data["total_region_area_m2"] = round(float(areas.sum()) * px_area_m2, 4) if areas.size else 0.0

        return ToolResult.success(
            self.name, data=data,
            provenance={
                "method": "cv2.connectedComponentsWithStats" if _HAVE_CV2 else "scipy.ndimage.label",
                "min_area_pixels": min_area_pixels,
            },
        )

    # ── Labelling backends ────────────────────────────────────────────────────
    def _label(self, binary: np.ndarray, connectivity: int):
        """
        Return (labels, stats, centroids) mirroring cv2's layout:
          stats[i]     = [x, y, width, height, area]  (row 0 = background)
          centroids[i] = [cx, cy]
        Uses OpenCV when available, else scipy.ndimage as a fallback.
        """
        if _HAVE_CV2:
            conn = 8 if connectivity == 8 else 4
            n, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=conn
            )
            return labels, stats, centroids

        # scipy fallback
        from scipy import ndimage
        structure = ndimage.generate_binary_structure(2, 2 if connectivity == 8 else 1)
        labels, n = ndimage.label(binary, structure=structure)
        stats = [[0, 0, binary.shape[1], binary.shape[0], int((labels == 0).sum())]]
        centroids = [[0.0, 0.0]]
        for lbl in range(1, n + 1):
            ys, xs = np.where(labels == lbl)
            x, y = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)
            stats.append([x, y, bw, bh, int(xs.size)])
            centroids.append([float(xs.mean()), float(ys.mean())])
        return labels, stats, centroids
