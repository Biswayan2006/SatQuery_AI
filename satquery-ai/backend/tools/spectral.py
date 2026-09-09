"""
SatQuery AI — Deterministic Tool Layer: Spectral Index & Statistics Tools
=========================================================================
Deterministic optical / multispectral computations:

  * :class:`NDVITool`    — Normalized Difference Vegetation Index  (NIR, RED)
  * :class:`NDWITool`    — Normalized Difference Water Index       (GREEN, NIR)
  * :class:`NDBITool`    — Normalized Difference Built-up Index    (SWIR1, NIR)
  * :class:`SpectralStatisticsTool` — per-band summary statistics

Every index follows the normalized-difference form ``(A − B) / (A + B)`` and is
computed only when *both* required bands can be resolved from the raster's
sensor / band metadata.  If a required band is missing the tool returns a
structured ``unsupported`` result naming the bands it looked for and the bands
it actually found — it never substitutes an arbitrary band and never fabricates
an index value.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .base import Tool, ToolResult, band_stats, finite_mask, round_dict, safe_ratio
from .raster import RasterInput


def _index_stats(index: np.ndarray) -> Dict[str, float]:
    """Summary stats over an index array, plus the valid-pixel percentage."""
    total = index.size
    stats = band_stats(index)
    valid_pct = (100.0 * stats["count"] / total) if total else 0.0
    stats["valid_pixel_percentage"] = round(valid_pct, 2)
    return stats


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a − b) / (a + b)`` with NaN where the sum is ~0 or inputs are invalid."""
    mask = finite_mask(a, b)
    a = np.where(mask, a, np.nan)
    b = np.where(mask, b, np.nan)
    return safe_ratio(a - b, a + b)


class _NormalizedDifferenceTool(Tool):
    """
    Shared machinery for the three normalized-difference index tools.

    Subclasses declare ``name``, ``high_band``/``low_band`` (the ``A`` and ``B``
    in ``(A − B)/(A + B)``) and a human ``index_label``.
    """

    high_band: str = ""
    low_band: str = ""
    index_label: str = ""

    def run(self, raster: RasterInput) -> ToolResult:  # type: ignore[override]
        needed = [self.high_band, self.low_band]
        missing = [b for b in needed if not raster.has_band(b)]
        if missing:
            return ToolResult.unsupported(
                self.name,
                reason=(
                    f"{self.index_label} requires the {', '.join(needed)} bands; "
                    f"missing {', '.join(missing)} for this raster"
                ),
                data={
                    "required_bands": needed,
                    "missing_bands": missing,
                    "available_bands": sorted(raster.resolved_bands().keys()),
                },
            )

        a = raster.band(self.high_band)
        b = raster.band(self.low_band)
        index = _normalized_difference(a, b)
        stats = _index_stats(index)

        if stats["count"] == 0:
            return ToolResult.unsupported(
                self.name,
                reason="no valid pixels (all nodata / non-finite after band read)",
                data={"required_bands": needed, "valid_pixel_percentage": 0.0},
            )

        return ToolResult.success(
            self.name,
            data=round_dict(stats),
            provenance={
                "formula": f"({self.high_band.upper()} - {self.low_band.upper()}) / "
                           f"({self.high_band.upper()} + {self.low_band.upper()})",
                "band_indices": {
                    self.high_band: raster.band_index(self.high_band),
                    self.low_band: raster.band_index(self.low_band),
                },
                "sensor": raster.sensor,
            },
        )


class NDVITool(_NormalizedDifferenceTool):
    name = "NDVI"
    description = "Normalized Difference Vegetation Index = (NIR - RED) / (NIR + RED)"
    high_band = "nir"
    low_band = "red"
    index_label = "NDVI"


class NDWITool(_NormalizedDifferenceTool):
    name = "NDWI"
    description = "Normalized Difference Water Index = (GREEN - NIR) / (GREEN + NIR)"
    high_band = "green"
    low_band = "nir"
    index_label = "NDWI"


class NDBITool(_NormalizedDifferenceTool):
    name = "NDBI"
    description = "Normalized Difference Built-up Index = (SWIR1 - NIR) / (SWIR1 + NIR)"
    high_band = "swir1"
    low_band = "nir"
    index_label = "NDBI"


class SpectralStatisticsTool(Tool):
    """
    Per-band summary statistics for every resolved band (or every raw band when
    no logical names resolve).  Purely descriptive: mean/std/min/max/percentiles
    per band, with each band's valid-pixel percentage.
    """

    name = "spectral_statistics"
    description = "Per-band summary statistics (mean/std/min/max/percentiles)"

    def run(self, raster: RasterInput) -> ToolResult:  # type: ignore[override]
        resolved = raster.resolved_bands()
        per_band: Dict[str, Dict[str, float]] = {}

        if resolved:
            # Named bands: report by logical name.
            for name in sorted(resolved.keys()):
                arr = raster.band(name)
                s = band_stats(arr)
                s["valid_pixel_percentage"] = round(
                    100.0 * s["count"] / arr.size if arr.size else 0.0, 2
                )
                per_band[name] = round_dict(s)
        else:
            # Fall back to positional band indices.
            for idx in range(raster.n_bands):
                arr = raster.array[:, :, idx].astype(np.float64)
                if raster.nodata is not None:
                    arr = np.where(arr == float(raster.nodata), np.nan, arr)
                s = band_stats(arr)
                s["valid_pixel_percentage"] = round(
                    100.0 * s["count"] / arr.size if arr.size else 0.0, 2
                )
                per_band[f"band_{idx}"] = round_dict(s)

        if not per_band:
            return ToolResult.unsupported(
                self.name, reason="raster has no readable bands"
            )

        return ToolResult.success(
            self.name,
            data={"n_bands": raster.n_bands, "bands": per_band},
            provenance={
                "resolved_band_names": resolved,
                "sensor": raster.sensor,
            },
        )
