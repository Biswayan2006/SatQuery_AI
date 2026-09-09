"""
SatQuery AI — Deterministic Tool Layer: Raster Input Abstraction
=================================================================
A ``RasterInput`` is the *validated input* stage of the tool pipeline: it wraps
a multi-band pixel array plus optional geospatial + sensor metadata and, most
importantly, resolves **which physical band is which** (RED, NIR, SWIR, …)
without assuming fixed band numbers for every sensor.

Why band resolution matters
---------------------------
NDVI = (NIR − RED) / (NIR + RED).  For Sentinel-2 the NIR band is B08 and RED is
B04; for Landsat-8 NIR is band 5 and RED is band 4; for a plain 3-band RGB image
there is *no* NIR band at all.  Hard-coding "NIR = band 4" silently computes
garbage on the wrong sensor.  So ``RasterInput`` resolves bands from, in order:

  1. an explicit ``band_map`` the caller supplies
     (e.g. ``{"red": 3, "nir": 7, "swir1": 11}`` — 0-indexed),
  2. a named ``sensor`` preset (sentinel-2, landsat-8/9, generic RGB/RGBN),
  3. a heuristic for the common 3-band RGB case.

If a required band cannot be resolved, the tool that needs it returns
``unsupported`` — it never falls back to an arbitrary band.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("satquery.tools.raster")


# ── Sensor band presets (0-indexed positions in the stored band array) ─────────
# Only the optical bands relevant to the indices we compute are listed.  These
# map a logical band name → the 0-based index within a full band stack for that
# sensor.  Callers with non-standard stacks should pass an explicit band_map.
SENSOR_PRESETS: Dict[str, Dict[str, int]] = {
    # Sentinel-2 L2A common 12-band stack order (B1..B12 minus B10),
    # but the widely-used ordering is B2,B3,B4,B8 for 10 m bands. We describe
    # the canonical full-resolution 13-band product by band NAME → index.
    "sentinel-2": {"blue": 1, "green": 2, "red": 3, "nir": 7, "swir1": 11, "swir2": 12},
    "sentinel2": {"blue": 1, "green": 2, "red": 3, "nir": 7, "swir1": 11, "swir2": 12},
    # Landsat 8/9 OLI: B2 blue, B3 green, B4 red, B5 NIR, B6 SWIR1, B7 SWIR2
    # (0-indexed within a B1..B7 stack → 1,2,3,4,5,6)
    "landsat-8": {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6},
    "landsat8": {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6},
    "landsat-9": {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6},
    # Plain optical RGB (no NIR/SWIR)
    "rgb": {"red": 0, "green": 1, "blue": 2},
    # RGB + NIR 4-band (e.g. NAIP, PlanetScope 4-band)
    "rgbn": {"red": 0, "green": 1, "blue": 2, "nir": 3},
    # Green-NIR ordering used by some analytic products; kept explicit.
    "bgrn": {"blue": 0, "green": 1, "red": 2, "nir": 3},
}

# SAR polarization presets: logical channel → 0-based index.
SAR_PRESETS: Dict[str, Dict[str, int]] = {
    "sentinel-1": {"vv": 0, "vh": 1},
    "sentinel1": {"vv": 0, "vh": 1},
    "vv-vh": {"vv": 0, "vh": 1},
    "vh-vv": {"vh": 0, "vv": 1},
    "vv": {"vv": 0},
    "vh": {"vh": 0},
}


@dataclass
class RasterInput:
    """
    Validated multi-band raster + metadata for the deterministic tools.

    Parameters
    ----------
    array : np.ndarray
        Pixel data in **[H, W, C]** (channels-last) or **[H, W]** layout.
        Stored internally as [H, W, C] float64.
    sensor : str or None
        Sensor preset name (see ``SENSOR_PRESETS`` / ``SAR_PRESETS``).
    band_map : dict or None
        Explicit logical-name → 0-based-index overrides.  Takes precedence over
        the sensor preset.  e.g. ``{"red": 3, "nir": 7}``.
    crs : str or None
        CRS string (WKT / PROJ / "EPSG:XXXX").  Enables geolocation tools.
    transform : list[float] or None
        6-element affine transform [a, b, c, d, e, f].  Enables geolocation.
    nodata : float or None
        Nodata sentinel value; matching pixels are excluded from statistics.
    modality : str
        "optical" | "sar" | "multispectral" | "unknown".
    """

    array: np.ndarray
    sensor: Optional[str] = None
    band_map: Optional[Dict[str, int]] = None
    crs: Optional[str] = None
    transform: Optional[List[float]] = None
    nodata: Optional[float] = None
    modality: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        arr = np.asarray(self.array)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim != 3:
            raise ValueError(f"RasterInput array must be 2-D or 3-D, got {arr.ndim}-D")
        self.array = arr.astype(np.float64)
        self._resolved = self._resolve_band_map()

    # ── Shape helpers ────────────────────────────────────────────────────────
    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.array.shape[2])

    @property
    def has_geo(self) -> bool:
        return bool(self.crs) and self.transform is not None and len(self.transform) >= 6

    # ── Band resolution ────────────────────────────────────────────────────────
    def _resolve_band_map(self) -> Dict[str, int]:
        """Merge sensor preset + explicit band_map + RGB heuristic into one map."""
        resolved: Dict[str, int] = {}

        preset_name = (self.sensor or "").lower().strip()
        if preset_name in SENSOR_PRESETS:
            resolved.update(SENSOR_PRESETS[preset_name])
        if preset_name in SAR_PRESETS:
            resolved.update(SAR_PRESETS[preset_name])

        # Heuristic: an unlabelled 3-band stack is almost always RGB.
        if not resolved and self.n_bands == 3 and self.modality != "sar":
            resolved.update(SENSOR_PRESETS["rgb"])
        # Heuristic: 4-band with no preset → assume RGBN (common analytic product).
        if not resolved and self.n_bands == 4 and self.modality != "sar":
            resolved.update(SENSOR_PRESETS["rgbn"])

        # Explicit band_map overrides everything.
        if self.band_map:
            resolved.update({k.lower(): int(v) for k, v in self.band_map.items()})

        # Drop any mapping that points outside the actual band count.
        return {k: v for k, v in resolved.items() if 0 <= v < self.n_bands}

    def has_band(self, name: str) -> bool:
        return name.lower() in self._resolved

    def band(self, name: str) -> np.ndarray:
        """
        Return the 2-D [H, W] float64 array for a logical band name, with nodata
        (and non-finite) pixels set to NaN so they're excluded from statistics.

        Raises KeyError if the band was not resolved — callers must check
        :meth:`has_band` first and return ``unsupported`` otherwise.
        """
        key = name.lower()
        if key not in self._resolved:
            raise KeyError(f"band '{name}' not available for this raster")
        idx = self._resolved[key]
        b = self.array[:, :, idx].astype(np.float64).copy()
        if self.nodata is not None:
            b[b == float(self.nodata)] = np.nan
        b[~np.isfinite(b)] = np.nan
        return b

    def band_index(self, name: str) -> Optional[int]:
        return self._resolved.get(name.lower())

    def resolved_bands(self) -> Dict[str, int]:
        return dict(self._resolved)

    # ── Construction from the pipeline's image_data dict ──────────────────────
    @classmethod
    def from_image_data(
        cls,
        image_data: Dict[str, Any],
        sensor: Optional[str] = None,
        band_map: Optional[Dict[str, int]] = None,
    ) -> "RasterInput":
        """
        Build a RasterInput from an ``image_utils.load_image()``-style dict.

        Pulls the numpy array (HWC), CRS/transform/nodata from ``metadata`` and
        the detected ``modality``.  Sensor / band_map may be supplied explicitly
        or read from ``metadata['sensor']`` / ``metadata['band_map']`` if present.
        """
        arr = image_data.get("numpy_array")
        if arr is None:
            pil = image_data.get("pil_image")
            if pil is None:
                raise ValueError("image_data has neither numpy_array nor pil_image")
            arr = np.array(pil)

        meta = image_data.get("metadata", {}) or {}
        return cls(
            array=arr,
            sensor=sensor or meta.get("sensor") or image_data.get("sensor"),
            band_map=band_map or meta.get("band_map"),
            crs=meta.get("crs"),
            transform=meta.get("transform"),
            nodata=meta.get("nodata"),
            modality=image_data.get("modality", "unknown"),
            metadata=meta,
        )
