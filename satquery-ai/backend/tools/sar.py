"""
SatQuery AI — Deterministic Tool Layer: SAR Backscatter Statistics
==================================================================
:class:`SARBackscatterStatisticsTool` reports intensity statistics for Synthetic
Aperture Radar imagery (per polarization: VV / VH / HH / HV) and the VV/VH ratio
when both co-pol and cross-pol channels exist.

Calibration honesty
-------------------
Raw SAR pixels straight from a product are *digital numbers* (DN) or intensity,
**not** calibrated backscatter coefficients (σ⁰ / γ⁰).  Reporting DN statistics
as "backscatter in dB" is a fabrication unless the source is radiometrically
calibrated.  So this tool:

  * reports intensity/DN statistics by default and flags ``calibrated=False``;
  * only claims calibration (``calibrated=True``, with the stated ``unit``) when
    the raster metadata explicitly says so — via ``metadata['calibrated'] is
    True`` and/or a recognised ``metadata['calibration']`` / ``metadata['unit']``
    marker (e.g. ``"sigma0"``, ``"gamma0"``, ``"dB"``).

The VV/VH ratio is reported in **linear intensity** space (mean(VV)/mean(VH));
we do not emit a dB ratio unless the inputs are known to be calibrated dB.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .base import Tool, ToolResult, band_stats, round_dict
from .raster import RasterInput

# Polarization channels we look for, in reporting order.
_SAR_POLS: List[str] = ["vv", "vh", "hh", "hv"]

# Metadata markers that prove the pixels are radiometrically calibrated.
_CALIBRATED_UNITS = {"sigma0", "sigma_naught", "gamma0", "gamma_naught", "db",
                     "beta0", "backscatter"}


def _detect_calibration(raster: RasterInput) -> Dict[str, Any]:
    """
    Inspect metadata for a trustworthy calibration marker.

    Returns ``{"calibrated": bool, "unit": str}``.  Default is uncalibrated
    intensity/DN — we never assume calibration the metadata doesn't assert.
    """
    meta = raster.metadata or {}
    if meta.get("calibrated") is True:
        unit = str(meta.get("unit") or meta.get("calibration") or "backscatter")
        return {"calibrated": True, "unit": unit}

    marker = str(meta.get("calibration") or meta.get("unit") or "").lower().strip()
    if marker in _CALIBRATED_UNITS:
        return {"calibrated": True, "unit": marker}

    return {"calibrated": False, "unit": "intensity_dn"}


class SARBackscatterStatisticsTool(Tool):
    name = "sar_backscatter_statistics"
    description = (
        "Per-polarization SAR intensity statistics (mean/std/percentiles) and "
        "VV/VH ratio; flags whether values are calibrated backscatter or raw DN"
    )

    def run(self, raster: RasterInput) -> ToolResult:  # type: ignore[override]
        # Which of the known polarization channels are resolvable?
        present = [p for p in _SAR_POLS if raster.has_band(p)]

        # If no named polarizations resolve but the modality is SAR, fall back to
        # treating positional channels as unnamed intensity bands.
        if not present:
            if raster.modality == "sar" or raster.n_bands <= 2:
                return self._unnamed_channels(raster)
            return ToolResult.unsupported(
                self.name,
                reason="no SAR polarization bands (vv/vh/hh/hv) resolved and "
                       "modality is not SAR",
                data={"available_bands": sorted(raster.resolved_bands().keys())},
            )

        calib = _detect_calibration(raster)
        per_pol: Dict[str, Dict[str, float]] = {}
        for pol in present:
            arr = raster.band(pol)
            s = band_stats(arr)
            s["valid_pixel_percentage"] = round(
                100.0 * s["count"] / arr.size if arr.size else 0.0, 2
            )
            per_pol[pol] = round_dict(s)

        data: Dict[str, Any] = {
            "polarizations": present,
            "calibrated": calib["calibrated"],
            "unit": calib["unit"],
            "per_polarization": per_pol,
        }

        # VV/VH ratio in linear intensity space (only if both present & VH mean>0).
        if "vv" in present and "vh" in present:
            vv_mean = per_pol["vv"]["mean"]
            vh_mean = per_pol["vh"]["mean"]
            if vh_mean != 0 and np.isfinite(vv_mean) and np.isfinite(vh_mean):
                data["vv_vh_ratio"] = round(float(vv_mean / vh_mean), 4)
                data["vv_vh_ratio_space"] = "linear_intensity" if not calib["calibrated"] else calib["unit"]
            else:
                data["vv_vh_ratio"] = None
                data["vv_vh_ratio_reason"] = "VH mean is zero or non-finite"

        return ToolResult.success(
            self.name,
            data=data,
            provenance={
                "band_indices": {p: raster.band_index(p) for p in present},
                "sensor": raster.sensor,
                "calibration_source": "metadata" if calib["calibrated"] else "assumed_uncalibrated",
            },
        )

    def _unnamed_channels(self, raster: RasterInput) -> ToolResult:
        """Report positional-channel intensity stats when polarizations are unlabelled."""
        calib = _detect_calibration(raster)
        per_channel: Dict[str, Dict[str, float]] = {}
        for idx in range(raster.n_bands):
            arr = raster.array[:, :, idx].astype(np.float64)
            if raster.nodata is not None:
                arr = np.where(arr == float(raster.nodata), np.nan, arr)
            s = band_stats(arr)
            s["valid_pixel_percentage"] = round(
                100.0 * s["count"] / arr.size if arr.size else 0.0, 2
            )
            per_channel[f"channel_{idx}"] = round_dict(s)

        if not per_channel:
            return ToolResult.unsupported(self.name, reason="SAR raster has no readable channels")

        return ToolResult.success(
            self.name,
            data={
                "polarizations": None,
                "calibrated": calib["calibrated"],
                "unit": calib["unit"],
                "per_channel": per_channel,
                "note": "polarization labels unknown; reported by positional channel",
            },
            provenance={
                "sensor": raster.sensor,
                "calibration_source": "metadata" if calib["calibrated"] else "assumed_uncalibrated",
            },
        )
