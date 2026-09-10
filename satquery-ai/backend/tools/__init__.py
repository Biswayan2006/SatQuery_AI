"""
SatQuery AI — Deterministic Tool Layer
======================================
Deterministic remote-sensing computations so the AI models are not responsible
for numbers they cannot actually compute (spectral indices, changed-area, pixel
geolocation, SAR intensity statistics, …).

Contract::

    Tool → validated RasterInput → deterministic numpy math → ToolResult
         → agentic controller → AI interpretation

Every tool returns a :class:`ToolResult` (``success`` / ``unsupported`` /
``error``) and never raises into the request path.

Public API
----------
    from tools import get_default_registry, RasterInput
    reg = get_default_registry()
    result = reg.run("NDVI", raster)          # → ToolResult
    result.to_dict()                          # → {"tool": "NDVI", "status": ..., ...}
"""
from __future__ import annotations

from .base import Tool, ToolResult, ToolStatus
from .change import ChangeAreaTool, ConnectedRegionStatisticsTool
from .geolocation import GroundResolutionTool, ImageBoundsTool, PixelToLatLonTool, ReverseGeocodingTool
from .raster import RasterInput
from .registry import ToolRegistry, get_default_registry
from .sar import SARBackscatterStatisticsTool
from .spectral import NDBITool, NDVITool, NDWITool, SpectralStatisticsTool

__all__ = [
    # Core contract
    "Tool", "ToolResult", "ToolStatus", "RasterInput",
    # Registry
    "ToolRegistry", "get_default_registry",
    # Spectral
    "NDVITool", "NDWITool", "NDBITool", "SpectralStatisticsTool",
    # SAR
    "SARBackscatterStatisticsTool",
    # Change / regions
    "ChangeAreaTool", "ConnectedRegionStatisticsTool",
    # Geolocation
    "PixelToLatLonTool", "ImageBoundsTool", "GroundResolutionTool",
    "ReverseGeocodingTool",
]
