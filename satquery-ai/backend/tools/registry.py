"""
SatQuery AI — Deterministic Tool Layer: Tool Registry
=====================================================
A tiny name → tool registry.  Unlike the model registry, tools are *stateless*
deterministic functions, so the registry is just a lookup table with a default
population of the ten built-in tools.  The agentic controller/planner asks the
registry for a tool by name and calls it; every call returns a :class:`ToolResult`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import Tool, ToolResult
from .change import ChangeAreaTool, ConnectedRegionStatisticsTool
from .geolocation import GroundResolutionTool, ImageBoundsTool, PixelToLatLonTool
from .sar import SARBackscatterStatisticsTool
from .spectral import NDBITool, NDVITool, NDWITool, SpectralStatisticsTool

logger = logging.getLogger("satquery.tools.registry")


class ToolRegistry:
    """Name-keyed registry of deterministic tools."""

    def __init__(self, populate_defaults: bool = True):
        self._tools: Dict[str, Tool] = {}
        if populate_defaults:
            self._register_defaults()

    def _register_defaults(self) -> None:
        for tool in (
            NDVITool(),
            NDWITool(),
            NDBITool(),
            SpectralStatisticsTool(),
            SARBackscatterStatisticsTool(),
            ChangeAreaTool(),
            PixelToLatLonTool(),
            ImageBoundsTool(),
            GroundResolutionTool(),
            ConnectedRegionStatisticsTool(),
        ):
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools.keys())

    def describe(self) -> Dict[str, str]:
        """name → one-line description, for planner prompts / docs."""
        return {n: self._tools[n].description for n in self.names()}

    def run(self, name: str, *args, **kwargs) -> ToolResult:
        """Look up and invoke a tool; unknown names yield an ``error`` result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.error(name, f"no such tool: {name!r}")
        return tool(*args, **kwargs)


# Process-wide default registry (tools are stateless, so sharing is safe).
_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry()
        logger.info("Deterministic tool registry initialised: %s",
                    ", ".join(_DEFAULT_REGISTRY.names()))
    return _DEFAULT_REGISTRY
