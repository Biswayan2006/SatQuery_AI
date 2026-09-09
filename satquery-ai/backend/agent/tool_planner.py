"""
SatQuery AI — Deterministic Tool Planner
========================================
Bridges the agentic controller and the deterministic tool layer.  Given the
classified task, the query text and the raster(s), it decides *which* of the ten
deterministic tools can add real evidence, runs them, and returns a list of
structured tool results (``ToolResult.to_dict()``).

Design principles
-----------------
* **The AI never computes what a tool can.**  When a query is about vegetation,
  water, built-up area, change extent or location, the matching deterministic
  tool runs and its numbers become evidence the model then interprets.
* **Selection is data-aware.**  A tool is only planned when its required bands /
  metadata are present; otherwise it is skipped (or, if planned, self-reports
  ``unsupported`` — never a fabricated number).
* **Never raises into the request path.**  Any failure degrades to an ``error``
  ToolResult; a broken tool layer must not break analysis.

Worked example plans (from the product spec)
--------------------------------------------
"Has vegetation decreased?"  → NDVI(before) + NDVI(after) + change-area on the
    NDVI-drop mask, so the answer rests on measured index means, not a guess.
"Where has construction expanded?" → NDBI + connected-region stats on the
    built-up-gain mask + pixel→lat/lon on region centroids → grounded location.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from agent.task_classifier import TaskType

logger = logging.getLogger("satquery.tool_planner")


# ── Query intent keywords → candidate single-image tools ───────────────────────
_VEGETATION_KW = ("vegetation", "veget", "green", "forest", "crop", "farm",
                  "ndvi", "plant", "tree", "deforest")
_WATER_KW = ("water", "flood", "lake", "river", "ndwi", "wetland", "reservoir",
             "inundat", "moisture")
_BUILTUP_KW = ("build", "built", "urban", "construction", "settlement", "ndbi",
               "concrete", "development", "expansion", "sprawl", "city")
_LOCATION_KW = ("where", "location", "coordinate", "lat", "lon", "geograph",
                "bounds", "extent", "resolution", "scale")


def _match(query: str, keywords) -> bool:
    q = (query or "").lower()
    return any(k in q for k in keywords)


class ToolPlanner:
    """Selects and runs deterministic tools; returns their structured evidence."""

    def __init__(self, registry=None):
        # Import lazily so the tool layer is optional at controller import time.
        from tools import get_default_registry
        self.tools = registry or get_default_registry()

    # ── Public entry point ─────────────────────────────────────────────────────
    def run_for_task(
        self,
        task_type: TaskType,
        query: str,
        images: List[Dict[str, Any]],
        raw_extras: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of tool-result dicts appropriate to the task + query + data.

        ``raw_extras`` may carry pipeline artefacts the tools consume, e.g.
        ``{"change_mask": np.ndarray, "pixel_resolution_m": float}``.
        """
        try:
            rasters = [self._raster(img) for img in images]
        except Exception as exc:  # never break the request path
            logger.warning("Tool planner could not build rasters: %s", exc)
            return []

        rasters = [r for r in rasters if r is not None]
        if not rasters:
            return []

        try:
            if task_type in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION):
                return self._plan_change(query, rasters, raw_extras or {})
            if task_type == TaskType.SAR_OPTICAL_FUSION:
                return self._plan_fusion(query, rasters)
            # Single-image tasks (VQA / captioning / grounding).
            return self._plan_single(query, rasters[0], task_type)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tool planning failed: %s", exc)
            return []

    # ── Raster construction ─────────────────────────────────────────────────────
    def _raster(self, image_data: Dict[str, Any]):
        from tools import RasterInput
        try:
            return RasterInput.from_image_data(image_data)
        except Exception as exc:
            logger.debug("RasterInput build failed: %s", exc)
            return None

    # ── Single-image plans ──────────────────────────────────────────────────────
    def _plan_single(self, query: str, raster, task_type: TaskType) -> List[Dict[str, Any]]:
        selected: List[str] = []

        # Intent-driven spectral indices (only added when the query is about them
        # OR the task is open-ended captioning, where a scene summary helps).
        if _match(query, _VEGETATION_KW) and raster.has_band("nir") and raster.has_band("red"):
            selected.append("NDVI")
        if _match(query, _WATER_KW) and raster.has_band("green") and raster.has_band("nir"):
            selected.append("NDWI")
        if _match(query, _BUILTUP_KW) and raster.has_band("swir1") and raster.has_band("nir"):
            selected.append("NDBI")

        # SAR statistics when the raster is radar.
        if raster.modality == "sar":
            selected.append("sar_backscatter_statistics")

        # Location / extent questions.
        if _match(query, _LOCATION_KW) and raster.has_geo:
            selected.extend(["image_bounds", "ground_resolution"])

        # Captioning with no specific intent → a compact scene summary via
        # spectral statistics (always safe, purely descriptive).
        if task_type == TaskType.CAPTIONING and not selected:
            selected.append("spectral_statistics")

        # De-dup while preserving order.
        seen, ordered = set(), []
        for name in selected:
            if name not in seen:
                seen.add(name)
                ordered.append(name)

        return [self.tools.run(name, raster).to_dict() for name in ordered]

    # ── Change plans (bi-temporal) ──────────────────────────────────────────────
    def _plan_change(self, query: str, rasters, extras: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        r1 = rasters[0]
        r2 = rasters[1] if len(rasters) > 1 else None

        pixel_res = extras.get("pixel_resolution_m")
        if pixel_res is None:
            pixel_res = self._resolution_from(r1) or (self._resolution_from(r2) if r2 else None)

        # 1) Index-based before/after evidence, chosen by intent.
        index_tool = None
        band_ok = lambda r, *bs: r is not None and all(r.has_band(b) for b in bs)
        if _match(query, _VEGETATION_KW):
            index_tool = ("NDVI", ("nir", "red"))
        elif _match(query, _WATER_KW):
            index_tool = ("NDWI", ("green", "nir"))
        elif _match(query, _BUILTUP_KW):
            index_tool = ("NDBI", ("swir1", "nir"))

        if index_tool:
            name, bands = index_tool
            if band_ok(r1, *bands):
                evidence.append(self.tools.run(name, r1).to_dict())
            if r2 is not None and band_ok(r2, *bands):
                evidence.append(self.tools.run(name, r2).to_dict())

        # 2) Change-area + connected regions on an upstream change mask, if given.
        mask = extras.get("change_mask")
        if mask is not None:
            evidence.append(
                self.tools.run("change_area", mask, pixel_resolution_m=pixel_res).to_dict()
            )
            region_res = self.tools.run(
                "connected_region_statistics", mask, pixel_resolution_m=pixel_res
            )
            region_dict = region_res.to_dict()

            # 3) Geolocate region centroids when geo metadata exists → grounded
            #    "where" answers (the construction-expansion example plan).
            if r1.has_geo and region_res.ok:
                self._geolocate_regions(r1, region_dict)
            evidence.append(region_dict)

        return evidence

    # ── Fusion plans (SAR + optical) ────────────────────────────────────────────
    def _plan_fusion(self, query: str, rasters) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for r in rasters:
            if r.modality == "sar":
                evidence.append(self.tools.run("sar_backscatter_statistics", r).to_dict())
            else:
                # Optical partner: run an intent-matched index if bands allow.
                if _match(query, _VEGETATION_KW) and r.has_band("nir") and r.has_band("red"):
                    evidence.append(self.tools.run("NDVI", r).to_dict())
                elif _match(query, _WATER_KW) and r.has_band("green") and r.has_band("nir"):
                    evidence.append(self.tools.run("NDWI", r).to_dict())
        return evidence

    # ── Helpers ─────────────────────────────────────────────────────────────────
    def _resolution_from(self, raster) -> Optional[float]:
        if raster is None or not raster.has_geo:
            return None
        res = self.tools.run("ground_resolution", raster)
        if res.ok:
            return res.to_dict().get("mean_resolution_m")
        return None

    def _geolocate_regions(self, raster, region_dict: Dict[str, Any]) -> None:
        """Attach geographic centroid + bbox to each region using pixel_to_latlon."""
        for region in region_dict.get("regions", []):
            centroid = region.get("centroid_pixels", {})
            bbox = region.get("bbox_pixels")
            res = self.tools.run(
                "pixel_to_latlon", raster,
                centroid.get("x", 0), centroid.get("y", 0),
                pixel_bbox=bbox,
            )
            if res.ok:
                d = res.to_dict()
                region["geographic"] = d.get("geographic")
                if "geographic_bbox" in d:
                    region["geographic_bbox"] = d["geographic_bbox"]
