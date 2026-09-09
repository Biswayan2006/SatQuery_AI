"""
SatQuery AI — Deterministic Tool Layer: Base Contract
======================================================
The tool layer performs calculations that can be done **deterministically** from
satellite pixel data and geospatial metadata, so the AI models are not asked to
invent numbers they cannot actually compute (NDVI means, changed-area in m²,
lat/lon of a pixel, …).

Data-flow contract
------------------
    Tool
      ↓  validate_input()      — check the raster / metadata actually supports it
      ↓  compute()             — pure, deterministic numpy math
      ↓  ToolResult            — structured evidence (status + numbers + provenance)
      ↓  agentic controller    — collects evidence from several tools
      ↓  AI interpretation     — the model explains the numbers, it does not fabricate them

Every tool returns a :class:`ToolResult`.  A tool never raises into the request
path: on bad input it returns ``status="unsupported"`` (the data genuinely can't
support the computation, e.g. NDVI with no NIR band) or ``status="error"`` (an
unexpected failure).  ``status="success"`` carries the computed evidence.

Honesty guarantees
------------------
* A tool that cannot compute its quantity returns ``unsupported`` with a reason —
  it never emits a plausible-looking fabricated number.
* SAR intensity statistics are reported as *intensity* (DN) statistics and are
  explicitly flagged ``calibrated=False`` unless metadata proves calibration —
  we never relabel raw DN as calibrated backscatter (σ⁰).
* Geographic coordinates are only produced when a CRS + affine transform exist.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("satquery.tools")


class ToolStatus(str, Enum):
    SUCCESS = "success"          # computed real evidence
    UNSUPPORTED = "unsupported"  # input genuinely can't support this computation
    ERROR = "error"             # unexpected failure


@dataclass
class ToolResult:
    """
    Structured evidence returned by every deterministic tool.

    The public dict (``to_dict``) always contains at least ``tool`` and
    ``status``; on success it is flattened with the tool's numeric outputs so
    the controller / AI layer sees a clean evidence record, e.g.::

        {"tool": "NDVI", "status": "success",
         "mean": 0.61, "min": -0.18, "max": 0.89, "valid_pixel_percentage": 97.3}
    """

    tool: str
    status: ToolStatus
    data: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None                 # why unsupported / errored
    provenance: Dict[str, Any] = field(default_factory=dict)  # how numbers were derived

    @property
    def ok(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"tool": self.tool, "status": self.status.value}
        if self.status == ToolStatus.SUCCESS:
            out.update(self.data)
            if self.provenance:
                out["provenance"] = self.provenance
        else:
            if self.reason:
                out["reason"] = self.reason
            # Preserve any partial data (e.g. which bands *were* found).
            out.update(self.data)
        return out

    # ── Convenience constructors ────────────────────────────────────────────
    @classmethod
    def success(cls, tool: str, data: Dict[str, Any],
                provenance: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(tool=tool, status=ToolStatus.SUCCESS, data=data,
                   provenance=provenance or {})

    @classmethod
    def unsupported(cls, tool: str, reason: str,
                    data: Optional[Dict[str, Any]] = None) -> "ToolResult":
        return cls(tool=tool, status=ToolStatus.UNSUPPORTED, reason=reason,
                   data=data or {})

    @classmethod
    def error(cls, tool: str, reason: str) -> "ToolResult":
        return cls(tool=tool, status=ToolStatus.ERROR, reason=reason)


class Tool:
    """
    Base class for a deterministic tool.

    Subclasses set ``name`` and implement :meth:`run`.  :meth:`__call__` wraps
    ``run`` so no tool ever raises into the request path — an unexpected
    exception becomes a structured ``error`` result.
    """

    name: str = "tool"
    #: Short human description used by the planner / docs.
    description: str = ""

    def run(self, *args, **kwargs) -> ToolResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> ToolResult:
        try:
            return self.run(*args, **kwargs)
        except Exception as exc:  # never propagate
            logger.warning("Tool '%s' failed: %s", self.name, exc)
            return ToolResult.error(self.name, f"unexpected error: {exc}")


# ── Numeric helpers shared across tools ────────────────────────────────────────

def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask of positions that are finite (not NaN/inf) in every input."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def safe_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Elementwise normalized-difference-style ratio ``numer / denom`` that yields
    NaN (not inf, not a divide warning) where the denominator is ~0, so invalid
    pixels are cleanly excluded downstream rather than fabricated as 0 or ±inf.
    """
    numer = numer.astype(np.float64)
    denom = denom.astype(np.float64)
    out = np.full(numer.shape, np.nan, dtype=np.float64)
    valid = np.abs(denom) > 1e-12
    np.divide(numer, denom, out=out, where=valid)
    return out


def band_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Summary statistics over the finite entries of ``values``.

    Returns mean/std/min/max/median plus 5th/25th/75th/95th percentiles and the
    finite-pixel count.  Empty / all-NaN input returns an all-zero bundle with
    ``count=0`` (callers should treat count==0 as "no valid pixels").
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "median": 0.0, "p5": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0,
                "count": 0}
    p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "median": float(np.median(v)),
        "p5": float(p5), "p25": float(p25), "p75": float(p75), "p95": float(p95),
        "count": n,
    }


def round_dict(d: Dict[str, Any], ndigits: int = 4) -> Dict[str, Any]:
    """Round float values in a flat dict for stable JSON output."""
    out: Dict[str, Any] = {}
    for k, val in d.items():
        if isinstance(val, float):
            out[k] = round(val, ndigits)
        else:
            out[k] = val
    return out
