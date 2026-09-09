"""
SatQuery AI — Execution Step Tracing
====================================
Structured, user-safe execution metadata for the agentic controller.  Each
executed step (model call or deterministic tool) records a :class:`StepRecord`::

    {
      "step": "NDVI",
      "status": "success",
      "duration_ms": 143,
      "output_summary": {"mean": 0.61, "valid_pixel_percentage": 97.3}
    }

What this deliberately does NOT contain
--------------------------------------
No hidden chain-of-thought, no prompts, no raw model logits.  Only concise,
inspectable metadata: the tool/model name, the parameters it ran with, a
status, a wall-clock duration and a *summary* of the output (never the full
tensor payload).  This is safe to return to API clients.

The controller stamps ``duration_ms`` itself (the workflow/runtime forbids
``Date.now()`` inside deterministic scripts, but the controller runs in normal
Python and uses ``time.perf_counter``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

STEP_SUCCESS = "success"
STEP_UNSUPPORTED = "unsupported"
STEP_SKIPPED = "skipped"
STEP_ERROR = "error"


@dataclass
class StepRecord:
    """One concise, user-safe record of an executed plan step."""

    step: str
    status: str = STEP_SUCCESS
    kind: str = "model"                 # "model" | "tool" | "stage"
    parameters: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    output_summary: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None        # populated for skipped/unsupported/error

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "step": self.step,
            "status": self.status,
            "kind": self.kind,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.parameters:
            out["parameters"] = self.parameters
        if self.output_summary:
            out["output_summary"] = self.output_summary
        if self.reason:
            out["reason"] = self.reason
        return out


class ExecutionTrace:
    """Ordered collection of :class:`StepRecord` with convenience helpers."""

    def __init__(self):
        self._records: List[StepRecord] = []

    def add(self, record: StepRecord) -> StepRecord:
        self._records.append(record)
        return record

    def record(
        self,
        step: str,
        status: str = STEP_SUCCESS,
        kind: str = "model",
        parameters: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
        output_summary: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
    ) -> StepRecord:
        return self.add(StepRecord(
            step=step, status=status, kind=kind,
            parameters=parameters or {}, duration_ms=duration_ms,
            output_summary=output_summary or {}, reason=reason,
        ))

    def to_list(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records]

    @property
    def records(self) -> List[StepRecord]:
        return list(self._records)


def summarize_tool_output(tool_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact ``output_summary`` from a ToolResult.to_dict() payload,
    keeping only the salient scalar fields (never big arrays / region lists).
    """
    if not isinstance(tool_dict, dict):
        return {}
    keep_keys = (
        "mean", "min", "max", "median", "valid_pixel_percentage",
        "changed_pixels", "changed_percentage", "area_m2", "area_km2",
        "region_count", "largest_region_pixels",
        "calibrated", "unit", "vv_vh_ratio",
        "resolution_x_m", "resolution_y_m", "mean_resolution_m",
        "west", "south", "east", "north",
        "n_bands",
    )
    summary = {k: tool_dict[k] for k in keep_keys if k in tool_dict}
    # For geolocation single-point results, surface the coordinate.
    if "geographic" in tool_dict and isinstance(tool_dict["geographic"], dict):
        summary["geographic"] = tool_dict["geographic"]
    return summary
