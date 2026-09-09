"""
SatQuery AI — Task-Aware Confidence Framework
==============================================
A dedicated package that replaces the old length-based confidence heuristics
with an honest, four-layer, task-aware confidence estimate:

    raw model confidence → evidence confidence → calibrated confidence → final

Public API
----------
* :class:`ConfidenceService` / :class:`ConfidenceConfig` / :class:`ConfidenceReport`
  — extract per-task signals and finalise them (see ``confidence_service.py``).
* :class:`TemperatureScaler` / :class:`CalibrationStore` / :func:`fit_from_validation`
  — optional post-hoc calibration fit on a **validation** split only
  (see ``calibration.py``).
* Calibration metrics — :func:`expected_calibration_error`,
  :func:`maximum_calibration_error`, :func:`brier_score`,
  :func:`reliability_curve`, :func:`reliability_stats` (see ``metrics.py``).
"""
from __future__ import annotations

from confidence.calibration import (
    CalibrationStore,
    TemperatureScaler,
    fit_from_validation,
)
from confidence.confidence_service import (
    CONF_CALIBRATED,
    CONF_UNAVAILABLE,
    CONF_UNCALIBRATED,
    TASK_CHANGE,
    TASK_CAPTIONING,
    TASK_FUSION,
    TASK_GROUNDING,
    TASK_VQA,
    ConfidenceConfig,
    ConfidenceReport,
    ConfidenceService,
)
from confidence.metrics import (
    box_iou,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    mean_best_iou,
    reliability_curve,
    reliability_stats,
)

__all__ = [
    # service
    "ConfidenceService",
    "ConfidenceConfig",
    "ConfidenceReport",
    "TASK_VQA",
    "TASK_CAPTIONING",
    "TASK_GROUNDING",
    "TASK_CHANGE",
    "TASK_FUSION",
    "CONF_CALIBRATED",
    "CONF_UNCALIBRATED",
    "CONF_UNAVAILABLE",
    # calibration
    "TemperatureScaler",
    "CalibrationStore",
    "fit_from_validation",
    # metrics
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "reliability_curve",
    "reliability_stats",
    "box_iou",
    "mean_best_iou",
]
