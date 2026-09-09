"""
SatQuery AI — Calibration Metrics
==================================
Pure, dependency-light (numpy only) functions for measuring how well a set of
confidence scores is *calibrated* against observed correctness.

A model is **calibrated** when, among all predictions it makes with confidence
``p``, a fraction ``p`` are actually correct.  These metrics quantify the gap:

  * ``expected_calibration_error``  — average |confidence − accuracy| over bins
  * ``maximum_calibration_error``   — worst-case bin gap
  * ``brier_score``                 — mean squared error of the confidence
  * ``reliability_curve``           — per-bin stats for a reliability diagram
  * ``reliability_stats``           — convenience bundle of all of the above

Localisation agreement (used by the grounding confidence path when reference
boxes exist) lives here too:

  * ``box_iou``       — intersection-over-union of two boxes
  * ``mean_best_iou`` — mean over predictions of the best IoU against references

Inputs
------
``confidences`` : sequence of floats in [0, 1] (predicted confidence)
``correct``     : sequence of 0/1 or bool (was the prediction correct)

All functions validate that the two inputs are the same length and degrade
gracefully on empty input (returning ``0.0`` / empty structures) rather than
raising — callers frequently evaluate on small or empty slices.

These functions never fabricate values; they only summarise the numbers passed
in.  Fitting a calibrator lives in ``calibration.py``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


# ── Input handling ────────────────────────────────────────────────────────────

def _as_arrays(confidences: Sequence[float], correct: Sequence[Any]):
    conf = np.asarray(list(confidences), dtype=np.float64)
    corr = np.asarray(list(correct), dtype=np.float64)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"confidences and correct must be the same length "
            f"({conf.shape[0]} vs {corr.shape[0]})"
        )
    # Clamp confidences into [0, 1]; coerce correctness to {0, 1}.
    conf = np.clip(conf, 0.0, 1.0)
    corr = (corr > 0.5).astype(np.float64)
    return conf, corr


def _bin_edges(n_bins: int) -> np.ndarray:
    n_bins = max(1, int(n_bins))
    return np.linspace(0.0, 1.0, n_bins + 1)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[Any],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).

    Bin predictions by confidence; within each bin compute the absolute gap
    between mean confidence and empirical accuracy, then average across bins
    weighted by the number of samples in each bin.

    Returns 0.0 for empty input.
    """
    conf, corr = _as_arrays(confidences, correct)
    n = conf.shape[0]
    if n == 0:
        return 0.0

    edges = _bin_edges(n_bins)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Last bin is inclusive of the right edge so confidence==1.0 lands.
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(corr[mask].mean())
        ece += (count / n) * abs(avg_conf - acc)
    return float(ece)


def maximum_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[Any],
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE) — the worst bin gap. 0.0 for empty input."""
    conf, corr = _as_arrays(confidences, correct)
    if conf.shape[0] == 0:
        return 0.0

    edges = _bin_edges(n_bins)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        gap = abs(float(conf[mask].mean()) - float(corr[mask].mean()))
        worst = max(worst, gap)
    return float(worst)


def brier_score(confidences: Sequence[float], correct: Sequence[Any]) -> float:
    """
    Brier score — mean squared error between confidence and the 0/1 outcome.

    Lower is better; ranges in [0, 1].  Returns 0.0 for empty input.
    """
    conf, corr = _as_arrays(confidences, correct)
    if conf.shape[0] == 0:
        return 0.0
    return float(np.mean((conf - corr) ** 2))


def reliability_curve(
    confidences: Sequence[float],
    correct: Sequence[Any],
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """
    Per-bin statistics for plotting a reliability diagram.

    Returns one dict per bin::

        {bin_lower, bin_upper, avg_confidence, accuracy, count}

    Bins with no samples are still returned (count=0, avg_confidence/accuracy
    equal to the bin midpoint / 0.0) so the curve has a stable shape.  The sum
    of ``count`` across bins equals the number of input samples.
    """
    conf, corr = _as_arrays(confidences, correct)
    edges = _bin_edges(n_bins)
    bins: List[Dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(mask.sum())
        if count > 0:
            avg_conf = float(conf[mask].mean())
            acc = float(corr[mask].mean())
        else:
            avg_conf = float((lo + hi) / 2.0)
            acc = 0.0
        bins.append({
            "bin_lower": round(float(lo), 4),
            "bin_upper": round(float(hi), 4),
            "avg_confidence": round(avg_conf, 4),
            "accuracy": round(acc, 4),
            "count": count,
        })
    return bins


# ── Localisation agreement (grounding) ────────────────────────────────────────

def box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """
    Intersection-over-union of two axis-aligned boxes ``[x1, y1, x2, y2]``.

    Coordinates may be normalised or absolute as long as both boxes use the
    same convention.  Degenerate or non-overlapping boxes give ``0.0``; malformed
    input (wrong length, non-numeric) also gives ``0.0`` rather than raising, so
    a bad reference annotation cannot break the request path.
    """
    try:
        ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
        bx1, by1, bx2, by2 = (float(v) for v in box_b)
    except (TypeError, ValueError):
        return 0.0

    # Normalise corner order so a reversed box still measures correctly.
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)

    inter_w = min(ax2, bx2) - max(ax1, bx1)
    inter_h = min(ay2, by2) - max(ay1, by1)
    if inter_w <= 0.0 or inter_h <= 0.0:
        return 0.0

    inter = inter_w * inter_h
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if union <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, inter / union)))


def mean_best_iou(
    predicted: Sequence[Sequence[float]],
    reference: Sequence[Sequence[float]],
) -> float:
    """
    Mean over predicted boxes of the best IoU against any reference box.

    A localisation-agreement summary in [0, 1] for the grounding confidence
    path.  Returns ``0.0`` when either side is empty — meaning "no agreement
    measured", so callers must check emptiness themselves before treating the
    result as evidence (``ConfidenceService.for_grounding`` does).
    """
    preds = [p for p in (predicted or []) if p is not None]
    refs = [r for r in (reference or []) if r is not None]
    if not preds or not refs:
        return 0.0
    return float(
        sum(max(box_iou(p, r) for r in refs) for p in preds) / len(preds)
    )


def reliability_stats(
    confidences: Sequence[float],
    correct: Sequence[Any],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Convenience bundle: ECE, MCE, Brier score, per-bin reliability curve, and
    the overall accuracy / mean confidence.  Safe on empty input.
    """
    conf, corr = _as_arrays(confidences, correct)
    n = int(conf.shape[0])
    return {
        "n_samples": n,
        "accuracy": round(float(corr.mean()), 4) if n else 0.0,
        "mean_confidence": round(float(conf.mean()), 4) if n else 0.0,
        "ece": round(expected_calibration_error(confidences, correct, n_bins), 4),
        "mce": round(maximum_calibration_error(confidences, correct, n_bins), 4),
        "brier": round(brier_score(confidences, correct), 4),
        "bins": reliability_curve(confidences, correct, n_bins),
    }
