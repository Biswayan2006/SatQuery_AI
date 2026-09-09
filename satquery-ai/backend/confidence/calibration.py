"""
SatQuery AI — Confidence Calibration
=====================================
Optional **temperature scaling** for confidence scores, plus a small on-disk
store mapping a task name to its fitted calibrator.

Temperature scaling is a single-parameter post-hoc calibration method: given raw
confidences ``p`` it converts them to logits ``z = log(p / (1 - p))``, divides by
a learned scalar ``T`` (the *temperature*), and maps back through the sigmoid::

    p_calibrated = sigmoid( logit(p) / T )

``T = 1`` is the identity (no change).  ``T > 1`` softens over-confident scores;
``T < 1`` sharpens under-confident ones.  A single scalar cannot invent
information — it only rescales — so it never fabricates a probability.

IMPORTANT — validation only
---------------------------
The temperature MUST be fit on a held-out **validation** split, never on the
test set (fitting on test would leak the evaluation labels and inflate reported
calibration).  The only fitting entry point is :func:`fit_from_validation`,
named to make that contract explicit.  ``metrics.py`` (which the *test* split
uses) contains no fitting code.
"""
from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from confidence.metrics import expected_calibration_error

logger = logging.getLogger("satquery.confidence.calibration")

_EPS = 1e-6


# ── Temperature scaler ──────────────────────────────────────────────────────────

class TemperatureScaler:
    """
    Single-parameter temperature-scaling calibrator.

    A freshly-constructed scaler has ``temperature = 1.0`` and ``fitted =
    False`` — i.e. it is the identity and is treated by the store as
    *not available* until :meth:`fit` succeeds.
    """

    method = "temperature_scaling"

    def __init__(self, temperature: float = 1.0, fitted: bool = False):
        self.temperature = float(temperature)
        self.fitted = bool(fitted)
        self.n_val_samples: int = 0
        self.ece_before: Optional[float] = None
        self.ece_after: Optional[float] = None

    # ── Fitting (validation set only) ────────────────────────────────────────

    def fit(
        self,
        confidences: Sequence[float],
        correct: Sequence[Any],
        max_iter: int = 200,
        n_bins: int = 10,
    ) -> "TemperatureScaler":
        """
        Learn the temperature that minimises binary NLL of the confidences
        against the 0/1 correctness labels.

        Parameters MUST come from a validation split — never the test set.

        Returns ``self`` (so callers can chain).  On degenerate input (empty,
        single-class, or optimisation failure) the scaler falls back to the
        identity ``T = 1`` and remains usable.
        """
        conf = [float(c) for c in confidences]
        corr = [1.0 if float(c) > 0.5 else 0.0 for c in correct]
        self.n_val_samples = len(conf)

        if self.n_val_samples == 0:
            logger.warning("TemperatureScaler.fit called with no samples — identity.")
            self.temperature, self.fitted = 1.0, False
            return self

        self.ece_before = expected_calibration_error(conf, corr, n_bins)

        # If all labels are identical, temperature is not identifiable.
        if len(set(corr)) < 2:
            logger.info("Single-class validation set — temperature left at 1.0.")
            self.temperature = 1.0
            self.fitted = True
            self.ece_after = self.ece_before
            return self

        temp = self._optimize_temperature(conf, corr, max_iter)
        self.temperature = float(max(temp, _EPS))
        self.fitted = True
        self.ece_after = expected_calibration_error(
            [self._apply(p) for p in conf], corr, n_bins
        )
        logger.info(
            "Temperature fit: T=%.4f (ECE %.4f -> %.4f, n=%d)",
            self.temperature, self.ece_before, self.ece_after, self.n_val_samples,
        )
        return self

    def _optimize_temperature(
        self, conf: List[float], corr: List[float], max_iter: int
    ) -> float:
        """Optimise T via torch LBFGS; fall back to a coarse grid search."""
        try:
            import torch

            logits = torch.tensor(
                [self._to_logit(p) for p in conf], dtype=torch.float64
            )
            labels = torch.tensor(corr, dtype=torch.float64)
            log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # T=exp(0)=1
            optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
            bce = torch.nn.BCEWithLogitsLoss()

            def _closure():
                optimizer.zero_grad()
                scaled = logits / torch.exp(log_t)
                loss = bce(scaled, labels)
                loss.backward()
                return loss

            optimizer.step(_closure)
            return float(torch.exp(log_t.detach()).item())
        except Exception as exc:  # torch missing or optimisation failed
            logger.info("LBFGS temperature fit unavailable (%s) — grid search.", exc)
            return self._grid_search(conf, corr)

    @staticmethod
    def _grid_search(conf: List[float], corr: List[float]) -> float:
        """Coarse-to-fine NLL grid search over T (numpy/pure-python fallback)."""
        def nll(t: float) -> float:
            total = 0.0
            for p, y in zip(conf, corr):
                z = math.log(min(max(p, _EPS), 1 - _EPS) / (1 - min(max(p, _EPS), 1 - _EPS)))
                q = 1.0 / (1.0 + math.exp(-z / t))
                q = min(max(q, _EPS), 1 - _EPS)
                total -= y * math.log(q) + (1 - y) * math.log(1 - q)
            return total / len(conf)

        best_t, best = 1.0, nll(1.0)
        grid = [0.1 * i for i in range(1, 51)]  # 0.1 .. 5.0
        for t in grid:
            val = nll(t)
            if val < best:
                best, best_t = val, t
        return best_t

    # ── Transform ────────────────────────────────────────────────────────────

    def transform(self, confidence: float) -> float:
        """Apply the fitted temperature to a single confidence in [0, 1]."""
        return self._apply(float(confidence))

    def transform_many(self, confidences: Sequence[float]) -> List[float]:
        return [self._apply(float(c)) for c in confidences]

    def _apply(self, p: float) -> float:
        z = self._to_logit(p)
        t = self.temperature if self.temperature > _EPS else 1.0
        return float(1.0 / (1.0 + math.exp(-z / t)))

    @staticmethod
    def _to_logit(p: float) -> float:
        p = min(max(p, _EPS), 1.0 - _EPS)
        return math.log(p / (1.0 - p))

    # ── Persistence ────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "temperature": round(self.temperature, 6),
            "fitted": self.fitted,
            "n_val_samples": self.n_val_samples,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemperatureScaler":
        obj = cls(
            temperature=float(data.get("temperature", 1.0)),
            fitted=bool(data.get("fitted", False)),
        )
        obj.n_val_samples = int(data.get("n_val_samples", 0))
        obj.ece_before = data.get("ece_before")
        obj.ece_after = data.get("ece_after")
        return obj


# ── Store ─────────────────────────────────────────────────────────────────────

class CalibrationStore:
    """
    Maps a task name → its fitted :class:`TemperatureScaler`.

    Persisted as ``calibration.json`` inside a directory (``calibration_dir``
    from app config).  When the file is absent every task is *uncalibrated*
    (the default deployment state) and :meth:`get` returns ``None``.
    """

    FILENAME = "calibration.json"

    def __init__(self, scalers: Optional[Dict[str, TemperatureScaler]] = None):
        self._scalers: Dict[str, TemperatureScaler] = scalers or {}

    # ── Access ──────────────────────────────────────────────────────────────
    def get(self, task: str) -> Optional[TemperatureScaler]:
        """Return the fitted scaler for a task, or None if none / not fitted."""
        scaler = self._scalers.get(task)
        if scaler is not None and scaler.fitted:
            return scaler
        return None

    def set(self, task: str, scaler: TemperatureScaler) -> None:
        self._scalers[task] = scaler

    def tasks(self) -> List[str]:
        return sorted(self._scalers.keys())

    # ── Persistence ────────────────────────────────────────────────────────────
    @classmethod
    def from_dir(cls, calibration_dir: Optional[str]) -> "CalibrationStore":
        """Load a store from ``<calibration_dir>/calibration.json`` if present."""
        if not calibration_dir:
            return cls()
        path = os.path.join(calibration_dir, cls.FILENAME)
        if not os.path.exists(path):
            logger.info("No calibration file at %s — running uncalibrated.", path)
            return cls()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            scalers = {
                task: TemperatureScaler.from_dict(sc)
                for task, sc in data.get("tasks", {}).items()
            }
            logger.info("Loaded calibration for tasks: %s", sorted(scalers))
            return cls(scalers)
        except Exception as exc:
            logger.warning("Failed to load calibration file %s: %s", path, exc)
            return cls()

    def save(self, calibration_dir: str) -> str:
        os.makedirs(calibration_dir, exist_ok=True)
        path = os.path.join(calibration_dir, self.FILENAME)
        payload = {"tasks": {t: s.to_dict() for t, s in self._scalers.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path


def fit_from_validation(
    task: str,
    confidences: Sequence[float],
    correct: Sequence[Any],
    store: Optional[CalibrationStore] = None,
    n_bins: int = 10,
) -> TemperatureScaler:
    """
    Fit a temperature for ``task`` on **validation** data and register it in a
    store.

    This is the single sanctioned entry point for calibration fitting.  The
    ``confidences``/``correct`` arrays must originate from a validation split;
    passing test-set labels here would leak the evaluation set and is a misuse
    of the API.

    Returns the fitted scaler.  Persist it with ``store.save(dir)``.
    """
    scaler = TemperatureScaler().fit(confidences, correct, n_bins=n_bins)
    if store is not None:
        store.set(task, scaler)
    return scaler
