"""
SatQuery AI — Task-Aware Confidence Framework Tests
====================================================
Covers the five areas the confidence phase requires:

  1. Confidence calculation   — per-task extraction + component combination
  2. Calibration              — temperature scaling (validation-only), metrics
  3. Low-confidence behaviour — abstention threshold + uncertainty bands
  4. Missing confidence signals — graceful omission, never fabricated
  5. Schema compatibility     — ConfidenceReport → AnalysisResponse fields

These tests exercise only pure-python / numpy paths and do not download or run
any real model, so they run fast and offline.

Run with:
    cd satquery-ai/backend
    python -m pytest tests/test_confidence.py -v
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np
import pytest

# ── Make backend importable ───────────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from confidence.confidence_service import (
    CONF_CALIBRATED,
    CONF_UNAVAILABLE,
    CONF_UNCALIBRATED,
    TASK_CAPTIONING,
    TASK_CHANGE,
    TASK_FUSION,
    TASK_GROUNDING,
    TASK_VQA,
    ConfidenceConfig,
    ConfidenceReport,
    ConfidenceService,
)
from confidence.calibration import (
    CalibrationStore,
    TemperatureScaler,
    fit_from_validation,
)
from confidence.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
    reliability_stats,
)


# ── Test doubles ──────────────────────────────────────────────────────────────

class _FakeCLIP:
    """Stub RS-CLIP encoder returning a fixed image↔text similarity."""

    def __init__(self, sim: float = 0.8):
        self._sim = sim
        self.calls = 0

    def image_text_similarity(self, image, text: str) -> float:
        self.calls += 1
        return self._sim


class _RaisingCLIP:
    """CLIP encoder whose similarity call blows up — must be caught, not raised."""

    def image_text_similarity(self, image, text: str) -> float:
        raise RuntimeError("boom")


def _gen_out(seq_score, answer="a lake"):
    """A VQA/captioning-style model output with a generation evidence block."""
    return {
        "answer": answer,
        "caption": answer,
        "confidence": 0.5,
        "confidence_is_calibrated": False,
        "evidence": {"sequence_score": seq_score, "token_logprobs": None},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIDENCE CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

class TestConfidenceCalculation:
    def test_vqa_uses_generation_score_not_length(self):
        """Raw model confidence must be exp(sequence_score), independent of length."""
        svc = ConfidenceService()
        short = svc.assess(TASK_VQA, _gen_out(-0.2, answer="lake"))
        long = svc.assess(TASK_VQA, _gen_out(-0.2, answer="a large lake " * 20))
        # Same generation score → same model confidence regardless of verbosity.
        assert short.components["model"] == pytest.approx(long.components["model"])
        assert short.components["model"] == pytest.approx(math.exp(-0.2), abs=1e-6)

    def test_vqa_adds_clip_consistency_component(self):
        svc = ConfidenceService()
        clip = _FakeCLIP(sim=0.9)
        r = svc.assess(TASK_VQA, _gen_out(-0.3), image=object(), clip_encoder=clip)
        assert clip.calls == 1
        assert r.components["consistency"] == pytest.approx(0.9)
        assert "model" in r.components
        # evidence layer = mean of non-model components (here just consistency)
        assert r.evidence == pytest.approx(0.9)

    def test_captioning_generation_plus_clip(self):
        svc = ConfidenceService()
        clip = _FakeCLIP(sim=0.7)
        r = svc.assess(
            TASK_CAPTIONING, _gen_out(-0.1, answer="urban area"),
            image=object(), clip_encoder=clip,
        )
        assert r.components["model"] == pytest.approx(math.exp(-0.1), abs=1e-6)
        assert r.components["consistency"] == pytest.approx(0.7)

    def test_grounding_top_score_and_detection_consistency(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_GROUNDING, {"scores": [0.9, 0.7, 0.8]})
        assert r.components["model"] == pytest.approx(0.9)          # top score
        assert r.components["consistency"] == pytest.approx(0.8)    # mean score
        assert "evidence" not in r.components                       # no IoU without refs
        assert any("IoU" in n for n in r.notes)                     # IoU noted, not invented

    def test_grounding_iou_when_reference_boxes_present(self):
        """IoU is computed (not invented) only when reference boxes are supplied."""
        svc = ConfidenceService()
        r = svc.assess(TASK_GROUNDING, {
            "scores": [0.9],
            "boxes": [[0, 0, 10, 10]],
            "reference_boxes": [[0, 0, 10, 10]],  # perfect overlap → IoU 1.0
        })
        assert r.components["evidence"] == pytest.approx(1.0, abs=1e-6)

    def test_change_threshold_margin_and_region_stability(self):
        svc = ConfidenceService()
        r = svc.assess(
            TASK_CHANGE,
            {"change_percentage": 50.0,
             "changed_regions": [{"area_pct": 30.0}, {"area_pct": 10.0}]},
            threshold=0.35,
        )
        # margin = |0.50 - 0.35| = 0.15 → raw = 0.5 + 0.15 = 0.65
        assert r.components["model"] == pytest.approx(0.65, abs=1e-6)
        # stability = largest region / total = 30 / 40 = 0.75
        assert r.components["evidence"] == pytest.approx(0.75, abs=1e-6)
        assert any("model-agreement" in n for n in r.notes)

    def test_fusion_modality_agreement_and_coherence(self):
        svc = ConfidenceService()
        out = {
            "vqa_evidence": {"sequence_score": -0.3},
            "sar_analytics": {"estimated_surface": "vegetation"},
            "optical_analytics": {"dominant_feature": "vegetation", "ndvi_proxy": 0.2},
        }
        r = svc.assess(TASK_FUSION, out)
        assert r.components["model"] == pytest.approx(math.exp(-0.3), abs=1e-6)
        assert r.components["consistency"] == pytest.approx(0.85)   # veg↔veg agreement
        assert r.components["evidence"] == pytest.approx(0.85)      # ndvi-backed coherence

    def test_combine_is_normalised_weighted_mean(self):
        """Combination weights are normalised over present components only."""
        cfg = ConfidenceConfig(weight_model=0.5, weight_evidence=0.25, weight_consistency=0.25)
        svc = ConfidenceService(cfg)
        # Only model + consistency present: weights 0.5 & 0.25 → normalised 2:1.
        comps = {"model": 0.9, "consistency": 0.6}
        combined = svc._combine(comps)
        assert combined == pytest.approx((0.5 * 0.9 + 0.25 * 0.6) / 0.75, abs=1e-6)

    def test_disabled_framework_emits_no_fabricated_number(self):
        """Switched off, the service must NOT invent a placeholder probability."""
        svc = ConfidenceService(ConfidenceConfig(enabled=False))
        r = svc.assess(TASK_VQA, _gen_out(-0.1))
        assert r.confidence_type == CONF_UNAVAILABLE
        assert r.final == 0.0
        assert r.requires_verification is True
        assert any("disabled" in n.lower() for n in r.notes)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibration:
    def test_identity_scaler_is_not_available_until_fit(self):
        ts = TemperatureScaler()
        assert ts.temperature == 1.0
        assert ts.fitted is False
        # transform is identity at T=1
        assert ts.transform(0.8) == pytest.approx(0.8, abs=1e-6)

    def test_fit_reduces_ece_on_overconfident_data(self):
        # Overconfident: high confidences but only ~half correct.
        conf = [0.95, 0.9, 0.92, 0.88, 0.91, 0.93, 0.9, 0.94]
        corr = [1, 0, 1, 0, 0, 1, 0, 1]
        ts = TemperatureScaler().fit(conf, corr)
        assert ts.fitted is True
        assert ts.n_val_samples == len(conf)
        assert ts.ece_after <= ts.ece_before + 1e-9

    def test_fit_empty_falls_back_to_identity(self):
        ts = TemperatureScaler().fit([], [])
        assert ts.temperature == 1.0
        assert ts.fitted is False

    def test_fit_single_class_leaves_temperature_at_one(self):
        ts = TemperatureScaler().fit([0.8, 0.7, 0.9], [1, 1, 1])
        assert ts.temperature == 1.0
        assert ts.fitted is True  # usable, but identity

    def test_transform_monotonic_and_bounded(self):
        ts = TemperatureScaler(temperature=2.0, fitted=True)
        vals = [ts.transform(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert vals == sorted(vals)              # order preserved
        assert ts.transform(0.5) == pytest.approx(0.5, abs=1e-6)  # 0.5 is a fixed point

    def test_roundtrip_serialisation(self):
        ts = TemperatureScaler().fit([0.9, 0.8, 0.4, 0.3], [1, 1, 0, 0])
        restored = TemperatureScaler.from_dict(ts.to_dict())
        # to_dict() rounds temperature to 6 places, so allow that quantisation.
        assert restored.temperature == pytest.approx(ts.temperature, abs=1e-6)
        assert restored.fitted == ts.fitted
        assert restored.transform(0.7) == pytest.approx(ts.transform(0.7), abs=1e-5)

    def test_store_get_returns_none_when_unfitted(self):
        store = CalibrationStore()
        assert store.get(TASK_VQA) is None
        store.set(TASK_VQA, TemperatureScaler())  # unfitted
        assert store.get(TASK_VQA) is None         # not surfaced until fitted

    def test_store_save_and_load_roundtrip(self):
        store = CalibrationStore()
        fit_from_validation(TASK_VQA, [0.9, 0.8, 0.4, 0.3], [1, 1, 0, 0], store=store)
        with tempfile.TemporaryDirectory() as d:
            path = store.save(d)
            assert os.path.exists(path)
            loaded = CalibrationStore.from_dir(d)
            assert loaded.get(TASK_VQA) is not None
            assert loaded.get(TASK_VQA).temperature == pytest.approx(
                store.get(TASK_VQA).temperature, abs=1e-6
            )

    def test_from_dir_missing_file_is_uncalibrated(self):
        with tempfile.TemporaryDirectory() as d:
            store = CalibrationStore.from_dir(d)  # no calibration.json
            assert store.get(TASK_VQA) is None

    def test_service_marks_calibrated_only_with_fitted_scaler(self):
        store = CalibrationStore()
        fit_from_validation(
            TASK_VQA, [0.9, 0.85, 0.4, 0.35, 0.6, 0.55], [1, 1, 0, 0, 1, 0], store=store
        )
        svc = ConfidenceService(calibration_store=store)
        r = svc.assess(TASK_VQA, _gen_out(-0.2))
        assert r.confidence_type == CONF_CALIBRATED
        assert r.calibrated is True
        # A task with no fitted scaler stays uncalibrated.
        r2 = svc.assess(TASK_GROUNDING, {"scores": [0.9, 0.8]})
        assert r2.confidence_type == CONF_UNCALIBRATED

    # ── Metrics ────────────────────────────────────────────────────────────────

    def test_ece_perfectly_calibrated_is_low(self):
        # 100 samples where accuracy matches confidence per group.
        conf = [0.9] * 10 + [0.1] * 10
        corr = [1] * 9 + [0] * 1 + [0] * 9 + [1] * 1  # 90% and 10% correct
        assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.0, abs=0.05)

    def test_ece_overconfident_is_high(self):
        conf = [0.99] * 10
        corr = [0] * 10  # always wrong but max confidence
        assert expected_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.99, abs=1e-6)

    def test_brier_bounds(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
        assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)

    def test_mce_is_worst_bin_gap(self):
        # 10 bins: conf 0.1 → bin [0.1,0.2] (acc 1.0, gap 0.9);
        #          conf 0.9 → bin [0.9,1.0] (acc 1.0, gap 0.1). Worst = 0.9.
        conf = [0.9, 0.9, 0.1, 0.1]
        corr = [1, 1, 1, 1]
        assert maximum_calibration_error(conf, corr, n_bins=10) == pytest.approx(0.9, abs=1e-6)

    def test_reliability_curve_counts_sum_to_n(self):
        conf = [0.05, 0.25, 0.55, 0.95]
        corr = [0, 0, 1, 1]
        bins = reliability_curve(conf, corr, n_bins=10)
        assert sum(b["count"] for b in bins) == len(conf)

    def test_reliability_stats_bundle_keys(self):
        stats = reliability_stats([0.9, 0.1], [1, 0], n_bins=5)
        for key in ("n_samples", "accuracy", "mean_confidence", "ece", "mce", "brier", "bins"):
            assert key in stats
        assert stats["n_samples"] == 2

    def test_metrics_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            expected_calibration_error([0.9, 0.1], [1])

    def test_metrics_empty_input_is_zero(self):
        assert expected_calibration_error([], []) == 0.0
        assert brier_score([], []) == 0.0
        assert reliability_stats([], [])["n_samples"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOW-CONFIDENCE BEHAVIOUR (abstention + bands)
# ══════════════════════════════════════════════════════════════════════════════

class TestLowConfidenceBehaviour:
    def test_below_threshold_requires_verification(self):
        svc = ConfidenceService(ConfidenceConfig(abstain_threshold=0.45))
        # sequence_score = log(0.3) → model confidence 0.3 < 0.45
        r = svc.assess(TASK_VQA, _gen_out(math.log(0.3)))
        assert r.final < 0.45
        assert r.requires_verification is True
        assert any("verification" in n.lower() for n in r.notes)

    def test_above_threshold_does_not_require_verification(self):
        svc = ConfidenceService(ConfidenceConfig(abstain_threshold=0.45))
        r = svc.assess(TASK_VQA, _gen_out(math.log(0.9)))
        assert r.final >= 0.45
        assert r.requires_verification is False

    def test_uncertainty_bands(self):
        svc = ConfidenceService(ConfidenceConfig(low_uncertainty_band=0.70, medium_uncertainty_band=0.45))
        assert svc._uncertainty_band(0.85) == "low"
        assert svc._uncertainty_band(0.60) == "medium"
        assert svc._uncertainty_band(0.30) == "high"

    def test_configurable_threshold_changes_outcome(self):
        out = _gen_out(math.log(0.6))  # model confidence ≈ 0.6
        strict = ConfidenceService(ConfidenceConfig(abstain_threshold=0.7))
        lenient = ConfidenceService(ConfidenceConfig(abstain_threshold=0.4))
        assert strict.assess(TASK_VQA, out).requires_verification is True
        assert lenient.assess(TASK_VQA, out).requires_verification is False

    def test_config_from_settings(self):
        class _S:
            confidence_enabled = True
            confidence_abstain_threshold = 0.5
            confidence_low_band = 0.8
            confidence_medium_band = 0.5
        cfg = ConfidenceConfig.from_settings(_S())
        assert cfg.abstain_threshold == 0.5
        assert cfg.low_uncertainty_band == 0.8


# ══════════════════════════════════════════════════════════════════════════════
# 4. MISSING CONFIDENCE SIGNALS (never fabricate)
# ══════════════════════════════════════════════════════════════════════════════

class TestMissingSignals:
    def test_vqa_no_generation_score_is_unavailable(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_VQA, _gen_out(None))
        assert r.confidence_type == CONF_UNAVAILABLE
        assert r.final == 0.0
        assert r.requires_verification is True
        assert r.components == {}

    def test_grounding_no_detections_is_unavailable(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_GROUNDING, {"scores": []})
        assert r.confidence_type == CONF_UNAVAILABLE
        assert r.requires_verification is True

    def test_change_no_statistics_is_unavailable(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_CHANGE, {"change_percentage": None})
        assert r.confidence_type == CONF_UNAVAILABLE

    def test_clip_unavailable_omits_consistency_not_faked(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_VQA, _gen_out(-0.2), image=None, clip_encoder=None)
        assert "consistency" not in r.components
        assert any("consistency" in n.lower() for n in r.notes)

    def test_clip_failure_is_swallowed(self):
        svc = ConfidenceService()
        r = svc.assess(
            TASK_VQA, _gen_out(-0.2), image=object(), clip_encoder=_RaisingCLIP()
        )
        # model component still present; consistency omitted, no exception raised
        assert "model" in r.components
        assert "consistency" not in r.components

    def test_degraded_forces_unavailable_even_with_signal(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_VQA, _gen_out(-0.05), is_degraded=True)
        assert r.confidence_type == CONF_UNAVAILABLE
        assert r.requires_verification is True
        # raw layer still recorded for transparency even though final is unavailable
        assert r.raw is not None

    def test_non_dict_output_is_unavailable(self):
        svc = ConfidenceService()
        assert svc.assess(TASK_VQA, None).confidence_type == CONF_UNAVAILABLE

    def test_unknown_task_is_unavailable(self):
        svc = ConfidenceService()
        r = svc.assess("no_such_task", {"answer": "x"})
        assert r.confidence_type == CONF_UNAVAILABLE

    def test_assess_never_raises_on_bad_input(self):
        svc = ConfidenceService()
        # A malformed grounding output (scores not a list of numbers) must not raise.
        r = svc.assess(TASK_GROUNDING, {"scores": "not-a-list"})
        assert isinstance(r, ConfidenceReport)
        assert r.confidence_type == CONF_UNAVAILABLE

    def test_fusion_missing_analytics_notes_and_omits(self):
        svc = ConfidenceService()
        r = svc.assess(TASK_FUSION, {"vqa_evidence": {"sequence_score": -0.3}})
        assert "model" in r.components
        assert "consistency" not in r.components
        assert any("modality agreement" in n.lower() for n in r.notes)


# ══════════════════════════════════════════════════════════════════════════════
# 5. SCHEMA COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaCompatibility:
    def test_to_public_dict_shape(self):
        r = ConfidenceReport(
            final=0.82,
            confidence_type=CONF_CALIBRATED,
            components={"model": 0.84, "evidence": 0.79, "consistency": 0.83},
            uncertainty="low",
            requires_verification=False,
        )
        d = r.to_public_dict()
        assert d == {
            "confidence": 0.82,
            "confidence_type": "calibrated",
            "confidence_components": {"model": 0.84, "evidence": 0.79, "consistency": 0.83},
            "uncertainty": "low",
            "requires_verification": False,
        }

    def test_public_dict_components_none_when_empty(self):
        r = ConfidenceReport(final=0.0, confidence_type=CONF_UNAVAILABLE)
        assert r.to_public_dict()["confidence_components"] is None

    def test_maps_onto_analysis_response(self):
        """The public dict must populate AnalysisResponse without error."""
        from api.schemas import AnalysisResponse, ExecutionSummary

        report = ConfidenceReport(
            final=0.7,
            confidence_type=CONF_UNCALIBRATED,
            components={"model": 0.7, "consistency": 0.6},
            uncertainty="medium",
            requires_verification=False,
        )
        conf = report.to_public_dict()
        resp = AnalysisResponse(
            session_id="sess_x",
            task="SINGLE_VQA",
            answer="a lake",
            confidence=conf["confidence"],
            confidence_type=conf["confidence_type"],
            confidence_components=conf["confidence_components"],
            uncertainty=conf["uncertainty"],
            requires_verification=conf["requires_verification"],
            execution_summary=ExecutionSummary(
                selected_task="SINGLE_VQA",
                task_confidence=0.9,
                models_used=["RemoteSensingVQA"],
                processing_time_ms=12.3,
            ),
        )
        assert resp.confidence == 0.7
        assert resp.confidence_type == "uncalibrated"
        assert resp.confidence_components == {"model": 0.7, "consistency": 0.6}
        assert resp.uncertainty == "medium"

    def test_analysis_response_confidence_bounds_enforced(self):
        """Schema clamps are respected — confidence must stay in [0, 1]."""
        from api.schemas import AnalysisResponse, ExecutionSummary
        from pydantic import ValidationError

        summary = ExecutionSummary(
            selected_task="SINGLE_VQA", task_confidence=0.9,
            models_used=["m"], processing_time_ms=1.0,
        )
        with pytest.raises(ValidationError):
            AnalysisResponse(
                session_id="s", task="SINGLE_VQA", answer="x",
                confidence=1.5, execution_summary=summary,
            )

    def test_defaults_are_backward_compatible(self):
        """Old clients that omit the new confidence fields still validate."""
        from api.schemas import AnalysisResponse, ExecutionSummary

        resp = AnalysisResponse(
            session_id="s", task="SINGLE_VQA", answer="x", confidence=0.5,
            execution_summary=ExecutionSummary(
                selected_task="SINGLE_VQA", task_confidence=0.9,
                models_used=["m"], processing_time_ms=1.0,
            ),
        )
        assert resp.confidence_type == "uncalibrated"
        assert resp.confidence_components is None
        assert resp.uncertainty == "unknown"
        assert resp.requires_verification is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
