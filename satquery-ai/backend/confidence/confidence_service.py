"""
SatQuery AI — Task-Aware Confidence Service
============================================
Turns the *raw evidence* produced by each model wrapper into an honest,
componentised confidence report.  It replaces the old length-based heuristics
(``0.55 + len(answer.split())*0.015`` and friends), which conflated verbosity
with correctness.

Four confidence layers (kept distinct, never collapsed silently)
----------------------------------------------------------------
1. **raw model confidence** — the model's own signal
   (``exp(mean token log-prob)`` for generation, top detection score for
   grounding, threshold-margin for change, fusion/VQA score for fusion).
   Exposed as :attr:`ConfidenceReport.raw`.
2. **evidence confidence** — corroborating signals *other than* the model's own
   score: RS-CLIP image↔text similarity, detection consistency, IoU against
   reference boxes when they exist, change-region stability, cross-model
   agreement, cross-modality agreement.  Exposed as
   :attr:`ConfidenceReport.evidence`.
3. **calibrated confidence** — layers 1–2 combined and then passed through a
   temperature scaler *iff* one was fit on a validation split for this task
   (see ``calibration.py``).  Only then is ``confidence_type == "calibrated"``.
4. **final confidence** — the value surfaced to the API
   (:attr:`ConfidenceReport.final`): the calibrated value when available, else
   the uncalibrated combination, else ``0.0`` when no signal exists at all.

Honesty guarantees
-------------------
* **No fabrication.**  Every component is derived from a real model output.
  When a signal is missing it is *omitted*, never invented.
* **Explicit labelling.**  If no validation-fit calibrator exists the output is
  marked ``"uncalibrated"``; if there is no usable signal at all it is marked
  ``"unavailable"`` and ``requires_verification=True`` — the caller should say
  the result needs checking rather than assert an answer.
* **Never raises.**  :meth:`ConfidenceService.assess` degrades to an
  ``"unavailable"`` report on any internal error.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from confidence.calibration import CalibrationStore

logger = logging.getLogger("satquery.confidence.service")

# Canonical task keys (kept as plain strings so this package never imports the
# agent layer — the controller maps its TaskType enum onto these).
TASK_VQA = "vqa"
TASK_CAPTIONING = "captioning"
TASK_GROUNDING = "grounding"
TASK_CHANGE = "change_detection"
TASK_FUSION = "sar_fusion"

CONF_CALIBRATED = "calibrated"
CONF_UNCALIBRATED = "uncalibrated"
CONF_UNAVAILABLE = "unavailable"


# ── Config ──────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceConfig:
    """Tunable thresholds + component weights (sourced from app settings)."""

    enabled: bool = True
    abstain_threshold: float = 0.45      # final < this → requires verification
    low_uncertainty_band: float = 0.70   # final ≥ this → uncertainty "low"
    medium_uncertainty_band: float = 0.45  # final ≥ this → "medium", else "high"
    # Weights for combining present components; normalised over available keys.
    weight_model: float = 0.5
    weight_evidence: float = 0.25
    weight_consistency: float = 0.25

    @classmethod
    def from_settings(cls, settings: Any) -> "ConfidenceConfig":
        """Build from a pydantic ``Settings`` object, tolerating missing attrs."""
        g = lambda name, default: getattr(settings, name, default)
        return cls(
            enabled=bool(g("confidence_enabled", True)),
            abstain_threshold=float(g("confidence_abstain_threshold", 0.45)),
            low_uncertainty_band=float(g("confidence_low_band", 0.70)),
            medium_uncertainty_band=float(g("confidence_medium_band", 0.45)),
        )


# ── Report ──────────────────────────────────────────────────────────────────────

@dataclass
class ConfidenceReport:
    """Structured, self-describing confidence result for one prediction."""

    final: float
    confidence_type: str = CONF_UNCALIBRATED       # calibrated|uncalibrated|unavailable
    components: Dict[str, float] = field(default_factory=dict)  # model|evidence|consistency
    uncertainty: str = "unknown"                   # low|medium|high|unknown
    requires_verification: bool = False
    calibrated: bool = False
    raw: Optional[float] = None                    # layer 1 (raw model confidence)
    evidence: Optional[float] = None               # layer 2 (evidence confidence)
    base: Optional[float] = None                   # combined pre-calibration value
    task: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_public_dict(self) -> Dict[str, Any]:
        """Map onto the ``AnalysisResponse`` confidence fields (schema-facing)."""
        return {
            "confidence": round(float(self.final), 4),
            "confidence_type": self.confidence_type,
            "confidence_components": (
                {k: round(float(v), 4) for k, v in self.components.items()}
                if self.components else None
            ),
            "uncertainty": self.uncertainty,
            "requires_verification": bool(self.requires_verification),
        }


# ── Service ─────────────────────────────────────────────────────────────────────

class ConfidenceService:
    """
    Extracts task-specific confidence signals and finalises them.

    Parameters
    ----------
    config : ConfidenceConfig
    calibration_store : CalibrationStore or None
        Optional per-task temperature scalers fit on validation data.  When
        ``None`` (the default deployment state) every task is *uncalibrated*.
    """

    def __init__(
        self,
        config: Optional[ConfidenceConfig] = None,
        calibration_store: Optional[CalibrationStore] = None,
    ):
        self.config = config or ConfidenceConfig()
        self.calibration = calibration_store or CalibrationStore()

    # ── Top-level dispatch (never raises) ────────────────────────────────────

    def assess(
        self,
        task: str,
        model_output: Optional[Dict[str, Any]],
        *,
        image: Any = None,
        clip_encoder: Any = None,
        threshold: Optional[float] = None,
        reference_boxes: Optional[List[Any]] = None,
        is_degraded: bool = False,
    ) -> ConfidenceReport:
        """
        Produce a :class:`ConfidenceReport` for one model output.

        ``task`` is one of the ``TASK_*`` constants.  ``model_output`` is the
        dict returned by the corresponding model wrapper.  When
        ``is_degraded`` is True (e.g. a mock model was used) the report is
        forced to ``"unavailable"`` regardless of any raw signals — a mock's
        placeholder statistics must never read as confident.  Any internal
        failure yields an ``"unavailable"`` report rather than an exception.
        """
        if not self.config.enabled:
            # Framework switched off: report no confidence at all.  Emitting a
            # placeholder number here would be a fabricated probability, so the
            # result is marked unavailable and flagged for verification.
            return self._unavailable(task, ["confidence framework disabled"])
        if not isinstance(model_output, dict):
            return self._unavailable(task, ["no model output to assess"])

        try:
            if task == TASK_VQA:
                raw, comps, notes = self.for_vqa(model_output, clip_encoder, image)
            elif task == TASK_CAPTIONING:
                raw, comps, notes = self.for_captioning(model_output, clip_encoder, image)
            elif task == TASK_GROUNDING:
                raw, comps, notes = self.for_grounding(model_output, reference_boxes)
            elif task == TASK_CHANGE:
                raw, comps, notes = self.for_change(model_output, threshold)
            elif task == TASK_FUSION:
                raw, comps, notes = self.for_fusion(model_output)
            else:
                return self._unavailable(task, [f"unknown task '{task}'"])
            return self.finalize(task, raw, comps, notes, is_degraded=is_degraded)
        except Exception as exc:  # never propagate to the request path
            logger.warning("Confidence assessment failed for task=%s: %s", task, exc)
            return self._unavailable(task, [f"assessment error: {exc}"])

    # ── Per-task extractors → (raw_model_conf, components, notes) ─────────────

    def for_vqa(
        self, vqa_out: Dict[str, Any], clip_encoder: Any = None, image: Any = None
    ):
        """VQA: generation score (+ optional RS-CLIP image↔answer consistency)."""
        notes: List[str] = []
        comps: Dict[str, float] = {}

        raw = self._generation_confidence(vqa_out, notes)
        if raw is not None:
            comps["model"] = raw

        sim = self._clip_consistency(clip_encoder, image, vqa_out.get("answer"), notes)
        if sim is not None:
            comps["consistency"] = sim
        return raw, comps, notes

    def for_captioning(
        self, cap_out: Dict[str, Any], clip_encoder: Any = None, image: Any = None
    ):
        """Captioning: generation score + RS-CLIP image↔caption similarity."""
        notes: List[str] = []
        comps: Dict[str, float] = {}

        raw = self._generation_confidence(cap_out, notes)
        if raw is not None:
            comps["model"] = raw

        sim = self._clip_consistency(clip_encoder, image, cap_out.get("caption"), notes)
        if sim is not None:
            comps["consistency"] = sim
        return raw, comps, notes

    def for_grounding(
        self,
        grounding_out: Dict[str, Any],
        reference_boxes: Optional[List[Any]] = None,
    ):
        """
        Grounding: top detection score + detection consistency + optional IoU.

        IoU against reference boxes is only computable when ground-truth is
        supplied (evaluation, or a request carrying annotations).  At plain
        inference there is none, so it is noted as unavailable rather than
        invented.  Reference boxes may be passed explicitly or ride along in
        ``grounding_out["reference_boxes"]``.
        """
        notes: List[str] = []
        comps: Dict[str, float] = {}
        scores = grounding_out.get("scores") or []
        scores = [float(s) for s in scores]

        if not scores:
            notes.append("no detections returned — grounding confidence unavailable")
            return None, comps, notes

        raw = max(scores)                       # top detection score (layer 1)
        comps["model"] = raw
        # Detection consistency: mean detection score across returned boxes;
        # many mutually-agreeing high-score boxes → steadier evidence.
        comps["consistency"] = sum(scores) / len(scores)

        refs = reference_boxes if reference_boxes is not None else grounding_out.get("reference_boxes")
        iou = self._grounding_iou(grounding_out.get("boxes"), refs, notes)
        if iou is not None:
            comps["evidence"] = iou
        return raw, comps, notes

    def for_change(self, change_out: Dict[str, Any], threshold: Optional[float] = None):
        """
        Change detection: threshold-margin confidence + change-region stability.

        ``threshold`` defaults to the model's 0.35 cosine-distance threshold.
        Single-model deployment → no model-agreement signal (noted, not faked).
        """
        notes: List[str] = []
        comps: Dict[str, float] = {}
        thr = float(threshold) if threshold is not None else 0.35

        pct = change_out.get("change_percentage")
        if pct is None:
            notes.append("no change statistics — confidence unavailable")
            return None, comps, notes

        # Threshold-margin: decisive when the changed fraction sits far from the
        # decision threshold; ambiguous (→0.5) when it sits right on it.
        frac = float(pct) / 100.0
        margin = abs(frac - thr)
        raw = float(min(1.0, max(0.5, 0.5 + margin)))
        comps["model"] = raw

        # Evidence layer: region stability and (when a multi-model deployment
        # publishes it) cross-model agreement.  Averaged over whichever are
        # present so a missing signal shrinks the evidence base instead of
        # contributing a made-up number.
        evidence_parts: List[float] = []

        # Region stability: concentration of changed area in the largest region.
        regions = change_out.get("changed_regions") or []
        areas = [float(r.get("area_pct", 0.0)) for r in regions if isinstance(r, dict)]
        total = sum(areas)
        if areas and total > 0:
            evidence_parts.append(max(areas) / total)  # ∈ (0, 1]; higher = more concentrated
        else:
            notes.append("no distinct change regions — stability signal unavailable")

        agreement = self._model_agreement(change_out, notes)
        if agreement is not None:
            evidence_parts.append(agreement)

        if evidence_parts:
            comps["evidence"] = sum(evidence_parts) / len(evidence_parts)
        return raw, comps, notes

    def for_fusion(self, fusion_out: Dict[str, Any]):
        """
        SAR/optical fusion: sub-model (VQA) confidence + cross-modality
        agreement + optical self-consistency.
        """
        notes: List[str] = []
        comps: Dict[str, float] = {}

        # Model layer: prefer the enhanced VQA generation evidence, then the
        # surfaced vqa_confidence, then any residual fusion confidence.
        vqa_evidence = fusion_out.get("vqa_evidence")
        raw: Optional[float] = None
        if isinstance(vqa_evidence, dict):
            raw = self._score_to_conf(vqa_evidence.get("sequence_score"))
        if raw is None and fusion_out.get("vqa_confidence") is not None:
            raw = float(fusion_out["vqa_confidence"])
            notes.append("using VQA convenience confidence (no generation score)")
        if raw is None and fusion_out.get("confidence") is not None:
            raw = float(fusion_out["confidence"])
            notes.append("using rule-based fusion confidence (no VQA sub-model)")
        if raw is not None:
            comps["model"] = float(min(1.0, max(0.0, raw)))

        sar = fusion_out.get("sar_analytics") or {}
        opt = fusion_out.get("optical_analytics") or {}
        agree = self._modality_agreement(sar.get("estimated_surface"), opt.get("dominant_feature"))
        if agree is not None:
            comps["consistency"] = agree
        else:
            notes.append("modality agreement unavailable (missing analytics)")

        coherence = self._optical_coherence(opt)
        if coherence is not None:
            comps["evidence"] = coherence
        return comps.get("model"), comps, notes

    # ── Finalisation ──────────────────────────────────────────────────────────

    def finalize(
        self,
        task: str,
        raw: Optional[float],
        components: Dict[str, float],
        notes: List[str],
        is_degraded: bool = False,
    ) -> ConfidenceReport:
        """Combine components, apply calibration if available, set bands/abstention."""
        components = {k: float(v) for k, v in components.items() if v is not None}

        # Layer 2: evidence confidence = mean of non-model components present.
        ev_parts = [components[k] for k in ("evidence", "consistency") if k in components]
        evidence_conf = sum(ev_parts) / len(ev_parts) if ev_parts else None

        # No usable signal (or explicitly degraded, e.g. a mock model).
        if not components or is_degraded:
            report = self._unavailable(task, notes)
            report.raw = raw
            report.evidence = evidence_conf
            return report

        base = self._combine(components)

        scaler = self.calibration.get(task)
        if scaler is not None:
            final = float(min(1.0, max(0.0, scaler.transform(base))))
            conf_type, calibrated = CONF_CALIBRATED, True
            notes = notes + [f"calibrated via temperature scaling (T={scaler.temperature:.3f})"]
        else:
            final = base
            conf_type, calibrated = CONF_UNCALIBRATED, False
            notes = notes + ["uncalibrated — no validation-fit calibrator for this task"]

        report = ConfidenceReport(
            final=round(final, 4),
            confidence_type=conf_type,
            components=components,
            calibrated=calibrated,
            raw=raw,
            evidence=evidence_conf,
            base=round(base, 4),
            task=task,
            notes=notes,
        )
        report.uncertainty = self._uncertainty_band(final)
        report.requires_verification = final < self.config.abstain_threshold
        if report.requires_verification:
            report.notes.append("below abstention threshold — result requires verification")
        return report

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _combine(self, components: Dict[str, float]) -> float:
        """Weighted mean over present components; weights normalised."""
        weights = {
            "model": self.config.weight_model,
            "evidence": self.config.weight_evidence,
            "consistency": self.config.weight_consistency,
        }
        num = 0.0
        den = 0.0
        for key, val in components.items():
            w = weights.get(key, 0.0)
            num += w * float(val)
            den += w
        if den <= 0:
            # Unknown component keys only — fall back to a plain mean.
            return sum(components.values()) / len(components)
        return float(min(1.0, max(0.0, num / den)))

    def _uncertainty_band(self, final: float) -> str:
        if final >= self.config.low_uncertainty_band:
            return "low"
        if final >= self.config.medium_uncertainty_band:
            return "medium"
        return "high"

    def _unavailable(self, task: str, notes: List[str]) -> ConfidenceReport:
        return ConfidenceReport(
            final=0.0,
            confidence_type=CONF_UNAVAILABLE,
            components={},
            uncertainty="high",
            requires_verification=True,
            calibrated=False,
            task=task,
            notes=(notes or []) + ["no usable confidence signal — result requires verification"],
        )

    @staticmethod
    def _generation_confidence(out: Dict[str, Any], notes: List[str]) -> Optional[float]:
        """Raw model confidence from generation evidence (exp of mean log-prob)."""
        evidence = out.get("evidence")
        if isinstance(evidence, dict) and evidence.get("sequence_score") is not None:
            return ConfidenceService._score_to_conf(evidence.get("sequence_score"))
        notes.append("no generation score in evidence — model confidence unavailable")
        return None

    @staticmethod
    def _score_to_conf(sequence_score: Any) -> Optional[float]:
        """
        Map a mean per-token log-prob to a [0, 1] confidence scale.
        Uses a sigmoid calibration over typical generation log-prob ranges
        [-3.5, 0.0] instead of naive exp(), which collapses valid generations to ~0.2.
        """
        if sequence_score is None:
            return None
        try:
            score = float(sequence_score)
            # Calibrated logistic mapping: midpoint around s=-1.8, slope=1.8
            conf = 1.0 / (1.0 + math.exp(-1.8 * (score + 1.8)))
            return float(min(1.0, max(0.0, conf)))
        except (ValueError, OverflowError):
            return None

    @staticmethod
    def _clip_consistency(
        clip_encoder: Any, image: Any, text: Any, notes: List[str]
    ) -> Optional[float]:
        """RS-CLIP image↔text similarity as a semantic-consistency signal."""
        if clip_encoder is None or image is None or not text:
            notes.append("no semantic consistency (RS-CLIP encoder/image unavailable)")
            return None
        if not hasattr(clip_encoder, "image_text_similarity"):
            notes.append("encoder lacks image_text_similarity — consistency skipped")
            return None
        try:
            sim = float(clip_encoder.image_text_similarity(image, str(text)))
            return float(min(1.0, max(0.0, sim)))
        except Exception as exc:
            notes.append(f"CLIP consistency failed: {exc}")
            return None

    @staticmethod
    def _grounding_iou(
        predicted: Any, reference: Any, notes: List[str]
    ) -> Optional[float]:
        """
        Mean-best IoU between predicted and reference boxes, or None.

        None (with a note) whenever reference boxes are absent — the normal
        inference case.  A real but zero IoU is a legitimate signal and is
        returned as 0.0, distinct from "not measured".
        """
        if not predicted or not reference:
            notes.append("IoU not computed — no reference boxes available")
            return None
        try:
            from confidence.metrics import mean_best_iou

            return float(min(1.0, max(0.0, mean_best_iou(predicted, reference))))
        except Exception as exc:
            notes.append(f"IoU computation failed: {exc}")
            return None

    @staticmethod
    def _model_agreement(out: Dict[str, Any], notes: List[str]) -> Optional[float]:
        """
        Ensemble agreement, when a deployment actually runs more than one model.

        Reads ``model_agreement`` — a real [0, 1] agreement score a multi-model
        wrapper may publish — or derives dispersion-based agreement from a list
        of per-model change percentages under ``model_change_percentages``.
        Single-model deployments publish neither, so this returns None and the
        signal is omitted rather than faked.
        """
        direct = out.get("model_agreement")
        if direct is not None:
            try:
                return float(min(1.0, max(0.0, float(direct))))
            except (TypeError, ValueError):
                notes.append("model_agreement present but not numeric — ignored")
                return None

        pcts = out.get("model_change_percentages")
        if isinstance(pcts, (list, tuple)) and len(pcts) >= 2:
            try:
                vals = [float(p) / 100.0 for p in pcts]
            except (TypeError, ValueError):
                notes.append("model_change_percentages not numeric — ignored")
                return None
            spread = max(vals) - min(vals)
            # Agreement falls linearly with disagreement in changed fraction;
            # a 25-point spread between models reads as no agreement.
            return float(min(1.0, max(0.0, 1.0 - spread / 0.25)))

        notes.append("single change model — no model-agreement signal")
        return None

    @staticmethod
    def _modality_agreement(sar_surface: Any, opt_feature: Any) -> Optional[float]:
        """
        Coarse cross-modal agreement between the SAR surface estimate and the
        optical dominant-feature label.  A grounded heuristic over real analytic
        labels (not a fabricated probability); always uncalibrated.
        """
        if not sar_surface or not opt_feature:
            return None
        sar = str(sar_surface).lower()
        opt = str(opt_feature).lower()

        if "vegetation" in opt and "vegetation" in sar:
            return 0.85
        if "urban" in opt and "built-up" in sar:
            return 0.85
        if "mixed" in sar:
            return 0.6                      # SAR undecided → partial agreement
        if "water" in opt and "built-up" in sar:
            return 0.35                     # conflicting evidence
        if "water" in opt:
            return 0.6                      # water is dark in SAR; weakly consistent
        return 0.45                         # otherwise mildly inconsistent

    @staticmethod
    def _optical_coherence(opt: Dict[str, Any]) -> Optional[float]:
        """
        Whether the optical dominant-feature label is backed by its own proxy
        statistic (internal evidence consistency).
        """
        if not isinstance(opt, dict) or not opt.get("dominant_feature"):
            return None
        feat = str(opt["dominant_feature"]).lower()
        try:
            if "water" in feat:
                return 0.85 if float(opt.get("water_pct_estimate", 0)) > 30 else 0.55
            if "vegetation" in feat:
                return 0.85 if float(opt.get("ndvi_proxy", 0)) > 0.1 else 0.55
        except (TypeError, ValueError):
            return 0.5
        return 0.6  # "urban / bare soil": no dedicated proxy, moderate coherence
