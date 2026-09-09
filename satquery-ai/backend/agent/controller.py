"""
SatQuery AI — Agentic Orchestration Controller
Plans, executes, and integrates specialist model outputs.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent.execution_trace import (
    ExecutionTrace,
    STEP_ERROR,
    STEP_SKIPPED,
    STEP_SUCCESS,
    STEP_UNSUPPORTED,
    summarize_tool_output,
)
from agent.query_intent import IntentExtractor, QueryIntent
from agent.task_classifier import TaskClassifier, TaskType
from api.schemas import (
    AlignmentInfo,
    AnalysisResponse,
    BoundingBox,
    ChangeRegion,
    ExecutionSummary,
    GeoBBox,
)
from confidence import (
    CONF_UNAVAILABLE,
    TASK_CAPTIONING,
    TASK_CHANGE,
    TASK_FUSION,
    TASK_GROUNDING,
    TASK_VQA,
    ConfidenceReport,
    ConfidenceService,
)

logger = logging.getLogger("satquery.controller")

# Prefix prepended to answers whose confidence falls below the abstention
# threshold, so the API says the result needs checking rather than asserting it.
ABSTAIN_CAVEAT = "[Verification Recommended] Low confidence result - manual check suggested: "

# Map the planner's TaskType onto the confidence service's task keys.
_TASK_KEY = {
    TaskType.SINGLE_VQA: TASK_VQA,
    TaskType.LAND_COVER_CLASSIFICATION: TASK_VQA,
    TaskType.CAPTIONING: TASK_CAPTIONING,
    TaskType.GROUNDING: TASK_GROUNDING,
    TaskType.CHANGE_VQA: TASK_CHANGE,
    TaskType.CHANGE_DESCRIPTION: TASK_CHANGE,
    TaskType.SAR_OPTICAL_FUSION: TASK_FUSION,
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    task_type: TaskType
    task_confidence: float
    model_names: List[str]
    steps: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)
    # Structured query understanding attached to the plan (extended, not replaced).
    intent: Optional[QueryIntent] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Explicit, inspectable plan — produced *before* execution.

        Example::

            {"task": "CHANGE_VQA", "intent": "vegetation decrease",
             "steps": ["validate_images", "align_images", "compute_ndvi", ...]}
        """
        return {
            "task": self.task_type.value,
            "intent": self._intent_phrase(),
            "intent_detail": self.intent.to_dict() if self.intent else {},
            "models": list(self.model_names),
            "steps": list(self.steps),
            "parameters": dict(self.parameters),
        }

    def _intent_phrase(self) -> str:
        """A short human phrase like 'vegetation decrease' or 'water localize'."""
        if self.intent is None:
            return self.task_type.value.lower()
        parts = [p for p in (self.intent.concept, self.intent.operation) if p]
        return " ".join(parts) if parts else self.task_type.value.lower()


@dataclass
class RawResults:
    answer: str = ""
    confidence: float = 0.0
    visual_evidence_b64: Optional[str] = None
    change_map_b64: Optional[str] = None
    fusion_map_b64: Optional[str] = None
    grounding_boxes: Optional[List[Dict]] = None
    change_percentage: Optional[float] = None
    change_regions: Optional[List[Dict]] = None        # enriched region dicts
    alignment_info: Optional[Dict] = None              # from AlignmentResult.info
    warnings: List[str] = field(default_factory=list)
    model_outputs: Dict[str, Any] = field(default_factory=dict)
    is_degraded: bool = False                          # a mock model was used
    confidence_report: Optional[ConfidenceReport] = None  # task-aware confidence
    tool_evidence: Optional[List[Dict[str, Any]]] = None  # deterministic tool results
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)  # per-step metadata
    is_georeferenced: bool = True


# ── Controller ────────────────────────────────────────────────────────────────

class AgenticController:
    """
    Heart of the SatQuery AI system.

    Workflow:
      1. Classify task from query + image metadata
      2. Plan execution (which models, which order)
      3. Execute plan (call specialist models)
      4. Integrate results into a unified AnalysisResponse
    """

    def __init__(self, registry):
        self.registry = registry
        semantic_router = self._build_semantic_router()
        self.classifier = TaskClassifier(semantic_router=semantic_router)
        self.intent_extractor = IntentExtractor(encoder=self._get_clip_encoder())
        self.confidence = self._build_confidence_service()
        self.tool_planner = self._build_tool_planner()

    def _build_semantic_router(self):
        """
        Build a semantic router from the registry's RS-CLIP encoder when it is a
        real (non-mock) encoder; otherwise None so the classifier stays in
        keyword-only mode.  Never raises into construction.
        """
        try:
            from models.rs_clip.semantic_router import SemanticRouter
            enc = self._get_clip_encoder()
            if enc is not None:
                return SemanticRouter(encoder=enc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Semantic router init failed (%s) — keyword-only routing", exc)
        return None

    @staticmethod
    def _build_tool_planner():
        """
        Construct the deterministic tool planner.  Degrades to None on failure so
        the request path never breaks over the (optional) tool layer.
        """
        try:
            from agent.tool_planner import ToolPlanner
            return ToolPlanner()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tool planner init failed (%s) — tools disabled", exc)
            return None

    @staticmethod
    def _build_confidence_service() -> ConfidenceService:
        """
        Construct the task-aware confidence service from app settings, loading
        any validation-fit calibration.  Degrades to sane defaults on failure so
        the request path never breaks over confidence configuration.
        """
        try:
            from confidence import CalibrationStore, ConfidenceConfig
            from config import get_settings

            settings = get_settings()
            conf_config = ConfidenceConfig.from_settings(settings)
            store = CalibrationStore.from_dir(getattr(settings, "calibration_dir", None))
            return ConfidenceService(conf_config, store)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Confidence service init failed (%s) — using defaults", exc)
            return ConfidenceService()

    async def analyze(
        self,
        images: List[Dict],
        query: str,
        task_hint: Optional[str] = None,
    ) -> AnalysisResponse:
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        t_start = time.perf_counter()

        modalities = [img.get("modality", "unknown") for img in images]

        # ── 1. Classify ───────────────────────────────────────────────────────
        if task_hint:
            try:
                task_type = TaskType(task_hint.upper())
                task_confidence = 1.0
            except ValueError:
                logger.warning("Invalid task_hint '%s', using classifier", task_hint)
                task_type, task_confidence = self.classifier.classify(
                    query, len(images), modalities
                )
        else:
            task_type, task_confidence = self.classifier.classify(
                query, len(images), modalities
            )

        # Counting requires discrete detections, not generative VQA. Cross-modal
        # comparisons require fusion before any difference computation.
        if self._is_counting_query(query) and images:
            task_type = TaskType.GROUNDING
            task_confidence = max(task_confidence, 0.98)
        task_type, task_confidence = self._enforce_task_contract(
            task_type, task_confidence, images
        )

        logger.info("Task classified: %s (%.2f) | query='%s'", task_type, task_confidence, query[:80])

        # ── 1b. Structured query understanding ────────────────────────────────
        intent = self.intent_extractor.extract(query, len(images), modalities)
        logger.info("Query intent: %s", intent.to_dict())

        # ── 2. Plan (explicit, produced before execution) ─────────────────────
        plan = self._plan(task_type, task_confidence, images, query, intent)
        logger.info("Execution plan: %s", plan.to_dict())

        # ── 3. Execute ────────────────────────────────────────────────────────
        raw = await self._execute(plan, images, query)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # ── 4. Integrate ──────────────────────────────────────────────────────
        return self._integrate(raw, plan, session_id, elapsed_ms)

    # ── Plan ──────────────────────────────────────────────────────────────────

    def _plan(
        self,
        task_type: TaskType,
        task_confidence: float,
        images: List[Dict],
        query: str,
        intent: Optional[QueryIntent] = None,
    ) -> ExecutionPlan:
        """
        Build an explicit execution plan.  The base per-task template is
        preserved; the intent then *injects* deterministic-tool steps (compute
        NDVI/NDWI/NDBI, geographic conversion, SAR/spectral evidence) into the
        step list so the plan reflects exactly what will run.
        """
        base = {
            TaskType.SINGLE_VQA: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["RemoteSensingVQA"],
                steps=["validate_input", "preprocess_image", "run_vqa", "format_answer"],
                parameters={"min_new_tokens": 15, "max_new_tokens": 100, "num_beams": 4},
            ),
            TaskType.LAND_COVER_CLASSIFICATION: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["RSCLIPEncoder"],
                steps=["validate_input", "preprocess_image", "zero_shot_land_cover", "format_answer"],
                parameters={"candidate_labels": 4},
            ),
            TaskType.CAPTIONING: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["RemoteSensingCaptioning"],
                steps=["validate_input", "preprocess_image", "generate_caption", "format_answer"],
                parameters={"max_new_tokens": 200, "num_beams": 5},
            ),
            TaskType.GROUNDING: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["RemoteSensingGrounding"],
                steps=["validate_input", "preprocess_image", "run_grounding", "draw_boxes", "format_answer"],
                parameters={"score_threshold": 0.2, "nms_threshold": 0.5},
            ),
            TaskType.CHANGE_VQA: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["ChangeDetectionModel", "RemoteSensingVQA"],
                steps=["validate_pair", "align_images", "detect_changes", "run_change_vqa", "format_answer"],
                parameters={"change_threshold": 0.15, "max_new_tokens": 150},
            ),
            TaskType.CHANGE_DESCRIPTION: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["ChangeDetectionModel", "RemoteSensingCaptioning"],
                steps=["validate_pair", "align_images", "detect_changes", "describe_changes", "format_answer"],
                parameters={"change_threshold": 0.15},
            ),
            TaskType.SAR_OPTICAL_FUSION: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["SAROpticalFusionModel"],
                steps=["validate_pair", "align_images", "fuse_modalities", "run_fusion_vqa", "format_answer"],
                parameters={"fusion_method": "feature_concat"},
            ),
        }

        plan = base.get(task_type, base[TaskType.CAPTIONING])
        plan.intent = intent
        self._inject_intent_steps(plan, intent, images)
        return plan

    @staticmethod
    def _is_counting_query(query: str) -> bool:
        normalized = query.lower()
        return any(
            phrase in normalized
            for phrase in ("how many", "count", "number of", "count of")
        )

    @staticmethod
    def _modality_family(modality: str) -> str:
        if modality.lower() == "sar":
            return "sar"
        if modality.lower() in {"optical", "multispectral", "rgb"}:
            return "optical"
        return "unknown"

    @classmethod
    def _enforce_task_contract(
        cls,
        task_type: TaskType,
        confidence: float,
        images: List[Dict],
    ) -> tuple[TaskType, float]:
        if task_type not in (TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION):
            return task_type, confidence
        if len(images) < 2:
            return task_type, confidence

        families = [cls._modality_family(image.get("modality", "unknown")) for image in images[:2]]
        if families[0] == families[1] and families[0] in {"sar", "optical"}:
            return task_type, confidence
        if set(families) == {"sar", "optical"}:
            logger.info("Cross-modal change request rerouted to SAR-optical fusion")
            return TaskType.SAR_OPTICAL_FUSION, min(confidence, 0.9)
        return task_type, confidence

    def _inject_intent_steps(
        self,
        plan: ExecutionPlan,
        intent: Optional[QueryIntent],
        images: List[Dict],
    ) -> None:
        """
        Insert deterministic-tool step names into the plan's step list based on
        the extracted intent + what the raster data can actually support.  This
        keeps the *plan* honest: a step only appears if its tool can run.
        """
        if intent is None:
            return

        # Determine which logical bands / geo metadata are available up-front so
        # planned tool steps are not fabricated for unsupported data.
        raster0 = self._peek_raster(images[0]) if images else None
        modalities = [img.get("modality", "unknown") for img in images]
        has_sar = "sar" in modalities

        index = intent.index_tool()  # ("NDVI", ("nir","red")) etc, or None
        compute_step = {
            "NDVI": "compute_ndvi",
            "NDWI": "compute_ndwi",
            "NDBI": "compute_ndbi",
        }

        injected: List[str] = []

        # Spectral index — only if bands resolve.
        if index and raster0 is not None:
            name, bands = index
            if all(raster0.has_band(b) for b in bands):
                step = compute_step[name]
                injected.append(step)
                if intent.requires_comparison:
                    injected.append(f"compare_{name.lower()}")

        # SAR / spectral evidence for radar or fusion tasks.
        if has_sar or intent.requires_fusion:
            injected.append("compute_sar_statistics")

        # Change-area + region stats for comparison tasks.
        if intent.requires_comparison and plan.task_type in (
            TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION
        ):
            injected.append("compute_change_area")
            if intent.requires_grounding or intent.operation in ("increase", "localize"):
                injected.append("compute_connected_regions")

        # Geographic conversion where possible.
        wants_geo = intent.requires_grounding or (
            raster0 is not None and raster0.has_geo and plan.task_type == TaskType.GROUNDING
        )
        if wants_geo and raster0 is not None and raster0.has_geo:
            injected.append("convert_to_geographic")

        if not injected:
            return

        # Insert the deterministic steps just before the final answer/format step
        # so the plan reads validate → align → compute-evidence → answer.
        insert_at = len(plan.steps)
        for i, s in enumerate(plan.steps):
            if s in ("format_answer", "run_change_vqa", "run_vqa",
                     "describe_changes", "run_fusion_vqa"):
                insert_at = i
                break
        seen = set(plan.steps)
        deduped = [s for s in injected if s not in seen]
        plan.steps[insert_at:insert_at] = deduped

    def _peek_raster(self, image_data: Dict):
        """Build a RasterInput to inspect band/geo availability; None on failure."""
        if self.tool_planner is None:
            return None
        try:
            from tools import RasterInput
            return RasterInput.from_image_data(image_data)
        except Exception:
            return None

    # ── Execute ───────────────────────────────────────────────────────────────

    async def _execute(
        self, plan: ExecutionPlan, images: List[Dict], query: str
    ) -> RawResults:
        """Dispatch execution to the appropriate handler, then assess confidence."""
        handlers = {
            TaskType.SINGLE_VQA: self._exec_vqa,
            TaskType.LAND_COVER_CLASSIFICATION: self._exec_land_cover_classification,
            TaskType.CAPTIONING: self._exec_captioning,
            TaskType.GROUNDING: self._exec_grounding,
            TaskType.CHANGE_VQA: self._exec_change_vqa,
            TaskType.CHANGE_DESCRIPTION: self._exec_change_description,
            TaskType.SAR_OPTICAL_FUSION: self._exec_sar_fusion,
        }
        handler = handlers.get(plan.task_type, self._exec_captioning)

        trace = ExecutionTrace()
        # Time the model/handler stage as a single "stage" record.
        t0 = time.perf_counter()
        raw = await handler(plan, images, query)
        if raw.alignment_info is not None:
            raw.is_georeferenced = bool(
                raw.alignment_info.get("geographic_coordinates_available", False)
            )
        else:
            raw.is_georeferenced = all(self._image_has_geo_metadata(image) for image in images)
        trace.record(
            step=plan.task_type.value,
            kind="stage",
            status=STEP_SUCCESS if raw.model_outputs else STEP_ERROR,
            parameters=plan.parameters,
            duration_ms=(time.perf_counter() - t0) * 1000,
            output_summary={"models": plan.model_names},
            reason=None if raw.model_outputs else "no model output produced",
        )

        # ── Deterministic tool evidence (numbers the model must not fabricate) ─
        raw.tool_evidence = self._run_tools(plan, images, query, raw, trace)
        raw.execution_trace = trace.to_list()

        # ── Task-aware confidence (replaces per-model heuristics) ─────────────
        raw.is_degraded = raw.is_degraded or self._plan_uses_mock(plan)
        raw.confidence_report = self._assess_confidence(plan, raw, images)

        # Abstention: when the (real) result is low-confidence, say it needs
        # verification rather than asserting it.  Skip for pure error responses
        # (no model output) and when the framework is disabled.
        report = raw.confidence_report
        if (
            self.confidence.config.enabled
            and report is not None
            and report.requires_verification
            and raw.model_outputs
            and not raw.answer.startswith(ABSTAIN_CAVEAT)
        ):
            raw.answer = ABSTAIN_CAVEAT + raw.answer

        return raw

    # ── Deterministic tool layer ──────────────────────────────────────────────

    def _run_tools(
        self,
        plan: ExecutionPlan,
        images: List[Dict],
        query: str,
        raw: RawResults,
        trace: Optional[ExecutionTrace] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Run the deterministic tools appropriate to this task and return their
        structured evidence.  Never raises — a tool-layer failure yields None.
        Each tool result is also recorded as a concise step in ``trace``.
        """
        if self.tool_planner is None:
            return None
        try:
            # Surface upstream artefacts the change tools consume: the binary
            # change mask and (when derivable) the ground pixel resolution.
            extras: Dict[str, Any] = {}
            change_out = raw.model_outputs.get("change") or raw.model_outputs.get("change_vqa")
            if isinstance(change_out, dict):
                mask = change_out.get("binary_mask")
                if mask is not None:
                    extras["change_mask"] = mask

            t0 = time.perf_counter()
            evidence = self.tool_planner.run_for_task(
                plan.task_type, query, images, raw_extras=extras
            )
            tool_ms = (time.perf_counter() - t0) * 1000

            # Record each tool's outcome as a user-safe step (status carried from
            # the ToolResult; unsupported is evidence, not an error).
            if trace is not None and evidence:
                # Distribute the measured tool time evenly (fine-grained per-tool
                # timing would require instrumenting the planner; the aggregate is
                # accurate and the split is a reasonable approximation).
                per = tool_ms / max(len(evidence), 1)
                for ev in evidence:
                    status = ev.get("status", STEP_SUCCESS)
                    step_status = {
                        "success": STEP_SUCCESS,
                        "unsupported": STEP_UNSUPPORTED,
                        "error": STEP_ERROR,
                    }.get(status, status)
                    trace.record(
                        step=ev.get("tool", "tool"),
                        kind="tool",
                        status=step_status,
                        duration_ms=per,
                        output_summary=summarize_tool_output(ev),
                        reason=ev.get("reason"),
                    )

            return evidence or None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Deterministic tool run failed: %s", exc)
            if trace is not None:
                trace.record(step="deterministic_tools", kind="tool",
                             status=STEP_ERROR, reason=str(exc))
            return None

    # ── Confidence helpers ────────────────────────────────────────────────────

    def _plan_uses_mock(self, plan: ExecutionPlan) -> bool:
        """True if any model this plan relies on resolved to a mock fallback."""
        from models._mock import MockModel

        for name in plan.model_names:
            try:
                # Use the regular get method here since we're just checking if it's a mock
                if isinstance(self.registry.get(name), MockModel):
                    return True
            except Exception:
                return True
        return False

    def _get_clip_encoder(self):
        """
        Resolve the RS-CLIP encoder for semantic consistency, or None.

        A mock fallback (which lacks ``image_text_similarity``) is ignored so
        consistency is simply omitted rather than fabricated.
        """
        try:
            # For CLIP encoder, we don't need concurrency control since it's
            # used for similarity checking, not inference
            enc = self.registry.get("RSCLIPEncoder")
            if enc is not None and hasattr(enc, "image_text_similarity"):
                return enc
        except Exception:
            pass
        return None

    def _change_threshold(self) -> float:
        try:
            cm = self.registry.get("ChangeDetectionModel")
            return float(getattr(cm, "threshold", 0.35))
        except Exception:
            return 0.35

    def _assess_confidence(
        self, plan: ExecutionPlan, raw: RawResults, images: List[Dict]
    ) -> ConfidenceReport:
        """Route the raw model output to the task-specific confidence extractor."""
        task_key = _TASK_KEY.get(plan.task_type)
        if task_key is None:
            return ConfidenceReport(
                final=0.0, confidence_type=CONF_UNAVAILABLE,
                uncertainty="high", requires_verification=True,
                notes=["no confidence route for task"],
            )

        mo = raw.model_outputs
        if plan.task_type == TaskType.LAND_COVER_CLASSIFICATION:
            classification = mo.get("land_cover_classification", {})
            labels = classification.get("labels", [])
            top_score = float(labels[0].get("score", 0.0)) if labels else 0.0
            return ConfidenceReport(
                final=raw.confidence,
                confidence_type="uncalibrated",
                uncertainty="low" if raw.confidence >= 0.7 else "medium",
                requires_verification=raw.confidence < 0.5,
                notes=[f"RS-CLIP top candidate score={top_score:.3f}"],
            )
        image = self._get_pil(images[0]) if images else None
        clip = self._get_clip_encoder()

        if plan.task_type == TaskType.SINGLE_VQA:
            return self.confidence.assess(
                TASK_VQA, mo.get("vqa"), image=image, clip_encoder=clip,
                is_degraded=raw.is_degraded,
            )
        if plan.task_type == TaskType.CAPTIONING:
            return self.confidence.assess(
                TASK_CAPTIONING, mo.get("captioning"), image=image, clip_encoder=clip,
                is_degraded=raw.is_degraded,
            )
        if plan.task_type == TaskType.GROUNDING:
            return self.confidence.assess(
                TASK_GROUNDING, mo.get("grounding"), is_degraded=raw.is_degraded,
            )
        if plan.task_type == TaskType.CHANGE_VQA:
            return self.confidence.assess(
                TASK_CHANGE, mo.get("change_vqa"), threshold=self._change_threshold(),
                is_degraded=raw.is_degraded,
            )
        if plan.task_type == TaskType.CHANGE_DESCRIPTION:
            return self.confidence.assess(
                TASK_CHANGE, mo.get("change"), threshold=self._change_threshold(),
                is_degraded=raw.is_degraded,
            )
        if plan.task_type == TaskType.SAR_OPTICAL_FUSION:
            return self.confidence.assess(
                TASK_FUSION, mo.get("sar_fusion"), is_degraded=raw.is_degraded,
            )
        return ConfidenceReport(
            final=0.0, confidence_type=CONF_UNAVAILABLE,
            uncertainty="high", requires_verification=True,
            notes=["no confidence route for task"],
        )

    def _get_pil(self, image_data: Dict):
        """Extract PIL image from image_data dict."""
        pil = image_data.get("pil_image")
        if pil is None:
            from utils.image_utils import normalize_to_rgb
            arr = image_data.get("numpy_array")
            bands = image_data.get("bands", 3)
            if arr is not None:
                pil = normalize_to_rgb(arr, bands)
        return pil

    @staticmethod
    def _image_has_geo_metadata(image_data: Dict) -> bool:
        metadata = image_data.get("metadata", {})
        return bool(
            image_data.get("is_geotiff")
            and metadata.get("crs")
            and metadata.get("transform")
        )

    async def _exec_vqa(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        pil = self._get_pil(images[0])
        
        # Use concurrency-controlled inference
        def inference_func(model, **kwargs):
            return model.answer(kwargs['pil'], kwargs['query'])
        
        out = await self.registry.inference_with_context(
            name="RemoteSensingVQA",
            inference_func=inference_func,
            request_id=None,  # Will be set by middleware
            session_id=None,
            pil=pil,
            query=query
        )
        
        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            model_outputs={"vqa": out},
        )

    async def _exec_land_cover_classification(
        self, plan: ExecutionPlan, images: List[Dict], query: str
    ) -> RawResults:
        if not images:
            return RawResults(
                answer="Land-cover classification requires an image.",
                confidence=0.0,
                is_degraded=True,
            )

        labels = [
            "dense urban built-up area",
            "agricultural land",
            "forest and vegetation",
            "water bodies and wetlands",
        ]
        pil = self._get_pil(images[0])

        def inference_func(model, **kwargs):
            ranked = model.rank_texts_for_image(kwargs["image"], kwargs["labels"])
            return [
                {
                    "label": label,
                    "score": max(0.0, min(1.0, (float(score) + 1.0) / 2.0)),
                }
                for label, score in ranked
            ]

        ranked = await self.registry.inference_with_context(
            name="RSCLIPEncoder",
            inference_func=inference_func,
            request_id=None,
            session_id=None,
            image=pil,
            labels=labels,
        )
        ranked = ranked or []
        top = ranked[0] if ranked else {"label": "unknown land cover", "score": 0.0}
        total = sum(item["score"] for item in ranked) or 1.0
        confidence = top["score"] / total
        alternatives = ", ".join(
            f"{item['label']} ({item['score']:.2f})" for item in ranked[:3]
        )
        answer = (
            f"The image is most consistent with {top['label']} "
            f"(RS-CLIP score: {top['score']:.2f}). "
            f"Top candidate classes: {alternatives}."
        )
        return RawResults(
            answer=answer,
            confidence=float(max(0.0, min(1.0, confidence))),
            model_outputs={
                "land_cover_classification": {
                    "labels": ranked,
                    "candidate_labels": labels,
                }
            },
        )

    async def _exec_captioning(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        pil = self._get_pil(images[0])
        
        # Use concurrency-controlled inference
        def inference_func(model, **kwargs):
            return model.generate_caption(kwargs['pil'])
        
        out = await self.registry.inference_with_context(
            name="RemoteSensingCaptioning",
            inference_func=inference_func,
            request_id=None,
            session_id=None,
            pil=pil
        )
        
        return RawResults(
            answer=out["caption"],
            confidence=out["confidence"],
            model_outputs={"captioning": out},
        )

    async def _exec_grounding(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        pil = self._get_pil(images[0])
        
        # Use concurrency-controlled inference
        def inference_func(model, **kwargs):
            return model.ground(kwargs['pil'], kwargs['query'])
        
        out = await self.registry.inference_with_context(
            name="RemoteSensingGrounding",
            inference_func=inference_func,
            request_id=None,
            session_id=None,
            pil=pil,
            query=query
        )

        boxes = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
             "label": l, "score": s}
            for b, l, s in zip(out["boxes"], out["labels"], out["scores"])
        ]

        count = len(boxes)
        if count > 0:
            # Summarise what was found
            from collections import Counter
            label_counts = Counter(b["label"] for b in boxes)
            parts = [f"{n}x {lbl}" for lbl, n in label_counts.most_common()]
            answer = f"Detected {count} object(s): {', '.join(parts)}. Query: '{query}'"
        else:
            answer = (
                f"No objects matching '{query}' were detected. "
                "Try rephrasing with simpler terms like 'road', 'building', 'water', or 'tree'."
            )

        return RawResults(
            answer=answer,
            confidence=float(max(out["scores"])) if out["scores"] else 0.3,
            visual_evidence_b64=out.get("annotated_image"),
            grounding_boxes=boxes,
            model_outputs={"grounding": out},
        )

    async def _exec_change_vqa(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="Change detection requires two images. Only one was provided.",
                confidence=0.1,
            )
        if not self._change_pair_is_compatible(images):
            return RawResults(
                answer="Change detection requires two images from the same sensor family; use SAR-optical fusion for cross-modal pairs.",
                confidence=0.0,
                is_degraded=True,
            )

        # ── Geospatial alignment ──────────────────────────────────────────────
        images[0], images[1], alignment_info, warnings = self._align_pair(images[0], images[1])

        if alignment_info.get("_incompatible"):
            return RawResults(
                answer=alignment_info["_error"],
                confidence=0.0,
                alignment_info=alignment_info,
                warnings=warnings,
            )

        pil1 = self._get_pil(images[0])
        pil2 = self._get_pil(images[1])
        geo_meta = images[0].get("_geo_meta")

        # Use concurrency-controlled inference for both models
        # First get change model with concurrency control
        def inference_func_change(model, **kwargs):
            return model.answer_change_question(
                kwargs['pil1'], kwargs['pil2'], kwargs['query'], geo_meta=kwargs['geo_meta']
            )
        
        out = await self.registry.inference_with_context(
            name="ChangeDetectionModel",
            inference_func=inference_func_change,
            request_id=None,
            session_id=None,
            pil1=pil1,
            pil2=pil2,
            query=query,
            geo_meta=geo_meta
        )

        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            change_map_b64=out.get("change_map_b64"),
            change_percentage=out.get("change_percentage"),
            change_regions=out.get("changed_regions"),
            alignment_info=alignment_info,
            warnings=warnings,
            model_outputs={"change_vqa": out},
        )

    async def _exec_change_description(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="Change description requires two images.",
                confidence=0.1,
            )
        if not self._change_pair_is_compatible(images):
            return RawResults(
                answer="Change detection requires two images from the same sensor family; use SAR-optical fusion for cross-modal pairs.",
                confidence=0.0,
                is_degraded=True,
            )

        # ── Geospatial alignment ──────────────────────────────────────────────
        images[0], images[1], alignment_info, warnings = self._align_pair(images[0], images[1])

        if alignment_info.get("_incompatible"):
            return RawResults(
                answer=alignment_info["_error"],
                confidence=0.0,
                alignment_info=alignment_info,
                warnings=warnings,
            )

        pil1 = self._get_pil(images[0])
        pil2 = self._get_pil(images[1])
        geo_meta = images[0].get("_geo_meta")

        # Use concurrency-controlled inference for change detection
        def inference_func_change(model, **kwargs):
            return model.detect_changes(kwargs['pil1'], kwargs['pil2'], geo_meta=kwargs['geo_meta'])
        
        change_out = await self.registry.inference_with_context(
            name="ChangeDetectionModel",
            inference_func=inference_func_change,
            request_id=None,
            session_id=None,
            pil1=pil1,
            pil2=pil2,
            geo_meta=geo_meta
        )

        # Use concurrency-controlled inference for captioning
        def inference_func_caption(model, **kwargs):
            return model.generate_caption(kwargs['pil'])
        
        cap1 = await self.registry.inference_with_context(
            name="RemoteSensingCaptioning",
            inference_func=inference_func_caption,
            request_id=None,
            session_id=None,
            pil=pil1
        )
        
        cap2 = await self.registry.inference_with_context(
            name="RemoteSensingCaptioning",
            inference_func=inference_func_caption,
            request_id=None,
            session_id=None,
            pil=pil2
        )

        pct = change_out.get("change_percentage", 0.0)
        answer = (
            f"Image 1: {cap1['caption']}\n\n"
            f"Image 2: {cap2['caption']}\n\n"
            f"Change analysis: Approximately {pct:.1f}% of the scene changed between "
            f"the two images. {self._describe_change_level(pct)}"
        )

        from utils.visualization import create_change_map
        from utils.image_utils import image_to_base64
        import numpy as np
        change_arr = change_out.get("change_map", np.zeros((64, 64)))
        change_pil = create_change_map(change_arr)
        change_b64 = image_to_base64(change_pil)

        return RawResults(
            answer=answer,
            confidence=0.75,
            change_map_b64=change_b64,
            change_percentage=pct,
            change_regions=change_out.get("changed_regions"),
            alignment_info=alignment_info,
            warnings=warnings,
            model_outputs={"change": change_out, "cap1": cap1, "cap2": cap2},
        )

    async def _exec_sar_fusion(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="SAR-optical fusion requires two images (one SAR, one optical).",
                confidence=0.1,
                is_degraded=True,
            )

        # Modality validation: the pair must contain exactly one SAR image and
        # one optical/multispectral image.  SAR-only and optical-only pairs are
        # rejected honestly (surfaced as a degraded result, never a crash).
        optical_mods = {"optical", "multispectral", "rgb"}
        sar_idx = [i for i, im in enumerate(images) if im.get("modality") == "sar"]
        opt_idx = [i for i, im in enumerate(images) if im.get("modality") in optical_mods]

        if not sar_idx:
            return RawResults(
                answer=(
                    "SAR-optical fusion needs one SAR image, but none was provided. "
                    "Both inputs appear to be optical/multispectral."
                ),
                confidence=0.1,
                is_degraded=True,
            )
        if not opt_idx:
            return RawResults(
                answer=(
                    "SAR-optical fusion needs one optical/multispectral image, but "
                    "none was provided. Both inputs appear to be SAR."
                ),
                confidence=0.1,
                is_degraded=True,
            )

        # Pass the full image_data dicts (not PIL): the raw SAR array + metadata
        # enable physically-meaningful VV/VH preprocessing inside the model.
        sar_data, opt_data, alignment_info, warnings = self._align_pair(
            images[sar_idx[0]], images[opt_idx[0]]
        )

        if alignment_info.get("_incompatible"):
            return RawResults(
                answer=alignment_info["_error"],
                confidence=0.0,
                alignment_info=alignment_info,
                warnings=warnings,
                is_degraded=True,
            )

        # Use concurrency-controlled inference
        def inference_func(model, **kwargs):
            return model.fuse_and_analyze(kwargs['opt_data'], kwargs['sar_data'], kwargs['query'])
        
        out = await self.registry.inference_with_context(
            name="SAROpticalFusionModel",
            inference_func=inference_func,
            request_id=None,
            session_id=None,
            opt_data=opt_data,
            sar_data=sar_data,
            query=query
        )

        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            fusion_map_b64=out.get("fusion_map_b64"),
            is_degraded=bool(out.get("requires_verification", False)),
            alignment_info=alignment_info,
            warnings=warnings,
            model_outputs={"sar_fusion": out},
        )

    # ── Alignment helper ──────────────────────────────────────────────────────

    @classmethod
    def _change_pair_is_compatible(cls, images: List[Dict]) -> bool:
        if len(images) < 2:
            return False
        families = [cls._modality_family(image.get("modality", "unknown")) for image in images[:2]]
        return families[0] == families[1] and families[0] in {"sar", "optical"}

    @staticmethod
    def _align_pair(
        image_data_a: Dict,
        image_data_b: Dict,
    ):
        """
        Run the geospatial alignment pipeline on an image pair.

        Returns
        -------
        (image_data_a_updated, image_data_b_updated, alignment_info_dict, warnings_list)

        On incompatible bounds the returned alignment_info_dict will contain
        a special ``_incompatible: True`` key and ``_error`` message so the
        calling handler can return an informative error response immediately.
        """
        from utils.geospatial_aligner import align_image_pair

        result = align_image_pair(image_data_a, image_data_b)
        warnings: List[str] = []

        if not result.success:
            logger.warning("Image pair alignment failed: %s", result.error)
            info = {
                **result.info,
                "_incompatible": True,
                "_error": result.error,
            }
            return image_data_a, image_data_b, info, [result.error]

        # Inject aligned PIL images back into image_data dicts so downstream
        # _get_pil() picks them up transparently.
        a_updated = {**image_data_a, "pil_image": result.pil_a}
        b_updated = {**image_data_b, "pil_image": result.pil_b}

        # Fusion consumes numeric SAR channels, so preserve the aligned arrays
        # instead of silently handing the model the pre-alignment raster.
        if result.array_a is not None:
            a_updated["numpy_array"] = result.array_a
            a_updated["shape"] = list(result.array_a.shape)
            a_updated["bands"] = result.array_a.shape[2] if result.array_a.ndim == 3 else 1
        if result.array_b is not None:
            b_updated["numpy_array"] = result.array_b
            b_updated["shape"] = list(result.array_b.shape)
            b_updated["bands"] = result.array_b.shape[2] if result.array_b.ndim == 3 else 1

        # Attach geo-metadata so change_model can derive geographic coordinates.
        a_updated["_geo_meta"] = result.meta_a
        b_updated["_geo_meta"] = result.meta_b

        if result.info.get("note"):
            warnings.append(result.info["note"])

        logger.info(
            "Alignment complete: method=%s, geo=%s",
            result.info.get("method", "none"),
            result.info.get("geographic_coordinates_available", False),
        )
        return a_updated, b_updated, result.info, warnings

    # ── Integrate ─────────────────────────────────────────────────────────────

    def _integrate(
        self,
        raw: RawResults,
        plan: ExecutionPlan,
        session_id: str,
        elapsed_ms: float,
    ) -> AnalysisResponse:
        grounding_boxes = None
        if raw.grounding_boxes:
            grounding_boxes = [
                BoundingBox(
                    x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
                    label=b["label"], score=b["score"],
                )
                for b in raw.grounding_boxes
            ]

        # Build AlignmentInfo schema object from raw dict (if present)
        alignment_schema: Optional[AlignmentInfo] = None
        if raw.alignment_info:
            # Strip internal sentinel keys before constructing schema
            clean = {
                k: v for k, v in raw.alignment_info.items()
                if not k.startswith("_")
            }
            try:
                alignment_schema = AlignmentInfo(**clean)
            except Exception as exc:
                logger.warning("Could not build AlignmentInfo schema: %s", exc)

        # Build ChangeRegion list from raw change regions (if present)
        change_regions_schema = None
        if raw.change_regions:
            change_regions_schema = []
            for r in raw.change_regions:
                geo_raw = r.get("geo_bbox")
                geo_schema = GeoBBox(**geo_raw) if geo_raw else None

                # Compute absolute pixel_bbox if alignment grid is known
                pixel_bbox = None
                if alignment_schema and alignment_schema.output_grid:
                    out_w = alignment_schema.output_grid.get("width", 0)
                    out_h = alignment_schema.output_grid.get("height", 0)
                    if out_w and out_h:
                        pixel_bbox = [
                            round(r["x1"] * out_w),
                            round(r["y1"] * out_h),
                            round(r["x2"] * out_w),
                            round(r["y2"] * out_h),
                        ]

                change_regions_schema.append(
                    ChangeRegion(
                        x1=r["x1"], y1=r["y1"], x2=r["x2"], y2=r["y2"],
                        area_pct=r.get("area_pct", 0.0),
                        pixel_bbox=pixel_bbox,
                        geo_bbox=geo_schema,
                    )
                )

        summary = ExecutionSummary(
            selected_task=plan.task_type.value,
            task_confidence=plan.task_confidence,
            models_used=plan.model_names,
            parameters=plan.parameters,
            processing_time_ms=round(elapsed_ms, 2),
            steps=plan.steps,
            alignment=alignment_schema,
            warnings=raw.warnings,
            plan=plan.to_dict(),
            intent=plan.intent.to_dict() if plan.intent else None,
            step_trace=raw.execution_trace or None,
        )

        # ── Confidence: driven by the task-aware report, never answer length ──
        report = raw.confidence_report
        if report is None:
            report = ConfidenceReport(
                final=0.0, confidence_type=CONF_UNAVAILABLE,
                uncertainty="high", requires_verification=True,
                notes=["no confidence report produced"],
            )
        conf = report.to_public_dict()

        return AnalysisResponse(
            session_id=session_id,
            task=plan.task_type.value,
            answer=raw.answer or "No answer generated.",
            confidence=conf["confidence"],
            confidence_type=conf["confidence_type"],
            confidence_components=conf["confidence_components"],
            uncertainty=conf["uncertainty"],
            requires_verification=conf["requires_verification"],
            visual_evidence=raw.visual_evidence_b64,
            change_map=raw.change_map_b64,
            fusion_map=raw.fusion_map_b64,
            grounding_boxes=grounding_boxes,
            change_percentage=raw.change_percentage,
            change_regions=change_regions_schema,
            tool_evidence=raw.tool_evidence,
            execution_summary=summary,
            is_degraded=raw.is_degraded,
            is_georeferenced=raw.is_georeferenced,
        )

    @staticmethod
    def _describe_change_level(pct: float) -> str:
        if pct < 2:
            return "The scene appears largely unchanged."
        elif pct < 10:
            return "Minor changes are present, possibly seasonal or noise-related."
        elif pct < 30:
            return "Moderate changes detected, likely new construction or land-use shifts."
        elif pct < 60:
            return "Significant changes detected, indicating major land-cover transformation."
        else:
            return "Extensive changes across the scene — possible large-scale disturbance."
