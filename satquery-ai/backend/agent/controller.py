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

from agent.task_classifier import TaskClassifier, TaskType
from api.schemas import AnalysisResponse, BoundingBox, ExecutionSummary

logger = logging.getLogger("satquery.controller")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ExecutionPlan:
    task_type: TaskType
    task_confidence: float
    model_names: List[str]
    steps: List[str]
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawResults:
    answer: str = ""
    confidence: float = 0.0
    visual_evidence_b64: Optional[str] = None
    change_map_b64: Optional[str] = None
    fusion_map_b64: Optional[str] = None
    grounding_boxes: Optional[List[Dict]] = None
    change_percentage: Optional[float] = None
    model_outputs: Dict[str, Any] = field(default_factory=dict)


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
        self.classifier = TaskClassifier()

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

        logger.info("Task classified: %s (%.2f) | query='%s'", task_type, task_confidence, query[:80])

        # ── 2. Plan ───────────────────────────────────────────────────────────
        plan = self._plan(task_type, task_confidence, images, query)

        # ── 3. Execute ────────────────────────────────────────────────────────
        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._execute, plan, images, query
        )

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
    ) -> ExecutionPlan:
        plans = {
            TaskType.SINGLE_VQA: ExecutionPlan(
                task_type=task_type,
                task_confidence=task_confidence,
                model_names=["RemoteSensingVQA"],
                steps=["validate_input", "preprocess_image", "run_vqa", "format_answer"],
                parameters={"max_new_tokens": 150, "num_beams": 4},
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

        return plans.get(task_type, plans[TaskType.CAPTIONING])

    # ── Execute ───────────────────────────────────────────────────────────────

    def _execute(
        self, plan: ExecutionPlan, images: List[Dict], query: str
    ) -> RawResults:
        """Dispatch execution to the appropriate handler."""
        handlers = {
            TaskType.SINGLE_VQA: self._exec_vqa,
            TaskType.CAPTIONING: self._exec_captioning,
            TaskType.GROUNDING: self._exec_grounding,
            TaskType.CHANGE_VQA: self._exec_change_vqa,
            TaskType.CHANGE_DESCRIPTION: self._exec_change_description,
            TaskType.SAR_OPTICAL_FUSION: self._exec_sar_fusion,
        }
        handler = handlers.get(plan.task_type, self._exec_captioning)
        return handler(plan, images, query)

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

    def _exec_vqa(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        model = self.registry.get("RemoteSensingVQA")
        pil = self._get_pil(images[0])
        out = model.answer(pil, query)
        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            model_outputs={"vqa": out},
        )

    def _exec_captioning(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        model = self.registry.get("RemoteSensingCaptioning")
        pil = self._get_pil(images[0])
        out = model.generate_caption(pil)
        return RawResults(
            answer=out["caption"],
            confidence=out["confidence"],
            model_outputs={"captioning": out},
        )

    def _exec_grounding(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        model = self.registry.get("RemoteSensingGrounding")
        pil = self._get_pil(images[0])
        out = model.ground(pil, query)

        boxes = [
            {"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3],
             "label": l, "score": s}
            for b, l, s in zip(out["boxes"], out["labels"], out["scores"])
        ]

        count = len(boxes)
        answer = (
            f"Found {count} instance(s) of '{query}' in the image."
            if count > 0
            else f"No instances of '{query}' detected in the image."
        )

        return RawResults(
            answer=answer,
            confidence=float(max(out["scores"])) if out["scores"] else 0.3,
            visual_evidence_b64=out.get("annotated_image"),
            grounding_boxes=boxes,
            model_outputs={"grounding": out},
        )

    def _exec_change_vqa(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="Change detection requires two images. Only one was provided.",
                confidence=0.1,
            )
        change_model = self.registry.get("ChangeDetectionModel")
        pil1 = self._get_pil(images[0])
        pil2 = self._get_pil(images[1])
        out = change_model.answer_change_question(pil1, pil2, query)
        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            change_map_b64=out.get("change_map_b64"),
            change_percentage=out.get("change_percentage"),
            model_outputs={"change_vqa": out},
        )

    def _exec_change_description(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="Change description requires two images.",
                confidence=0.1,
            )
        change_model = self.registry.get("ChangeDetectionModel")
        cap_model = self.registry.get("RemoteSensingCaptioning")
        pil1 = self._get_pil(images[0])
        pil2 = self._get_pil(images[1])

        change_out = change_model.detect_changes(pil1, pil2)
        cap1 = cap_model.generate_caption(pil1)
        cap2 = cap_model.generate_caption(pil2)

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
            model_outputs={"change": change_out, "cap1": cap1, "cap2": cap2},
        )

    def _exec_sar_fusion(self, plan: ExecutionPlan, images: List[Dict], query: str) -> RawResults:
        if len(images) < 2:
            return RawResults(
                answer="SAR-optical fusion requires two images (one SAR, one optical).",
                confidence=0.1,
            )
        model = self.registry.get("SAROpticalFusionModel")
        # Determine which is SAR vs optical
        mod0 = images[0].get("modality", "unknown")
        mod1 = images[1].get("modality", "unknown")

        if mod0 == "sar":
            sar_img, opt_img = self._get_pil(images[0]), self._get_pil(images[1])
        else:
            sar_img, opt_img = self._get_pil(images[1]), self._get_pil(images[0])

        out = model.fuse_and_analyze(opt_img, sar_img, query)
        return RawResults(
            answer=out["answer"],
            confidence=out["confidence"],
            fusion_map_b64=out.get("fusion_map_b64"),
            model_outputs={"sar_fusion": out},
        )

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

        summary = ExecutionSummary(
            selected_task=plan.task_type.value,
            task_confidence=plan.task_confidence,
            models_used=plan.model_names,
            parameters=plan.parameters,
            processing_time_ms=round(elapsed_ms, 2),
            steps=plan.steps,
        )

        return AnalysisResponse(
            session_id=session_id,
            task=plan.task_type.value,
            answer=raw.answer or "No answer generated.",
            confidence=round(max(0.0, min(1.0, raw.confidence)), 4),
            visual_evidence=raw.visual_evidence_b64,
            change_map=raw.change_map_b64,
            fusion_map=raw.fusion_map_b64,
            grounding_boxes=grounding_boxes,
            change_percentage=raw.change_percentage,
            execution_summary=summary,
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
