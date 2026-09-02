"""
SatQuery AI — Task Classifier
Maps a natural-language query + image metadata to a TaskType.
Uses keyword heuristics with confidence scoring (no GPU required).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class TaskType(str, Enum):
    SINGLE_VQA = "SINGLE_VQA"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    CHANGE_VQA = "CHANGE_VQA"
    CHANGE_DESCRIPTION = "CHANGE_DESCRIPTION"
    SAR_OPTICAL_FUSION = "SAR_OPTICAL_FUSION"


# ── Keyword rule sets ─────────────────────────────────────────────────────────

_RULES: Dict[TaskType, Dict[str, float]] = {
    TaskType.CAPTIONING: {
        "describe": 0.8,
        "description": 0.8,
        "caption": 0.9,
        "summarize": 0.7,
        "summary": 0.7,
        "overview": 0.6,
        "what does this image show": 0.85,
        "tell me about": 0.6,
        "what is in": 0.5,
        "what can you see": 0.5,
    },
    TaskType.GROUNDING: {
        "where is": 0.85,
        "locate": 0.9,
        "find": 0.75,
        "detect": 0.8,
        "identify location": 0.9,
        "show me": 0.7,
        "point out": 0.8,
        "bounding box": 0.95,
        "highlight": 0.7,
        "mark": 0.65,
        "position of": 0.85,
    },
    TaskType.CHANGE_VQA: {
        "what changed": 0.95,
        "change between": 0.95,
        "difference between": 0.9,
        "how much changed": 0.9,
        "how many": 0.4,
        "before and after": 0.85,
        "temporal": 0.8,
        "deforestation": 0.75,
        "construction": 0.6,
    },
    TaskType.CHANGE_DESCRIPTION: {
        "describe the changes": 0.95,
        "what happened": 0.6,
        "summarize changes": 0.9,
        "change detection": 0.9,
        "evolution": 0.65,
        "progression": 0.65,
        "transformation": 0.65,
    },
    TaskType.SAR_OPTICAL_FUSION: {
        "sar": 0.9,
        "synthetic aperture": 0.95,
        "radar": 0.85,
        "backscatter": 0.9,
        "fuse": 0.85,
        "fusion": 0.85,
        "combine optical": 0.9,
        "multi-modal": 0.85,
        "multimodal": 0.85,
    },
    TaskType.SINGLE_VQA: {
        "how many": 0.7,
        "what type": 0.65,
        "is there": 0.6,
        "are there": 0.6,
        "which": 0.5,
        "what is the": 0.55,
        "land cover": 0.65,
        "land use": 0.65,
        "vegetation": 0.55,
        "water": 0.5,
        "urban": 0.5,
        "building": 0.55,
        "road": 0.5,
        "forest": 0.5,
        "crop": 0.5,
        "agriculture": 0.55,
        "flood": 0.65,
        "damage": 0.65,
    },
}


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    scores: Dict[str, float]
    reason: str


class TaskClassifier:
    """
    Classify a user query into a TaskType using keyword heuristics.

    Priority ordering (from most to least specific):
      SAR_OPTICAL_FUSION > GROUNDING > CHANGE_VQA/CHANGE_DESCRIPTION >
      CAPTIONING > SINGLE_VQA
    """

    def classify(
        self,
        query: str,
        num_images: int = 1,
        modalities: List[str] | None = None,
    ) -> Tuple[TaskType, float]:
        """
        Returns (TaskType, confidence).

        Modality and image-count signals can override keyword scores:
        - 2 images → change task boosted
        - SAR + optical in modalities → SAR_OPTICAL_FUSION boosted
        """
        modalities = modalities or []
        result = self._score(query, num_images, modalities)
        return result.task_type, result.confidence

    def classify_detailed(
        self,
        query: str,
        num_images: int = 1,
        modalities: List[str] | None = None,
    ) -> ClassificationResult:
        modalities = modalities or []
        return self._score(query, num_images, modalities)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score(
        self, query: str, num_images: int, modalities: List[str]
    ) -> ClassificationResult:
        q = query.lower().strip()
        q = re.sub(r"[^\w\s]", " ", q)

        scores: Dict[str, float] = {t.value: 0.0 for t in TaskType}

        for task_type, keywords in _RULES.items():
            for kw, weight in keywords.items():
                if kw in q:
                    scores[task_type.value] = max(scores[task_type.value], weight)

        # ── Structural signals ────────────────────────────────────────────────
        # Two images → likely a change or fusion task
        if num_images == 2:
            scores[TaskType.CHANGE_VQA.value] += 0.3
            scores[TaskType.CHANGE_DESCRIPTION.value] += 0.25
            scores[TaskType.SAR_OPTICAL_FUSION.value] += 0.15

        # SAR + optical pair → fusion
        has_sar = "sar" in modalities
        has_optical = "optical" in modalities or "multispectral" in modalities
        if has_sar and has_optical:
            scores[TaskType.SAR_OPTICAL_FUSION.value] += 0.5

        # Only SAR image but no explicit SAR query → treat as captioning
        if has_sar and not has_optical and scores[TaskType.SAR_OPTICAL_FUSION.value] < 0.3:
            scores[TaskType.CAPTIONING.value] += 0.2

        # If no keyword matched at all → default to captioning for 1 image,
        # change description for 2 images
        if all(v == 0.0 for v in scores.values()):
            if num_images == 2:
                scores[TaskType.CHANGE_DESCRIPTION.value] = 0.4
            else:
                scores[TaskType.CAPTIONING.value] = 0.4

        # ── Priority tiebreak (prefer more specific tasks) ────────────────────
        priority = [
            TaskType.SAR_OPTICAL_FUSION,
            TaskType.GROUNDING,
            TaskType.CHANGE_VQA,
            TaskType.CHANGE_DESCRIPTION,
            TaskType.CAPTIONING,
            TaskType.SINGLE_VQA,
        ]

        best_task = max(scores, key=lambda t: (scores[t], -priority.index(TaskType(t))))
        best_score = scores[best_task]

        # Clamp confidence to [0.3, 0.99]
        confidence = max(0.3, min(0.99, best_score))

        reason = self._build_reason(TaskType(best_task), scores, num_images, modalities)

        return ClassificationResult(
            task_type=TaskType(best_task),
            confidence=confidence,
            scores=scores,
            reason=reason,
        )

    @staticmethod
    def _build_reason(
        task: TaskType,
        scores: Dict[str, float],
        num_images: int,
        modalities: List[str],
    ) -> str:
        parts = [f"Selected task: {task.value} (score={scores[task.value]:.2f})"]
        if num_images == 2:
            parts.append("two images detected")
        if modalities:
            parts.append(f"modalities: {', '.join(modalities)}")
        return "; ".join(parts)
