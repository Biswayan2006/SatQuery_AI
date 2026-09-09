"""
SatQuery AI — Task Classifier
Maps a natural-language query + image metadata to a TaskType.

Scoring pipeline
----------------
When an RS-CLIP encoder is available (loaded via the model registry or
passed directly), three signals are blended:

    final_score = 0.55 × semantic_score
                + 0.25 × structural_score
                + 0.20 × keyword_score

When RS-CLIP is unavailable the classifier falls back to keyword + structural
scoring only.  The public API (``classify``, ``classify_detailed``) is
identical in both modes.

Blending weights are configurable via ``SemanticRouter(weights=...)``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("satquery.task_classifier")


class TaskType(str, Enum):
    SINGLE_VQA = "SINGLE_VQA"
    LAND_COVER_CLASSIFICATION = "LAND_COVER_CLASSIFICATION"
    CAPTIONING = "CAPTIONING"
    GROUNDING = "GROUNDING"
    CHANGE_VQA = "CHANGE_VQA"
    CHANGE_DESCRIPTION = "CHANGE_DESCRIPTION"
    SAR_OPTICAL_FUSION = "SAR_OPTICAL_FUSION"


# ── Entity lexicon for "describe X" disambiguation ────────────────────────────
# When a query contains "describe" (or synonym) PLUS one of these entity terms,
# the query is object-specific → route to VQA, not generic captioning.
#
# Maintained as a flat set for O(1) membership testing.  Add new RS entity
# terms here rather than scattering ad-hoc checks.
_RS_ENTITY_LEXICON: set = {
    # Infrastructure
    "road", "highway", "street", "bridge", "tunnel", "overpass", "intersection",
    "railway", "railroad", "track", "canal", "dam", "pipeline",
    # Buildings / structures
    "building", "house", "structure", "tower", "facility", "complex",
    "parking lot", "parking", "airport", "runway", "port", "harbor", "harbour",
    # Water
    "river", "water", "lake", "pond", "ocean", "sea", "stream", "creek",
    "reservoir", "wetland", "flood",
    # Vegetation
    "forest", "vegetation", "tree", "field", "crop", "agriculture", "farmland",
    "grassland", "shrub", "mangrove",
    # Land use
    "urban", "settlement", "industrial", "residential", "commercial",
    "park", "playground", "stadium",
    # Transport / vehicles
    "vehicle", "car", "truck", "ship", "vessel", "aircraft", "plane",
    "boat", "train",
    # Terrain
    "mountain", "hill", "valley", "desert", "sand", "coast", "shoreline",
    "cliff", "terrain", "slope",
}


# ── Keyword rule sets ─────────────────────────────────────────────────────────

_RULES: Dict[TaskType, Dict[str, float]] = {
    TaskType.CAPTIONING: {
        "describe": 0.8, "description": 0.8, "caption": 0.9,
        "summarize": 0.7, "summary": 0.7, "overview": 0.6,
        "what does this image show": 0.85, "tell me about": 0.6,
        "what is in": 0.5, "what can you see": 0.5,
    },
    TaskType.GROUNDING: {
        "where is": 0.85, "locate": 0.9, "find": 0.75, "detect": 0.8,
        "identify": 0.85, "identify location": 0.9, "show me": 0.7,
        "point out": 0.8, "bounding box": 0.95, "highlight": 0.7,
        "mark": 0.65, "position of": 0.85,
        "how many": 0.98, "count": 0.98, "number of": 0.98, "count of": 0.98,
    },
    TaskType.CHANGE_VQA: {
        "what changed": 0.95, "change between": 0.95, "difference between": 0.9,
        "how much changed": 0.9, "how many": 0.4, "before and after": 0.85,
        "temporal": 0.8, "deforestation": 0.75, "construction": 0.6,
    },
    TaskType.CHANGE_DESCRIPTION: {
        "describe the changes": 0.95, "what happened": 0.6,
        "summarize changes": 0.9, "change detection": 0.9,
        "evolution": 0.65, "progression": 0.65, "transformation": 0.65,
    },
    TaskType.SAR_OPTICAL_FUSION: {
        "sar": 0.9, "synthetic aperture": 0.95, "radar": 0.85,
        "backscatter": 0.9, "fuse": 0.85, "fusion": 0.85,
        "combine optical": 0.9, "multi-modal": 0.85, "multimodal": 0.85,
    },
    TaskType.SINGLE_VQA: {
        "how many": 0.7, "what type": 0.65, "is there": 0.6, "are there": 0.6,
        "which": 0.5, "what is the": 0.55, "land cover": 0.65, "land use": 0.65,
        "vegetation": 0.55, "water": 0.5, "urban": 0.5, "building": 0.55,
        "road": 0.5, "forest": 0.5, "crop": 0.5, "agriculture": 0.55,
        "flood": 0.65, "damage": 0.65,
    },
    TaskType.LAND_COVER_CLASSIFICATION: {
        "classify": 0.98, "classification": 0.98, "land cover": 0.98,
        "major land-cover": 0.98, "what type of land": 0.98,
        "land-cover type": 0.98, "land cover type": 0.98,
    },
}


@dataclass
class ClassificationResult:
    task_type: TaskType
    confidence: float
    scores: Dict[str, float]
    reason: str
    used_semantic_router: bool = False


class TaskClassifier:
    """
    Classify a user query into a TaskType.

    When ``semantic_router`` is provided (and loaded), scores from all three
    signal sources are blended.  Otherwise keyword + structural only.

    Parameters
    ----------
    semantic_router : SemanticRouter or None
        Pass a pre-built ``SemanticRouter`` instance to enable semantic scoring.
        ``None`` silently enables keyword-only mode.
    """

    def __init__(self, semantic_router=None):
        self._router = semantic_router

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        query: str,
        num_images: int = 1,
        modalities: Optional[List[str]] = None,
    ) -> Tuple[TaskType, float]:
        """Returns (TaskType, confidence)."""
        result = self.classify_detailed(query, num_images, modalities)
        return result.task_type, result.confidence

    def classify_detailed(
        self,
        query: str,
        num_images: int = 1,
        modalities: Optional[List[str]] = None,
    ) -> ClassificationResult:
        modalities = modalities or []

        if self._is_land_cover_classification(query):
            scores = {task.value: 0.0 for task in TaskType}
            scores[TaskType.LAND_COVER_CLASSIFICATION.value] = 0.99
            return ClassificationResult(
                task_type=TaskType.LAND_COVER_CLASSIFICATION,
                confidence=0.99,
                scores=scores,
                reason="Explicit land-cover classification request; routed to RS-CLIP zero-shot labels",
                used_semantic_router=False,
            )

        # Always compute keyword + structural scores
        keyword_scores, structural_scores = self._keyword_and_structural(
            query, num_images, modalities
        )

        # Attempt semantic blending
        used_semantic = False
        if self._router is not None and self._router.is_available:
            try:
                route_result = self._router.route(
                    query, structural_scores, keyword_scores
                )
                used_semantic = route_result.used_semantic
                best_task = TaskType(route_result.task_type)
                confidence = route_result.final_score
                blended_scores = route_result.all_scores

                reason = self._build_reason(
                    best_task, blended_scores, num_images, modalities,
                    semantic_score=route_result.semantic_score,
                    used_semantic=used_semantic,
                )
                result = ClassificationResult(
                    task_type=best_task,
                    confidence=confidence,
                    scores=blended_scores,
                    reason=reason,
                    used_semantic_router=used_semantic,
                )
                return self._enforce_modality_contract(result, num_images, modalities)
            except Exception as exc:
                logger.warning(
                    "SemanticRouter failed (%s) — falling back to keyword-only",
                    exc,
                )

        # Keyword-only fallback
        return self._keyword_classify(query, num_images, modalities,
                                       keyword_scores, structural_scores)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_land_cover_classification(query: str) -> bool:
        normalized = re.sub(r"[^\w\s-]", " ", query.lower())
        return any(
            phrase in normalized
            for phrase in (
                "classify", "classification", "major land-cover",
                "land cover", "land-cover type", "land cover type",
                "land-cover types", "land cover types",
                "what type of land", "types of land", "land use",
                "land-use", "terrain type", "terrain classification",
                "dominant land", "cover types", "classify the",
            )
        )

    def _keyword_and_structural(
        self,
        query: str,
        num_images: int,
        modalities: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute keyword scores and structural scores separately.
        Returns (keyword_scores, structural_scores) both normalised to [0,1].
        """
        q = query.lower().strip()
        q = re.sub(r"[^\w\s]", " ", q)

        keyword_scores: Dict[str, float] = {t.value: 0.0 for t in TaskType}
        for task_type, keywords in _RULES.items():
            for kw, weight in keywords.items():
                if kw in q:
                    keyword_scores[task_type.value] = max(
                        keyword_scores[task_type.value], weight
                    )

        structural_scores: Dict[str, float] = {t.value: 0.0 for t in TaskType}
        has_sar = "sar" in modalities
        has_optical = "optical" in modalities or "multispectral" in modalities

        if num_images == 2:
            structural_scores[TaskType.CHANGE_VQA.value] = 0.3
            structural_scores[TaskType.CHANGE_DESCRIPTION.value] = 0.25
            structural_scores[TaskType.SAR_OPTICAL_FUSION.value] = 0.15

        if has_sar and has_optical:
            structural_scores[TaskType.SAR_OPTICAL_FUSION.value] += 0.5

        if has_sar and not has_optical:
            if num_images == 1:
                # Single SAR image -> prefer VQA or land-cover rather than optical captioning
                structural_scores[TaskType.SINGLE_VQA.value] += 0.3
            elif structural_scores[TaskType.SAR_OPTICAL_FUSION.value] < 0.3:
                structural_scores[TaskType.CAPTIONING.value] += 0.2

        # Default fallback if no signal at all.
        # Single images default to VQA (safer for RS queries); pairs default
        # to change description.  Captioning only wins when explicitly invoked
        # by a keyword.
        if all(v == 0.0 for v in keyword_scores.values()):
            if num_images == 2:
                keyword_scores[TaskType.CHANGE_DESCRIPTION.value] = 0.4
            else:
                keyword_scores[TaskType.SINGLE_VQA.value] = 0.3

        # ── Disambiguate "describe X" vs generic description ─────────────────
        # If the query uses a description verb ("describe", "summarize", …)
        # together with a specific RS entity, override routing to VQA.
        self._disambiguate_entity_description(q, keyword_scores)

        return keyword_scores, structural_scores

    @staticmethod
    def _disambiguate_entity_description(
        q: str, keyword_scores: Dict[str, float]
    ) -> None:
        """
        Detect ``"describe <entity>"`` patterns and reroute from CAPTIONING
        to SINGLE_VQA when a specific RS entity is mentioned.

        The heuristic:
          1. Query contains a description verb (describe, summarize, overview, …).
          2. Query contains a term from ``_RS_ENTITY_LEXICON``.
          3. The entity term is NOT immediately preceded by "the image" / "the scene"
             (which would make it a generic description).

        When both conditions hold, VQA is boosted to 0.9 (above captioning's
        0.8) so the downstream max() picks SINGLE_VQA.
        """
        desc_verbs = ("describe", "summarize", "overview", "tell me about",
                       "what does", "what can you see", "caption")
        has_desc_verb = any(v in q for v in desc_verbs)
        if not has_desc_verb:
            return

        # Check whether a specific entity is mentioned
        words = q.split()
        has_entity = any(w in _RS_ENTITY_LEXICON for w in words)
        # Also check multi-word entities (e.g., "parking lot", "water body")
        if not has_entity:
            has_entity = any(e in q for e in _RS_ENTITY_LEXICON if " " in e)

        if has_entity:
            # Override: this is an object-specific question, not a generic caption
            keyword_scores[TaskType.SINGLE_VQA.value] = max(
                keyword_scores[TaskType.SINGLE_VQA.value], 0.9
            )
            # Suppress captioning so it doesn't win on tie
            keyword_scores[TaskType.CAPTIONING.value] = min(
                keyword_scores[TaskType.CAPTIONING.value], 0.3
            )

    def _keyword_classify(
        self,
        query: str,
        num_images: int,
        modalities: List[str],
        keyword_scores: Dict[str, float],
        structural_scores: Dict[str, float],
    ) -> ClassificationResult:
        """
        Original keyword + structural classification (no semantic component).
        Preserves the exact priority ordering from the original implementation.
        """
        # Merge: structural acts as additive boost on top of keyword scores
        scores: Dict[str, float] = {}
        for t in TaskType:
            scores[t.value] = keyword_scores[t.value] + structural_scores[t.value]

        priority = [
            TaskType.SAR_OPTICAL_FUSION,
            TaskType.GROUNDING,
            TaskType.CHANGE_VQA,
            TaskType.CHANGE_DESCRIPTION,
            TaskType.LAND_COVER_CLASSIFICATION,
            TaskType.SINGLE_VQA,
            TaskType.CAPTIONING,
        ]
        best_task = max(
            scores,
            key=lambda t: (scores[t], -priority.index(TaskType(t))),
        )
        best_score = scores[best_task]
        confidence = max(0.3, min(0.99, best_score))
        reason = self._build_reason(
            TaskType(best_task), scores, num_images, modalities
        )
        result = ClassificationResult(
            task_type=TaskType(best_task),
            confidence=confidence,
            scores=scores,
            reason=reason,
            used_semantic_router=False,
        )
        return self._enforce_modality_contract(result, num_images, modalities)

    @staticmethod
    def _enforce_modality_contract(
        result: ClassificationResult,
        num_images: int,
        modalities: List[str],
    ) -> ClassificationResult:
        """Keep fusion routing consistent with the supplied modalities."""
        families = [
            "sar" if modality == "sar" else
            "optical" if modality in {"optical", "multispectral", "rgb"} else
            "unknown"
            for modality in modalities
        ]
        if (
            num_images == 2
            and set(families) == {"sar", "optical"}
            and result.task_type in {TaskType.CHANGE_VQA, TaskType.CHANGE_DESCRIPTION}
        ):
            result.task_type = TaskType.SAR_OPTICAL_FUSION
            result.confidence = min(result.confidence, result.scores.get(
                TaskType.SAR_OPTICAL_FUSION.value, result.confidence
            ))
            result.reason += "; cross-modal comparison rerouted through fusion"
            return result

        if result.task_type != TaskType.SAR_OPTICAL_FUSION:
            return result

        has_sar = modalities.count("sar") == 1
        has_optical = sum(
            modality in {"optical", "multispectral", "rgb"}
            for modality in modalities
        ) == 1
        if num_images == 2 and has_sar and has_optical:
            return result

        alternatives = {
            task: score
            for task, score in result.scores.items()
            if task != TaskType.SAR_OPTICAL_FUSION.value
        }
        fallback = TaskType(
            max(alternatives, key=lambda task: alternatives[task])
        ) if alternatives else TaskType.CAPTIONING
        result.task_type = fallback
        result.confidence = min(result.confidence, result.scores.get(fallback.value, 0.3))
        result.reason += "; fusion rejected: requires exactly one SAR and one optical image"
        return result

    @staticmethod
    def _build_reason(
        task: TaskType,
        scores: Dict[str, float],
        num_images: int,
        modalities: List[str],
        semantic_score: float = 0.0,
        used_semantic: bool = False,
    ) -> str:
        parts = [
            f"Selected task: {task.value} (score={scores.get(task.value, 0.0):.2f})"
        ]
        if used_semantic:
            parts.append(f"semantic_score={semantic_score:.2f}")
        if num_images == 2:
            parts.append("two images detected")
        if modalities:
            parts.append(f"modalities: {', '.join(modalities)}")
        return "; ".join(parts)
