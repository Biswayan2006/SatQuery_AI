"""
SatQuery AI — Semantic Task Router
====================================
Combines three scoring signals to classify a user query into a TaskType:

  1. Semantic score   (RS-CLIP query ↔ task description embeddings)  — weight 0.55
  2. Structural score (image count, modality signals)                — weight 0.25
  3. Keyword score    (existing keyword heuristics)                  — weight 0.20

Weights are configurable.

Fallback behaviour
------------------
If the RS-CLIP encoder is unavailable (not loaded, import error, OOM):
  - SemanticRouter.classify() transparently falls back to the existing
    keyword-only TaskClassifier — NO exception propagates to the API.

Integration with task_classifier.py
------------------------------------
``TaskClassifier`` instantiates a ``SemanticRouter`` internally.
If the router is available it blends scores; if not it uses its own
keyword scoring unchanged.  The public API of ``TaskClassifier`` does
NOT change.

Thread safety
-------------
``SemanticRouter`` caches task description embeddings on first call
(lazy initialisation).  The cache is populated once under a lock and
then read-only for all subsequent calls — safe for concurrent requests.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("satquery.semantic_router")


# ── Task description templates ────────────────────────────────────────────────
# Each TaskType gets a list of natural-language descriptions.
# The semantic score for a query is its max cosine similarity to these.

TASK_DESCRIPTIONS: Dict[str, List[str]] = {
    "SINGLE_VQA": [
        "answer a question about a satellite image",
        "visual question answering on a remote sensing image",
        "what is the land cover type in this image",
        "describe the content of this satellite photograph",
        "how many buildings are in this aerial image",
        "identify features in this satellite scene",
    ],
    "CAPTIONING": [
        "describe this satellite image",
        "generate a caption for this remote sensing scene",
        "provide a summary of what is shown in this aerial image",
        "explain what this satellite image shows",
        "give a scene description of this satellite photograph",
    ],
    "GROUNDING": [
        "locate objects in this satellite image",
        "find and draw bounding boxes around features",
        "detect the position of buildings in the aerial image",
        "show me where the roads are in this satellite image",
        "identify the location of water bodies",
        "highlight the urban areas in this image",
    ],
    "CHANGE_VQA": [
        "what changed between these two satellite images",
        "answer questions about changes over time in satellite imagery",
        "detect differences between two remote sensing images",
        "what is different in the before and after satellite images",
        "how much deforestation occurred between these images",
    ],
    "CHANGE_DESCRIPTION": [
        "describe the changes between two satellite images",
        "explain what happened in this area over time",
        "summarize the differences between two temporal satellite images",
        "provide a change analysis of these two remote sensing scenes",
    ],
    "SAR_OPTICAL_FUSION": [
        "analyse SAR and optical satellite imagery together",
        "fuse synthetic aperture radar and optical images",
        "combine radar backscatter with optical reflectance for analysis",
        "interpret multimodal remote sensing data with SAR and optical",
        "use both SAR and multispectral imagery for scene understanding",
    ],
}

# Default blending weights
DEFAULT_WEIGHTS: Dict[str, float] = {
    "semantic": 0.55,
    "structural": 0.25,
    "keyword": 0.20,
}


@dataclass
class SemanticRouterResult:
    task_type: str
    final_score: float
    semantic_score: float
    structural_score: float
    keyword_score: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    used_semantic: bool = True


class SemanticRouter:
    """
    CLIP-based semantic task router with configurable score blending.

    Parameters
    ----------
    encoder : RSCLIPEncoder or None
        If None, semantic scoring is disabled and only keyword + structural
        scores are used.
    weights : dict, optional
        Override default blending weights. Keys: "semantic", "structural", "keyword".
    """

    def __init__(
        self,
        encoder=None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._encoder = encoder
        self._weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self._task_embs: Optional[Dict[str, np.ndarray]] = None
        self._emb_lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        """True if the semantic encoder is loaded and usable."""
        return self._encoder is not None

    # ── Embedding cache ────────────────────────────────────────────────────────

    def _get_task_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Build and cache mean text embeddings for each task type.
        Returns None if encoder is unavailable.
        Thread-safe lazy init.
        """
        if self._encoder is None:
            return None

        if self._task_embs is not None:
            return self._task_embs

        with self._emb_lock:
            if self._task_embs is not None:
                return self._task_embs
            try:
                embs: Dict[str, np.ndarray] = {}
                for task, descs in TASK_DESCRIPTIONS.items():
                    feat = self._encoder.encode_text(descs)   # [N, D]
                    embs[task] = feat.mean(axis=0)             # [D]
                    # Re-normalise the mean
                    norm = np.linalg.norm(embs[task])
                    if norm > 0:
                        embs[task] /= norm
                self._task_embs = embs
                logger.info(
                    "Semantic router: cached task embeddings for %d tasks", len(embs)
                )
            except Exception as exc:
                logger.warning("Semantic router: failed to build task embeddings: %s", exc)
                return None

        return self._task_embs

    # ── Scoring ────────────────────────────────────────────────────────────────

    def semantic_scores(self, query: str) -> Optional[Dict[str, float]]:
        """
        Compute cosine similarity between the query and each task description.
        Returns None if the encoder is unavailable.
        """
        task_embs = self._get_task_embeddings()
        if task_embs is None:
            return None

        try:
            query_emb = self._encoder.encode_text(query)   # [D]
            scores: Dict[str, float] = {}
            for task, task_emb in task_embs.items():
                scores[task] = float(np.dot(query_emb, task_emb))
            # Normalise to [0, 1] from cosine range [-1, 1]
            return {t: (s + 1.0) / 2.0 for t, s in scores.items()}
        except Exception as exc:
            logger.warning("Semantic router: encode_text failed: %s", exc)
            return None

    def blend_scores(
        self,
        semantic: Dict[str, float],
        structural: Dict[str, float],
        keyword: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Weighted blend of three score dicts.
        All dicts must have the same keys (TaskType values).
        """
        w_s = self._weights["semantic"]
        w_t = self._weights["structural"]
        w_k = self._weights["keyword"]

        blended: Dict[str, float] = {}
        all_tasks = set(semantic) | set(structural) | set(keyword)
        for task in all_tasks:
            blended[task] = (
                w_s * semantic.get(task, 0.0)
                + w_t * structural.get(task, 0.0)
                + w_k * keyword.get(task, 0.0)
            )
        return blended

    def route(
        self,
        query: str,
        structural_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
    ) -> SemanticRouterResult:
        """
        Main routing method.  Returns the best TaskType with blended score.

        Parameters
        ----------
        query : str
            The user's natural-language query.
        structural_scores : dict
            Per-task scores from structural signals (image count, modality).
        keyword_scores : dict
            Per-task scores from keyword matching.
        """
        sem_scores = self.semantic_scores(query)
        used_semantic = sem_scores is not None

        if not used_semantic:
            # Fallback: blend only structural + keyword, renormalised
            kw_weight = self._weights["keyword"] + self._weights["semantic"] / 2
            st_weight = self._weights["structural"] + self._weights["semantic"] / 2
            total = kw_weight + st_weight
            blended = {
                t: (
                    (st_weight / total) * structural_scores.get(t, 0.0)
                    + (kw_weight / total) * keyword_scores.get(t, 0.0)
                )
                for t in set(structural_scores) | set(keyword_scores)
            }
            sem_scores = {t: 0.0 for t in blended}
        else:
            blended = self.blend_scores(sem_scores, structural_scores, keyword_scores)

        best_task = max(blended, key=lambda t: blended[t])
        best_score = blended[best_task]

        return SemanticRouterResult(
            task_type=best_task,
            final_score=round(max(0.3, min(0.99, best_score)), 4),
            semantic_score=round(sem_scores.get(best_task, 0.0), 4),
            structural_score=round(structural_scores.get(best_task, 0.0), 4),
            keyword_score=round(keyword_scores.get(best_task, 0.0), 4),
            all_scores=blended,
            used_semantic=used_semantic,
        )

    # ── Singleton-style factory ────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        checkpoint_path: Optional[str] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> "SemanticRouter":
        """
        Convenience constructor: load RSCLIPEncoder and build a SemanticRouter.
        Returns a router with encoder=None (keyword-only fallback) if loading fails.
        """
        try:
            from models.rs_clip.encoder import RSCLIPEncoder
            encoder = RSCLIPEncoder.from_pretrained(
                model_name=model_name,
                pretrained=pretrained,
                checkpoint_path=checkpoint_path,
                device=device,
                cache_dir=cache_dir,
            )
            logger.info("SemanticRouter built with RSCLIPEncoder: %s", encoder)
            return cls(encoder=encoder, weights=weights)
        except Exception as exc:
            logger.warning(
                "SemanticRouter: failed to load RSCLIPEncoder (%s) — "
                "falling back to keyword-only routing.",
                exc,
            )
            return cls(encoder=None, weights=weights)
