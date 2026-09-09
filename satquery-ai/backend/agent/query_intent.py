"""
SatQuery AI — Structured Query Intent Extraction
================================================
Turns a natural-language query into a small, explicit **intent record** that the
planner uses to decide *which deterministic tools* and *which reasoning steps*
belong in the execution plan.  It answers three orthogonal questions:

  * **concept**   — what is the query about?  (vegetation / water / built_up /
    change / scene)
  * **operation** — what should we do with it?  (increase / decrease / localize /
    describe / compare / detect)
  * flags — does it need a **comparison** (bi-temporal), **grounding**
    (localisation), or cross-modal **fusion**?

Examples (from the product spec)::

    "Has the built-up area increased?"
      → {concept: built_up, operation: increase, requires_comparison: True}
    "Where are the water bodies?"
      → {concept: water, operation: localize, requires_grounding: True}
    "Use the optical and SAR images together to identify built-up regions."
      → {concept: built_up, requires_fusion: True}

Semantic vs keyword
-------------------
Concept detection is primarily lexical, but when an RS-CLIP encoder is available
the extractor blends a semantic concept score (query ↔ concept description) so it
does not rely on exact keywords.  Without the encoder it degrades cleanly to the
keyword rules — the public API is identical either way.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("satquery.query_intent")


# ── Concepts (the "what") ──────────────────────────────────────────────────────
CONCEPT_VEGETATION = "vegetation"
CONCEPT_WATER = "water"
CONCEPT_BUILTUP = "built_up"
CONCEPT_CHANGE = "change"
CONCEPT_SCENE = "scene"

# Concept → keyword set.  Order matters only for tie-breaking (earlier wins).
_CONCEPT_KEYWORDS: Dict[str, List[str]] = {
    CONCEPT_VEGETATION: [
        "vegetation", "veget", "greenery", "green cover", "forest", "crop",
        "farm", "farmland", "ndvi", "plant", "tree", "deforest", "foliage",
        "agriculture", "agricultural",
    ],
    CONCEPT_WATER: [
        "water", "flood", "lake", "river", "ndwi", "wetland", "reservoir",
        "inundat", "moisture", "coastline", "shoreline", "pond", "sea",
    ],
    CONCEPT_BUILTUP: [
        "built-up", "built up", "builtup", "building", "buildings", "urban",
        "construction", "settlement", "ndbi", "concrete", "development",
        "expansion", "sprawl", "city", "infrastructure", "road", "impervious",
    ],
    CONCEPT_CHANGE: [
        "change", "changed", "difference", "differ", "temporal", "evolution",
        "progression", "transformation", "before and after", "over time",
    ],
    CONCEPT_SCENE: [
        "describe", "caption", "summary", "summarize", "overview",
        "what does this", "what is in", "what can you see", "scene",
    ],
}

# One-line natural descriptions per concept for optional semantic scoring.
_CONCEPT_DESCRIPTIONS: Dict[str, List[str]] = {
    CONCEPT_VEGETATION: ["vegetation, forests, crops and green cover in a satellite image"],
    CONCEPT_WATER: ["water bodies, rivers, lakes and flooding in a satellite image"],
    CONCEPT_BUILTUP: ["urban built-up area, buildings and construction in a satellite image"],
    CONCEPT_CHANGE: ["temporal change between two satellite images"],
    CONCEPT_SCENE: ["a general description of a satellite scene"],
}


# ── Operations (the "how") ──────────────────────────────────────────────────────
OP_INCREASE = "increase"
OP_DECREASE = "decrease"
OP_LOCALIZE = "localize"
OP_DESCRIBE = "describe"
OP_COMPARE = "compare"
OP_DETECT = "detect"

_OP_KEYWORDS: Dict[str, List[str]] = {
    OP_INCREASE: ["increase", "increased", "expand", "expansion", "grow", "grew",
                  "growth", "more", "gain", "gained", "rise", "risen", "added"],
    OP_DECREASE: ["decrease", "decreased", "decline", "declined", "loss", "lost",
                  "less", "reduce", "reduced", "shrink", "shrank", "deforest",
                  "drop", "dropped", "disappear", "removed"],
    OP_LOCALIZE: ["where", "locate", "location", "position", "coordinate",
                  "which area", "which region", "point out", "pinpoint",
                  "bounding box", "mark"],
    OP_COMPARE: ["compare", "comparison", "versus", "vs", "difference between",
                 "changed between", "before and after", "over time", "temporal"],
    OP_DESCRIBE: ["describe", "description", "caption", "summarize", "summary",
                  "overview", "explain", "what does this", "tell me about"],
    OP_DETECT: ["detect", "identify", "is there", "are there", "how many",
                "count", "find", "presence", "how much"],
}


@dataclass
class QueryIntent:
    """Structured, explicit query intent used by the planner."""

    concept: Optional[str] = None
    operation: Optional[str] = None
    requires_comparison: bool = False
    requires_grounding: bool = False
    requires_fusion: bool = False
    used_semantic: bool = False
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """Compact public form — only the flags that are True are surfaced."""
        out: Dict[str, object] = {}
        if self.concept is not None:
            out["concept"] = self.concept
        if self.operation is not None:
            out["operation"] = self.operation
        if self.requires_comparison:
            out["requires_comparison"] = True
        if self.requires_grounding:
            out["requires_grounding"] = True
        if self.requires_fusion:
            out["requires_fusion"] = True
        return out

    # Concept → the index tool + bands it needs (None for non-index concepts).
    def index_tool(self):
        return {
            CONCEPT_VEGETATION: ("NDVI", ("nir", "red")),
            CONCEPT_WATER: ("NDWI", ("green", "nir")),
            CONCEPT_BUILTUP: ("NDBI", ("swir1", "nir")),
        }.get(self.concept)


class IntentExtractor:
    """
    Extract a :class:`QueryIntent` from a query + image structure + modalities.

    Parameters
    ----------
    encoder : RSCLIPEncoder or None
        Optional semantic encoder.  When present, concept detection blends a
        semantic similarity score with the keyword score.  When None, keyword
        rules alone are used.
    """

    def __init__(self, encoder=None):
        self._encoder = encoder
        self._concept_embs: Optional[Dict[str, np.ndarray]] = None

    # ── Public API ─────────────────────────────────────────────────────────────
    def extract(
        self,
        query: str,
        num_images: int = 1,
        modalities: Optional[List[str]] = None,
    ) -> QueryIntent:
        modalities = modalities or []
        q = self._normalize(query)

        concept, concept_scores = self._detect_concept(query, q)
        operation = self._detect_operation(q)

        has_sar = "sar" in modalities
        has_optical = "optical" in modalities or "multispectral" in modalities

        requires_fusion = (has_sar and has_optical) or self._has_any(
            q, ("fuse", "fusion", "combine optical", "combine sar",
                "together", "multimodal", "multi-modal")
        )
        requires_grounding = operation == OP_LOCALIZE or self._has_any(
            q, ("where", "locate", "bounding box", "point out")
        )
        requires_comparison = (
            num_images >= 2
            or operation in (OP_INCREASE, OP_DECREASE, OP_COMPARE)
            or concept == CONCEPT_CHANGE
        )

        return QueryIntent(
            concept=concept,
            operation=operation,
            requires_comparison=requires_comparison,
            requires_grounding=requires_grounding,
            requires_fusion=requires_fusion,
            used_semantic=self._encoder is not None and self._concept_embs is not None,
            scores=concept_scores,
        )

    # ── Concept detection ──────────────────────────────────────────────────────
    def _detect_concept(self, raw_query: str, q: str):
        keyword_scores = {c: 0.0 for c in _CONCEPT_KEYWORDS}
        for concept, kws in _CONCEPT_KEYWORDS.items():
            for kw in kws:
                if kw in q:
                    keyword_scores[concept] = max(keyword_scores[concept], 1.0)

        semantic_scores = self._semantic_concept_scores(raw_query)
        if semantic_scores is not None:
            # Blend: keyword is a strong prior (0.6), semantic refines (0.4).
            scores = {
                c: 0.6 * keyword_scores.get(c, 0.0) + 0.4 * semantic_scores.get(c, 0.0)
                for c in _CONCEPT_KEYWORDS
            }
        else:
            scores = keyword_scores

        best = max(scores, key=lambda c: scores[c])
        if scores[best] <= 0.0:
            # No concept signal at all → treat as a generic scene query.
            return CONCEPT_SCENE, scores
        return best, scores

    def _semantic_concept_scores(self, query: str) -> Optional[Dict[str, float]]:
        if self._encoder is None:
            return None
        try:
            if self._concept_embs is None:
                embs: Dict[str, np.ndarray] = {}
                for concept, descs in _CONCEPT_DESCRIPTIONS.items():
                    feat = self._encoder.encode_text(descs)
                    v = feat.mean(axis=0)
                    n = np.linalg.norm(v)
                    embs[concept] = v / n if n > 0 else v
                self._concept_embs = embs
            qv = self._encoder.encode_text(query)
            return {
                c: (float(np.dot(qv, e)) + 1.0) / 2.0
                for c, e in self._concept_embs.items()
            }
        except Exception as exc:
            logger.debug("Semantic concept scoring failed: %s", exc)
            return None

    # ── Operation detection ─────────────────────────────────────────────────────
    def _detect_operation(self, q: str) -> Optional[str]:
        # Priority order: localisation and directional change are the most
        # action-defining; describe/detect are the softer fallbacks.
        for op in (OP_LOCALIZE, OP_INCREASE, OP_DECREASE, OP_COMPARE,
                   OP_DESCRIBE, OP_DETECT):
            if self._has_any(q, _OP_KEYWORDS[op]):
                return op
        return None

    # ── Helpers ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _normalize(query: str) -> str:
        q = (query or "").lower().strip()
        # Keep hyphens (built-up) but drop other punctuation.
        return re.sub(r"[^\w\s\-]", " ", q)

    @staticmethod
    def _has_any(q: str, needles) -> bool:
        return any(n in q for n in needles)
