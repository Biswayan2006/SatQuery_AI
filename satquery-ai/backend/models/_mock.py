"""
Mock model fallback — used when a real model fails to load.
Returns plausible placeholder outputs so the API stays functional.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("satquery.mock")


class MockModel:
    """Minimal stub that mimics the interface of specialist models."""

    def __init__(self, name: str):
        self.name = name
        logger.warning("MockModel active for '%s' — real model failed to load.", name)

    # VQA / Captioning interface
    def answer(self, image, question: str) -> dict:
        return {
            "answer": (
                f"[Mock response] The model '{self.name}' is not loaded. "
                "Please check model configuration and GPU availability."
            ),
            "confidence": 0.1,
        }

    def generate_caption(self, image) -> dict:
        return {
            "caption": (
                f"[Mock caption] The captioning model '{self.name}' is unavailable. "
                "This is a placeholder response."
            ),
            "confidence": 0.1,
        }

    # Grounding interface
    def ground(self, image, text_query: str) -> dict:
        return {
            "boxes": [],
            "labels": [],
            "scores": [],
            "annotated_image": None,
        }

    # Change detection interface
    def detect_changes(self, img1, img2) -> dict:
        import numpy as np
        h, w = (getattr(img1, "size", (64, 64))[1], getattr(img1, "size", (64, 64))[0])
        return {
            "change_map": np.zeros((h, w), dtype=np.float32),
            "changed_regions": [],
            "change_percentage": 0.0,
        }

    def answer_change_question(self, img1, img2, question: str) -> dict:
        return {
            "answer": f"[Mock] Change detection model '{self.name}' is unavailable.",
            "confidence": 0.1,
            "change_map_b64": None,
            "change_percentage": 0.0,
        }

    # SAR Fusion interface
    def fuse_and_analyze(self, optical, sar, query: str) -> dict:
        return {
            "answer": f"[Mock] SAR fusion model '{self.name}' is unavailable.",
            "confidence": 0.1,
            "fusion_map_b64": None,
        }
