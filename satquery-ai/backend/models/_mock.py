"""
Mock model fallback — used when a real model fails to load.
Returns clear degradation reports with metadata.
"""
from __future__ import annotations

import logging
import time

logger = logging.getLogger("satquery.mock")


class MockModel:
    """Fallback model that provides clear degradation reporting."""

    def __init__(self, name: str):
        self.name = name
        self.is_mock = True
        self.mock_reason = "Model failed to load or is unavailable"
        self.created_at = time.time()
        logger.warning(
            f"MockModel active for '{name}' — system is in degraded mode",
            extra={
                "model": name,
                "mock_reason": self.mock_reason,
                "degraded": True,
            }
        )
    
    def _create_degraded_response(self, task_type: str, additional_info: str = "") -> dict:
        """Create a standardized degraded response."""
        base_message = f"System is in degraded mode: Model '{self.name}' is unavailable."
        
        if additional_info:
            message = f"{base_message} {additional_info}"
        else:
            message = base_message
        
        return {
            "answer": message,
            "confidence": 0.01,  # Very low confidence for mock responses
            "confidence_is_calibrated": False,
            "confidence_type": "unavailable",
            "degraded": True,
            "mock_model_used": self.name,
            "mock_reason": self.mock_reason,
            "requires_verification": True,
            "uncertainty": "high",
            "evidence": {
                "sequence_score": None,
                "token_logprobs": None,
                "answer_length": 0,
                "decoding": {},
                "model_agreement": None,
                "degradation_reason": self.mock_reason,
            },
        }

    # VQA / Captioning interface
    def answer(self, image, question: str) -> dict:
        response = self._create_degraded_response("VQA")
        response["answer"] = (
            f"[Degraded Mode] Visual Question Answering is currently unavailable. "
            f"Model '{self.name}' failed to load. Please check: "
            "1) GPU memory availability, 2) Model configuration, 3) Network connectivity for model downloads."
        )
        return response

    def generate_caption(self, image) -> dict:
        response = self._create_degraded_response("Captioning")
        response["caption"] = (
            f"[Degraded Mode] Image captioning is currently unavailable. "
            f"Model '{self.name}' failed to load. Please check system resources."
        )
        # Rename key for captioning response
        response["caption"] = response.pop("answer")
        return response

    # Grounding interface
    def ground(self, image, text_query: str) -> dict:
        return {
            "boxes": [],
            "labels": [],
            "scores": [],
            "annotated_image": None,
            "degraded": True,
            "mock_model_used": self.name,
            "mock_reason": self.mock_reason,
            "warning": f"Grounding model '{self.name}' is unavailable. No objects detected.",
        }

    # Change detection interface
    def detect_changes(self, img1, img2, *, geo_meta=None) -> dict:
        import numpy as np
        h, w = (getattr(img1, "size", (64, 64))[1], getattr(img1, "size", (64, 64))[0])
        return {
            "change_map": np.zeros((h, w), dtype=np.float32),
            "binary_mask": np.zeros((h, w), dtype=np.float32),
            "changed_regions": [],
            "change_percentage": 0.0,
            "degraded": True,
            "mock_model_used": self.name,
            "mock_reason": self.mock_reason,
            "warning": f"Change detection model '{self.name}' is unavailable. No changes detected.",
        }

    def answer_change_question(self, img1, img2, question: str, *, geo_meta=None) -> dict:
        response = self._create_degraded_response("Change Detection")
        response["answer"] = (
            f"[Degraded Mode] Change detection analysis is currently unavailable. "
            f"Model '{self.name}' failed to load. Temporal analysis cannot be performed."
        )
        response["change_map_b64"] = None
        response["change_percentage"] = 0.0
        response["changed_regions"] = []
        return response

    # SAR Fusion interface
    def fuse_and_analyze(self, optical, sar, query: str) -> dict:
        response = self._create_degraded_response("SAR Fusion")
        response["answer"] = (
            f"[Degraded Mode] SAR-optical fusion analysis is currently unavailable. "
            f"Model '{self.name}' failed to load. Cross-modal analysis cannot be performed."
        )
        response["fusion_map_b64"] = None
        response["fusion_trained"] = False
        return response
    
    def cleanup(self):
        """Cleanup method for consistency with real models."""
        logger.info(f"Mock model '{self.name}' cleanup called")
