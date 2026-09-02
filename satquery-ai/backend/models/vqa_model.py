"""
SatQuery AI — Remote Sensing VQA Model
Uses BLIP / BLIP-2 for visual question answering on satellite imagery.
Auto-detects model type from model_name.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("satquery.vqa")


class RemoteSensingVQA:
    """
    Remote sensing Visual Question Answering.

    Supports:
      - Salesforce/blip-vqa-base  (small, fast ~990 MB)
      - Salesforce/blip2-opt-2.7b (large, accurate ~6 GB)
      - Any BLIP or BLIP-2 compatible model
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 150,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._is_blip2 = "blip2" in model_name.lower()

        self._load()

    def _load(self) -> None:
        if self._is_blip2:
            self._load_blip2()
        else:
            self._load_blip()

    def _load_blip2(self) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        logger.info("Loading BLIP-2 VQA: %s on %s", self.model_name, self.device)
        load_kwargs: dict = {"cache_dir": self.cache_dir}
        if self.device == "cuda" and torch.cuda.is_available():
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.processor = Blip2Processor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        if self.device == "cpu":
            self.model = self.model.to("cpu")
        self.model.eval()
        logger.info("BLIP-2 VQA loaded")

    def _load_blip(self) -> None:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        logger.info("Loading BLIP VQA: %s on %s", self.model_name, self.device)
        self.processor = BlipProcessor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = BlipForQuestionAnswering.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("BLIP VQA loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, image: Image.Image, question: str) -> dict:
        if image is None:
            return {"answer": "No image provided.", "confidence": 0.0}

        image = self._preprocess(image)

        try:
            if self._is_blip2:
                return self._infer_blip2(image, question)
            else:
                return self._infer_blip(image, question)
        except Exception as exc:
            logger.error("VQA inference failed: %s", exc)
            return {"answer": f"Inference error: {str(exc)}", "confidence": 0.0}

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_blip2(self, image: Image.Image, question: str) -> dict:
        prompted_q = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompted_q, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                early_stopping=True,
            )
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        confidence = min(0.95, 0.5 + len(answer.split()) * 0.02)
        return {"answer": answer, "confidence": round(confidence, 4)}

    def _infer_blip(self, image: Image.Image, question: str) -> dict:
        inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
        confidence = min(0.92, 0.55 + len(answer.split()) * 0.02)
        return {"answer": answer, "confidence": round(confidence, 4)}

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        max_side = 1024
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        return image
