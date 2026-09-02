"""
SatQuery AI — Remote Sensing Image Captioning
Generates rich scene descriptions using BLIP / BLIP-2.
Auto-detects model type from model_name.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger("satquery.captioning")

RS_CAPTION_PREFIX = "A satellite image showing"


class RemoteSensingCaptioning:
    """
    Generates scene-level captions for remote sensing imagery.

    Supports:
      - Salesforce/blip-image-captioning-base  (small, fast ~990 MB)
      - Salesforce/blip2-opt-2.7b              (large, accurate ~6 GB)
      - microsoft/git-base-coco                (fallback)
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 200,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._is_blip2 = "blip2" in model_name.lower()
        self._is_git = False

        self._load()

    def _load(self) -> None:
        if self._is_blip2:
            self._load_blip2()
        else:
            self._load_blip()

    def _load_blip2(self) -> None:
        from transformers import Blip2ForConditionalGeneration, Blip2Processor

        logger.info("Loading BLIP-2 captioning: %s on %s", self.model_name, self.device)
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
        logger.info("BLIP-2 captioning loaded")

    def _load_blip(self) -> None:
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            logger.info("Loading BLIP captioning: %s on %s", self.model_name, self.device)
            self.processor = BlipProcessor.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            logger.info("BLIP captioning loaded")
        except Exception as exc:
            logger.warning("BLIP captioning failed, falling back to GIT: %s", exc)
            self._load_git_fallback()

    def _load_git_fallback(self) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        fallback = "microsoft/git-base-coco"
        logger.info("Loading fallback captioning: %s", fallback)
        self.processor = AutoProcessor.from_pretrained(fallback, cache_dir=self.cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(fallback, cache_dir=self.cache_dir)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_name = fallback
        self._is_git = True
        logger.info("GIT captioning fallback loaded")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_caption(self, image: Image.Image) -> dict:
        if image is None:
            return {"caption": "No image provided.", "confidence": 0.0}

        image = self._preprocess(image)

        try:
            if self._is_git:
                return self._git_caption(image)
            elif self._is_blip2:
                return self._blip2_caption(image)
            else:
                return self._blip_caption(image)
        except Exception as exc:
            logger.error("Captioning inference failed: %s", exc)
            return {"caption": f"Caption error: {str(exc)}", "confidence": 0.0}

    # ── Inference ─────────────────────────────────────────────────────────────

    def _blip_caption(self, image: Image.Image) -> dict:
        # Conditional captioning with RS prefix
        inputs = self.processor(
            images=image,
            text=RS_CAPTION_PREFIX,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=5,
                repetition_penalty=1.3,
                length_penalty=1.0,
                early_stopping=True,
            )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        if not caption.lower().startswith("a satellite"):
            caption = RS_CAPTION_PREFIX + " " + caption

        confidence = min(0.95, 0.55 + len(caption.split()) * 0.015)
        return {"caption": caption, "confidence": round(confidence, 4)}

    def _blip2_caption(self, image: Image.Image) -> dict:
        inputs = self.processor(
            images=image,
            text=RS_CAPTION_PREFIX,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=5,
                repetition_penalty=1.3,
                early_stopping=True,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if not caption.lower().startswith("a satellite"):
            caption = RS_CAPTION_PREFIX + " " + caption

        confidence = min(0.95, 0.55 + len(caption.split()) * 0.015)
        return {"caption": caption, "confidence": round(confidence, 4)}

    def _git_caption(self, image: Image.Image) -> dict:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        confidence = min(0.90, 0.50 + len(caption.split()) * 0.015)
        return {"caption": caption, "confidence": round(confidence, 4)}

    @staticmethod
    def _preprocess(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        max_side = 1024
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        return image
