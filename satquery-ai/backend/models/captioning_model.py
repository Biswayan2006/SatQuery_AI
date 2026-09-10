"""
SatQuery AI — Remote Sensing Image Captioning
Generates rich scene descriptions using BLIP / BLIP-2.
Auto-detects model type from model_name.

Confidence / evidence
---------------------
Like the VQA wrapper, this model does NOT fabricate a calibrated confidence and
does NOT use caption length as a confidence proxy.  Each caption is returned
with the raw generation ``evidence`` (mean per-token log-probability and the
per-token log-probs) needed for downstream calibration, plus a convenience
``confidence`` derived from that real signal and flagged
``confidence_is_calibrated=False``.

Geographic claim guard
----------------------
BLIP is trained on web-crawled data (COCO, Conceptual Captions) and
frequently memorizes place names as texture patterns (e.g. "aerial view of
london" for ANY urban aerial image).  The geographic claim guard detects
these unsupported location assertions and strips them, preserving honest
visual descriptions while preventing geographic hallucinations.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import torch
from PIL import Image

from models.geographic_guard import apply_geographic_guard, GeographicGuardResult

logger = logging.getLogger("satquery.captioning")

RS_CAPTION_PREFIX = "A satellite image showing"


def _extract_caption_scores(gen, seq):
    """
    Mean per-token log-prob of a generated caption (and the per-token list).

    Returns ``(sequence_score, token_logprobs)`` where either may be ``None``
    if the model did not expose generation scores.  Mirrors
    ``vqa_model.RemoteSensingVQA._extract_scores``.
    """
    try:
        scores = getattr(gen, "scores", None)
        if not scores:
            return None, None
        gen_tokens = seq[0][-len(scores):]
        logprobs = []
        for step_logits, tok in zip(scores, gen_tokens):
            lp = torch.log_softmax(step_logits[0].float(), dim=-1)
            logprobs.append(float(lp[tok]))
        if not logprobs:
            return None, None
        return float(np.mean(logprobs)), [round(x, 4) for x in logprobs]
    except Exception as exc:
        logger.debug("Caption score extraction failed: %s", exc)
        return None, None


def _caption_evidence(seq_score, token_logprobs, caption: str, decoding: dict) -> dict:
    # Caption length is deliberately absent: verbosity is not evidence of
    # correctness, and the old length-based heuristic is gone.
    return {
        "sequence_score": seq_score,       # mean token log-prob (or None)
        "token_logprobs": token_logprobs,  # list[float] or None
        "n_tokens": len(token_logprobs) if token_logprobs else 0,
        "decoding": decoding,
        "model_agreement": None,           # reserved for ensemble signals
    }


def _empty_caption_evidence() -> dict:
    return {
        "sequence_score": None,
        "token_logprobs": None,
        "n_tokens": 0,
        "decoding": {},
        "model_agreement": None,
    }


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

    def generate_caption(
        self, image: Image.Image, image_data: Optional[dict] = None,
    ) -> dict:
        if image is None:
            return {
                "caption": "No image provided.",
                "confidence": 0.0,
                "confidence_is_calibrated": False,
                "evidence": _empty_caption_evidence(),
            }

        image = self._preprocess(image)

        try:
            if self._is_git:
                return self._git_caption(image, image_data)
            elif self._is_blip2:
                return self._blip2_caption(image, image_data)
            else:
                return self._blip_caption(image, image_data)
        except Exception as exc:
            logger.error("Captioning inference failed: %s", exc)
            return {
                "caption": f"Caption error: {str(exc)}",
                "confidence": 0.0,
                "confidence_is_calibrated": False,
                "evidence": _empty_caption_evidence(),
            }

    # ── Inference ─────────────────────────────────────────────────────────────

    def _build_caption_result(
        self, caption: str, gen, seq, decoding: dict,
        image_data: Optional[dict] = None,
    ) -> dict:
        """
        Assemble caption + raw generation evidence + geographic guard.

        ``confidence`` is a convenience value derived from the mean token
        log-probability (a real model signal, NOT caption length);
        ``confidence_is_calibrated`` is always False.
        """
        # Apply geographic claim guard
        guard_result = apply_geographic_guard(caption, image_data)
        safe_caption = guard_result.sanitized_caption

        seq_score, token_logprobs = _extract_caption_scores(gen, seq)
        evidence = _caption_evidence(seq_score, token_logprobs, safe_caption, decoding)

        # Geographic guard metadata
        evidence["geographic_guard"] = {
            "original_caption": guard_result.original_caption,
            "geographic_claim_detected": guard_result.has_geographic_claims,
            "geographic_claim_verified": guard_result.has_independent_evidence,
            "verification_reason": guard_result.verification_reason,
            "stripped_claims": guard_result.stripped_claims,
            "claims_count": len(guard_result.geographic_claims),
        }

        if guard_result.stripped_claims:
            evidence["location_note"] = (
                f"Geographic claims ({', '.join(guard_result.stripped_claims)}) were removed "
                "because the captioning model has no independent geolocation evidence."
            )

        if seq_score is not None:
            confidence = float(np.clip(math.exp(seq_score), 0.0, 1.0))
        else:
            confidence = 0.5  # neutral placeholder; flagged uncalibrated + score-less

        return {
            "caption": safe_caption,
            "confidence": round(confidence, 4),
            "confidence_is_calibrated": False,
            "evidence": evidence,
        }

    def _blip_caption(self, image: Image.Image, image_data: Optional[dict] = None) -> dict:
        # Conditional captioning with RS prefix
        inputs = self.processor(
            images=image,
            text=RS_CAPTION_PREFIX,
            return_tensors="pt",
        ).to(self.device)

        decoding = {
            "num_beams": 5, "repetition_penalty": 1.3, "length_penalty": 1.0,
            "early_stopping": True, "max_new_tokens": self.max_new_tokens,
        }
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=5,
                repetition_penalty=1.3,
                length_penalty=1.0,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        caption = self.processor.decode(seq[0], skip_special_tokens=True).strip()
        if not caption.lower().startswith("a satellite"):
            caption = RS_CAPTION_PREFIX + " " + caption

        return self._build_caption_result(caption, gen, seq, decoding, image_data)

    def _blip2_caption(self, image: Image.Image, image_data: Optional[dict] = None) -> dict:
        inputs = self.processor(
            images=image,
            text=RS_CAPTION_PREFIX,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        decoding = {
            "num_beams": 5, "repetition_penalty": 1.3,
            "early_stopping": True, "max_new_tokens": self.max_new_tokens,
        }
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=5,
                repetition_penalty=1.3,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        caption = self.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        if not caption.lower().startswith("a satellite"):
            caption = RS_CAPTION_PREFIX + " " + caption

        return self._build_caption_result(caption, gen, seq, decoding, image_data)

    def _git_caption(self, image: Image.Image, image_data: Optional[dict] = None) -> dict:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        decoding = {"max_new_tokens": self.max_new_tokens}
        with torch.no_grad():
            gen = self.model.generate(
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        caption = self.processor.batch_decode(seq, skip_special_tokens=True)[0].strip()
        return self._build_caption_result(caption, gen, seq, decoding, image_data)

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
