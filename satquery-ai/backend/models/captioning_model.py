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
"""
from __future__ import annotations

import logging
import math
import re
from typing import Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("satquery.captioning")

RS_CAPTION_PREFIX = "A satellite image showing"

# Generic image-captioning checkpoints frequently emit memorized place names.
# The captioning model has no geolocation input, so location claims are not
# evidence and must not be presented as observations.
_LOCATION_CLAIM_PATTERNS = (
    re.compile(r"\s+(?:in|near|around)\s+(?:the\s+)?(?:city\s+of\s+)?[A-Z][\w-]*(?:,\s*[A-Z][\w-]*)?"),
    re.compile(r"\s+(?:the\s+)?city\s+of\s+[A-Z][\w-]*(?:,\s*[A-Z][\w-]*)?"),
)


def _sanitize_location_claims(caption: str) -> tuple[str, bool]:
    """Remove unsupported place-name claims from generic model captions."""
    sanitized = caption
    changed = False
    for pattern in _LOCATION_CLAIM_PATTERNS:
        sanitized, substitutions = pattern.subn("", sanitized)
        changed = changed or substitutions > 0
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip(" ,.-")
    return sanitized, changed


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

    def generate_caption(self, image: Image.Image) -> dict:
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
                return self._git_caption(image)
            elif self._is_blip2:
                return self._blip2_caption(image)
            else:
                return self._blip_caption(image)
        except Exception as exc:
            logger.error("Captioning inference failed: %s", exc)
            return {
                "caption": f"Caption error: {str(exc)}",
                "confidence": 0.0,
                "confidence_is_calibrated": False,
                "evidence": _empty_caption_evidence(),
            }

    # ── Inference ─────────────────────────────────────────────────────────────

    def _build_caption_result(self, caption: str, gen, seq, decoding: dict) -> dict:
        """
        Assemble caption + raw generation evidence.  ``confidence`` is a
        convenience value derived from the mean token log-probability (a real
        model signal, NOT caption length); ``confidence_is_calibrated`` is
        always False.
        """
        caption, location_claim_suppressed = _sanitize_location_claims(caption)
        seq_score, token_logprobs = _extract_caption_scores(gen, seq)
        evidence = _caption_evidence(seq_score, token_logprobs, caption, decoding)
        evidence["location_claim_suppressed"] = location_claim_suppressed
        if location_claim_suppressed:
            evidence["location_note"] = (
                "Location claims were removed because captioning does not provide geolocation evidence."
            )

        if seq_score is not None:
            confidence = float(np.clip(math.exp(seq_score), 0.0, 1.0))
        else:
            confidence = 0.5  # neutral placeholder; flagged uncalibrated + score-less

        return {
            "caption": caption,
            "confidence": round(confidence, 4),
            "confidence_is_calibrated": False,
            "evidence": evidence,
        }

    def _blip_caption(self, image: Image.Image) -> dict:
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

        return self._build_caption_result(caption, gen, seq, decoding)

    def _blip2_caption(self, image: Image.Image) -> dict:
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

        return self._build_caption_result(caption, gen, seq, decoding)

    def _git_caption(self, image: Image.Image) -> dict:
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
        return self._build_caption_result(caption, gen, seq, decoding)

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
