"""
SatQuery AI — Remote Sensing VQA Model
Uses BLIP / BLIP-2 for visual question answering on satellite imagery.
Auto-detects model type from model_name.

Model source
------------
The wrapper can load either:

  * a **base pretrained** model (``model_source="pretrained"``), or
  * a **fine-tuned checkpoint** produced by ``training/train_vqa.py``
    (``model_source="finetuned"`` + ``finetuned_checkpoint=<dir>``).

The checkpoint directory is supplied via configuration / environment
(``VQA_MODEL_SOURCE`` / ``VQA_FINETUNED_CHECKPOINT``) and is NEVER hardcoded.

Confidence / evidence
---------------------
This wrapper does NOT fabricate a calibrated confidence score.  It returns the
raw generation evidence needed for later calibration:

  * ``sequence_score``   — mean per-token log-probability of the generated
                           answer (higher = more confident), when the model
                           exposes generation scores; else None
  * ``token_logprobs``   — per-token log-probabilities when available
  * ``answer_length``    — number of tokens / words in the answer
  * ``decoding``         — the decoding parameters used (num_beams, etc.)

For backward compatibility a ``confidence`` field is still returned, but it is
derived from ``sequence_score`` (a real model signal) when available and
clearly flagged as uncalibrated via ``confidence_is_calibrated=False``.
"""
from __future__ import annotations

import logging
import math
import os
import re
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
      - Fine-tuned checkpoints (LoRA via peft / manual, or full) produced by
        training/train_vqa.py
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-vqa-base",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        max_new_tokens: int = 150,
        model_source: str = "pretrained",
        finetuned_checkpoint: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.model_source = (model_source or "pretrained").lower()
        self.finetuned_checkpoint = finetuned_checkpoint or None
        self.model = None
        self.processor = None
        self.is_finetuned = False
        self._is_blip2 = "blip2" in model_name.lower()

        self._load()

    # ── Loading ─────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.model_source == "finetuned" and self.finetuned_checkpoint:
            if self._load_finetuned():
                return
            logger.warning(
                "Falling back to base pretrained weights after fine-tuned "
                "load failure."
            )
        # Base pretrained path.
        if self._is_blip2:
            self._load_blip2()
        else:
            self._load_blip()

    def _load_finetuned(self) -> bool:
        """
        Load base weights + fine-tuned adapter/checkpoint.

        Returns True on success.  The checkpoint dir layout is produced by
        training/train_vqa.py (model_version.json + adapter weights + processor).
        """
        ckpt = self.finetuned_checkpoint
        if not ckpt or not os.path.isdir(ckpt):
            logger.warning("Fine-tuned checkpoint dir not found: %s", ckpt)
            return False

        try:
            import json
            meta_path = os.path.join(ckpt, "model_version.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)

            base_model = meta.get("base_model", self.model_name)
            self._is_blip2 = "blip2" in base_model.lower()
            lora_backend = meta.get("lora_backend", "manual")
            lora_cfg = meta.get("lora_config", {})

            # Load base model + processor.
            if self._is_blip2:
                from transformers import Blip2ForConditionalGeneration, Blip2Processor
                self.processor = Blip2Processor.from_pretrained(base_model, cache_dir=self.cache_dir)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    base_model, cache_dir=self.cache_dir, torch_dtype=torch.float32,
                )
            else:
                from transformers import BlipForQuestionAnswering, BlipProcessor
                self.processor = BlipProcessor.from_pretrained(base_model, cache_dir=self.cache_dir)
                model = BlipForQuestionAnswering.from_pretrained(base_model, cache_dir=self.cache_dir)

            # Apply the fine-tuned weights according to the training backend.
            if lora_backend == "peft":
                from peft import PeftModel  # type: ignore
                model = PeftModel.from_pretrained(model, ckpt)
            elif lora_backend == "manual":
                from training.vqa_lora import rebuild_manual_lora
                model = rebuild_manual_lora(model, lora_cfg)
                weights = torch.load(
                    os.path.join(ckpt, "adapter_weights.pt"), map_location=self.device
                )
                model.load_state_dict(weights, strict=False)
            elif lora_backend == "full":
                weights = torch.load(
                    os.path.join(ckpt, "adapter_weights.pt"), map_location=self.device
                )
                model.load_state_dict(weights, strict=False)

            # Prefer the checkpoint's processor (tokenizer parity).
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(ckpt)
            except Exception:
                pass

            self.model = model.to(self.device).eval()
            self.model_name = base_model
            self.is_finetuned = True
            logger.info(
                "Loaded fine-tuned VQA model (%s backend) from %s",
                lora_backend, ckpt,
            )
            return True
        except Exception as exc:
            logger.error("Fine-tuned VQA load failed: %s", exc)
            return False

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

    # ── Prompt construction ──────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(question: str) -> str:
        """
        Construct the VQA prompt sent to BLIP.

        For overhead satellite imagery, prepending a short domain hint helps
        BLIP orient its visual attention toward aerial/overhead structures
        rather than natural-scene defaults.  The prefix is intentionally
        compact — a full paragraph would dilute the actual question.

        The user's question is ALWAYS preserved verbatim after the prefix.
        """
        q = question.strip()
        # Don't double-prefix if the caller already included context
        _satellite_markers = (
            "satellite", "overhead", "aerial", "remote sensing",
            "satellite image", "overhead image",
        )
        if any(m in q.lower() for m in _satellite_markers):
            return q
        return f"This is an overhead satellite image. {q}"

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, image: Image.Image, question: str) -> dict:
        if image is None:
            return {
                "answer": "No image provided.",
                "confidence": 0.0,
                "confidence_is_calibrated": False,
                "evidence": self._empty_evidence(),
            }

        image = self._preprocess(image)

        try:
            if self._is_blip2:
                return self._infer_blip2(image, question)
            else:
                return self._infer_blip(image, question)
        except Exception as exc:
            logger.error("VQA inference failed: %s", exc)
            return {
                "answer": f"Inference error: {str(exc)}",
                "confidence": 0.0,
                "confidence_is_calibrated": False,
                "evidence": self._empty_evidence(),
            }

    def answer_with_visual_tokens(
        self,
        visual_tokens: "torch.Tensor",
        question: str,
    ) -> dict:
        """
        Answer a question conditioned on externally-supplied **visual tokens**.

        This is the genuine multimodal injection point used by the SAR-optical
        fusion model.  Instead of running the vision encoder on a single image,
        the caller supplies fused ``[B, N, H]`` tokens that already combine SAR
        and optical information; those tokens are fed to BLIP's cross-attention
        pathway exactly where ``image_embeds`` normally goes::

            text_encoder(input_ids, encoder_hidden_states=visual_tokens)
              → question_embeds
            text_decoder.generate(encoder_hidden_states=question_embeds)

        The fused representation therefore conditions the whole answer — it is
        NOT flattened into a text prompt.

        Only the BLIP (``BlipForQuestionAnswering``) architecture exposes this
        split cleanly; for other backends the method raises
        ``NotImplementedError`` so the caller can fall back honestly.
        """
        if self.model is None or self._is_blip2:
            raise NotImplementedError(
                "Visual-token injection is only supported for BLIP VQA models."
            )
        if not hasattr(self.model, "text_encoder") or not hasattr(self.model, "text_decoder"):
            raise NotImplementedError(
                "Loaded model does not expose BLIP's text_encoder/text_decoder."
            )

        model = self.model
        device = self.device
        vt = visual_tokens.to(device)

        # Tokenise the question text only (no image — tokens come from fusion).
        enc = self.processor(text=question, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        img_attn = torch.ones(vt.size()[:-1], dtype=torch.long, device=device)
        decoding = {
            "num_beams": 1,
            "min_new_tokens": 1,
            "max_new_tokens": 50,
            "repetition_penalty": 1.2,
            "do_sample": False,
        }

        with torch.no_grad():
            question_outputs = model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=vt,
                encoder_attention_mask=img_attn,
                return_dict=False,
            )
            question_embeds = question_outputs[0]
            q_attn = torch.ones(
                question_embeds.size()[:-1], dtype=torch.long, device=device
            )
            bos_ids = torch.full(
                (question_embeds.size(0), 1),
                fill_value=model.decoder_start_token_id,
                device=device,
                dtype=torch.long,
            )
            gen = model.text_decoder.generate(
                input_ids=bos_ids,
                eos_token_id=model.config.text_config.sep_token_id,
                pad_token_id=model.config.text_config.pad_token_id,
                encoder_hidden_states=question_embeds,
                encoder_attention_mask=q_attn,
                min_new_tokens=1,
                max_new_tokens=50,
                repetition_penalty=1.2,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        answer = self._clean_generated_answer(
            self.processor.decode(seq[0], skip_special_tokens=True), question
        )
        result = self._build_result(answer, gen, seq, decoding)
        result["conditioned_on"] = "fused_visual_tokens"
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer_blip2(self, image: Image.Image, question: str) -> dict:
        prompted_q = f"Question: {self._build_prompt(question)} Answer:"
        inputs = self.processor(images=image, text=prompted_q, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        decoding = {
            "num_beams": 4,
            "early_stopping": True,
            "min_new_tokens": 1,
            "max_new_tokens": 50,
            "repetition_penalty": 1.2,
            "do_sample": False,
        }

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                min_new_tokens=1,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seq = gen.sequences
        answer = self._clean_generated_answer(
            self.processor.batch_decode(seq, skip_special_tokens=True)[0], question
        )
        return self._build_result(answer, gen, seq, decoding)

    def _infer_blip(self, image: Image.Image, question: str) -> dict:
        prompted_q = self._build_prompt(question)
        inputs = self.processor(images=image, text=prompted_q, return_tensors="pt").to(self.device)
        decoding = {
            "num_beams": 4,
            "min_new_tokens": 1,
            "max_new_tokens": 50,
            "repetition_penalty": 1.2,
            "do_sample": False,
        }

        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                min_new_tokens=1,
                max_new_tokens=50,
                num_beams=4,
                repetition_penalty=1.2,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )
        seq = gen.sequences
        answer = self._clean_generated_answer(
            self.processor.decode(seq[0], skip_special_tokens=True), question
        )
        return self._build_result(answer, gen, seq, decoding)

    @staticmethod
    def _clean_generated_answer(text: str, question: str) -> str:
        """Remove BLIP prompt echoes while retaining the generated answer."""
        answer = " ".join((text or "").strip().split())
        answer = re.sub(r"^question\s*:\s*.*?\s*answer\s*:\s*", "", answer, flags=re.I)
        answer = re.sub(r"^answer\s*:\s*", "", answer, flags=re.I)
        question_text = " ".join(question.strip().split())
        if question_text and answer.lower().startswith(question_text.lower()):
            answer = answer[len(question_text):].lstrip(" :.-")
        return answer or "The image requires further visual verification."

    # ── Evidence assembly ────────────────────────────────────────────────────────

    def _build_result(self, answer: str, gen, seq, decoding: dict) -> dict:
        """
        Assemble the answer plus raw generation evidence for later calibration.

        We do NOT claim calibrated confidence. ``confidence`` is a convenience
        value derived from the mean token log-probability (a genuine model
        signal), normalised to [0, 1] via exp(); ``confidence_is_calibrated``
        is always False.
        """
        seq_score, token_logprobs = self._extract_scores(gen, seq)
        answer_length = len(answer.split())

        evidence = {
            "sequence_score": seq_score,       # mean token log-prob (or None)
            "token_logprobs": token_logprobs,  # list[float] or None
            "answer_length": answer_length,
            "decoding": decoding,
            "model_agreement": None,           # reserved for ensemble signals
        }

        # Convenience confidence from a REAL signal (not answer length).
        if seq_score is not None:
            confidence = float(np.clip(math.exp(seq_score), 0.0, 1.0))
        else:
            # No score available (e.g. some model configs) — expose 0.5 as a
            # neutral placeholder and flag it as uncalibrated + score-less.
            confidence = 0.5

        return {
            "answer": answer,
            "confidence": round(confidence, 4),
            "confidence_is_calibrated": False,
            "evidence": evidence,
        }

    def _extract_scores(self, gen, seq):
        """
        Compute mean per-token log-prob of the generated sequence.

        Returns (sequence_score, token_logprobs) where either may be None if
        the model did not return scores.
        """
        try:
            # transformers exposes .scores (tuple of [batch, vocab] logits per
            # generated step) when output_scores=True.
            scores = getattr(gen, "scores", None)
            if not scores:
                return None, None

            # Generated tokens are the tail of the sequence.
            gen_tokens = seq[0][-len(scores):]
            logprobs = []
            for step_logits, tok in zip(scores, gen_tokens):
                lp = torch.log_softmax(step_logits[0].float(), dim=-1)
                logprobs.append(float(lp[tok]))
            if not logprobs:
                return None, None
            return float(np.mean(logprobs)), [round(x, 4) for x in logprobs]
        except Exception as exc:
            logger.debug("Score extraction failed: %s", exc)
            return None, None

    @staticmethod
    def _empty_evidence() -> dict:
        return {
            "sequence_score": None,
            "token_logprobs": None,
            "answer_length": 0,
            "decoding": {},
            "model_agreement": None,
        }

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
