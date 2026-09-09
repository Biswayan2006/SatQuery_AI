"""
SatQuery AI — RS-CLIP Inference Encoder
========================================
Clean, self-contained inference interface for the fine-tuned RS-CLIP model.

This module is the ONLY entry-point for inference.
It does NOT import anything from training/ — the training/inference
boundary is enforced by design.

Public API
----------
::

    encoder = RSCLIPEncoder.from_pretrained(
        model_name="ViT-B-32",
        pretrained="openai",           # base weights tag
        checkpoint_path="./checkpoints/rs_clip/rs_clip_best.pt",  # optional
        device="cpu",
        cache_dir="./model_cache",
    )

    # Single PIL image → normalised embedding [D]
    emb = encoder.encode_image(pil_image)

    # String or list of strings → normalised embedding [D] or [N, D]
    emb = encoder.encode_text("a satellite image showing urban areas")

    # Cosine similarity ∈ [−1, 1] (rescaled to [0, 1] for convenience)
    score = encoder.image_text_similarity(pil_image, "flood damage")

    # Batch similarity matrix
    mat = encoder.batch_similarity(list_of_pil, list_of_texts)  # [N, M]

Model card
----------
See ``models/rs_clip/MODEL_CARD.md`` for base model, training details,
and limitations.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger("satquery.rs_clip.encoder")

# Default base model — matches training/configs/rs_clip_base.yaml
_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "openai"


class RSCLIPEncoder:
    """
    Remote-Sensing CLIP inference encoder.

    Wraps an OpenCLIP model (optionally with LoRA adapters loaded from a
    fine-tuned checkpoint) and exposes a simple encode / similarity API.

    Attributes
    ----------
    model_name : str
    embed_dim  : int  — embedding dimension (auto-detected after load)
    device     : str
    is_finetuned : bool — True if a RS checkpoint was loaded
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        preprocess,
        device: str = "cpu",
        model_name: str = _DEFAULT_MODEL,
        is_finetuned: bool = False,
    ):
        self._model = model.to(device).eval()
        self._tokenizer = tokenizer
        self._preprocess = preprocess
        self.device = device
        self.model_name = model_name
        self.is_finetuned = is_finetuned
        self.embed_dim: int = self._detect_embed_dim()

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = _DEFAULT_MODEL,
        pretrained: str = _DEFAULT_PRETRAINED,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
    ) -> "RSCLIPEncoder":
        """
        Load the encoder.

        Parameters
        ----------
        model_name : str
            Any model name accepted by ``open_clip.create_model_and_transforms``.
        pretrained : str
            Base pretrained weights tag (e.g. "openai", "laion400m_e32").
        checkpoint_path : str or None
            Path to a fine-tuned RS-CLIP ``.pt`` checkpoint produced by
            ``training/train_clip.py``.  When None the base OpenCLIP weights
            are used (still useful for semantic similarity without RS adaptation).
        device : str
            "cpu" | "cuda" | "cuda:0" etc.
        cache_dir : str or None
            Directory for caching HuggingFace / OpenCLIP downloads.
        """
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open-clip-torch is required for RSCLIPEncoder. "
                "Install with: pip install open-clip-torch"
            ) from exc

        if cache_dir:
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", cache_dir)

        logger.info(
            "Loading RSCLIPEncoder: model=%s, pretrained=%s, checkpoint=%s, device=%s",
            model_name, pretrained, checkpoint_path or "none (base weights)", device,
        )

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        is_finetuned = False

        if checkpoint_path and os.path.exists(checkpoint_path):
            is_finetuned = cls._load_checkpoint(model, checkpoint_path, device)
        elif checkpoint_path:
            logger.warning(
                "RS-CLIP checkpoint not found at '%s' — using base weights only.",
                checkpoint_path,
            )

        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocess=preprocess,
            device=device,
            model_name=model_name,
            is_finetuned=is_finetuned,
        )

    @staticmethod
    def _load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        device: str,
    ) -> bool:
        """
        Load fine-tuned weights (with optional LoRA adapters) from a checkpoint.
        Returns True if loading succeeded.
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            lora_enabled = ckpt.get("lora_enabled", False)

            if lora_enabled:
                # LoRA layers must be inserted before loading state dict
                lora_cfg = ckpt.get("config", {}).get("lora", {})
                if lora_cfg:
                    # Import only the LoRA machinery, NOT the full training module
                    from training.train_clip import apply_lora
                    for p in model.parameters():
                        p.requires_grad_(False)
                    model = apply_lora(model, lora_cfg)

            model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            metrics = ckpt.get("metrics", {})
            logger.info(
                "RS-CLIP checkpoint loaded: epoch=%s, metrics=%s",
                epoch, metrics,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to load RS-CLIP checkpoint: %s", exc)
            return False

    # ── Embedding API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        """
        Encode one or more PIL images to L2-normalised embeddings.

        Parameters
        ----------
        image : PIL.Image or list of PIL.Image

        Returns
        -------
        np.ndarray
            Shape [D] for a single image, [N, D] for a list.
        """
        single = isinstance(image, Image.Image)
        images = [image] if single else image

        tensors = torch.stack([
            self._preprocess(img.convert("RGB"))
            for img in images
        ]).to(self.device)

        feats = self._model.encode_image(tensors)
        feats = F.normalize(feats, dim=-1)
        result = feats.cpu().numpy()
        return result[0] if single else result

    @torch.no_grad()
    def encode_text(
        self,
        text: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Encode one or more text strings to L2-normalised embeddings.

        Parameters
        ----------
        text : str or list of str

        Returns
        -------
        np.ndarray
            Shape [D] for a single string, [N, D] for a list.
        """
        single = isinstance(text, str)
        texts = [text] if single else text

        tokens = self._tokenizer(texts).to(self.device)
        feats = self._model.encode_text(tokens)
        feats = F.normalize(feats, dim=-1)
        result = feats.cpu().numpy()
        return result[0] if single else result

    def image_text_similarity(
        self,
        image: Image.Image,
        text: str,
    ) -> float:
        """
        Cosine similarity between one image and one text string.

        Returns a float in [0, 1] (cosine similarity rescaled from [-1,1]).
        Since both embeddings are L2-normalised their dot product is already
        the cosine similarity ∈ [-1, 1]; we rescale to [0, 1] for convenience.
        """
        img_emb = self.encode_image(image)      # [D]
        txt_emb = self.encode_text(text)         # [D]
        cosine = float(np.dot(img_emb, txt_emb))
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    @torch.no_grad()
    def batch_similarity(
        self,
        images: List[Image.Image],
        texts: List[str],
    ) -> np.ndarray:
        """
        Compute a similarity matrix between N images and M texts.

        Returns
        -------
        np.ndarray  shape [N, M], values ∈ [-1, 1]
        """
        img_embs = self.encode_image(images)   # [N, D]
        txt_embs = self.encode_text(texts)     # [M, D]
        return img_embs @ txt_embs.T           # [N, M]

    @torch.no_grad()
    def rank_texts_for_image(
        self,
        image: Image.Image,
        texts: List[str],
    ) -> List[tuple]:
        """
        Rank a list of texts by their similarity to an image.

        Returns
        -------
        list of (text, score) sorted by descending score.
        """
        img_emb = self.encode_image(image)     # [D]
        txt_embs = self.encode_text(texts)     # [N, D]
        scores = txt_embs @ img_emb            # [N]
        ranked = sorted(zip(texts, scores.tolist()), key=lambda x: -x[1])
        return ranked

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_embed_dim(self) -> int:
        """Run a dummy forward pass to detect the embedding dimension."""
        try:
            import open_clip
            dummy_img = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                feat = self._model.encode_image(dummy_img)
            return feat.shape[-1]
        except Exception:
            return 512   # ViT-B-32 default

    def __repr__(self) -> str:
        status = "fine-tuned" if self.is_finetuned else "base (not fine-tuned)"
        return (
            f"RSCLIPEncoder("
            f"model={self.model_name}, "
            f"embed_dim={self.embed_dim}, "
            f"device={self.device}, "
            f"status={status})"
        )
