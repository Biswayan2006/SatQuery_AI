"""
SatQuery AI — VQA LoRA Adapters
================================
Parameter-efficient fine-tuning for BLIP / BLIP-2 VQA models.

Two backends are supported, tried in order:

  1. **peft** (preferred) — HuggingFace PEFT LoRA, if installed.
  2. **manual** (fallback) — a self-contained LoRALinear wrapper that requires
     no extra dependency.  Same math as train_clip.py's LoRALinear.

The public entry point is :func:`apply_vqa_lora`, which returns
``(model, backend_name)``.  The chosen backend is recorded in the checkpoint so
inference can reconstruct the adapters correctly.

Training / inference separation
--------------------------------
Imported by training scripts AND by the inference wrapper (models/vqa_model.py)
purely for the *manual* adapter classes needed to rebuild a fine-tuned model.
It does NOT import from models/ or api/, so no cycle is created.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("satquery.vqa_lora")


# ── Manual LoRA ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper around a frozen nn.Linear.

        W_eff = W_frozen + (B @ A) * (alpha / rank)

    Only A and B are trainable; the wrapped Linear (incl. bias) is frozen.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / rank

        in_f = linear.in_features
        out_f = linear.out_features
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + delta * self.scaling


def _apply_manual_lora(model: nn.Module, cfg: Dict) -> nn.Module:
    """Replace matching Linear layers with LoRALinear wrappers (substring match)."""
    rank = int(cfg.get("rank", 8))
    alpha = float(cfg.get("alpha", 16))
    dropout = float(cfg.get("dropout", 0.0))
    patterns: List[str] = cfg.get("target_modules", ["query", "value"])

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Match on the leaf attribute name OR full path substring.
        leaf = name.split(".")[-1]
        if not (any(p in leaf for p in patterns) or any(p in name for p in patterns)):
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout))
        replaced += 1

    if replaced == 0:
        raise RuntimeError(
            "Manual LoRA replaced 0 layers. Check lora.target_modules against "
            "the model's Linear layer names."
        )
    logger.info(
        "Manual LoRA: replaced %d Linear layers (rank=%d, alpha=%.0f)",
        replaced, rank, alpha,
    )
    return model


# ── peft LoRA ────────────────────────────────────────────────────────────────

def _apply_peft_lora(model: nn.Module, cfg: Dict):
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore

    lora_config = LoraConfig(
        r=int(cfg.get("rank", 8)),
        lora_alpha=int(cfg.get("alpha", 16)),
        lora_dropout=float(cfg.get("dropout", 0.0)),
        target_modules=cfg.get("target_modules", ["query", "value"]),
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = _count_params(peft_model)
    logger.info(
        "peft LoRA applied: trainable=%d / total=%d (%.3f%%)",
        trainable, total, 100.0 * trainable / max(total, 1),
    )
    return peft_model


# ── Public entry point ──────────────────────────────────────────────────────────

def apply_vqa_lora(
    model: nn.Module,
    cfg: Dict,
    prefer_peft: bool = True,
) -> Tuple[nn.Module, str]:
    """
    Apply LoRA adapters to a VQA model.

    Returns
    -------
    (model, backend)  where backend is "peft" or "manual".
    """
    # Freeze all base parameters first; adapters unfreeze their own.
    for p in model.parameters():
        p.requires_grad_(False)

    if prefer_peft:
        try:
            import peft  # noqa: F401
            return _apply_peft_lora(model, cfg), "peft"
        except ImportError:
            logger.info("peft not installed — falling back to manual LoRA adapters.")

    return _apply_manual_lora(model, cfg), "manual"


def rebuild_manual_lora(model: nn.Module, cfg: Dict) -> nn.Module:
    """
    Re-insert manual LoRA layers into a fresh base model so a saved manual-LoRA
    state dict can be loaded.  Used at inference time.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    return _apply_manual_lora(model, cfg)


def trainable_parameters(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
