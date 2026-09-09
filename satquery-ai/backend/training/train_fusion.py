"""
SatQuery AI — SAR-Optical Fusion Training
=========================================
Parameter-efficient training for the multimodal fusion adapter that conditions
VQA answer generation on fused SAR + optical features.

What is trained (and what is NOT)
---------------------------------
FROZEN:
  * The BLIP vision tower (optical encoder) — reused as-is so optical tokens
    stay in BLIP's native embedding space.
  * The BLIP language model (text encoder + decoder) — untouched, unless LoRA
    is explicitly enabled on its cross-attention (see ``lora`` in the config).

TRAINED:
  * The SAR encoder (``SAREncoder``) — from scratch.
  * The ``MultimodalFusionAdapter`` — cross-attention fusion + auxiliary heads.
  * Optionally, LoRA adapters on BLIP cross-attention (via training.vqa_lora).

Objectives (combined loss)
--------------------------
  1. Contrastive modality alignment (symmetric InfoNCE) between pooled SAR and
     optical tokens — pulls paired S1/S2 representations together.
  2. Land-cover multi-label classification on the pooled FUSED tokens against
     the BigEarthNet-43 label set — grounds fusion in real semantics.
  3. (Optional) Multimodal QA loss — ONLY if a paired-QA dataset is supplied.
     Skipped entirely otherwise; nothing is fabricated.

Data
----
Uses ``BigEarthNetAdapter(use_sar=True)`` which returns paired Sentinel-1 (VV/VH)
and Sentinel-2 imagery plus the 43-label multi-hot vector, with a deterministic
MD5 80/10/10 train/val/test split — so test data never leaks into training.

Honesty
-------
This script PRODUCES a checkpoint.  Until it is actually run on data and a
checkpoint file exists, all inference stays ``fusion_trained=False``.  Running
this script is the ONLY thing that makes the fusion adapter "trained".  A 1-step
dry run (``--max-samples 4 --epochs 1``) proves the gradients flow and a
checkpoint is written; it does NOT produce a usefully-trained model, and the
script never claims otherwise.

Usage
-----
    cd satquery-ai/backend
    python training/train_fusion.py --config training/configs/fusion_bigearthnet.yaml

    # Quick dry run (proves gradients + checkpoint, NOT a trained model)
    python training/train_fusion.py \\
        --config training/configs/fusion_bigearthnet.yaml \\
        --max-samples 4 --epochs 1

Training / inference separation
--------------------------------
This file is ONLY imported by the training pipeline.  It imports the SAR encoder
and fusion adapter from ``models/sar_fusion_model.py`` (the same classes used at
inference) so the trained checkpoint loads back cleanly — but nothing in
``models/`` or ``api/`` imports this script.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

logger = logging.getLogger("satquery.train_fusion")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml")
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_samples is not None:
        cfg["datasets"]["bigearthnet"]["max_samples"] = args.max_samples
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    return cfg


# ── Loss ──────────────────────────────────────────────────────────────────────

def contrastive_loss(
    optical_features: torch.Tensor,
    sar_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE between paired optical and SAR pooled features."""
    logits = logit_scale.exp() * optical_features @ sar_features.T
    labels = torch.arange(optical_features.shape[0], device=optical_features.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


# ── Optical tokens from a frozen BLIP vision tower ──────────────────────────────

class OpticalTokenizer:
    """
    Wraps a frozen BLIP vision tower + image processor to turn RGB tensors into
    BLIP-native visual tokens ``[B, N, H]``.  Runs under no_grad — the optical
    encoder is NOT trained.
    """

    def __init__(self, vqa_model_name: str, cache_dir: Optional[str], device: torch.device):
        from transformers import BlipForQuestionAnswering, BlipProcessor
        self.device = device
        self.processor = BlipProcessor.from_pretrained(vqa_model_name, cache_dir=cache_dir)
        model = BlipForQuestionAnswering.from_pretrained(vqa_model_name, cache_dir=cache_dir)
        self.vision = model.vision_model.to(device).eval()
        self.model = model  # kept for optional LoRA on cross-attention
        for p in self.vision.parameters():
            p.requires_grad_(False)
        self.hidden = int(model.config.vision_config.hidden_size)

    @torch.no_grad()
    def __call__(self, rgb_uint8_list: List["Image.Image"]) -> torch.Tensor:
        proc = self.processor(images=rgb_uint8_list, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(self.device)
        return self.vision(pixel_values=pixel_values)[0]


# ── Collation ───────────────────────────────────────────────────────────────────

def _rgb_tensor_to_pil(img: torch.Tensor):
    """[3,H,W] float in [0,1] → PIL RGB (uint8)."""
    from PIL import Image
    arr = (img.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy())
    return Image.fromarray(arr, mode="RGB")


def _collate_fn(batch: List[Dict]) -> Dict:
    pils = [_rgb_tensor_to_pil(b["image"]) for b in batch]
    sar = torch.stack([b["metadata"]["sar_tensor"] for b in batch])       # [B, 2, H, W]
    labels = torch.stack([b["metadata"]["label_tensor"] for b in batch])  # [B, 43]
    texts = [b["text"] for b in batch]
    return {"optical_pil": pils, "sar": sar, "labels": labels, "text": texts}


def _sar_to_fusion_channels(sar_batch: torch.Tensor) -> torch.Tensor:
    """
    Convert a 2-channel VV/VH batch into the 3-channel [VV, VH, VV/VH] tensor the
    SAR encoder expects — matching the physically-meaningful preprocessing used
    at inference.  Input is already dB-normalised by the dataset adapter, so the
    ratio channel is the standardised VV-VH difference.

    The ratio channel is standardised PER SAMPLE over spatial dims (not over the
    whole tensor), so the 3-channel stack is independent of batch size: a patch
    yields the same tensor whether fed alone (evaluation, batch size 1) or inside
    a batch (training, batch size 8).  A global reduction would make each patch's
    ratio channel depend on the other patches in its batch and diverge between the
    batched training path and the per-sample eval path (evaluation/fusion/run.py).
    """
    vv, vh = sar_batch[:, 0:1], sar_batch[:, 1:2]
    ratio = vv - vh
    mu = ratio.mean(dim=(2, 3), keepdim=True)
    sd = ratio.std(dim=(2, 3), keepdim=True)
    ratio = (ratio - mu) / (sd + 1e-6)
    return torch.cat([vv, vh, ratio], dim=1)


# ── Dataset ─────────────────────────────────────────────────────────────────────

def build_datasets(cfg: Dict) -> Tuple[Any, Any]:
    from training.datasets.bigearthnet import BigEarthNetAdapter

    bc = cfg["datasets"]["bigearthnet"]
    if not bc.get("enabled", True):
        raise RuntimeError("BigEarthNet must be enabled for fusion training.")

    train_ds = BigEarthNetAdapter(
        data_dir=bc["data_dir"], split="train", use_sar=True,
        image_size=bc.get("image_size", 120), clip_preprocess=None,
        max_samples=bc.get("max_samples"),
    )
    val_ds = BigEarthNetAdapter(
        data_dir=bc["data_dir"], split="val", use_sar=True,
        image_size=bc.get("image_size", 120), clip_preprocess=None,
        max_samples=bc.get("max_samples"),
    )
    return train_ds, val_ds


# ── Train / eval ────────────────────────────────────────────────────────────────

def _run_epoch(
    adapter: nn.Module,
    optical_tok: OpticalTokenizer,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cfg: Dict,
    train: bool,
) -> Dict[str, float]:
    adapter.train(train)
    w = cfg["training"]["loss_weights"]
    w_contrast = float(w.get("contrastive", 1.0))
    w_class = float(w.get("classification", 1.0))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))

    totals = {"loss": 0.0, "contrastive": 0.0, "classification": 0.0}
    n = 0

    for batch in loader:
        opt_tokens = optical_tok(batch["optical_pil"])            # [B, N_o, H] (frozen)
        sar_in = _sar_to_fusion_channels(batch["sar"]).to(device)
        labels = batch["labels"].to(device)

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            sar_tokens = adapter.encode_sar(sar_in)                # [B, N_s, H] (trained)
            fused, _ = adapter.fuse(opt_tokens, sar_tokens)        # [B, N_o, H] (trained)

            opt_feat, sar_feat = adapter.contrastive_features(opt_tokens, sar_tokens)
            loss_contrast = contrastive_loss(opt_feat, sar_feat, adapter.logit_scale)

            logits = adapter.classify(fused)                       # [B, 43]
            loss_class = F.binary_cross_entropy_with_logits(logits, labels)

            loss = w_contrast * loss_contrast + w_class * loss_class

        if train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in adapter.parameters() if p.requires_grad], max_grad_norm
            )
            optimizer.step()
            with torch.no_grad():
                adapter.logit_scale.clamp_(0.0, 4.6052)

        totals["loss"] += float(loss.item())
        totals["contrastive"] += float(loss_contrast.item())
        totals["classification"] += float(loss_class.item())
        n += 1

    return {k: round(v / max(n, 1), 4) for k, v in totals.items()}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(adapter: nn.Module, cfg: Dict, epoch: int,
                    metrics: Dict, output_dir: str, is_best: bool) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ckpt = {
        "adapter_state_dict": adapter.state_dict(),
        "hidden": adapter.hidden,
        "epoch": epoch,
        "metrics": metrics,
        "config": cfg,
        "base_vqa_model": cfg["model"]["name"],
        # Honest provenance: this flag says a checkpoint was produced by running
        # this script — it does NOT assert the model is well-trained.
        "produced_by": "training/train_fusion.py",
    }
    path = os.path.join(output_dir, f"fusion_epoch{epoch:02d}.pt")
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s", path)
    if is_best:
        best = os.path.join(output_dir, "fusion_best.pt")
        torch.save(ckpt, best)
        logger.info("New best fusion adapter: %s", best)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — SAR-Optical Fusion Training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    logging.basicConfig(
        level=getattr(logging, cfg.get("logging", {}).get("level", "INFO").upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    torch.manual_seed(cfg["training"].get("seed", 42))
    dev_str = cfg["training"].get("device", "auto")
    device = torch.device("cuda" if (dev_str == "auto" and torch.cuda.is_available())
                          else (dev_str if dev_str != "auto" else "cpu"))
    logger.info("Device: %s", device)

    # ── Frozen optical tower + adapter ────────────────────────────────────────
    from models.sar_fusion_model import MultimodalFusionAdapter

    optical_tok = OpticalTokenizer(cfg["model"]["name"], cfg["model"].get("cache_dir"), device)
    adapter = MultimodalFusionAdapter(hidden=optical_tok.hidden).to(device)

    n_train = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    logger.info("Trainable adapter parameters: %d (BLIP vision + LM stay frozen)", n_train)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg)
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    t_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg.get("num_workers", 0), collate_fn=_collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=t_cfg["batch_size"], shuffle=False,
        num_workers=t_cfg.get("num_workers", 0), collate_fn=_collate_fn,
    )

    optimizer = AdamW(
        [p for p in adapter.parameters() if p.requires_grad],
        lr=float(t_cfg["learning_rate"]), weight_decay=float(t_cfg.get("weight_decay", 0.05)),
    )

    output_dir = t_cfg["output_dir"]
    epochs = t_cfg["epochs"]
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = _run_epoch(adapter, optical_tok, train_loader, optimizer, device, cfg, train=True)
        val_metrics = (
            _run_epoch(adapter, optical_tok, val_loader, None, device, cfg, train=False)
            if len(val_ds) > 0 else {}
        )
        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%s | %.1fs",
            epoch, epochs, train_metrics["loss"],
            val_metrics.get("loss", "n/a"), time.time() - t0,
        )

        val_loss = val_metrics.get("loss", train_metrics["loss"])
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        save_checkpoint(adapter, cfg, epoch,
                        {"train": train_metrics, "val": val_metrics},
                        output_dir, is_best)

    # ── Final metrics ─────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "fusion_training_metrics.json"), "w") as f:
        json.dump({
            "base_vqa_model": cfg["model"]["name"],
            "epochs_trained": epochs,
            "best_val_loss": best_val,
            "trainable_params": n_train,
            "note": (
                "A checkpoint was produced. This does NOT by itself certify a "
                "usefully-trained model — evaluate before trusting fused answers."
            ),
        }, f, indent=2)
    logger.info("Fusion training complete. Best val loss: %.4f", best_val)


if __name__ == "__main__":
    main()
