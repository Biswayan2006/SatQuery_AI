"""
SatQuery AI — RS-CLIP Fine-tuning Script
========================================
Refactored from training/finetune_clip.py.

New capabilities over the original:
  - YAML config file driven (no more argparse sprawl)
  - Mixed-precision training (fp16 / bf16 via torch.amp)
  - Gradient accumulation
  - LoRA-style parameter-efficient adapters on ViT attention layers
  - Checkpoint resume
  - Early stopping
  - Multiple dataset mixing (BigEarthNet + VRSBench)
  - Saves model card metadata alongside checkpoint

Usage
-----
    cd satquery-ai/backend
    python training/train_clip.py --config training/configs/rs_clip_base.yaml

    # Resume from a checkpoint
    python training/train_clip.py \\
        --config training/configs/rs_clip_base.yaml \\
        --resume checkpoints/rs_clip/rs_clip_epoch05.pt

    # Quick smoke test (small data)
    python training/train_clip.py \\
        --config training/configs/rs_clip_base.yaml \\
        --max-samples 128 --epochs 2

Training / inference separation
---------------------------------
This file is ONLY imported by the training pipeline.
It is NOT imported by FastAPI routes or models/.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset

logger = logging.getLogger("satquery.train_clip")

# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML training config. Requires PyYAML."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for config loading. "
            "Install with: pip install pyyaml"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    """Apply CLI overrides on top of YAML config."""
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_samples is not None:
        for ds in cfg["datasets"].values():
            ds["max_samples"] = args.max_samples
    if args.resume is not None:
        cfg["resume"]["from_checkpoint"] = args.resume
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    return cfg


# ── LoRA implementation ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation of a frozen nn.Linear layer.

    During forward:  W_eff = W_frozen + (B @ A) * (alpha / rank)
    Only A and B are trainable.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features
        self.linear = linear              # frozen original weights
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        delta = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + delta * self.scaling


def apply_lora(model: nn.Module, cfg: Dict) -> nn.Module:
    """
    Replace matching Linear sub-modules with LoRALinear wrappers.
    Patterns in cfg["lora"]["target_modules"] are matched as substrings
    against the full dotted parameter path.
    """
    rank = cfg.get("rank", 8)
    alpha = float(cfg.get("alpha", 16))
    dropout = float(cfg.get("dropout", 0.0))
    patterns: List[str] = cfg.get("target_modules", ["attn"])

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(pat in name for pat in patterns):
            continue

        # Navigate to parent and replace
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        original = getattr(parent, child_name)
        setattr(parent, child_name, LoRALinear(original, rank, alpha, dropout))
        replaced += 1

    logger.info("LoRA: replaced %d Linear layers (rank=%d, alpha=%.0f)", replaced, rank, alpha)
    return model


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only the trainable (non-frozen) parameters."""
    return [p for p in model.parameters() if p.requires_grad]


# ── Loss ──────────────────────────────────────────────────────────────────────

def contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE / CLIP contrastive loss.
    image_features, text_features: [B, D] L2-normalised.
    """
    logits_per_image = logit_scale.exp() * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(image_features.shape[0], device=image_features.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    tokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    epoch: int,
    cfg: Dict,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    accum_steps: int = cfg["training"].get("gradient_accumulation_steps", 1)
    logit_scale_max: float = cfg["training"].get("logit_scale_max", 4.6052)
    log_interval: int = cfg["training"].get("log_interval", 20)
    max_grad_norm: float = cfg["training"].get("max_grad_norm", 1.0)
    mp_dtype = _resolve_amp_dtype(cfg["training"].get("mixed_precision", "none"))

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        texts: List[str] = batch["text"]
        text_tokens = tokenizer(texts).to(device)

        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=mp_dtype)
            if mp_dtype is not None
            else nullcontext()
        )

        with amp_ctx:
            image_features = F.normalize(model.encode_image(images), dim=-1)
            text_features = F.normalize(model.encode_text(text_tokens), dim=-1)
            loss = contrastive_loss(image_features, text_features, model.logit_scale)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum_steps
        n_batches += 1

        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(get_trainable_params(model), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(get_trainable_params(model), max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                model.logit_scale.clamp_(0.0, logit_scale_max)

            optimizer.zero_grad()

            if ((step + 1) // accum_steps) % log_interval == 0:
                avg = total_loss / n_batches
                logger.info(
                    "Epoch %d | Step %d/%d | Loss: %.4f",
                    epoch, step + 1, len(loader), avg,
                )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    recall_at_k: List[int] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Image→text retrieval metrics (R@k).
    Returns a dict like {"R@1": 45.2, "R@5": 72.1, "R@10": 83.0}.
    """
    model.eval()
    all_img: List[torch.Tensor] = []
    all_txt: List[torch.Tensor] = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        texts: List[str] = batch["text"]
        text_tokens = tokenizer(texts).to(device)

        img_f = F.normalize(model.encode_image(images), dim=-1)
        txt_f = F.normalize(model.encode_text(text_tokens), dim=-1)
        all_img.append(img_f.cpu())
        all_txt.append(txt_f.cpu())

    img_feats = torch.cat(all_img)    # [N, D]
    txt_feats = torch.cat(all_txt)    # [N, D]
    sims = img_feats @ txt_feats.T    # [N, N]
    n = sims.shape[0]
    labels = torch.arange(n)

    metrics: Dict[str, float] = {}
    for k in recall_at_k:
        k_eff = min(k, n)
        top_k = sims.topk(k_eff, dim=1).indices
        hit = (top_k == labels.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"R@{k}"] = round(hit * 100, 2)

    logger.info("Validation: %s", metrics)
    return metrics


# ── Early stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = (
            (value > self.best + self.min_delta) if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement "
                "(best=%.4f, last=%.4f)",
                self.patience, self.best, value,
            )
            return True
        return False


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    metrics: Dict,
    cfg: Dict,
    output_dir: str,
    is_best: bool,
    extra_meta: Optional[Dict] = None,
) -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save only LoRA/trainable parameters to keep checkpoint small
    trainable_state = {
        k: v for k, v in model.state_dict().items()
        if any(k == n for n, p in model.named_parameters() if p.requires_grad)
    }

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics,
        "config": cfg,
        "model_name": cfg["model"]["name"],
        "pretrained": cfg["model"]["pretrained"],
        "lora_enabled": cfg["lora"].get("enabled", False),
        **(extra_meta or {}),
    }

    filename = f"rs_clip_epoch{epoch:02d}.pt"
    path = os.path.join(output_dir, filename)
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s", path)

    if is_best:
        best_path = os.path.join(output_dir, cfg["checkpoint"]["best_filename"])
        shutil.copy2(path, best_path)
        logger.info("New best model: %s", best_path)

    # Rotate old checkpoints
    keep_last: int = cfg["checkpoint"].get("keep_last", 3)
    if keep_last > 0:
        _rotate_checkpoints(output_dir, keep_last)

    return path


def _rotate_checkpoints(output_dir: str, keep_last: int) -> None:
    """Delete oldest epoch checkpoints keeping only `keep_last`."""
    pattern = re.compile(r"rs_clip_epoch(\d+)\.pt$")
    ckpts = sorted(
        [(int(pattern.search(f).group(1)), os.path.join(output_dir, f))
         for f in os.listdir(output_dir) if pattern.search(f)],
        key=lambda x: x[0],
    )
    for _, path in ckpts[:-keep_last]:
        os.remove(path)
        logger.debug("Removed old checkpoint: %s", path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler) -> int:
    """Load checkpoint and return the epoch to resume from."""
    logger.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    logger.info("Resumed from epoch %d", ckpt["epoch"])
    return start_epoch


# ── Dataset building ──────────────────────────────────────────────────────────

def build_datasets(cfg: Dict, preprocess) -> Tuple[Any, Any]:
    """
    Build train and val datasets from config.
    Returns (train_dataset, val_dataset).
    Mixing: if multiple datasets enabled, ConcatDataset with even mixing.
    """
    from training.datasets.bigearthnet import BigEarthNetAdapter
    from training.datasets.vrsbench import VRSBenchAdapter

    train_parts, val_parts = [], []
    ds_cfg = cfg["datasets"]

    if ds_cfg.get("bigearthnet", {}).get("enabled", False):
        bc = ds_cfg["bigearthnet"]
        for split, parts_list in (("train", train_parts), ("val", val_parts)):
            parts_list.append(BigEarthNetAdapter(
                data_dir=bc["data_dir"],
                split=split,
                use_sar=bc.get("use_sar", False),
                image_size=bc.get("image_size", 120),
                clip_preprocess=preprocess,
                max_samples=bc.get("max_samples"),
            ))
        logger.info("BigEarthNet dataset enabled (dir=%s)", bc["data_dir"])

    if ds_cfg.get("vrsbench", {}).get("enabled", False):
        vc = ds_cfg["vrsbench"]
        for split, parts_list in (("train", train_parts), ("val", val_parts)):
            parts_list.append(VRSBenchAdapter(
                data_dir=vc["data_dir"],
                split=split,
                clip_preprocess=preprocess,
                image_size=vc.get("image_size", 224),
                max_samples=vc.get("max_samples"),
            ))
        logger.info("VRSBench dataset enabled (dir=%s)", vc["data_dir"])

    if not train_parts:
        raise RuntimeError(
            "No datasets enabled. Set datasets.bigearthnet.enabled or "
            "datasets.vrsbench.enabled to true in the config."
        )

    train_ds = ConcatDataset(train_parts) if len(train_parts) > 1 else train_parts[0]
    val_ds = ConcatDataset(val_parts) if len(val_parts) > 1 else val_parts[0]
    return train_ds, val_ds


def _collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of sample dicts into a batch dict."""
    images = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    metadata = [b["metadata"] for b in batch]
    return {"image": images, "text": texts, "metadata": metadata}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_amp_dtype(mixed_precision: str) -> Optional[torch.dtype]:
    mp = mixed_precision.lower()
    if mp == "fp16":
        return torch.float16
    if mp == "bf16":
        return torch.bfloat16
    return None


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — RS-CLIP Fine-tuning")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--output-dir", default=None, help="Override checkpoint output dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args)

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level = cfg.get("logging", {}).get("level", "INFO")
    log_file = cfg.get("logging", {}).get("log_file")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
    )

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)

    device = _resolve_device(cfg["training"].get("device", "auto"))
    logger.info("Device: %s", device)

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip_torch is required. pip install open-clip-torch")
        sys.exit(1)

    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]
    logger.info("Loading OpenCLIP %s (pretrained=%s)", model_name, pretrained)

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", False):
        logger.info("Applying LoRA adapters …")
        # Freeze everything first, then apply LoRA (which unfreezes its own params)
        for p in model.parameters():
            p.requires_grad_(False)
        model = apply_lora(model, lora_cfg)
    else:
        logger.info("Full fine-tuning (LoRA disabled)")

    trainable = get_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    logger.info(
        "Parameters: total=%d, trainable=%d (%.2f%%)",
        total_params, trainable_params, 100.0 * trainable_params / max(total_params, 1),
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds, val_ds = build_datasets(cfg, preprocess)
    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    t_cfg = cfg["training"]
    batch_size = t_cfg["batch_size"]
    num_workers = t_cfg.get("num_workers", 4)
    pin_memory = t_cfg.get("pin_memory", True) and (device.type == "cuda")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=_collate_fn,
    )

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    lr = float(t_cfg["learning_rate"])
    wd = float(t_cfg.get("weight_decay", 0.1))
    optimizer = AdamW(trainable, lr=lr, weight_decay=wd)

    epochs = t_cfg["epochs"]
    scheduler_name = t_cfg.get("lr_scheduler", "cosine")
    if scheduler_name == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    # ── AMP scaler ────────────────────────────────────────────────────────────
    mp_dtype = _resolve_amp_dtype(t_cfg.get("mixed_precision", "none"))
    scaler = (
        torch.cuda.amp.GradScaler()
        if (mp_dtype == torch.float16 and device.type == "cuda")
        else None
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    resume_path = cfg["resume"].get("from_checkpoint")
    if resume_path and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, scaler)

    # ── Early stopping ────────────────────────────────────────────────────────
    es_cfg = cfg.get("early_stopping", {})
    early_stop = (
        EarlyStopping(
            patience=es_cfg.get("patience", 4),
            mode=es_cfg.get("mode", "max"),
            min_delta=float(es_cfg.get("min_delta", 0.001)),
        )
        if es_cfg.get("enabled", False)
        else None
    )

    output_dir = t_cfg["output_dir"]
    recall_at_k: List[int] = cfg["evaluation"].get("recall_at_k", [1, 5, 10])
    eval_interval: int = t_cfg.get("eval_interval", 1)
    save_every: int = cfg["checkpoint"].get("save_every", 1)
    monitor_key = es_cfg.get("monitor", "R@1")

    best_metric = -float("inf") if es_cfg.get("mode", "max") == "max" else float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, tokenizer, train_loader, optimizer, scaler, device, epoch, cfg
        )
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Time: %.1fs",
            epoch, epochs, train_loss, elapsed,
        )

        # Validation
        val_metrics: Dict = {"train_loss": round(train_loss, 4)}
        if epoch % eval_interval == 0:
            vm = evaluate(model, tokenizer, val_loader, device, recall_at_k)
            val_metrics.update({"val_" + k.replace("@", ""): v for k, v in vm.items()})
            # Also keep original keys for monitor lookup
            val_metrics.update({k: v for k, v in vm.items()})

        monitor_val = val_metrics.get(monitor_key, val_metrics.get("val_r1", 0.0))
        is_best = (
            (monitor_val > best_metric) if es_cfg.get("mode", "max") == "max"
            else (monitor_val < best_metric)
        )
        if is_best:
            best_metric = monitor_val

        # Checkpoint
        if epoch % save_every == 0 or is_best:
            save_checkpoint(
                model, optimizer, scaler, epoch, val_metrics, cfg,
                output_dir, is_best,
                extra_meta={"train_loss": train_loss},
            )

        # Early stopping check
        if early_stop is not None and epoch % eval_interval == 0:
            if early_stop.step(monitor_val):
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training complete. Best %s: %.2f%%", monitor_key, best_metric)

    # Save final metrics alongside config
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model_name": cfg["model"]["name"],
                "pretrained": cfg["model"]["pretrained"],
                "lora_enabled": cfg["lora"].get("enabled", False),
                "lora_rank": cfg["lora"].get("rank"),
                "epochs_trained": epoch,
                "best_metric": {monitor_key: best_metric},
                "config": cfg,
            },
            f, indent=2,
        )
    logger.info("Metrics saved: %s", metrics_path)


if __name__ == "__main__":
    main()
