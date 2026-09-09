"""
SatQuery AI — Remote-Sensing VQA Fine-tuning
=============================================
Fine-tunes a BLIP / BLIP-2 VQA model on remote-sensing VQA datasets
(VRSBench, RSVQA, and any adapter registered in
``training/datasets/vqa/__init__.py``).

Capabilities
------------
  * YAML-config driven — dataset locations never hardcoded
  * LoRA / parameter-efficient fine-tuning by default (peft or manual backend)
  * Full-model training only with an explicit two-key opt-in
  * train / val / test split handling with NO test-set training
  * dynamic padding, mixed precision, gradient accumulation, gradient clipping
  * checkpoint saving + resume, validation after each epoch, early stopping
  * configurable learning rate / batch size / epochs
  * reproducible random seed
  * model versioning: checkpoint + processor + config + metrics + dataset
    version + git commit hash

Usage
-----
    cd satquery-ai/backend
    python training/train_vqa.py --config training/configs/vqa_base.yaml

    # Resume
    python training/train_vqa.py --config training/configs/vqa_base.yaml \\
        --resume checkpoints/vqa/epoch02

    # Smoke test
    python training/train_vqa.py --config training/configs/vqa_base.yaml \\
        --max-samples 32 --epochs 1

Training / inference separation
--------------------------------
Only run as a script / imported by evaluate_vqa.py.  NOT imported by
FastAPI routes or the model registry.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

# Make backend importable when run as a script.
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from training.vqa_lora import apply_vqa_lora, trainable_parameters  # noqa: E402
from training.vqa_utils import (  # noqa: E402
    VQACollator,
    aggregate_metrics,
    dataset_version,
    derive_splits,
    git_commit_hash,
    set_seed,
)

logger = logging.getLogger("satquery.train_vqa")


# ── Config ──────────────────────────────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML required: pip install pyyaml") from exc
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_cli_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.max_samples is not None:
        for ds in cfg["datasets"].values():
            ds["max_samples"] = args.max_samples
    if args.resume is not None:
        cfg.setdefault("resume", {})["from_checkpoint"] = args.resume
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = args.output_dir
    if args.device is not None:
        cfg["training"]["device"] = args.device
    return cfg


# ── Model loading ───────────────────────────────────────────────────────────────

def is_blip2(model_name: str) -> bool:
    return "blip2" in model_name.lower()


def load_model_and_processor(model_name: str, cache_dir: Optional[str], device: torch.device):
    """Load a BLIP or BLIP-2 VQA model + processor."""
    if is_blip2(model_name):
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        processor = Blip2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.float32,
        )
    else:
        from transformers import BlipForQuestionAnswering, BlipProcessor
        processor = BlipProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = BlipForQuestionAnswering.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device)
    return model, processor


# ── Dataset building ──────────────────────────────────────────────────────────

def build_split_datasets(cfg: Dict) -> Dict[str, Dataset]:
    """
    Build train/val/test datasets from config.

    For each enabled dataset we attempt to load explicit train/val/test splits
    via the adapter.  If a split yields zero samples for *every* enabled
    dataset (i.e. the dataset has no native splits), we fall back to a
    deterministic derived split of the train adapter — never touching an
    explicit test file.
    """
    from training.datasets.vqa import build_vqa_dataset

    ds_cfg = cfg["datasets"]
    enabled = {k: v for k, v in ds_cfg.items() if v.get("enabled", False)}
    if not enabled:
        raise RuntimeError(
            "No VQA datasets enabled. Set datasets.<name>.enabled=true and a "
            "valid data_dir in the config."
        )

    split_parts: Dict[str, List[Dataset]] = {"train": [], "val": [], "test": []}
    provenance: List[Dict[str, Any]] = []

    for name, dc in enabled.items():
        native_ok = False
        per_ds_splits: Dict[str, Dataset] = {}
        for split in ("train", "val", "test"):
            adapter = build_vqa_dataset(
                name,
                data_dir=dc["data_dir"],
                split=split,
                max_samples=dc.get("max_samples"),
                image_size=dc.get("image_size"),
                file_prefix=dc.get("file_prefix"),
                image_subdir=dc.get("image_subdir"),
                image_ext=dc.get("image_ext"),
            )
            per_ds_splits[split] = adapter
            if len(adapter) > 0:
                native_ok = True

        if native_ok and len(per_ds_splits["train"]) > 0 and len(per_ds_splits["val"]) > 0:
            # Dataset ships native splits — use them directly (no leakage).
            for split in ("train", "val", "test"):
                if len(per_ds_splits[split]) > 0:
                    split_parts[split].append(per_ds_splits[split])
            logger.info(
                "[%s] native splits: train=%d val=%d test=%d",
                name, len(per_ds_splits["train"]), len(per_ds_splits["val"]),
                len(per_ds_splits["test"]),
            )
        else:
            # No usable native val split — derive deterministically from train.
            base = per_ds_splits["train"]
            if len(base) == 0:
                logger.warning("[%s] produced 0 samples — skipping.", name)
                continue
            sp_cfg = cfg.get("split", {})
            derived = derive_splits(
                base,
                seed=cfg["training"].get("seed", 42),
                val_fraction=sp_cfg.get("val_fraction", 0.1),
                test_fraction=sp_cfg.get("test_fraction", 0.1),
            )
            for split in ("train", "val", "test"):
                split_parts[split].append(derived[split])
            logger.info(
                "[%s] derived splits: train=%d val=%d test=%d",
                name, len(derived["train"]), len(derived["val"]), len(derived["test"]),
            )

        provenance.append({
            "name": name,
            "data_dir": dc["data_dir"],
            "n_samples": sum(len(per_ds_splits[s]) for s in ("train", "val", "test")),
        })

    def _concat(parts: List[Dataset]) -> Optional[Dataset]:
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    result = {
        "train": _concat(split_parts["train"]),
        "val": _concat(split_parts["val"]),
        "test": _concat(split_parts["test"]),
        "_provenance": provenance,
    }
    if result["train"] is None or len(result["train"]) == 0:
        raise RuntimeError("Training split is empty. Check dataset paths.")
    return result


# ── Train / eval loops ──────────────────────────────────────────────────────────

def _resolve_amp_dtype(mixed_precision: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None  # AMP only meaningful on CUDA here
    mp = str(mixed_precision).lower()
    if mp == "fp16":
        return torch.float16
    if mp == "bf16":
        return torch.bfloat16
    return None


def train_one_epoch(
    model, loader, optimizer, scaler, scheduler, device, epoch, cfg,
) -> float:
    model.train()
    accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    log_interval = int(cfg["training"].get("log_interval", 20))
    mp_dtype = _resolve_amp_dtype(cfg["training"].get("mixed_precision", "none"), device)

    total_loss, n = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        model_inputs = _batch_to_device(batch, device)
        amp_ctx = (
            torch.autocast(device_type=device.type, dtype=mp_dtype)
            if mp_dtype is not None else nullcontext()
        )
        with amp_ctx:
            out = model(**model_inputs)
            loss = out.loss / accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accum
        n += 1

        if (step + 1) % accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ((step + 1) // accum) % log_interval == 0:
                logger.info(
                    "Epoch %d | step %d/%d | loss %.4f",
                    epoch, step + 1, len(loader), total_loss / n,
                )

    return total_loss / max(n, 1)


def _batch_to_device(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Move only tensor inputs to device (drop raw text / metadata carriers)."""
    skip = {"raw_questions", "raw_answers", "metadata"}
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
        if k not in skip and isinstance(v, torch.Tensor)
    }


@torch.no_grad()
def evaluate(model, processor, loader, device, cfg, max_samples: Optional[int] = None) -> Dict:
    """Generate answers on a loader and compute VQA metrics."""
    model.eval()
    gen_cfg = cfg.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens", 20))
    num_beams = int(gen_cfg.get("num_beams", 3))
    blip2 = is_blip2(cfg["model"]["name"])

    preds: List[str] = []
    golds: List[str] = []
    cats: List[Optional[str]] = []
    seen = 0

    for batch in loader:
        gen_inputs = {
            k: v.to(device) for k, v in batch.items()
            if k in ("pixel_values", "input_ids", "attention_mask")
            and isinstance(v, torch.Tensor)
        }
        generated = model.generate(**gen_inputs, max_new_tokens=max_new, num_beams=num_beams)
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        for i, ans in enumerate(decoded):
            preds.append(ans.strip())
            golds.append(batch["raw_answers"][i])
            cats.append(batch["metadata"][i].get("question_type"))
        seen += len(decoded)
        if max_samples and seen >= max_samples:
            break

    return aggregate_metrics(preds, golds, categories=cats)


# ── Checkpointing ───────────────────────────────────────────────────────────────

def save_checkpoint(
    model, processor, optimizer, scaler, epoch, metrics, cfg,
    lora_backend: str, provenance: List[Dict], is_best: bool,
) -> str:
    output_dir = cfg["training"]["output_dir"]
    ckpt_dir = os.path.join(output_dir, f"epoch{epoch:02d}")
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Model weights + processor (versioned artefacts).
    _save_model_weights(model, ckpt_dir, lora_backend)
    processor.save_pretrained(ckpt_dir)

    # Trainer state (for resume) — kept separate from the servable artefacts.
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "trainable_state_dict": {
                k: v.cpu() for k, v in model.state_dict().items()
                if any(k == n for n, p in model.named_parameters() if p.requires_grad)
            },
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )

    # Provenance / model version metadata.
    version_meta = {
        "base_model": cfg["model"]["name"],
        "lora_enabled": cfg["lora"].get("enabled", True),
        "lora_backend": lora_backend,
        "lora_config": cfg.get("lora", {}),
        "epoch": epoch,
        "metrics": metrics,
        "dataset_version": dataset_version(provenance),
        "datasets": provenance,
        "git_commit": git_commit_hash(),
        "config": cfg,
    }
    with open(os.path.join(ckpt_dir, "model_version.json"), "w", encoding="utf-8") as f:
        json.dump(version_meta, f, indent=2)

    logger.info("Checkpoint saved: %s", ckpt_dir)

    if is_best:
        best_dir = os.path.join(output_dir, cfg["checkpoint"].get("best_filename", "best"))
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(ckpt_dir, best_dir)
        logger.info("New best model: %s", best_dir)

    _rotate_checkpoints(output_dir, int(cfg["checkpoint"].get("keep_last", 3)))
    return ckpt_dir


def _save_model_weights(model, ckpt_dir: str, lora_backend: str) -> None:
    if lora_backend == "peft":
        # peft models expose save_pretrained → saves only adapter weights.
        model.save_pretrained(ckpt_dir)
    else:
        # Manual LoRA (or full) — save only trainable params to stay small,
        # plus a full fallback for full-finetune runs.
        trainable = {
            k: v.cpu() for k, v in model.state_dict().items()
            if any(k == n for n, p in model.named_parameters() if p.requires_grad)
        }
        torch.save(trainable, os.path.join(ckpt_dir, "adapter_weights.pt"))


def _rotate_checkpoints(output_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    import re
    pat = re.compile(r"^epoch(\d+)$")
    entries = []
    for name in os.listdir(output_dir):
        m = pat.match(name)
        full = os.path.join(output_dir, name)
        if m and os.path.isdir(full):
            entries.append((int(m.group(1)), full))
    entries.sort(key=lambda x: x[0])
    for _, path in entries[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)
        logger.debug("Removed old checkpoint: %s", path)


def maybe_resume(cfg: Dict, model, optimizer, scaler) -> int:
    """Load trainer state from a checkpoint dir; returns next epoch to run."""
    resume_path = cfg.get("resume", {}).get("from_checkpoint")
    if not resume_path or not os.path.isdir(resume_path):
        return 1
    state_file = os.path.join(resume_path, "trainer_state.pt")
    if not os.path.exists(state_file):
        logger.warning("Resume: trainer_state.pt not found in %s", resume_path)
        return 1
    logger.info("Resuming from %s", resume_path)
    state = torch.load(state_file, map_location="cpu")
    # Restore trainable weights.
    trainable_state = state.get("trainable_state_dict", {})
    if trainable_state:
        missing, unexpected = model.load_state_dict(trainable_state, strict=False)
        logger.info("Resume: loaded %d trainable tensors", len(trainable_state))
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scaler is not None and state.get("scaler_state_dict"):
        scaler.load_state_dict(state["scaler_state_dict"])
    return int(state["epoch"]) + 1


# ── Main ──────────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, name: str, epochs: int, steps_per_epoch: int, warmup_ratio: float):
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = int(total_steps * warmup_ratio)
    if name == "cosine":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)
    if name == "linear":
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            return max(0.0, (total_steps - step) / max(total_steps - warmup_steps, 1))

        return LambdaLR(optimizer, lr_lambda)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="SatQuery AI — RS VQA fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), args)

    # Logging.
    log_cfg = cfg.get("logging", {})
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_cfg.get("log_file"):
        handlers.append(logging.FileHandler(log_cfg["log_file"]))
    logging.basicConfig(
        level=getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=handlers,
    )

    t_cfg = cfg["training"]
    set_seed(t_cfg.get("seed", 42))

    device_str = t_cfg.get("device", "auto")
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else
        ("cpu" if device_str == "auto" else device_str)
    )
    logger.info("Device: %s", device)

    # ── Model + processor ──────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"].get("cache_dir")
    logger.info("Loading VQA model: %s", model_name)
    model, processor = load_model_and_processor(model_name, cache_dir, device)

    # ── Fine-tuning strategy ─────────────────────────────────────────────────────
    lora_cfg = cfg.get("lora", {})
    lora_backend = "none"
    if lora_cfg.get("enabled", True):
        model, lora_backend = apply_vqa_lora(model, lora_cfg)
        model = model.to(device)
    else:
        if not t_cfg.get("allow_full_finetune", False):
            raise RuntimeError(
                "Full-model fine-tuning requested (lora.enabled=false) but "
                "training.allow_full_finetune is not true. Refusing to run full "
                "fine-tuning without the explicit two-key opt-in."
            )
        logger.warning("FULL fine-tuning enabled — all parameters trainable.")
        for p in model.parameters():
            p.requires_grad_(True)
        lora_backend = "full"

    n_trainable = sum(p.numel() for p in trainable_parameters(model))
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %d / %d (%.3f%%)",
        n_trainable, n_total, 100.0 * n_trainable / max(n_total, 1),
    )

    # ── Datasets ─────────────────────────────────────────────────────────────────
    splits = build_split_datasets(cfg)
    provenance = splits["_provenance"]
    logger.info(
        "Samples — train=%d val=%d test=%d",
        len(splits["train"]),
        len(splits["val"]) if splits["val"] else 0,
        len(splits["test"]) if splits["test"] else 0,
    )

    max_len = int(t_cfg.get("max_text_length", 64))
    train_collate = VQACollator(processor, max_length=max_len, train=True)
    eval_collate = VQACollator(processor, max_length=max_len, train=False)

    num_workers = int(t_cfg.get("num_workers", 2))
    pin = bool(t_cfg.get("pin_memory", True)) and device.type == "cuda"

    train_loader = DataLoader(
        splits["train"], batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=True,
        collate_fn=train_collate,
    )
    val_loader = None
    if splits["val"] and len(splits["val"]) > 0:
        val_loader = DataLoader(
            splits["val"], batch_size=t_cfg["batch_size"], shuffle=False,
            num_workers=num_workers, pin_memory=pin, collate_fn=eval_collate,
        )

    # ── Optimiser / scheduler / scaler ───────────────────────────────────────────
    lr = float(t_cfg["learning_rate"])
    wd = float(t_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(trainable_parameters(model), lr=lr, weight_decay=wd)

    epochs = int(t_cfg["epochs"])
    accum = int(t_cfg.get("gradient_accumulation_steps", 1))
    steps_per_epoch = max(len(train_loader) // accum, 1)
    scheduler = build_scheduler(
        optimizer, t_cfg.get("lr_scheduler", "cosine"),
        epochs, steps_per_epoch, float(t_cfg.get("warmup_ratio", 0.0)),
    )

    mp_dtype = _resolve_amp_dtype(t_cfg.get("mixed_precision", "none"), device)
    scaler = (
        torch.cuda.amp.GradScaler()
        if (mp_dtype == torch.float16 and device.type == "cuda") else None
    )

    # ── Resume ───────────────────────────────────────────────────────────────────
    start_epoch = maybe_resume(cfg, model, optimizer, scaler)

    # ── Early stopping ────────────────────────────────────────────────────────────
    es_cfg = cfg.get("early_stopping", {})
    monitor = es_cfg.get("monitor", "vqa_accuracy")
    es_mode = es_cfg.get("mode", "max")
    patience = int(es_cfg.get("patience", 3))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    best_metric = -float("inf") if es_mode == "max" else float("inf")
    no_improve = 0

    eval_interval = int(t_cfg.get("eval_interval", 1))
    max_val = cfg.get("evaluation", {}).get("max_val_samples")

    # ── Training loop ──────────────────────────────────────────────────────────
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        last_epoch = epoch
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler, device, epoch, cfg
        )
        logger.info(
            "Epoch %d/%d | train_loss %.4f | %.1fs",
            epoch, epochs, train_loss, time.time() - t0,
        )

        metrics: Dict[str, Any] = {"train_loss": round(train_loss, 4)}
        monitor_val = None
        if val_loader is not None and epoch % eval_interval == 0:
            val_metrics = evaluate(model, processor, val_loader, device, cfg, max_val)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()
                            if not isinstance(v, dict)})
            metrics["val_detail"] = val_metrics
            monitor_val = val_metrics.get(monitor)
            logger.info(
                "Epoch %d validation | EM=%.4f VQAacc=%.4f F1=%.4f",
                epoch, val_metrics["exact_match"], val_metrics["vqa_accuracy"],
                val_metrics["token_f1"],
            )

        # Best-model tracking.
        is_best = False
        if monitor_val is not None:
            improved = (
                monitor_val > best_metric + min_delta if es_mode == "max"
                else monitor_val < best_metric - min_delta
            )
            if improved:
                best_metric = monitor_val
                is_best = True
                no_improve = 0
            else:
                no_improve += 1

        if epoch % int(cfg["checkpoint"].get("save_every", 1)) == 0 or is_best:
            save_checkpoint(
                model, processor, optimizer, scaler, epoch, metrics, cfg,
                lora_backend, provenance, is_best,
            )

        if es_cfg.get("enabled", False) and monitor_val is not None and no_improve >= patience:
            logger.info("Early stopping at epoch %d (no improvement in %d epochs)",
                        epoch, patience)
            break

    logger.info("Training complete. Best %s = %.4f", monitor, best_metric)

    # Final training summary.
    summary = {
        "base_model": model_name,
        "lora_backend": lora_backend,
        "epochs_trained": last_epoch,
        "best_metric": {monitor: best_metric},
        "dataset_version": dataset_version(provenance),
        "git_commit": git_commit_hash(),
    }
    out_dir = t_cfg["output_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary written to %s", os.path.join(out_dir, "training_summary.json"))


if __name__ == "__main__":
    main()
