"""
SatQuery AI — CLIP Fine-tuning on BigEarthNet
Adapts OpenCLIP for remote sensing scene understanding via contrastive learning.

Usage:
  python training/finetune_clip.py \\
    --data-dir /data/BigEarthNet \\
    --output-dir ./checkpoints \\
    --epochs 10 \\
    --batch-size 64 \\
    --model-name ViT-B-32
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

logger = logging.getLogger("satquery.finetune")

# ── Loss ──────────────────────────────────────────────────────────────────────

def contrastive_loss(image_features: torch.Tensor, text_features: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE / CLIP contrastive loss.
    image_features, text_features: [B, D] — L2 normalised
    logit_scale: learned temperature parameter
    """
    logits_per_image = logit_scale.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    return (loss_i + loss_t) / 2


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    tokenizer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_interval: int = 20,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(loader):
        optical, sar, labels, descriptions = batch

        # Use RGB composite (bands 3, 2, 1) as model input
        # optical: [B, 12, H, W] → take B04, B03, B02
        rgb = optical[:, [3, 2, 1], :, :]  # [B, 3, H, W]

        # Normalize to [0,1] from S2 standardised range
        rgb = torch.clamp((rgb * 0.2 + 0.5), 0.0, 1.0)
        rgb = rgb.to(device)

        text_tokens = tokenizer(list(descriptions)).to(device)

        optimizer.zero_grad()

        image_features = model.encode_image(rgb)
        text_features = model.encode_text(text_tokens)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        logit_scale = model.logit_scale
        loss = contrastive_loss(image_features, text_features, logit_scale)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Clamp logit_scale
        with torch.no_grad():
            model.logit_scale.clamp_(0.0, 4.6052)  # max ln(100)

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % log_interval == 0:
            avg = total_loss / n_batches
            logger.info("Epoch %d | Step %d/%d | Loss: %.4f", epoch, step + 1, len(loader), avg)

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Zero-shot evaluation: image→text retrieval R@1, R@5."""
    model.eval()
    all_img_feats = []
    all_txt_feats = []

    for batch in loader:
        optical, _, labels, descriptions = batch
        rgb = optical[:, [3, 2, 1], :, :]
        rgb = torch.clamp((rgb * 0.2 + 0.5), 0.0, 1.0).to(device)
        text_tokens = tokenizer(list(descriptions)).to(device)

        img_f = F.normalize(model.encode_image(rgb), dim=-1)
        txt_f = F.normalize(model.encode_text(text_tokens), dim=-1)

        all_img_feats.append(img_f.cpu())
        all_txt_feats.append(txt_f.cpu())

    img_feats = torch.cat(all_img_feats)
    txt_feats = torch.cat(all_txt_feats)

    # Compute similarity matrix
    sims = img_feats @ txt_feats.t()  # [N, N]
    n = sims.shape[0]
    labels = torch.arange(n)

    r1 = (sims.topk(1, dim=1).indices.squeeze(1) == labels).float().mean().item()
    r5 = (sims.topk(min(5, n), dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    return {"R@1": round(r1 * 100, 2), "R@5": round(r5 * 100, 2)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP on BigEarthNet")
    parser.add_argument("--data-dir", required=True, help="Path to BigEarthNet root")
    parser.add_argument("--output-dir", default="./checkpoints", help="Checkpoint save dir")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-name", default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained weights")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Load OpenCLIP
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip_torch is required. Install with: pip install open-clip-torch")
        sys.exit(1)

    logger.info("Loading OpenCLIP %s (pretrained=%s)", args.model_name, args.pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model = model.to(device)

    # Dataset
    from training.bigearthnet_dataset import BigEarthNetDataset

    train_ds = BigEarthNetDataset(
        data_dir=args.data_dir,
        split="train",
        use_sar=False,  # VLP on optical only for CLIP
        max_samples=args.max_samples,
    )
    val_ds = BigEarthNetDataset(
        data_dir=args.data_dir,
        split="val",
        use_sar=False,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info("Train samples: %d | Val samples: %d", len(train_ds), len(val_ds))

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    best_r1 = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, tokenizer, train_loader, optimizer, device, epoch, args.log_interval
        )
        scheduler.step()

        val_metrics = evaluate(model, tokenizer, val_loader, device)
        elapsed = time.time() - t0

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val R@1: %.2f%% | R@5: %.2f%% | Time: %.1fs",
            epoch, args.epochs, train_loss,
            val_metrics["R@1"], val_metrics["R@5"], elapsed,
        )

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"clip_bigearthnet_epoch{epoch:02d}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_r1": val_metrics["R@1"],
            "model_name": args.model_name,
            "pretrained": args.pretrained,
        }, ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)

        if val_metrics["R@1"] > best_r1:
            best_r1 = val_metrics["R@1"]
            best_path = os.path.join(args.output_dir, "clip_bigearthnet_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_r1": val_metrics["R@1"],
                "model_name": args.model_name,
            }, best_path)
            logger.info("New best model saved (R@1=%.2f%%)", best_r1)

    logger.info("Fine-tuning complete. Best R@1: %.2f%%", best_r1)


if __name__ == "__main__":
    main()
