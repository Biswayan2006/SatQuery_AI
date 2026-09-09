"""
SatQuery AI — RS-CLIP Evaluation Script
========================================
Evaluates a trained RS-CLIP checkpoint on:
  1. Image→text retrieval  (R@1, R@5, R@10)
  2. Text→image retrieval  (R@1, R@5, R@10)
  3. Zero-shot scene classification (when labels are available)

Saves a JSON results file alongside the checkpoint.

Usage
-----
    cd satquery-ai/backend
    python training/evaluate_clip.py \\
        --checkpoint checkpoints/rs_clip/rs_clip_best.pt \\
        --config    training/configs/rs_clip_base.yaml \\
        --output    checkpoints/rs_clip/eval_results.json

    # Evaluate on a specific dataset split
    python training/evaluate_clip.py \\
        --checkpoint checkpoints/rs_clip/rs_clip_best.pt \\
        --config    training/configs/rs_clip_base.yaml \\
        --split     test \\
        --max-samples 5000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger("satquery.evaluate_clip")


# ── Metrics ────────────────────────────────────────────────────────────────────

def retrieval_recall(
    query_feats: np.ndarray,    # [Q, D]
    gallery_feats: np.ndarray,  # [G, D]
    query_labels: np.ndarray,   # [Q] int index (diagonal = ground truth)
    gallery_labels: np.ndarray, # [G] int index
    k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Compute Recall@k for cross-modal retrieval.
    For image-text pairs where query index == gallery index is the match.
    """
    sims = query_feats @ gallery_feats.T  # [Q, G]
    results: Dict[str, float] = {}

    for k in k_values:
        k_eff = min(k, gallery_feats.shape[0])
        top_k = np.argsort(-sims, axis=1)[:, :k_eff]     # [Q, k]
        hits = 0
        for q_idx, top in enumerate(top_k):
            q_lbl = query_labels[q_idx]
            # Multi-label aware: hit if any retrieved item shares a label
            retrieved_lbls = gallery_labels[top]
            if hasattr(q_lbl, "__len__"):
                # Multi-hot: hit if label overlap
                if isinstance(q_lbl, np.ndarray) and q_lbl.ndim > 0:
                    if (q_lbl.astype(bool) & retrieved_lbls.astype(bool)).any():
                        hits += 1
                    continue
            # Single-label: exact match
            if q_lbl in retrieved_lbls:
                hits += 1
        results[f"R@{k}"] = round(hits / len(query_feats) * 100, 2)

    return results


def zero_shot_accuracy(
    image_feats: np.ndarray,   # [N, D]
    label_feats: np.ndarray,   # [C, D]
    true_labels: np.ndarray,   # [N] int
) -> Dict[str, float]:
    """
    Zero-shot classification: assign each image the label whose text
    embedding is most similar.
    """
    sims = image_feats @ label_feats.T   # [N, C]
    pred = np.argmax(sims, axis=1)       # [N]
    acc = float((pred == true_labels).mean())
    return {"zero_shot_top1": round(acc * 100, 2)}


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract image and text features from a DataLoader.

    Returns
    -------
    img_feats  : np.ndarray [N, D]
    txt_feats  : np.ndarray [N, D]
    label_vecs : np.ndarray [N, C] or None  (multi-hot, if available)
    """
    model.eval()
    all_img, all_txt, all_lbls = [], [], []
    has_labels = False

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        texts: List[str] = batch["text"]
        text_tokens = tokenizer(texts).to(device)

        img_f = F.normalize(model.encode_image(images), dim=-1)
        txt_f = F.normalize(model.encode_text(text_tokens), dim=-1)

        all_img.append(img_f.cpu().numpy())
        all_txt.append(txt_f.cpu().numpy())

        # Collect multi-hot labels if provided
        metas = batch.get("metadata", [])
        if metas:
            lbl_tensors = [m.get("label_tensor") for m in metas if "label_tensor" in m]
            if lbl_tensors:
                has_labels = True
                all_lbls.append(
                    torch.stack(lbl_tensors).numpy()
                    if isinstance(lbl_tensors[0], torch.Tensor)
                    else np.stack(lbl_tensors)
                )

    img_feats = np.concatenate(all_img)
    txt_feats = np.concatenate(all_txt)
    label_vecs = np.concatenate(all_lbls) if (has_labels and all_lbls) else None

    return img_feats, txt_feats, label_vecs


# ── Zero-shot label templates ──────────────────────────────────────────────────

def build_label_features(
    model: torch.nn.Module,
    tokenizer,
    label_names: List[str],
    device: torch.device,
    templates: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute text embeddings for each label using prompt templates.
    Returns [C, D] normalised array.
    """
    if templates is None:
        templates = [
            "a satellite image of {}.",
            "an aerial photograph showing {}.",
            "remote sensing image of {}.",
            "a satellite view of {}.",
        ]

    model.eval()
    label_feats = []

    with torch.no_grad():
        for label in label_names:
            texts = [t.format(label.lower()) for t in templates]
            tokens = tokenizer(texts).to(device)
            feats = F.normalize(model.encode_text(tokens), dim=-1)
            # Average over templates
            label_feats.append(feats.mean(dim=0).cpu().numpy())

    return np.stack(label_feats)   # [C, D]


# ── Main ───────────────────────────────────────────────────────────────────────

def _collate_fn(batch: List[Dict]) -> Dict:
    images = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    metadata = [b["metadata"] for b in batch]
    return {"image": images, "text": texts, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RS-CLIP checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None, help="Path to save JSON results")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # ── Load config ───────────────────────────────────────────────────────────
    from training.train_clip import load_config, _resolve_device
    cfg = load_config(args.config)
    device = _resolve_device(args.device)
    logger.info("Evaluation device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        import open_clip
    except ImportError:
        logger.error("open_clip_torch required. pip install open-clip-torch")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt.get("model_name", cfg["model"]["name"])
    pretrained = ckpt.get("pretrained", cfg["model"]["pretrained"])

    logger.info("Loading model: %s (pretrained=%s)", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    # LoRA layers may be present — load full state
    lora_enabled = ckpt.get("lora_enabled", False)
    if lora_enabled:
        from training.train_clip import apply_lora
        for p in model.parameters():
            p.requires_grad_(False)
        model = apply_lora(model, cfg.get("lora", {}))

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Checkpoint loaded (epoch %d)", ckpt.get("epoch", -1))

    # ── Dataset ───────────────────────────────────────────────────────────────
    from training.datasets.bigearthnet import BigEarthNetAdapter, BIGEARTHNET_43_LABELS
    from training.datasets.vrsbench import VRSBenchAdapter

    eval_datasets = []
    ds_cfg = cfg["datasets"]

    if ds_cfg.get("bigearthnet", {}).get("enabled", False):
        bc = ds_cfg["bigearthnet"]
        eval_datasets.append(BigEarthNetAdapter(
            data_dir=bc["data_dir"],
            split=args.split,
            clip_preprocess=preprocess,
            max_samples=args.max_samples,
        ))

    if ds_cfg.get("vrsbench", {}).get("enabled", False):
        vc = ds_cfg["vrsbench"]
        eval_datasets.append(VRSBenchAdapter(
            data_dir=vc["data_dir"],
            split=args.split,
            clip_preprocess=preprocess,
            max_samples=args.max_samples,
        ))

    if not eval_datasets:
        logger.error("No datasets enabled in config.")
        sys.exit(1)

    from torch.utils.data import ConcatDataset
    eval_ds = ConcatDataset(eval_datasets) if len(eval_datasets) > 1 else eval_datasets[0]
    logger.info("Eval samples: %d", len(eval_ds))

    loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
    )

    # ── Feature extraction ─────────────────────────────────────────────────────
    logger.info("Extracting features …")
    img_feats, txt_feats, label_vecs = extract_features(model, tokenizer, loader, device)
    n = img_feats.shape[0]
    logger.info("Features extracted: %d samples, dim=%d", n, img_feats.shape[1])

    # ── Retrieval metrics ──────────────────────────────────────────────────────
    k_values = tuple(cfg["evaluation"].get("recall_at_k", [1, 5, 10]))
    labels_idx = np.arange(n)   # diagonal: image i matches text i

    i2t = retrieval_recall(img_feats, txt_feats, labels_idx, labels_idx, k_values)
    t2i = retrieval_recall(txt_feats, img_feats, labels_idx, labels_idx, k_values)

    logger.info("Image→Text: %s", i2t)
    logger.info("Text→Image: %s", t2i)

    results: Dict = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_samples": n,
        "embed_dim": int(img_feats.shape[1]),
        "image_to_text": i2t,
        "text_to_image": t2i,
    }

    # ── Zero-shot classification ───────────────────────────────────────────────
    if cfg["evaluation"].get("zero_shot_classification", False) and label_vecs is not None:
        logger.info("Running zero-shot classification …")
        label_feats = build_label_features(
            model, tokenizer, BIGEARTHNET_43_LABELS, device
        )
        # For zero-shot: assign argmax of multi-hot label to each sample
        true_cls = np.argmax(label_vecs, axis=1)
        zs = zero_shot_accuracy(img_feats, label_feats, true_cls)
        results["zero_shot_classification"] = zs
        logger.info("Zero-shot: %s", zs)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = args.output
    if out_path is None:
        out_path = str(Path(args.checkpoint).parent / "eval_results.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Evaluation results saved to %s", out_path)


if __name__ == "__main__":
    main()
