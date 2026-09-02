"""
SatQuery AI — Benchmark Evaluation
Evaluates model performance on BigEarthNet and custom RS benchmarks.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger("satquery.evaluate")


# ── Metrics ───────────────────────────────────────────────────────────────────

def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute multi-label classification metrics."""
    y_bin = (y_pred >= threshold).astype(int)

    metrics = {
        "macro_f1": f1_score(y_true, y_bin, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_bin, average="micro", zero_division=0),
        "macro_precision": precision_score(y_true, y_bin, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_bin, average="macro", zero_division=0),
        "mean_ap": average_precision_score(y_true, y_pred, average="macro"),
        "subset_accuracy": float((y_true == y_bin).all(axis=1).mean()),
    }
    return {k: round(float(v), 4) for k, v in metrics.items()}


def retrieval_metrics(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    k_values: List[int] = (1, 5, 10),
) -> Dict:
    """Compute image retrieval metrics (R@k)."""
    sims = query_feats @ gallery_feats.T  # [Q, G]
    results = {}

    for k in k_values:
        top_k_indices = np.argsort(-sims, axis=1)[:, :k]
        hits = 0
        for q_idx, indices in enumerate(top_k_indices):
            q_lbl = query_labels[q_idx]
            retrieved_lbls = gallery_labels[indices]
            # Multi-label: hit if any label overlap
            if (q_lbl & retrieved_lbls).any(axis=-1).any():
                hits += 1
        results[f"R@{k}"] = round(hits / len(query_feats) * 100, 2)

    return results


# ── Classification Evaluator ──────────────────────────────────────────────────

class ClassificationEvaluator:
    """Evaluate a multi-label scene classification model on BigEarthNet."""

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device

    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for optical, sar, labels, _ in loader:
                optical = optical.to(self.device)
                labels_np = labels.numpy()

                # Use RGB composite
                rgb = optical[:, [3, 2, 1], :, :]
                rgb = torch.clamp((rgb * 0.2 + 0.5), 0.0, 1.0)

                logits = self.model(rgb)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_preds.append(probs)
                all_labels.append(labels_np)

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_labels, axis=0)

        return multilabel_metrics(y_true, y_pred)


# ── CLIP Retrieval Evaluator ──────────────────────────────────────────────────

class CLIPRetrievalEvaluator:
    """Evaluate CLIP zero-shot retrieval on BigEarthNet."""

    def __init__(self, clip_model, tokenizer, device: torch.device):
        self.model = clip_model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, loader: DataLoader, label_names: List[str]) -> Dict:
        self.model.eval()
        img_feats, txt_feats, all_labels = [], [], []

        with torch.no_grad():
            for optical, _, labels, descriptions in loader:
                optical = optical.to(self.device)
                rgb = optical[:, [3, 2, 1], :, :]
                rgb = torch.clamp((rgb * 0.2 + 0.5), 0.0, 1.0)
                text_tokens = self.tokenizer(list(descriptions)).to(self.device)

                img_f = F.normalize(self.model.encode_image(rgb), dim=-1)
                txt_f = F.normalize(self.model.encode_text(text_tokens), dim=-1)

                img_feats.append(img_f.cpu().numpy())
                txt_feats.append(txt_f.cpu().numpy())
                all_labels.append(labels.numpy())

        img_feats = np.concatenate(img_feats)
        txt_feats = np.concatenate(txt_feats)
        all_labels = np.concatenate(all_labels)

        retrieval = retrieval_metrics(img_feats, txt_feats, all_labels, all_labels)

        # Also compute image-to-image retrieval
        img_retrieval = retrieval_metrics(img_feats, img_feats, all_labels, all_labels)

        return {
            "image_to_text": retrieval,
            "image_to_image": img_retrieval,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate SatQuery AI models on BigEarthNet")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default="eval_results.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )
    logger.info("Evaluation device: %s", device)

    # Load model
    try:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
        tokenizer = open_clip.get_tokenizer(args.model_name)

        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        logger.info("Checkpoint loaded from epoch %d (Val R@1=%.2f%%)", ckpt.get("epoch", -1), ckpt.get("val_r1", 0))
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return

    # Dataset
    from training.bigearthnet_dataset import BigEarthNetDataset, BIGEARTHNET_43_LABELS
    test_ds = BigEarthNetDataset(data_dir=args.data_dir, split="test", use_sar=False)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    logger.info("Test samples: %d", len(test_ds))

    # Evaluate
    evaluator = CLIPRetrievalEvaluator(model, tokenizer, device)
    results = evaluator.evaluate(test_loader, BIGEARTHNET_43_LABELS)

    # Print results
    logger.info("=== Evaluation Results ===")
    for metric_group, scores in results.items():
        logger.info("%s:", metric_group)
        for k, v in scores.items():
            logger.info("  %s: %.2f%%", k, v)

    # Save
    out_path = args.output
    with open(out_path, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "results": results}, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
