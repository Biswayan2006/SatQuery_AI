"""
SatQuery AI — Remote-Sensing VQA Evaluation
============================================
Evaluates a base or fine-tuned VQA model on a dataset's **test** split and
writes a JSON report:

    {
        "model": "...",
        "dataset": "...",
        "split": "test",
        "metrics": { "exact_match": ..., "vqa_accuracy": ..., "token_f1": ...,
                     "per_category": {...} },
        "error_examples": [ {question, gold, prediction, ...}, ... ],
        "provenance": { git_commit, dataset_version, checkpoint }
    }

Usage
-----
    cd satquery-ai/backend

    # Evaluate the BASE pretrained model
    python training/evaluate_vqa.py --config training/configs/vqa_base.yaml \\
        --model-source pretrained --output reports/vqa_base_eval.json

    # Evaluate a FINE-TUNED checkpoint
    python training/evaluate_vqa.py --config training/configs/vqa_base.yaml \\
        --model-source finetuned --checkpoint checkpoints/vqa/best \\
        --output reports/vqa_finetuned_eval.json

Training / inference separation
--------------------------------
A standalone script. Imports train_vqa helpers but not models/ or api/.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from training.vqa_utils import (  # noqa: E402
    VQACollator,
    aggregate_metrics,
    dataset_version,
    git_commit_hash,
    set_seed,
)

logger = logging.getLogger("satquery.evaluate_vqa")


# ── Model loading (base or fine-tuned) ────────────────────────────────────────

def load_eval_model(
    cfg: Dict,
    model_source: str,
    checkpoint: Optional[str],
    device: torch.device,
):
    """
    Load the model to evaluate.

    model_source == "pretrained" → base HuggingFace weights.
    model_source == "finetuned"  → base weights + adapter/checkpoint at
                                   ``checkpoint`` (dir produced by train_vqa.py).
    """
    from training.train_vqa import is_blip2, load_model_and_processor

    model_name = cfg["model"]["name"]
    cache_dir = cfg["model"].get("cache_dir")

    if model_source == "finetuned":
        if not checkpoint or not os.path.isdir(checkpoint):
            raise FileNotFoundError(
                f"--checkpoint dir required for finetuned eval, got: {checkpoint}"
            )
        version_meta = _read_version_meta(checkpoint)
        model_name = version_meta.get("base_model", model_name)
        model, processor = _load_finetuned(model_name, checkpoint, cache_dir, device, version_meta)
    else:
        model, processor = load_model_and_processor(model_name, cache_dir, device)

    model.eval()
    return model, processor, model_name


def _read_version_meta(checkpoint_dir: str) -> Dict[str, Any]:
    path = os.path.join(checkpoint_dir, "model_version.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_finetuned(model_name, checkpoint, cache_dir, device, version_meta):
    """Reconstruct a fine-tuned model (peft, manual-LoRA, or full)."""
    from training.train_vqa import load_model_and_processor

    lora_backend = version_meta.get("lora_backend", "manual")
    lora_cfg = version_meta.get("lora_config", {})

    model, processor = load_model_and_processor(model_name, cache_dir, device)

    if lora_backend == "peft":
        from peft import PeftModel  # type: ignore
        model = PeftModel.from_pretrained(model, checkpoint)
        logger.info("Loaded peft adapter from %s", checkpoint)
    elif lora_backend == "manual":
        from training.vqa_lora import rebuild_manual_lora
        model = rebuild_manual_lora(model, lora_cfg)
        weights = torch.load(
            os.path.join(checkpoint, "adapter_weights.pt"), map_location=device
        )
        missing, unexpected = model.load_state_dict(weights, strict=False)
        logger.info("Loaded manual LoRA adapter (%d tensors) from %s",
                    len(weights), checkpoint)
    elif lora_backend == "full":
        weights = torch.load(
            os.path.join(checkpoint, "adapter_weights.pt"), map_location=device
        )
        model.load_state_dict(weights, strict=False)
        logger.info("Loaded full fine-tuned weights from %s", checkpoint)

    # Prefer the processor saved with the checkpoint (tokenizer parity).
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(checkpoint)
    except Exception:
        pass

    return model.to(device), processor


# ── Test dataset ────────────────────────────────────────────────────────────

def build_test_dataset(cfg: Dict) -> "tuple[Dataset, List[Dict]]":
    """Build the concatenated test split across enabled datasets."""
    from training.datasets.vqa import build_vqa_dataset

    ds_cfg = cfg["datasets"]
    enabled = {k: v for k, v in ds_cfg.items() if v.get("enabled", False)}
    if not enabled:
        raise RuntimeError("No datasets enabled in config.")

    parts: List[Dataset] = []
    provenance: List[Dict] = []
    for name, dc in enabled.items():
        adapter = build_vqa_dataset(
            name, data_dir=dc["data_dir"], split="test",
            max_samples=dc.get("max_samples"),
            image_size=dc.get("image_size"),
            file_prefix=dc.get("file_prefix"),
            image_subdir=dc.get("image_subdir"),
            image_ext=dc.get("image_ext"),
        )
        if len(adapter) > 0:
            parts.append(adapter)
        provenance.append({"name": name, "data_dir": dc["data_dir"], "n_samples": len(adapter)})
        logger.info("[%s] test samples: %d", name, len(adapter))

    if not parts:
        raise RuntimeError(
            "Test split is empty for all enabled datasets. Ensure test "
            "annotation files exist, or evaluate a dataset with native splits."
        )
    ds = parts[0] if len(parts) == 1 else ConcatDataset(parts)
    return ds, provenance


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model,
    processor,
    loader,
    device,
    cfg,
    return_scores: bool = False,
) -> "tuple[List[str], List[str], List[Dict]] | tuple[List[str], List[str], List[Dict], List[Optional[float]]]":
    """
    Run model.generate over a DataLoader and collect predictions.

    Parameters
    ----------
    return_scores : bool
        When True a fourth element is returned: a list of per-sample mean
        token log-probability scores (``float`` when available, else ``None``).
        These are raw model evidence — NOT calibrated confidence values.
        Use them as inputs to calibration metrics, never as final scores.
    """
    import math
    import numpy as np
    from training.train_vqa import is_blip2  # noqa: F401

    gen_cfg = cfg.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens", 20))
    num_beams = int(gen_cfg.get("num_beams", 3))

    preds: List[str] = []
    golds: List[str] = []
    metas: List[Dict] = []
    scores: List[Optional[float]] = []

    for batch in loader:
        gen_inputs = {
            k: v.to(device) for k, v in batch.items()
            if k in ("pixel_values", "input_ids", "attention_mask")
            and isinstance(v, torch.Tensor)
        }

        # Always request output_scores so we can surface generation evidence.
        gen_out = model.generate(
            **gen_inputs,
            max_new_tokens=max_new,
            num_beams=num_beams,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Decode text predictions.
        sequences = gen_out.sequences
        decoded = processor.batch_decode(sequences, skip_special_tokens=True)

        # Extract per-sample mean token log-probability (evidence, not calibrated).
        batch_scores = _extract_batch_scores(gen_out, sequences, len(decoded))

        for i, ans in enumerate(decoded):
            preds.append(ans.strip())
            golds.append(batch["raw_answers"][i])
            m = dict(batch["metadata"][i])
            m["question"] = batch["raw_questions"][i]
            metas.append(m)
            scores.append(batch_scores[i] if i < len(batch_scores) else None)

    if return_scores:
        return preds, golds, metas, scores
    return preds, golds, metas


def _extract_batch_scores(
    gen_out,
    sequences: "torch.Tensor",
    batch_size: int,
) -> "List[Optional[float]]":
    """
    Extract mean per-token log-probabilities for each item in a batch.

    Returns a list of length ``batch_size``.  Each element is a ``float``
    (the mean token log-prob for that sample's generated tokens) or ``None``
    when scores are unavailable.

    This is raw model evidence only.  It correlates with correctness but is
    NOT a calibrated probability.
    """
    import math
    import numpy as np

    try:
        step_logits = getattr(gen_out, "scores", None)
        if not step_logits:
            return [None] * batch_size

        n_steps = len(step_logits)
        # gen_out.sequences shape: [batch, seq_len]
        # generated tokens are the last n_steps tokens of each sequence
        gen_tokens = sequences[:, -n_steps:]  # [batch, n_steps]

        result: List[Optional[float]] = []
        for b in range(batch_size):
            logprobs = []
            for t in range(n_steps):
                # step_logits[t]: [batch, vocab]
                logit_t = step_logits[t][b].float()
                lp = torch.log_softmax(logit_t, dim=-1)
                tok_id = gen_tokens[b, t].item()
                logprobs.append(float(lp[tok_id]))
            mean_lp = float(np.mean(logprobs)) if logprobs else None
            result.append(mean_lp)
        return result
    except Exception as exc:
        logger.debug("Batch score extraction failed: %s", exc)
        return [None] * batch_size


def collect_error_examples(
    preds: List[str], golds: List[str], metas: List[Dict], limit: int,
) -> List[Dict]:
    from training.vqa_utils import normalize_text
    errors = []
    for pred, gold, meta in zip(preds, golds, metas):
        if normalize_text(pred) != normalize_text(gold):
            errors.append({
                "image_id": meta.get("image_id"),
                "question_id": meta.get("question_id"),
                "question": meta.get("question"),
                "question_type": meta.get("question_type"),
                "gold": gold,
                "prediction": pred,
            })
            if len(errors) >= limit:
                break
    return errors


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RS VQA model on test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-source", choices=["pretrained", "finetuned"], default="pretrained")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint dir for finetuned eval")
    parser.add_argument("--output", default=None, help="Path to write JSON report")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    from training.train_vqa import load_config
    cfg = load_config(args.config)
    if args.max_samples is not None:
        for ds in cfg["datasets"].values():
            ds["max_samples"] = args.max_samples

    set_seed(cfg["training"].get("seed", 42))
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    logger.info("Eval device: %s | source: %s", device, args.model_source)

    model, processor, model_name = load_eval_model(
        cfg, args.model_source, args.checkpoint, device
    )

    test_ds, provenance = build_test_dataset(cfg)
    logger.info("Total test samples: %d", len(test_ds))

    collate = VQACollator(
        processor, max_length=int(cfg["training"].get("max_text_length", 64)), train=False
    )
    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate,
    )

    preds, golds, metas = run_inference(model, processor, loader, device, cfg)
    metrics = aggregate_metrics(preds, golds, categories=[m.get("question_type") for m in metas])

    num_err = int(cfg.get("evaluation", {}).get("num_error_examples", 30))
    error_examples = collect_error_examples(preds, golds, metas, num_err)

    dataset_names = "+".join(p["name"] for p in provenance)
    report = {
        "model": model_name,
        "model_source": args.model_source,
        "dataset": dataset_names,
        "split": "test",
        "metrics": metrics,
        "error_examples": error_examples,
        "provenance": {
            "checkpoint": args.checkpoint if args.model_source == "finetuned" else None,
            "dataset_version": dataset_version(provenance),
            "git_commit": git_commit_hash(),
            "datasets": provenance,
        },
    }

    out_path = args.output or os.path.join(
        BACKEND_DIR, "reports", f"vqa_{args.model_source}_eval.json"
    )
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=== VQA Evaluation (%s) ===", args.model_source)
    logger.info("Exact match : %.4f", metrics["exact_match"])
    logger.info("VQA accuracy: %.4f", metrics["vqa_accuracy"])
    logger.info("Token F1    : %.4f", metrics["token_f1"])
    logger.info("Report saved: %s", out_path)


if __name__ == "__main__":
    main()
