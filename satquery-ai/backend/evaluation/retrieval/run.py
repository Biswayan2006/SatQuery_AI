"""
SatQuery AI — Retrieval Evaluation Runner
=========================================
    python -m evaluation.retrieval.run --config evaluation/configs/evaluation.yaml

Cross-modal image↔text Recall@1/5/10 on the VRSBench test split, comparing the
generic **OpenAI CLIP** baseline against an **RS-CLIP** fine-tuned checkpoint
(adapted).  Reuses ``training.evaluate_clip``'s ``extract_features`` /
``retrieval_recall`` / ``_collate_fn`` — the retrieval math is never
re-implemented here.

Honesty:
  * No dataset configured / open_clip missing → explicit ``no_data`` report.
  * No RS-CLIP checkpoint (``RS_CLIP_CKPT`` unset) → adapted column is ``n/a``.
  * Test captions are only used for scoring, never for training.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json
from evaluation.common.variants import Variant, resolve_variants

logger = logging.getLogger("satquery.eval.retrieval")

DOMAIN = "retrieval"
TITLE = "Retrieval (OpenAI CLIP vs RS-CLIP)"


def _no_data(output_dir: str, notes: List[str], prov=None) -> Dict[str, Any]:
    report = {"domain": DOMAIN, "status": "no_data", "dataset": "vrsbench",
              "variants": {}, "comparison": [], "notes": notes, "provenance": provenance(prov)}
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, [], notes)
    logger.warning("Retrieval: %s", notes[-1] if notes else "no data")
    return report


def _build_clip(variant: Variant, model_cfg: Dict[str, Any], device: str):
    """Construct (model, preprocess, tokenizer, label) for a variant."""
    import open_clip
    import torch

    cache_dir = model_cfg.get("cache_dir") or None
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    if variant.kind == "adapted":
        ckpt = torch.load(variant.checkpoint, map_location=device, weights_only=False)
        name = ckpt.get("model_name", model_cfg.get("name", "ViT-B-32"))
        pretrained = ckpt.get("pretrained", model_cfg.get("pretrained", "openai"))
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)
        _, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)
        tokenizer = open_clip.get_tokenizer(name)
        if ckpt.get("lora_enabled", False):
            from training.train_clip import apply_lora
            for p in model.parameters():
                p.requires_grad_(False)
            model = apply_lora(model, {})
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        label = f"RS-CLIP ({name}, epoch {ckpt.get('epoch', '?')})"
    else:
        name = model_cfg.get("name", "ViT-B-32")
        pretrained = model_cfg.get("pretrained", "openai")
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained, **kwargs)
        tokenizer = open_clip.get_tokenizer(name)
        label = f"OpenAI CLIP ({name})"

    model = model.to(device).eval()
    return model, preprocess, tokenizer, label


def _evaluate_variant(variant: Variant, data_dir: str, model_cfg: Dict[str, Any],
                      k_values: Tuple[int, ...], limit: Optional[int], device: str):
    """Return (metrics_dict|None, label, n_samples)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from training.datasets.vrsbench import VRSBenchAdapter
    from training.evaluate_clip import _collate_fn, extract_features, retrieval_recall

    model, preprocess, tokenizer, label = _build_clip(variant, model_cfg, device)

    # one_caption_per_image=True → N image-text pairs; diagonal is ground truth.
    ds = VRSBenchAdapter(data_dir=data_dir, split="test", clip_preprocess=preprocess,
                         max_samples=limit, one_caption_per_image=True)
    if len(ds) == 0:
        return None, label, 0

    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=_collate_fn)
    img_feats, txt_feats, _ = extract_features(model, tokenizer, loader, torch.device(device))
    n = int(img_feats.shape[0])
    idx = np.arange(n)
    return {
        "i2t": retrieval_recall(img_feats, txt_feats, idx, idx, k_values),
        "t2i": retrieval_recall(txt_feats, img_feats, idx, idx, k_values),
    }, label, n


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    rcfg = cfg.get("retrieval", {})
    ds_cfg = (rcfg.get("datasets", {}) or {}).get("vrsbench", {})
    data_dir = ds_cfg.get("data_dir")
    model_cfg = rcfg.get("model", {})
    k_values = tuple(rcfg.get("k_values", [1, 5, 10]))
    notes: List[str] = []

    if not is_configured(data_dir):
        notes.append("skipped: retrieval dataset not configured (set VRSBENCH_DIR).")
        return _no_data(output_dir, notes)

    try:
        import open_clip  # noqa: F401
    except Exception:
        notes.append("skipped: open_clip_torch not installed (pip install open-clip-torch).")
        return _no_data(output_dir, notes, prov=[{"name": "vrsbench", "data_dir": data_dir}])

    limit = max_samples if max_samples is not None else ds_cfg.get("max_samples")
    variants = resolve_variants(
        cfg, DOMAIN,
        baseline_label="OpenAI CLIP", adapted_label="RS-CLIP (SatQuery)",
        adapted_needs_checkpoint=True,
    )

    results: Dict[str, Dict[str, Any]] = {}
    labels: Dict[str, str] = {}
    n_used = 0
    for v in variants:
        if not v.available:
            notes.append(f"{v.name} (adapted): {v.reason}")
            continue
        try:
            res, label, n = _evaluate_variant(v, data_dir, model_cfg, k_values, limit, device)
        except Exception as exc:
            logger.exception("retrieval variant '%s' failed", v.name)
            notes.append(f"{v.name}: evaluation failed ({type(exc).__name__}: {exc}).")
            continue
        if res is None:
            notes.append(f"{v.name}: VRSBench test split empty (no annotations found).")
            continue
        results[v.kind] = res
        labels[v.kind] = label
        n_used = max(n_used, n)

    base = results.get("baseline")
    adpt = results.get("adapted")
    if not base:
        notes.append("skipped: retrieval baseline produced no results (no VRSBench test data "
                     "or CLIP weights unavailable offline).")
        return _no_data(output_dir, notes, prov=[{"name": "vrsbench", "data_dir": data_dir, "n_samples": n_used}])

    model_label = labels.get("baseline", "OpenAI CLIP")
    rows: List[Dict[str, Any]] = []
    for k in k_values:
        key = f"R@{k}"
        a = adpt["i2t"].get(key) if adpt else None
        rows.append(comparison_row(model_label, "vrsbench (i2t)", key, base["i2t"].get(key), a))

    notes.append(f"Image→text and text→image Recall@{list(k_values)} over {n_used} VRSBench test "
                 "pairs (one caption per image; diagonal ground truth).")
    if not adpt:
        notes.append("Adapted = n/a: set RS_CLIP_CKPT to a trained RS-CLIP checkpoint.")

    report = {
        "domain": DOMAIN,
        "status": "ok",
        "dataset": "vrsbench",
        "k_values": list(k_values),
        "variants": {
            k: {"label": labels.get(k), "image_to_text": r["i2t"], "text_to_image": r["t2i"]}
            for k, r in results.items()
        },
        "comparison": rows,
        "notes": notes,
        "provenance": provenance([{"name": "vrsbench", "data_dir": data_dir, "n_samples": n_used}]),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    logger.info("Retrieval i2t R@1 baseline=%s adapted=%s (n=%d)",
                base["i2t"].get("R@1"), adpt["i2t"].get("R@1") if adpt else "n/a", n_used)
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
