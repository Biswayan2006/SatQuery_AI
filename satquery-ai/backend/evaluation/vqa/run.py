"""
SatQuery AI — VQA Evaluation Runner
===================================
    python -m evaluation.vqa.run --config evaluation/configs/evaluation.yaml

Compares the BLIP **pretrained** baseline against a **fine-tuned** SatQuery
checkpoint (adapted) on the VQA test split, reusing the existing inference
harness in ``training/evaluate_vqa.py`` (no re-implementation).

Honesty:
  * No dataset configured  → writes an explicit ``no_data`` report, exits 0.
  * No adapted checkpoint   → SatQuery column is ``n/a`` (never fabricated).
  * Test labels are only used for scoring, never for inference/training.

Also emits ``confidence_pairs`` (emitted-confidence, correctness) for the
confidence-calibration runner to consume.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json
from evaluation.common.variants import resolve_variants

logger = logging.getLogger("satquery.eval.vqa")

DOMAIN = "vqa"
TITLE = "VQA (BLIP baseline vs SatQuery fine-tuned)"
_METRICS = ["exact_match", "vqa_accuracy", "token_f1"]


def _evaluate_variant(vqa_cfg: Dict[str, Any], variant, test_ds, device: str) -> Dict[str, Any]:
    """Run one variant (pretrained/finetuned) over the shared test set."""
    import torch
    from torch.utils.data import DataLoader

    from training.evaluate_vqa import load_eval_model, run_inference
    from training.vqa_utils import VQACollator
    from evaluation.common.metrics import vqa_metrics_module

    source = "finetuned" if variant.kind == "adapted" else "pretrained"
    torch_device = torch.device(device)
    model, processor, model_name = load_eval_model(vqa_cfg, source, variant.checkpoint, torch_device)

    collate = VQACollator(
        processor,
        max_length=int(vqa_cfg.get("training", {}).get("max_text_length", 64)),
        train=False,
    )
    loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, collate_fn=collate)

    preds, golds, metas, scores = run_inference(
        model, processor, loader, torch_device, vqa_cfg, return_scores=True
    )

    vqamod = vqa_metrics_module()
    metrics = vqamod["aggregate_metrics"](
        preds, golds, categories=[m.get("question_type") for m in metas]
    )

    exact_match = vqamod["exact_match"]
    pairs: List[Dict[str, Any]] = []
    for pred, gold, score in zip(preds, golds, scores):
        if score is None:
            continue
        conf = math.exp(score) if score <= 0 else 1.0
        pairs.append({"confidence": round(max(0.0, min(1.0, conf)), 6),
                      "correct": int(exact_match(pred, gold))})

    return {
        "variant_name": variant.name,
        "model_name": model_name,
        "metrics": metrics,
        "confidence_pairs": pairs,
        "n_samples": len(preds),
    }


def _write(output_dir: str, report: Dict[str, Any], rows: List[Dict[str, Any]], notes: List[str]) -> None:
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
    vqa_cfg = dict(cfg.get("vqa", {}))
    datasets = {k: dict(v) for k, v in (vqa_cfg.get("datasets", {}) or {}).items()}
    if max_samples is not None:
        for dc in datasets.values():
            dc["max_samples"] = max_samples

    configured = {
        name: dc for name, dc in datasets.items()
        if dc.get("enabled", False) and is_configured(dc.get("data_dir"))
    }
    notes: List[str] = []

    if not configured:
        notes.append("skipped: no VQA dataset configured. Set VRSBENCH_DIR and/or RSVQA_DIR "
                     "(see evaluation/configs/evaluation.yaml).")
        report = {"domain": DOMAIN, "status": "no_data", "dataset": None,
                  "variants": {}, "comparison": [], "confidence_pairs": [],
                  "notes": notes, "provenance": provenance()}
        _write(output_dir, report, [], notes)
        logger.warning("VQA: %s", notes[-1])
        return report

    vqa_cfg["datasets"] = configured

    # Build the test split ONCE — both variants score identical samples.
    try:
        from training.evaluate_vqa import build_test_dataset
        test_ds, prov = build_test_dataset(vqa_cfg)
    except RuntimeError as exc:
        notes.append(f"skipped: {exc}")
        report = {"domain": DOMAIN, "status": "no_data", "dataset": None,
                  "variants": {}, "comparison": [], "confidence_pairs": [],
                  "notes": notes,
                  "provenance": provenance([{"name": n, "data_dir": dc.get("data_dir")}
                                            for n, dc in configured.items()])}
        _write(output_dir, report, [], notes)
        logger.warning("VQA: %s", notes[-1])
        return report

    dataset_name = "+".join(p["name"] for p in prov)
    variants = resolve_variants(
        cfg, DOMAIN,
        baseline_label="BLIP pretrained",
        adapted_label="BLIP fine-tuned (SatQuery)",
        adapted_needs_checkpoint=True,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for v in variants:
        if not v.available:
            notes.append(f"{v.name} (adapted): {v.reason}")
            continue
        try:
            results[v.kind] = _evaluate_variant(vqa_cfg, v, test_ds, device)
        except Exception as exc:  # model download / load / inference failure
            logger.exception("VQA variant '%s' failed", v.name)
            notes.append(f"{v.name}: evaluation failed ({type(exc).__name__}: {exc}).")

    base = results.get("baseline")
    adpt = results.get("adapted")

    rows: List[Dict[str, Any]] = []
    if base:
        model_label = base["model_name"]
        for metric in _METRICS:
            b = base["metrics"].get(metric)
            a = adpt["metrics"].get(metric) if adpt else None
            rows.append(comparison_row(model_label, dataset_name, metric, b, a))

    report = {
        "domain": DOMAIN,
        "status": "ok" if base else "error",
        "dataset": dataset_name,
        "variants": {
            k: {"name": r["variant_name"], "model": r["model_name"],
                "metrics": r["metrics"], "n_samples": r["n_samples"]}
            for k, r in results.items()
        },
        "comparison": rows,
        # Consumed by the confidence runner (baseline variant's emitted confidence).
        "confidence_pairs": base["confidence_pairs"] if base else [],
        "notes": notes,
        "provenance": provenance(prov),
    }
    _write(output_dir, report, rows, notes)
    if base:
        logger.info("VQA baseline exact_match=%.4f | adapted=%s",
                    base["metrics"].get("exact_match"),
                    adpt["metrics"].get("exact_match") if adpt else "n/a")
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
