"""
SatQuery AI — Routing Evaluation Runner
=======================================
    python -m evaluation.routing.run --config evaluation/configs/evaluation.yaml

Scores the ``TaskClassifier`` router against a curated in-house gold set
(``routing_eval_set.json`` — NOT a benchmark's hidden test labels), reporting
accuracy, per-task precision/recall/F1, and a confusion matrix.

Baseline vs SatQuery: the honest baseline is a majority-class predictor (always
guess the most frequent task); SatQuery is the keyword+structural router.  Fully
offline (semantic router disabled by default).
"""
from __future__ import annotations

import json
import logging
import os
from collections import Counter
from typing import Any, Dict, List

from evaluation.common.cli import build_parser, init_run
from evaluation.common.config import is_configured
from evaluation.common.reporting import append_markdown, comparison_row, provenance, write_json

logger = logging.getLogger("satquery.eval.routing")

DOMAIN = "routing"
TITLE = "Routing (task classification)"
_TASK_LABELS = [
    "SINGLE_VQA", "CAPTIONING", "GROUNDING",
    "CHANGE_VQA", "CHANGE_DESCRIPTION", "SAR_OPTICAL_FUSION",
]


def _load_eval_set(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data.get("samples", [])
    return data if isinstance(data, list) else []


def run(cfg: Dict[str, Any], output_dir: str, device: str, max_samples: int | None = None) -> Dict[str, Any]:
    rcfg = cfg.get("routing", {})
    eval_set_path = rcfg.get("eval_set")
    notes: List[str] = []

    if not is_configured(eval_set_path) or not os.path.isfile(eval_set_path):
        notes.append(f"skipped: routing eval set not found ({eval_set_path!r}). "
                     "Set ROUTING_EVAL_SET or ship routing_eval_set.json.")
        report = {"domain": DOMAIN, "status": "no_data", "notes": notes, "provenance": provenance()}
        write_json(output_dir, DOMAIN, report)
        append_markdown(output_dir, DOMAIN, TITLE, [], notes)
        logger.warning("Routing: %s", notes[-1])
        return report

    samples = _load_eval_set(eval_set_path)
    if max_samples is not None:
        samples = samples[:max_samples]
    if not samples:
        notes.append(f"skipped: routing eval set {eval_set_path!r} contains no samples.")
        report = {"domain": DOMAIN, "status": "no_data", "notes": notes, "provenance": provenance()}
        write_json(output_dir, DOMAIN, report)
        append_markdown(output_dir, DOMAIN, TITLE, [], notes)
        logger.warning("Routing: %s", notes[-1])
        return report

    from agent.task_classifier import TaskClassifier
    from evaluation.common.metrics import aggregate_routing_metrics, per_task_prf, routing_accuracy

    # Offline keyword+structural router (semantic router disabled by default).
    clf = TaskClassifier(semantic_router=None)

    gold: List[str] = []
    pred: List[str] = []
    per_sample: List[Dict[str, Any]] = []
    for s in samples:
        g = s.get("gold_task")
        if g is None:
            continue
        query = s.get("query", "")
        # `s.get("num_images", 1)` returns None when the key is present but null,
        # and int(None) raises; coerce defensively so one malformed curated entry
        # skips gracefully instead of aborting the whole routing domain.
        try:
            num_images = int(s.get("num_images") or 1)
        except (TypeError, ValueError):
            num_images = 1
        modalities = s.get("modalities", []) or []
        task, conf = clf.classify(query, num_images=num_images, modalities=modalities)
        gold.append(str(g))
        pred.append(task.value)
        per_sample.append({
            "query": query, "num_images": num_images, "modalities": modalities,
            "gold": str(g), "pred": task.value,
            "confidence": round(float(conf), 4), "correct": int(task.value == str(g)),
        })

    metrics = aggregate_routing_metrics(gold, pred, labels=_TASK_LABELS)

    # Majority-class baseline: always predict the most frequent gold label.
    majority = Counter(gold).most_common(1)[0][0] if gold else _TASK_LABELS[0]
    base_pred = [majority] * len(gold)
    base_acc = routing_accuracy(gold, base_pred)
    base_macro_f1 = per_task_prf(gold, base_pred, labels=_TASK_LABELS)["macro"]["f1"]

    rows = [
        comparison_row("TaskClassifier", "routing_eval_set", "accuracy", base_acc, metrics["accuracy"]),
        comparison_row("TaskClassifier", "routing_eval_set", "macro_f1", base_macro_f1, metrics["macro"]["f1"]),
    ]
    notes.append(f"Baseline = majority-class predictor (always '{majority}').")
    notes.append("SatQuery = keyword+structural TaskClassifier (offline; semantic router disabled).")
    notes.append(f"Evaluated on {len(gold)} curated in-house gold queries (not benchmark test labels).")

    report = {
        "domain": DOMAIN,
        "status": "ok",
        "n_samples": len(gold),
        "router": metrics,
        "baseline_majority_class": {"label": majority, "accuracy": base_acc, "macro_f1": base_macro_f1},
        "comparison": rows,
        "per_sample": per_sample,
        "notes": notes,
        "provenance": provenance(),
    }
    write_json(output_dir, DOMAIN, report)
    append_markdown(output_dir, DOMAIN, TITLE, rows, notes)
    logger.info("Routing accuracy: %.4f (majority baseline %.4f) over %d queries",
                metrics["accuracy"], base_acc, len(gold))
    return report


def main() -> None:
    args = build_parser(DOMAIN).parse_args()
    cfg, output_dir, device = init_run(args, DOMAIN)
    run(cfg, output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
