"""
SatQuery AI — Routing / Task-Classification Metrics
===================================================
Evaluates the ``TaskClassifier`` router: how often it maps a query (+ image
count + modalities) to the correct ``TaskType``.

Metrics
-------
  * ``routing_accuracy``  — overall fraction correct.
  * ``per_task_prf``      — precision / recall / F1 per task label, plus macro
                            and micro (weighted) averages.
  * ``confusion_matrix``  — labelled gold×predicted count matrix.

Labels are plain strings (the ``TaskType.value``s), so this module has no
dependency on the model code and is fully offline-testable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence


def routing_accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Overall classification accuracy. 0.0 on empty / mismatched input."""
    n = len(gold)
    if n == 0 or len(pred) != n:
        return 0.0
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return round(correct / n, 4)


def confusion_matrix(
    gold: Sequence[str],
    pred: Sequence[str],
    labels: Sequence[str] = None,
) -> Dict[str, Any]:
    """
    Confusion matrix as a nested dict.

    Returns ``{"labels": [...], "matrix": [[...]]}`` where ``matrix[i][j]`` is
    the count of samples with gold label ``labels[i]`` predicted as
    ``labels[j]``.  Label order is sorted unless ``labels`` is given.
    """
    if labels is None:
        labels = sorted(set(gold) | set(pred))
    idx = {lab: i for i, lab in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    for g, p in zip(gold, pred):
        if g in idx and p in idx:
            matrix[idx[g]][idx[p]] += 1
    return {"labels": list(labels), "matrix": matrix}


def per_task_prf(
    gold: Sequence[str],
    pred: Sequence[str],
    labels: Sequence[str] = None,
) -> Dict[str, Any]:
    """
    Per-label precision/recall/F1 with macro and weighted (micro-by-support)
    averages.

    Returns::

        {
          "per_task": {label: {precision, recall, f1, support}},
          "macro": {precision, recall, f1},
          "weighted": {precision, recall, f1},
          "n_samples": int,
        }
    """
    n = len(gold)
    if n == 0 or len(pred) != n:
        return {"per_task": {}, "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                "weighted": {"precision": 0.0, "recall": 0.0, "f1": 0.0}, "n_samples": 0}

    if labels is None:
        labels = sorted(set(gold) | set(pred))

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = Counter(gold)

    for g, p in zip(gold, pred):
        if g == p:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1

    per_task: Dict[str, Dict[str, float]] = {}
    macro_p = macro_r = macro_f = 0.0
    wt_p = wt_r = wt_f = 0.0
    for lab in labels:
        t, f_p, f_n = tp[lab], fp[lab], fn[lab]
        prec = t / (t + f_p) if (t + f_p) else 0.0
        rec = t / (t + f_n) if (t + f_n) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        sup = support.get(lab, 0)
        per_task[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sup,
        }
        macro_p += prec
        macro_r += rec
        macro_f += f1
        wt_p += prec * sup
        wt_r += rec * sup
        wt_f += f1 * sup

    k = len(labels) or 1
    macro = {"precision": round(macro_p / k, 4), "recall": round(macro_r / k, 4),
             "f1": round(macro_f / k, 4)}
    weighted = {"precision": round(wt_p / n, 4), "recall": round(wt_r / n, 4),
                "f1": round(wt_f / n, 4)}
    return {"per_task": per_task, "macro": macro, "weighted": weighted, "n_samples": n}


def aggregate_routing_metrics(
    gold: Sequence[str],
    pred: Sequence[str],
    labels: Sequence[str] = None,
) -> Dict[str, Any]:
    """Full routing bundle: accuracy + per-task PRF + confusion matrix."""
    prf = per_task_prf(gold, pred, labels)
    return {
        "accuracy": routing_accuracy(gold, pred),
        "per_task": prf["per_task"],
        "macro": prf["macro"],
        "weighted": prf["weighted"],
        "confusion_matrix": confusion_matrix(gold, pred, labels),
        "n_samples": prf["n_samples"],
    }
