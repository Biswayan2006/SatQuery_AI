"""
SatQuery AI — Change-Detection Mask Metrics
===========================================
Pixel-level binary change-mask metrics: precision, recall, F1, IoU (Jaccard),
and overall accuracy.  These evaluate a predicted binary change map against a
ground-truth mask (the "changed" class is positive).

Inputs are array-likes (numpy arrays, nested lists) of the same shape; values
are treated as binary via ``> threshold``.  Safe on empty / mismatched input
(returns zeros with a ``valid=False`` flag rather than raising).

The positive class is *change*.  Precision/recall/F1/IoU are computed on that
class; overall accuracy counts both classes.  Metrics aggregate correctly
across a whole dataset when TP/FP/FN/TN are summed (use ``accumulate_confusion``
then ``metrics_from_confusion``).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


def _binarize(mask: Any, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    return arr > threshold


def accumulate_confusion(pred: Any, gold: Any, threshold: float = 0.5) -> Dict[str, int]:
    """Return TP/FP/FN/TN pixel counts for one (pred, gold) mask pair."""
    p = _binarize(pred, threshold).ravel()
    g = _binarize(gold, threshold).ravel()
    if p.shape[0] == 0 or p.shape[0] != g.shape[0]:
        return {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "valid": False}
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "valid": True}


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute change metrics from summed confusion counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    total = tp + fp + fn + tn
    overall_acc = (tp + tn) / total if total else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "overall_accuracy": round(overall_acc, 4),
    }


def change_mask_metrics(pred: Any, gold: Any, threshold: float = 0.5) -> Dict[str, float]:
    """
    Precision/recall/F1/IoU/overall-accuracy for a single change-mask pair.

    For dataset-level aggregation, sum ``accumulate_confusion`` across pairs and
    call ``metrics_from_confusion`` on the totals (correct micro-average).
    """
    c = accumulate_confusion(pred, gold, threshold)
    if not c["valid"]:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0,
                "overall_accuracy": 0.0, "valid": False}
    m = metrics_from_confusion(c["tp"], c["fp"], c["fn"], c["tn"])
    m["valid"] = True
    return m
