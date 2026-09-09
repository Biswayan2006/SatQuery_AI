"""
SatQuery AI — Detection / Grounding Metrics
===========================================
Bounding-box metrics for the grounding task, built on the existing
``confidence.metrics.box_iou`` primitive (never re-implemented).

Boxes are ``[x1, y1, x2, y2]`` in a consistent coordinate convention
(normalised or absolute — both sides must match).

Metrics
-------
  * ``acc_at_iou``  — fraction of samples whose best predicted box overlaps a
                      reference box at IoU ≥ threshold (Acc@0.5 by default).
  * ``mean_iou``    — mean over samples of the best IoU between any predicted
                      and any reference box (mIoU for single-object grounding).
  * ``pr_at_iou``   — precision / recall / F1 at an IoU threshold via greedy
                      one-to-one matching across a set of samples.

All functions are safe on empty input and never raise on malformed boxes
(``box_iou`` already degrades to 0.0).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from confidence.metrics import box_iou

Box = Sequence[float]


# ── Per-sample helpers ──────────────────────────────────────────────────────────

def _best_iou(pred_boxes: Sequence[Box], ref_boxes: Sequence[Box]) -> float:
    """Best IoU between any predicted box and any reference box (0.0 if empty)."""
    preds = [p for p in (pred_boxes or []) if p is not None]
    refs = [r for r in (ref_boxes or []) if r is not None]
    if not preds or not refs:
        return 0.0
    return max(box_iou(p, r) for p in preds for r in refs)


def mean_iou(
    predictions: Sequence[Sequence[Box]],
    references: Sequence[Sequence[Box]],
) -> float:
    """Mean best-IoU over samples. ``predictions[i]`` vs ``references[i]``."""
    n = len(predictions)
    if n == 0 or len(references) != n:
        return 0.0
    total = sum(_best_iou(predictions[i], references[i]) for i in range(n))
    return round(total / n, 4)


def acc_at_iou(
    predictions: Sequence[Sequence[Box]],
    references: Sequence[Sequence[Box]],
    threshold: float = 0.5,
) -> float:
    """Fraction of samples with best IoU ≥ ``threshold`` (Acc@IoU)."""
    n = len(predictions)
    if n == 0 or len(references) != n:
        return 0.0
    hits = sum(1 for i in range(n) if _best_iou(predictions[i], references[i]) >= threshold)
    return round(hits / n, 4)


# ── Detection PR (greedy one-to-one matching) ───────────────────────────────────

def pr_at_iou(
    predictions: Sequence[Sequence[Box]],
    references: Sequence[Sequence[Box]],
    threshold: float = 0.5,
    scores: Sequence[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Precision / recall / F1 at an IoU threshold across all samples.

    Within each sample, predicted boxes are matched greedily (highest IoU first,
    or by descending ``scores`` when provided) to reference boxes one-to-one; a
    match counts as a TP when IoU ≥ threshold.  Unmatched predictions are FP,
    unmatched references are FN.  Aggregated over the whole set.
    """
    n = len(predictions)
    if n == 0 or len(references) != n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "threshold": threshold}

    tp = fp = fn = 0
    for i in range(n):
        preds = [p for p in (predictions[i] or []) if p is not None]
        refs = [r for r in (references[i] or []) if r is not None]

        # Order predictions by score (desc) when available, else by best IoU.
        if scores is not None and i < len(scores) and scores[i] is not None \
                and len(scores[i]) == len(preds):
            order = sorted(range(len(preds)), key=lambda j: scores[i][j], reverse=True)
        else:
            order = list(range(len(preds)))

        used_ref = set()
        for j in order:
            best_iou_val, best_r = 0.0, -1
            for ri, r in enumerate(refs):
                if ri in used_ref:
                    continue
                iou = box_iou(preds[j], r)
                if iou > best_iou_val:
                    best_iou_val, best_r = iou, ri
            if best_r >= 0 and best_iou_val >= threshold:
                tp += 1
                used_ref.add(best_r)
            else:
                fp += 1
        fn += len(refs) - len(used_ref)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "threshold": threshold,
    }


def aggregate_detection_metrics(
    predictions: Sequence[Sequence[Box]],
    references: Sequence[Sequence[Box]],
    thresholds: Sequence[float] = (0.5, 0.75),
    scores: Sequence[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Full grounding metric bundle: mIoU, Acc@each-threshold, PR@0.5.

    Returns ``{"m_iou", "acc@0.5", "acc@0.75", ..., "pr@0.5": {...}, "n_samples"}``.
    """
    n = len(predictions)
    out: Dict[str, Any] = {"m_iou": mean_iou(predictions, references), "n_samples": n}
    for t in thresholds:
        out[f"acc@{t}"] = acc_at_iou(predictions, references, threshold=t)
    out["pr@0.5"] = pr_at_iou(predictions, references, threshold=0.5, scores=scores)
    return out
