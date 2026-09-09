"""
SatQuery AI — Evaluation Metrics
================================
Single import surface for every metric the framework uses.

Existing metrics are RE-EXPORTED from their canonical locations (never
re-implemented):

  * VQA            — ``training.vqa_utils``          (exact_match, vqa_accuracy, token_f1, aggregate_metrics)
  * Retrieval      — ``training.evaluate_clip``      (retrieval_recall, zero_shot_accuracy)
  * Multilabel     — ``training.evaluate``           (multilabel_metrics)
  * Calibration    — ``confidence.metrics``          (ECE, MCE, Brier, reliability_stats)
  * Box IoU        — ``confidence.metrics``          (box_iou, mean_best_iou)

New metrics implemented here (pure Python + numpy, offline-testable):

  * Captioning     — bleu, rouge_l, meteor, cider, aggregate_caption_metrics
  * Detection      — acc_at_iou, mean_iou, pr_at_iou, aggregate_detection_metrics
  * Change mask    — change_mask_metrics
  * Routing        — accuracy, per_task_prf, confusion_matrix, aggregate_routing_metrics
"""
from __future__ import annotations

# ── Re-exports of existing metrics (canonical implementations) ──────────────────
from confidence.metrics import (  # noqa: F401
    box_iou,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    mean_best_iou,
    reliability_curve,
    reliability_stats,
)

# ── New metrics implemented in this package ─────────────────────────────────────
from evaluation.common.metrics.captioning import (  # noqa: F401
    aggregate_caption_metrics,
    bleu,
    cider,
    meteor,
    rouge_l,
)
from evaluation.common.metrics.change_mask import (  # noqa: F401
    accumulate_confusion,
    change_mask_metrics,
    metrics_from_confusion,
)
from evaluation.common.metrics.detection import (  # noqa: F401
    acc_at_iou,
    aggregate_detection_metrics,
    mean_iou,
    pr_at_iou,
)
from evaluation.common.metrics.routing import (  # noqa: F401
    aggregate_routing_metrics,
    confusion_matrix,
    per_task_prf,
    routing_accuracy,
)


def vqa_metrics_module():
    """Lazy accessor for VQA metrics (kept lazy: pulls in torch-adjacent deps)."""
    from training.vqa_utils import (  # noqa: F401
        aggregate_metrics,
        exact_match,
        token_f1,
        vqa_accuracy,
    )
    return {
        "aggregate_metrics": aggregate_metrics,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "vqa_accuracy": vqa_accuracy,
    }


__all__ = [
    # calibration + iou (re-exported)
    "box_iou", "mean_best_iou", "expected_calibration_error",
    "maximum_calibration_error", "brier_score", "reliability_curve", "reliability_stats",
    # captioning
    "bleu", "rouge_l", "meteor", "cider", "aggregate_caption_metrics",
    # detection
    "acc_at_iou", "mean_iou", "pr_at_iou", "aggregate_detection_metrics",
    # change
    "change_mask_metrics", "accumulate_confusion", "metrics_from_confusion",
    # routing
    "routing_accuracy", "per_task_prf", "confusion_matrix", "aggregate_routing_metrics",
    # lazy vqa
    "vqa_metrics_module",
]
