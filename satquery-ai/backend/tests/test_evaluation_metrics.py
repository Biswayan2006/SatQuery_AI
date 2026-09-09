"""
SatQuery AI — Evaluation Framework Unit Tests
==============================================
Offline tests for every metric and support function in the evaluation package.
No datasets, no models, no network access required.

Coverage
--------
  1.  Captioning metrics: BLEU-1..4, ROUGE-L, METEOR, CIDEr, aggregate
  2.  Change-mask metrics: accumulate_confusion, metrics_from_confusion,
                           change_mask_metrics
  3.  Detection metrics:   acc_at_iou, mean_iou, pr_at_iou,
                           aggregate_detection_metrics
  4.  Routing metrics:     routing_accuracy, per_task_prf, confusion_matrix,
                           aggregate_routing_metrics
  5.  Config resolution:   load_config (${ENV:-default}), is_configured,
                           path_exists, resolve_device
  6.  Reporting:           comparison_row, build_comparison_table, write_json,
                           append_markdown
  7.  Variants:            resolve_variants (baseline available, adapted n/a,
                           adapted available with checkpoint)
  8.  Metrics __init__:    re-export surface (accumulate_confusion,
                           metrics_from_confusion now included — the fixed bug)
  9.  Smoke imports:       every evaluation/<domain>/run.py importable without
                           touching torch / HuggingFace / datasets

All expected values are computed from the canonical implementations, not from
external references, so the tests pin behaviour rather than benchmark accuracy.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

# ── Make the backend package importable ───────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Captioning metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestBLEU:
    """Tests for evaluation.common.metrics.captioning.bleu"""

    def _bleu(self, hyp, refs, max_n=4):
        from evaluation.common.metrics.captioning import bleu
        return bleu(hyp, refs, max_n=max_n)

    def test_perfect_match_all_ones(self):
        b = self._bleu("the cat sat on the mat", ["the cat sat on the mat"])
        assert b["bleu_1"] == 1.0
        assert b["bleu_2"] == 1.0
        assert b["bleu_4"] == 1.0

    def test_one_token_differs_bleu1(self):
        # "the cat sat on the mat" vs "the cat sat on a mat"
        # 5 of 6 unigrams clipped; BP=1 (equal length)  → 5/6 ≈ 0.8333
        b = self._bleu("the cat sat on the mat", ["the cat sat on a mat"])
        assert abs(b["bleu_1"] - 0.8333) < 0.0005

    def test_one_token_differs_bleu4(self):
        # Computed value from implementation
        b = self._bleu("the cat sat on the mat", ["the cat sat on a mat"])
        assert abs(b["bleu_4"] - 0.5373) < 0.001

    def test_empty_hypothesis_all_zero(self):
        b = self._bleu("", ["the cat sat on the mat"])
        for k in range(1, 5):
            assert b[f"bleu_{k}"] == 0.0

    def test_empty_reference_all_zero(self):
        b = self._bleu("the cat sat", [""])
        for k in range(1, 5):
            assert b[f"bleu_{k}"] == 0.0

    def test_no_overlap_all_zero(self):
        b = self._bleu("foo bar baz", ["xyz abc def"])
        assert b["bleu_1"] == 0.0

    def test_brevity_penalty_applied_for_short_hypothesis(self):
        # hypothesis much shorter than reference → BP < 1
        b = self._bleu("cat", ["the cat sat on the mat"])
        # BP = exp(1 - 6/1) = exp(-5) ≈ 0.0067 — BLEU-1 should be well below 1
        assert b["bleu_1"] < 0.1

    def test_returns_all_four_keys(self):
        b = self._bleu("hello world", ["hello world"])
        assert set(b.keys()) == {"bleu_1", "bleu_2", "bleu_3", "bleu_4"}

    def test_multiple_references_uses_best(self):
        # One ref is a perfect match, one is completely different
        b = self._bleu("the cat sat", ["xyz abc def", "the cat sat"])
        assert b["bleu_1"] == 1.0


class TestROUGEL:
    """Tests for evaluation.common.metrics.captioning.rouge_l"""

    def _rl(self, hyp, refs):
        from evaluation.common.metrics.captioning import rouge_l
        return rouge_l(hyp, refs)

    def test_perfect_match(self):
        assert self._rl("the cat sat on the mat", ["the cat sat on the mat"]) == 1.0

    def test_one_token_differs(self):
        # LCS = ["the","cat","sat","on","mat"] len=5, prec=5/6, rec=5/6
        # β=1.2, F = (1+1.44)*(5/6)*(5/6) / ((5/6)+1.44*(5/6)) = (5/6)
        val = self._rl("the cat sat on the mat", ["the cat sat on a mat"])
        assert abs(val - 0.8333) < 0.0005

    def test_empty_hypothesis(self):
        assert self._rl("", ["the cat sat"]) == 0.0

    def test_empty_reference(self):
        assert self._rl("the cat sat", [""]) == 0.0

    def test_no_overlap(self):
        assert self._rl("foo bar", ["xyz abc"]) == 0.0

    def test_partial_overlap(self):
        # "a b c d" vs "a b x y" → LCS=["a","b"] len=2, prec=2/4=0.5, rec=2/4=0.5
        val = self._rl("a b c d", ["a b x y"])
        assert 0.0 < val < 1.0

    def test_best_reference_used(self):
        val_best = self._rl("the cat sat", ["the cat sat", "xyz abc"])
        assert val_best == 1.0


class TestMETEOR:
    """Tests for evaluation.common.metrics.captioning.meteor"""

    def _m(self, hyp, refs):
        from evaluation.common.metrics.captioning import meteor
        return meteor(hyp, refs)

    def test_perfect_match_near_one(self):
        # Penalty γ*(chunks/matches)^β with chunks=1, matches=4 → small penalty
        val = self._m("the cat sat on the mat", ["the cat sat on the mat"])
        assert val > 0.99

    def test_empty_hypothesis(self):
        assert self._m("", ["the cat sat"]) == 0.0

    def test_no_overlap(self):
        assert self._m("foo bar", ["xyz abc"]) == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        val = self._m("the cat sat", ["the cat sat on the mat"])
        assert 0.0 < val <= 1.0

    def test_best_reference_used(self):
        # "the cat sat" vs ["xyz abc", "the cat sat"] — the perfect match ref
        # should be chosen, giving a high score.  On a 3-token hypothesis the
        # fragmentation penalty is non-trivial (1 chunk / 3 matches ≈ 0.33^3 * 0.5),
        # so the score is 0.98+ rather than 1.0.
        val = self._m("the cat sat", ["xyz abc", "the cat sat"])
        assert val > 0.95


class TestCIDEr:
    """Tests for evaluation.common.metrics.captioning.cider"""

    def _c(self, hyps, refs, **kw):
        from evaluation.common.metrics.captioning import cider
        return cider(hyps, refs, **kw)

    def test_empty_returns_zero(self):
        assert self._c([], [])["cider"] == 0.0

    def test_mismatched_lengths_returns_zero(self):
        assert self._c(["a b"], [["a b"], ["c d"]])["cider"] == 0.0

    def test_two_sample_distinct_captions(self):
        # Two images, each described by unique captions → high consensus within each
        result = self._c(["a b c", "d e f"], [["a b c"], ["d e f"]])
        assert result["cider"] > 0.0  # non-zero corpus CIDEr
        assert result["cider"] == pytest.approx(7.5, abs=0.1)

    def test_per_image_scores_length(self):
        result = self._c(["a b c", "d e f"], [["a b c"], ["d e f"]])
        assert len(result["cider_per"]) == 2


class TestAggregateCaptionMetrics:
    """Tests for aggregate_caption_metrics corpus bundling"""

    def test_perfect_corpus(self):
        from evaluation.common.metrics.captioning import aggregate_caption_metrics
        hyps = ["the cat sat on the mat", "a dog ran in the park"]
        refs = [["the cat sat on the mat"], ["a dog ran in the park"]]
        m = aggregate_caption_metrics(hyps, refs)
        assert m["bleu_1"] == 1.0
        assert m["bleu_4"] == 1.0
        assert m["rouge_l"] == 1.0
        assert m["meteor"] > 0.99
        assert m["n_samples"] == 2

    def test_empty_corpus(self):
        from evaluation.common.metrics.captioning import aggregate_caption_metrics
        m = aggregate_caption_metrics([], [])
        assert m["bleu_1"] == 0.0
        assert m["n_samples"] == 0

    def test_output_keys_complete(self):
        from evaluation.common.metrics.captioning import aggregate_caption_metrics
        m = aggregate_caption_metrics(["hello"], [["hello"]])
        expected_keys = {"bleu_1", "bleu_2", "bleu_3", "bleu_4",
                         "rouge_l", "meteor", "cider", "n_samples"}
        assert expected_keys.issubset(set(m.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# 2. Change-mask metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestAccumulateConfusion:
    """Tests for accumulate_confusion (now properly exported from metrics.__init__)"""

    def _ac(self, pred, gold, threshold=0.5):
        from evaluation.common.metrics import accumulate_confusion
        return accumulate_confusion(pred, gold, threshold)

    def test_all_true_positives(self):
        pred = np.ones((3, 3))
        gold = np.ones((3, 3))
        c = self._ac(pred, gold)
        assert c["tp"] == 9
        assert c["fp"] == 0
        assert c["fn"] == 0
        assert c["tn"] == 0
        assert c["valid"] is True

    def test_all_true_negatives(self):
        pred = np.zeros((3, 3))
        gold = np.zeros((3, 3))
        c = self._ac(pred, gold)
        assert c["tp"] == 0
        assert c["tn"] == 9
        assert c["valid"] is True

    def test_half_positive_half_negative(self):
        # pred: [[1,1],[0,0]]   gold: [[1,1],[1,1]]
        # TP=2 FP=0 FN=2 TN=0
        pred = np.array([[1, 1], [0, 0]], dtype=float)
        gold = np.array([[1, 1], [1, 1]], dtype=float)
        c = self._ac(pred, gold)
        assert c["tp"] == 2
        assert c["fp"] == 0
        assert c["fn"] == 2
        assert c["tn"] == 0

    def test_mismatched_shapes_valid_false(self):
        c = self._ac(np.zeros((2, 2)), np.zeros((3, 3)))
        assert c["valid"] is False

    def test_boolean_input(self):
        pred = np.array([True, False, True])
        gold = np.array([True, True, False])
        c = self._ac(pred, gold)
        assert c["tp"] == 1
        assert c["fp"] == 1
        assert c["fn"] == 1
        assert c["tn"] == 0


class TestMetricsFromConfusion:
    """Tests for metrics_from_confusion (now properly exported)"""

    def _mfc(self, tp, fp, fn, tn):
        from evaluation.common.metrics import metrics_from_confusion
        return metrics_from_confusion(tp, fp, fn, tn)

    def test_perfect_precision_recall(self):
        m = self._mfc(tp=4, fp=0, fn=0, tn=0)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["iou"] == 1.0
        assert m["overall_accuracy"] == 1.0

    def test_half_recall(self):
        # tp=2, fp=0, fn=2, tn=0 → P=1.0, R=0.5, F1=0.6667, IoU=0.5, OA=0.5
        m = self._mfc(tp=2, fp=0, fn=2, tn=0)
        assert m["precision"] == 1.0
        assert abs(m["recall"] - 0.5) < 1e-4
        assert abs(m["f1"] - 0.6667) < 0.001
        assert abs(m["iou"] - 0.5) < 1e-4
        assert abs(m["overall_accuracy"] - 0.5) < 1e-4

    def test_zero_division_safe(self):
        m = self._mfc(tp=0, fp=0, fn=0, tn=0)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["iou"] == 0.0
        assert m["overall_accuracy"] == 0.0

    def test_round_to_4_decimals(self):
        m = self._mfc(tp=1, fp=2, fn=0, tn=0)
        # precision = 1/3 ≈ 0.3333
        assert m["precision"] == pytest.approx(0.3333, abs=0.0001)


class TestChangeMaskMetrics:
    """Tests for change_mask_metrics"""

    def _cmm(self, pred, gold, threshold=0.5):
        from evaluation.common.metrics.change_mask import change_mask_metrics
        return change_mask_metrics(pred, gold, threshold)

    def test_perfect_mask(self):
        m = self._cmm(np.ones((4, 4)), np.ones((4, 4)))
        assert m["f1"] == 1.0
        assert m["iou"] == 1.0
        assert m.get("valid", True) is True

    def test_invalid_shape_returns_zeros(self):
        m = self._cmm(np.zeros((2, 2)), np.zeros((3, 3)))
        assert m["f1"] == 0.0
        assert m.get("valid") is False

    def test_all_false_positive(self):
        # pred = all 1, gold = all 0 → FP only → precision=0, recall=0
        m = self._cmm(np.ones((3, 3)), np.zeros((3, 3)))
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. Detection metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestAccAtIoU:
    def test_perfect_boxes(self):
        from evaluation.common.metrics.detection import acc_at_iou
        preds = [[[0.1, 0.1, 0.5, 0.5]]]
        refs  = [[[0.1, 0.1, 0.5, 0.5]]]
        assert acc_at_iou(preds, refs, threshold=0.5) == 1.0

    def test_no_overlap(self):
        from evaluation.common.metrics.detection import acc_at_iou
        preds = [[[0.0, 0.0, 0.2, 0.2]]]
        refs  = [[[0.8, 0.8, 1.0, 1.0]]]
        assert acc_at_iou(preds, refs, threshold=0.5) == 0.0

    def test_empty_input(self):
        from evaluation.common.metrics.detection import acc_at_iou
        assert acc_at_iou([], [], threshold=0.5) == 0.0

    def test_partial_hit(self):
        from evaluation.common.metrics.detection import acc_at_iou
        # First sample hits, second misses
        preds = [[[0.1, 0.1, 0.5, 0.5]], [[0.0, 0.0, 0.1, 0.1]]]
        refs  = [[[0.1, 0.1, 0.5, 0.5]], [[0.8, 0.8, 1.0, 1.0]]]
        assert acc_at_iou(preds, refs, threshold=0.5) == 0.5


class TestMeanIoU:
    def test_perfect(self):
        from evaluation.common.metrics.detection import mean_iou
        preds = [[[0.1, 0.1, 0.5, 0.5]]]
        refs  = [[[0.1, 0.1, 0.5, 0.5]]]
        assert mean_iou(preds, refs) == 1.0

    def test_no_overlap(self):
        from evaluation.common.metrics.detection import mean_iou
        preds = [[[0.0, 0.0, 0.2, 0.2]]]
        refs  = [[[0.8, 0.8, 1.0, 1.0]]]
        assert mean_iou(preds, refs) == 0.0

    def test_empty(self):
        from evaluation.common.metrics.detection import mean_iou
        assert mean_iou([], []) == 0.0


class TestPRAtIoU:
    def test_two_perfect_samples(self):
        from evaluation.common.metrics.detection import pr_at_iou
        preds = [[[0.1, 0.1, 0.5, 0.5]], [[0.3, 0.3, 0.7, 0.7]]]
        refs  = [[[0.1, 0.1, 0.5, 0.5]], [[0.3, 0.3, 0.7, 0.7]]]
        result = pr_at_iou(preds, refs, threshold=0.5)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["tp"] == 2
        assert result["fp"] == 0

    def test_all_false_positives(self):
        from evaluation.common.metrics.detection import pr_at_iou
        # predictions don't overlap with references
        preds = [[[0.0, 0.0, 0.1, 0.1]]]
        refs  = [[[0.8, 0.8, 1.0, 1.0]]]
        result = pr_at_iou(preds, refs, threshold=0.5)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["fp"] == 1
        assert result["fn"] == 1

    def test_empty(self):
        from evaluation.common.metrics.detection import pr_at_iou
        result = pr_at_iou([], [], threshold=0.5)
        assert result["precision"] == 0.0
        assert result["f1"] == 0.0


class TestAggregateDetectionMetrics:
    def test_bundle_keys(self):
        from evaluation.common.metrics.detection import aggregate_detection_metrics
        preds = [[[0.1, 0.1, 0.5, 0.5]]]
        refs  = [[[0.1, 0.1, 0.5, 0.5]]]
        m = aggregate_detection_metrics(preds, refs, thresholds=[0.5, 0.75])
        assert "m_iou" in m
        assert "acc@0.5" in m
        assert "acc@0.75" in m
        assert "pr@0.5" in m
        assert m["n_samples"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Routing metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestRoutingAccuracy:
    def test_three_of_four_correct(self):
        from evaluation.common.metrics.routing import routing_accuracy
        gold = ["SINGLE_VQA", "CAPTIONING", "GROUNDING", "SINGLE_VQA"]
        pred = ["SINGLE_VQA", "CAPTIONING", "SINGLE_VQA", "SINGLE_VQA"]
        assert routing_accuracy(gold, pred) == 0.75

    def test_all_correct(self):
        from evaluation.common.metrics.routing import routing_accuracy
        gold = ["A", "B", "C"]
        assert routing_accuracy(gold, gold) == 1.0

    def test_all_wrong(self):
        from evaluation.common.metrics.routing import routing_accuracy
        gold = ["A", "B", "C"]
        pred = ["B", "C", "A"]
        assert routing_accuracy(gold, pred) == 0.0

    def test_empty(self):
        from evaluation.common.metrics.routing import routing_accuracy
        assert routing_accuracy([], []) == 0.0

    def test_mismatched_length_zero(self):
        from evaluation.common.metrics.routing import routing_accuracy
        assert routing_accuracy(["A", "B"], ["A"]) == 0.0


class TestPerTaskPRF:
    """Hand-verified expected values for the 3-sample routing scenario."""

    def _prf(self):
        from evaluation.common.metrics.routing import per_task_prf
        gold = ["SINGLE_VQA", "CAPTIONING", "GROUNDING", "SINGLE_VQA"]
        pred = ["SINGLE_VQA", "CAPTIONING", "SINGLE_VQA", "SINGLE_VQA"]
        return per_task_prf(gold, pred)

    def test_single_vqa_precision(self):
        # TP=2, FP=1 (GROUNDING predicted as SINGLE_VQA) → P = 2/3 ≈ 0.6667
        prf = self._prf()
        assert prf["per_task"]["SINGLE_VQA"]["precision"] == pytest.approx(0.6667, abs=0.0001)

    def test_single_vqa_recall_one(self):
        # TP=2, FN=0 → R=1.0
        prf = self._prf()
        assert prf["per_task"]["SINGLE_VQA"]["recall"] == 1.0

    def test_captioning_perfect(self):
        prf = self._prf()
        assert prf["per_task"]["CAPTIONING"]["precision"] == 1.0
        assert prf["per_task"]["CAPTIONING"]["recall"] == 1.0
        assert prf["per_task"]["CAPTIONING"]["f1"] == 1.0

    def test_grounding_zero_f1(self):
        # Only FN — never predicted → P=0.0, R=0.0, F1=0.0
        prf = self._prf()
        assert prf["per_task"]["GROUNDING"]["f1"] == 0.0

    def test_n_samples(self):
        prf = self._prf()
        assert prf["n_samples"] == 4

    def test_empty_input(self):
        from evaluation.common.metrics.routing import per_task_prf
        prf = per_task_prf([], [])
        assert prf["n_samples"] == 0
        assert prf["macro"]["f1"] == 0.0


class TestConfusionMatrix:
    def test_shape_and_labels(self):
        from evaluation.common.metrics.routing import confusion_matrix
        gold = ["CAPTIONING", "GROUNDING", "SINGLE_VQA", "SINGLE_VQA"]
        pred = ["CAPTIONING", "SINGLE_VQA", "SINGLE_VQA", "SINGLE_VQA"]
        cm = confusion_matrix(gold, pred)
        assert cm["labels"] == ["CAPTIONING", "GROUNDING", "SINGLE_VQA"]
        # matrix[0][0] = CAPTIONING→CAPTIONING = 1
        assert cm["matrix"][0][0] == 1
        # matrix[1][2] = GROUNDING→SINGLE_VQA = 1
        assert cm["matrix"][1][2] == 1
        # matrix[2][2] = SINGLE_VQA→SINGLE_VQA = 2
        assert cm["matrix"][2][2] == 2

    def test_custom_labels_preserves_order(self):
        from evaluation.common.metrics.routing import confusion_matrix
        gold = ["A", "B"]
        pred = ["B", "A"]
        cm = confusion_matrix(gold, pred, labels=["B", "A"])
        assert cm["labels"] == ["B", "A"]

    def test_empty_input(self):
        from evaluation.common.metrics.routing import confusion_matrix
        cm = confusion_matrix([], [])
        assert cm["labels"] == []
        assert cm["matrix"] == []


class TestAggregateRoutingMetrics:
    def test_keys_present(self):
        from evaluation.common.metrics.routing import aggregate_routing_metrics
        gold = ["SINGLE_VQA", "CAPTIONING"]
        pred = ["SINGLE_VQA", "CAPTIONING"]
        m = aggregate_routing_metrics(gold, pred)
        assert "accuracy" in m
        assert "per_task" in m
        assert "macro" in m
        assert "weighted" in m
        assert "confusion_matrix" in m
        assert "n_samples" in m

    def test_perfect_accuracy_and_f1(self):
        from evaluation.common.metrics.routing import aggregate_routing_metrics
        labels = ["A", "B", "C"]
        gold = labels * 3
        m = aggregate_routing_metrics(gold, gold, labels=labels)
        assert m["accuracy"] == 1.0
        assert m["macro"]["f1"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Config resolution
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadConfig:
    """Tests for evaluation/common/config.py"""

    def _write_yaml(self, content: str) -> str:
        """Write YAML to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            return f.name

    def test_resolves_env_var(self, monkeypatch):
        from evaluation.common.config import load_config
        monkeypatch.setenv("TEST_EVAL_VAR", "/my/dataset/path")
        yaml_text = 'datasets:\n  bigearthnet:\n    data_dir: "${TEST_EVAL_VAR}"\n'
        path = self._write_yaml(yaml_text)
        try:
            cfg = load_config(path)
            assert cfg["datasets"]["bigearthnet"]["data_dir"] == "/my/dataset/path"
        finally:
            os.unlink(path)

    def test_uses_default_when_var_unset(self, monkeypatch):
        from evaluation.common.config import load_config
        monkeypatch.delenv("MISSING_EVAL_VAR", raising=False)
        yaml_text = 'data_dir: "${MISSING_EVAL_VAR:-/fallback/path}"\n'
        path = self._write_yaml(yaml_text)
        try:
            cfg = load_config(path)
            assert cfg["data_dir"] == "/fallback/path"
        finally:
            os.unlink(path)

    def test_empty_default_when_var_unset(self, monkeypatch):
        from evaluation.common.config import load_config
        monkeypatch.delenv("ALSO_MISSING", raising=False)
        yaml_text = 'data_dir: "${ALSO_MISSING:-}"\n'
        path = self._write_yaml(yaml_text)
        try:
            cfg = load_config(path)
            assert cfg["data_dir"] == ""
        finally:
            os.unlink(path)

    def test_missing_file_raises(self):
        from evaluation.common.config import load_config
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_config("/nonexistent/path/config.yaml")

    def test_is_configured_true_for_non_empty(self):
        from evaluation.common.config import is_configured
        assert is_configured("/some/real/path") is True

    def test_is_configured_false_for_empty(self):
        from evaluation.common.config import is_configured
        assert is_configured("") is False
        assert is_configured(None) is False

    def test_resolve_device_cpu(self):
        from evaluation.common.config import resolve_device
        assert resolve_device("cpu") == "cpu"

    def test_resolve_device_auto_returns_string(self):
        from evaluation.common.config import resolve_device
        result = resolve_device("auto")
        assert result in ("cpu", "cuda")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Reporting
# ══════════════════════════════════════════════════════════════════════════════

class TestComparisonRow:
    def _row(self, baseline, adapted, higher=True):
        from evaluation.common.reporting import comparison_row
        return comparison_row("BLIP", "vrsbench", "Exact Match",
                               baseline, adapted, higher_is_better=higher)

    def test_keys(self):
        row = self._row(0.42, 0.55)
        assert "model" in row
        assert "dataset" in row
        assert "metric" in row
        assert "baseline" in row
        assert "satquery" in row
        assert "improvement" in row

    def test_positive_improvement_higher_is_better(self):
        row = self._row(0.42, 0.55)
        assert abs(row["improvement"] - 0.13) < 0.001

    def test_negative_improvement_higher_is_better(self):
        row = self._row(0.55, 0.42)
        assert row["improvement"] < 0

    def test_adapted_none_returns_na(self):
        row = self._row(0.42, None)
        assert row["improvement"] == "n/a"
        assert row["satquery"] is None

    def test_baseline_none_returns_na(self):
        row = self._row(None, 0.55)
        assert row["improvement"] == "n/a"

    def test_lower_is_better_improvement_positive_when_adapted_lower(self):
        # ECE: baseline=0.15, adapted=0.08 → improvement = baseline-adapted = +0.07
        row = self._row(0.15, 0.08, higher=False)
        assert abs(row["improvement"] - 0.07) < 0.001

    def test_lower_is_better_improvement_negative_when_adapted_higher(self):
        row = self._row(0.08, 0.15, higher=False)
        assert row["improvement"] < 0


class TestBuildComparisonTable:
    def test_produces_markdown_table(self):
        from evaluation.common.reporting import build_comparison_table, comparison_row
        rows = [
            comparison_row("BLIP", "vrsbench", "Exact Match", 0.42, 0.55, higher_is_better=True),
            comparison_row("BLIP", "vrsbench", "Token F1", 0.60, None, higher_is_better=True),
        ]
        table = build_comparison_table(rows)
        assert "| Model |" in table or "|Model|" in table or "Model" in table
        assert "BLIP" in table
        assert "Exact Match" in table
        assert "n/a" in table  # for the None adapted row

    def test_empty_rows_returns_string(self):
        from evaluation.common.reporting import build_comparison_table
        result = build_comparison_table([])
        assert isinstance(result, str)


class TestWriteJson:
    def test_writes_valid_json(self, tmp_path):
        from evaluation.common.reporting import write_json
        report = {"domain": "vqa", "metrics": {"exact_match": 0.42}}
        write_json(str(tmp_path), "vqa", report)
        out_file = tmp_path / "vqa_results.json"
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert data["domain"] == "vqa"
        assert data["metrics"]["exact_match"] == pytest.approx(0.42)

    def test_creates_output_dir_if_missing(self, tmp_path):
        from evaluation.common.reporting import write_json
        nested = str(tmp_path / "nested" / "dir")
        write_json(nested, "routing", {"domain": "routing"})
        assert os.path.isfile(os.path.join(nested, "routing_results.json"))


class TestAppendMarkdown:
    def test_appends_table_to_report(self, tmp_path):
        from evaluation.common.reporting import append_markdown, comparison_row
        rows = [comparison_row("BLIP", "vrsbench", "Exact Match", 0.42, 0.55, higher_is_better=True)]
        append_markdown(str(tmp_path), "vqa", "VQA Test", rows, ["note1"])
        report_file = tmp_path / "benchmark_report.md"
        assert report_file.exists()
        content = report_file.read_text()
        assert "VQA Test" in content
        assert "note1" in content

    def test_multiple_appends_accumulate(self, tmp_path):
        from evaluation.common.reporting import append_markdown, comparison_row
        rows = [comparison_row("M", "d", "acc", 0.5, 0.6, higher_is_better=True)]
        append_markdown(str(tmp_path), "vqa", "Section 1", rows, [])
        append_markdown(str(tmp_path), "routing", "Section 2", rows, [])
        content = (tmp_path / "benchmark_report.md").read_text()
        assert "Section 1" in content
        assert "Section 2" in content


# ══════════════════════════════════════════════════════════════════════════════
# 7. Variants
# ══════════════════════════════════════════════════════════════════════════════

class TestResolveVariants:
    def _cfg(self, ckpt: str) -> dict:
        return {
            "vqa": {
                "variants": {"adapted_checkpoint": ckpt}
            }
        }

    def test_baseline_always_available(self):
        from evaluation.common.variants import resolve_variants
        variants = resolve_variants(self._cfg(""), "vqa",
                                    baseline_label="Base", adapted_label="Tuned")
        baseline = next(v for v in variants if v.kind == "baseline")
        assert baseline.available is True

    def test_adapted_unavailable_when_no_checkpoint(self):
        from evaluation.common.variants import resolve_variants
        variants = resolve_variants(self._cfg(""), "vqa",
                                    baseline_label="Base", adapted_label="Tuned")
        adapted = next(v for v in variants if v.kind == "adapted")
        assert adapted.available is False
        assert adapted.reason != ""

    def test_adapted_available_when_checkpoint_exists(self, tmp_path):
        from evaluation.common.variants import resolve_variants
        ckpt = str(tmp_path)  # tmp_path exists on disk
        variants = resolve_variants(self._cfg(ckpt), "vqa",
                                    baseline_label="Base", adapted_label="Tuned")
        adapted = next(v for v in variants if v.kind == "adapted")
        assert adapted.available is True
        assert adapted.checkpoint == ckpt

    def test_adapted_unavailable_when_path_missing(self):
        from evaluation.common.variants import resolve_variants
        variants = resolve_variants(self._cfg("/nonexistent/checkpoint"), "vqa",
                                    baseline_label="Base", adapted_label="Tuned")
        adapted = next(v for v in variants if v.kind == "adapted")
        assert adapted.available is False

    def test_returns_two_variants(self):
        from evaluation.common.variants import resolve_variants
        variants = resolve_variants(self._cfg(""), "vqa",
                                    baseline_label="Base", adapted_label="Tuned")
        assert len(variants) == 2

    def test_unavailable_adapted_helper(self):
        from evaluation.common.variants import unavailable_adapted
        v = unavailable_adapted("RS-CLIP router", "no checkpoint")
        assert v.available is False
        assert "no checkpoint" in v.reason


# ══════════════════════════════════════════════════════════════════════════════
# 8. Metrics __init__ re-export surface (critical bug fix verification)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsInitExports:
    """
    Verify the previously-broken exports are now present.

    The critical bug: accumulate_confusion and metrics_from_confusion were
    defined in change_mask.py but not re-exported from __init__.py, causing
    evaluation/change/run.py to fail at import time with ImportError.
    """

    def test_accumulate_confusion_importable_from_init(self):
        from evaluation.common.metrics import accumulate_confusion
        assert callable(accumulate_confusion)

    def test_metrics_from_confusion_importable_from_init(self):
        from evaluation.common.metrics import metrics_from_confusion
        assert callable(metrics_from_confusion)

    def test_change_mask_metrics_importable(self):
        from evaluation.common.metrics import change_mask_metrics
        assert callable(change_mask_metrics)

    def test_all_captioning_metrics_importable(self):
        from evaluation.common.metrics import bleu, rouge_l, meteor, cider, aggregate_caption_metrics
        for fn in (bleu, rouge_l, meteor, cider, aggregate_caption_metrics):
            assert callable(fn)

    def test_all_detection_metrics_importable(self):
        from evaluation.common.metrics import acc_at_iou, mean_iou, pr_at_iou, aggregate_detection_metrics
        for fn in (acc_at_iou, mean_iou, pr_at_iou, aggregate_detection_metrics):
            assert callable(fn)

    def test_all_routing_metrics_importable(self):
        from evaluation.common.metrics import (
            routing_accuracy, per_task_prf, confusion_matrix, aggregate_routing_metrics
        )
        for fn in (routing_accuracy, per_task_prf, confusion_matrix, aggregate_routing_metrics):
            assert callable(fn)

    def test_calibration_metrics_importable(self):
        from evaluation.common.metrics import (
            expected_calibration_error, maximum_calibration_error,
            brier_score, reliability_stats, box_iou, mean_best_iou,
        )
        for fn in (expected_calibration_error, maximum_calibration_error,
                   brier_score, reliability_stats, box_iou, mean_best_iou):
            assert callable(fn)

    def test_vqa_metrics_module_callable(self):
        from evaluation.common.metrics import vqa_metrics_module
        m = vqa_metrics_module()
        assert "exact_match" in m
        assert callable(m["exact_match"])

    def test_all_listed_in_dunder_all(self):
        import evaluation.common.metrics as em
        for name in ["accumulate_confusion", "metrics_from_confusion",
                     "change_mask_metrics", "bleu", "rouge_l",
                     "acc_at_iou", "routing_accuracy"]:
            assert name in em.__all__, f"{name} missing from __all__"


# ══════════════════════════════════════════════════════════════════════════════
# 9. Smoke imports — every run.py must be importable without heavy deps
# ══════════════════════════════════════════════════════════════════════════════

class TestSmokeImports:
    """
    Each evaluation/<domain>/run.py defers all heavy imports (torch,
    transformers, datasets) to inside functions.  Importing the module
    at the top level must never trigger a model download or GPU allocation.
    """

    RUNNERS = [
        "evaluation.vqa.run",
        "evaluation.captioning.run",
        "evaluation.grounding.run",
        "evaluation.retrieval.run",
        "evaluation.change.run",
        "evaluation.fusion.run",
        "evaluation.routing.run",
        "evaluation.confidence.run",
    ]

    @pytest.mark.parametrize("module_path", RUNNERS)
    def test_module_importable(self, module_path: str):
        """Import must succeed without raising any exception."""
        import importlib
        mod = importlib.import_module(module_path)
        assert mod is not None

    @pytest.mark.parametrize("module_path", RUNNERS)
    def test_module_has_main(self, module_path: str):
        """Every runner must expose a main() function for -m invocation."""
        import importlib
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "main"), f"{module_path} missing main()"
        assert callable(mod.main)

    def test_routing_run_main_accepts_config_arg(self):
        """
        routing.run.main() must handle --config without error (using the
        shipped gold set, no model download needed).
        """
        import importlib, sys
        mod = importlib.import_module("evaluation.routing.run")
        config_path = os.path.join(BACKEND_DIR, "evaluation", "configs", "evaluation.yaml")
        assert os.path.isfile(config_path), f"evaluation.yaml not found at {config_path}"
        # Patch sys.argv and run — should exit 0 (no exception)
        with tempfile.TemporaryDirectory() as tmpdir:
            old_argv = sys.argv
            sys.argv = [
                "evaluation.routing.run",
                "--config", config_path,
                "--output", tmpdir,
            ]
            try:
                mod.main()
                # main() must write at least routing_results.json
                assert os.path.isfile(os.path.join(tmpdir, "routing_results.json"))
            finally:
                sys.argv = old_argv

    def test_vqa_run_skips_cleanly_without_datasets(self):
        """
        vqa.run.main() with no dataset paths configured must exit 0 and write
        a JSON file showing the unavailable variant (not raise an exception).
        """
        import importlib, sys
        mod = importlib.import_module("evaluation.vqa.run")
        config_path = os.path.join(BACKEND_DIR, "evaluation", "configs", "evaluation.yaml")
        with tempfile.TemporaryDirectory() as tmpdir:
            old_argv = sys.argv
            old_env = {k: os.environ.pop(k, None)
                       for k in ("VRSBENCH_DIR", "RSVQA_DIR", "VQA_ADAPTED_CKPT")}
            sys.argv = [
                "evaluation.vqa.run",
                "--config", config_path,
                "--output", tmpdir,
                "--max-samples", "4",
            ]
            try:
                mod.main()
                out = os.path.join(tmpdir, "vqa_results.json")
                assert os.path.isfile(out), "vqa_results.json not written"
                with open(out) as f:
                    report = json.load(f)
                assert report["domain"] == "vqa"
            except SystemExit as e:
                # argparse may exit — only fail on non-zero
                if e.code not in (0, None):
                    raise
            finally:
                sys.argv = old_argv
                for k, v in old_env.items():
                    if v is not None:
                        os.environ[k] = v


class TestRunAllOrchestrator:
    """
    evaluation/run_all.py runs every domain in ONE process so the shared
    benchmark_report.md accumulates all sections (a per-domain `python -m`
    truncates it each process — the bug run_all exists to avoid).
    """

    def test_importable_and_has_main(self):
        import importlib
        mod = importlib.import_module("evaluation.run_all")
        assert hasattr(mod, "main") and callable(mod.main)
        assert hasattr(mod, "run_all") and callable(mod.run_all)

    def test_covers_every_domain_runner(self):
        """The orchestrator must list exactly the eight domain runners."""
        import importlib
        mod = importlib.import_module("evaluation.run_all")
        assert set(mod._DOMAINS) == {
            "vqa", "retrieval", "captioning", "grounding",
            "change", "fusion", "routing", "confidence",
        }
        # VQA must precede confidence (confidence consumes VQA's emitted pairs).
        assert mod._DOMAINS.index("vqa") < mod._DOMAINS.index("confidence")

    def test_offline_run_accumulates_all_sections(self):
        """
        Offline (no dataset env vars) run_all.main() must produce ONE
        benchmark_report.md holding every domain section — routing with real
        numbers, the dataset-backed domains as honest no-data — proving the
        single-process accumulation works and no domain aborts the run.
        """
        import importlib, sys
        mod = importlib.import_module("evaluation.run_all")
        config_path = os.path.join(BACKEND_DIR, "evaluation", "configs", "evaluation.yaml")
        env_keys = ("VRSBENCH_DIR", "RSVQA_DIR", "VQA_ADAPTED_CKPT", "RS_CLIP_CKPT",
                    "LEVIR_CD_DIR", "CDVQA_DIR", "BIGEARTHNET_DIR", "SAR_FUSION_CKPT")
        with tempfile.TemporaryDirectory() as tmpdir:
            old_argv = sys.argv
            old_env = {k: os.environ.pop(k, None) for k in env_keys}
            sys.argv = ["evaluation.run_all", "--config", config_path,
                        "--output", tmpdir, "--max-samples", "4", "--device", "cpu"]
            try:
                mod.main()
                report_md = os.path.join(tmpdir, "benchmark_report.md")
                assert os.path.isfile(report_md), "combined benchmark_report.md not written"
                content = open(report_md, encoding="utf-8").read()
                # Every domain contributes a section header, in one report.
                assert "Routing (task classification)" in content
                assert "VQA" in content
                assert "Confidence calibration" in content
                # Routing runs on the shipped gold set → a real comparison row.
                assert os.path.isfile(os.path.join(tmpdir, "routing_results.json"))
                with open(os.path.join(tmpdir, "routing_results.json")) as f:
                    routing = json.load(f)
                assert routing["status"] == "ok"
                assert routing["router"]["accuracy"] > 0.0
            except SystemExit as e:
                if e.code not in (0, None):
                    raise
            finally:
                sys.argv = old_argv
                for k, v in old_env.items():
                    if v is not None:
                        os.environ[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# 10. Regression locks for the adversarial-review fixes
# ══════════════════════════════════════════════════════════════════════════════

class TestSarFusionChannelsBatchInvariant:
    """
    ``train_fusion._sar_to_fusion_channels`` is SHARED by training (batch size 8)
    and the fusion eval (per-sample, batch size 1).  The VV/VH→ratio channel must
    be standardised PER SAMPLE over spatial dims, so a patch yields the same
    3-channel stack whether fed alone or inside a batch — otherwise a trained
    adapter would be scored on a different ratio distribution than it saw in
    training.  These lock that batch-invariance (the review-confirmed fix; a
    global reduction would fail the first test).
    """

    def test_ratio_channel_is_batch_composition_invariant(self):
        torch = pytest.importorskip("torch")
        from training.train_fusion import _sar_to_fusion_channels

        torch.manual_seed(0)
        p0 = torch.randn(2, 8, 8)                # [VV, VH] patch A
        p1 = torch.randn(2, 8, 8) * 3.0 + 1.0    # patch B — very different stats

        solo0 = _sar_to_fusion_channels(p0.unsqueeze(0))          # eval path (bs 1)
        solo1 = _sar_to_fusion_channels(p1.unsqueeze(0))
        batched = _sar_to_fusion_channels(torch.stack([p0, p1]))  # training path (bs 2)

        # A patch's output must not depend on what else shares its batch.
        assert torch.allclose(batched[0], solo0[0], atol=1e-5)
        assert torch.allclose(batched[1], solo1[0], atol=1e-5)

    def test_ratio_channel_is_per_sample_standardised(self):
        torch = pytest.importorskip("torch")
        from training.train_fusion import _sar_to_fusion_channels

        torch.manual_seed(1)
        batch = torch.randn(4, 2, 12, 12)
        out = _sar_to_fusion_channels(batch)

        assert out.shape == (4, 3, 12, 12)
        # VV/VH pass through untouched; only the ratio channel is normalised.
        assert torch.allclose(out[:, 0:2], batch, atol=1e-6)
        ratio = out[:, 2]
        for i in range(ratio.shape[0]):
            assert abs(float(ratio[i].mean())) < 1e-4
            assert abs(float(ratio[i].std()) - 1.0) < 1e-2


class TestRoutingRobustNumImages:
    """
    ``routing.run`` must not crash on a curated entry whose ``num_images`` is
    null or non-numeric — it coerces to 1 and still scores the entry (the
    review-confirmed fix for the bare ``int(s.get('num_images', 1))`` that raised
    on a present-but-null value).
    """

    def test_null_and_nonnumeric_num_images_do_not_crash(self):
        import importlib
        mod = importlib.import_module("evaluation.routing.run")
        config_path = os.path.join(BACKEND_DIR, "evaluation", "configs", "evaluation.yaml")
        eval_set = {"samples": [
            {"query": "describe this scene", "gold_task": "CAPTIONING", "num_images": None},
            {"query": "how many planes are there", "gold_task": "SINGLE_VQA", "num_images": "two"},
            {"query": "locate the runway", "gold_task": "GROUNDING", "num_images": 1},
        ]}
        with tempfile.TemporaryDirectory() as tmpdir:
            set_path = os.path.join(tmpdir, "routing_eval_set.json")
            with open(set_path, "w", encoding="utf-8") as f:
                json.dump(eval_set, f)
            old_argv = sys.argv
            old_env = os.environ.pop("ROUTING_EVAL_SET", None)
            os.environ["ROUTING_EVAL_SET"] = set_path
            sys.argv = ["evaluation.routing.run", "--config", config_path, "--output", tmpdir]
            try:
                mod.main()
                out = os.path.join(tmpdir, "routing_results.json")
                assert os.path.isfile(out)
                with open(out) as f:
                    report = json.load(f)
                assert report["status"] == "ok"
                # All three entries survived coercion — none dropped, none crashed.
                assert report["n_samples"] == 3
            finally:
                sys.argv = old_argv
                if old_env is not None:
                    os.environ["ROUTING_EVAL_SET"] = old_env
                else:
                    os.environ.pop("ROUTING_EVAL_SET", None)


class TestChangeVqaTrackRobust:
    """
    ``change._vqa_track`` must skip an unreadable/failed CDVQA sample instead of
    letting it abort the whole change domain (which would also discard the
    mask-track results computed first).  Review-confirmed fix: per-sample
    try/except + honest degradation to ``None`` when nothing can be scored.
    """

    class _DS:
        def __init__(self, fail_indices, n=3):
            self._fail = set(fail_indices)
            self._n = n

        def __len__(self):
            return self._n

        def __getitem__(self, i):
            if i in self._fail:
                raise OSError("image file is truncated")  # PIL-style failure
            return {"image_a": object(), "image_b": object(),
                    "question": "what changed?", "answer": "buildings"}

    class _Model:
        def __init__(self, *a, **k):
            pass

        def answer_change_question(self, a, b, q):
            return {"answer": "buildings"}

    def _run_track(self, ds, notes, tmpdir):
        from unittest import mock
        from evaluation.change import run as change_run
        ds_cfg = {"data_dir": tmpdir, "max_samples": None}
        ccfg = {"model": {}}
        with mock.patch("evaluation.datasets.cdvqa.CDVQADataset", return_value=ds), \
             mock.patch("models.change_model.ChangeDetectionModel", self._Model):
            return change_run._vqa_track(ccfg, ds_cfg, "cpu", None, notes)

    def test_one_bad_sample_is_skipped_not_fatal(self):
        notes: List[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_track(self._DS(fail_indices=[1]), notes, tmpdir)
        assert result is not None
        assert result["n_samples"] == 2  # index 1 skipped; 0 and 2 kept
        assert any("skipped 1" in n for n in notes)

    def test_all_bad_samples_degrade_to_none(self):
        notes: List[str] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self._run_track(self._DS(fail_indices=[0, 1, 2]), notes, tmpdir)
        assert result is None
        assert any("no CDVQA samples could be opened" in n for n in notes)
