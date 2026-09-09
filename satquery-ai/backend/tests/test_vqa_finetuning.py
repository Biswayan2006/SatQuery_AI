"""
SatQuery AI — VQA Fine-tuning Pipeline Tests
=============================================
Covers (no network / no model downloads):

  1. VQA metrics — exact match, VQA accuracy, token F1, per-category
  2. Answer normalisation
  3. Dataset adapters — normalised sample schema (VRSBench, RSVQA)
  4. Split derivation — no overlap, deterministic
  5. Dynamic-padding collator with a fake processor
  6. Manual LoRA — only adapter params trainable, forward shape, changes output
  7. apply_vqa_lora backend selection
  8. Base model loading path (RemoteSensingVQA, mocked transformers)
  9. Fine-tuned model loading path (mocked)
 10. answer() schema compatibility — returns answer/confidence/evidence,
     confidence NOT derived from answer length, confidence_is_calibrated=False
 11. Failure behavior — None image, inference exception
 12. Evaluation report shape

Run with:
    cd satquery-ai/backend
    python -m pytest tests/test_vqa_finetuning.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — Metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics:

    def test_exact_match_positive(self):
        from training.vqa_utils import exact_match
        assert exact_match("Yes", "yes") == 1.0
        assert exact_match("A forest.", "forest") == 1.0  # article + punct stripped

    def test_exact_match_negative(self):
        from training.vqa_utils import exact_match
        assert exact_match("urban", "rural") == 0.0

    def test_vqa_accuracy_single_ref(self):
        from training.vqa_utils import vqa_accuracy
        assert vqa_accuracy("water", ["water"]) == pytest.approx(1 / 3)
        assert vqa_accuracy("water", ["water", "water", "water"]) == 1.0
        assert vqa_accuracy("water", ["forest"]) == 0.0

    def test_token_f1_partial_overlap(self):
        from training.vqa_utils import token_f1
        f1 = token_f1("a large urban area", "urban area")
        assert 0.0 < f1 < 1.0
        assert token_f1("forest", "forest") == 1.0
        assert token_f1("forest", "water") == 0.0

    def test_aggregate_overall(self):
        from training.vqa_utils import aggregate_metrics
        preds = ["yes", "no", "forest"]
        golds = ["yes", "yes", "forest"]
        m = aggregate_metrics(preds, golds)
        assert m["n_samples"] == 3
        assert m["exact_match"] == pytest.approx(2 / 3, abs=1e-3)
        assert "vqa_accuracy" in m and "token_f1" in m

    def test_aggregate_per_category(self):
        from training.vqa_utils import aggregate_metrics
        preds = ["yes", "no", "3"]
        golds = ["yes", "yes", "3"]
        cats = ["presence", "presence", "count"]
        m = aggregate_metrics(preds, golds, categories=cats)
        assert "per_category" in m
        assert set(m["per_category"]) == {"presence", "count"}
        assert m["per_category"]["count"]["exact_match"] == 1.0
        assert m["per_category"]["presence"]["n_samples"] == 2

    def test_aggregate_empty(self):
        from training.vqa_utils import aggregate_metrics
        m = aggregate_metrics([], [])
        assert m["n_samples"] == 0

    def test_normalize_text(self):
        from training.vqa_utils import normalize_text
        assert normalize_text("The Forest.") == "forest"
        assert normalize_text("  YES!! ") == "yes"


# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — Dataset adapters (schema)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVRSBenchVQAAdapter:

    def _make_dataset(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "images"), exist_ok=True)
        from PIL import Image
        Image.new("RGB", (32, 32)).save(os.path.join(tmpdir, "images", "1.jpg"))
        # Layout B: split files
        with open(os.path.join(tmpdir, "train_questions.json"), "w") as f:
            json.dump([{"id": 10, "image_id": 1, "question": "What is shown?",
                        "type": "description"}], f)
        with open(os.path.join(tmpdir, "train_answers.json"), "w") as f:
            json.dump([{"question_id": 10, "answer": "a forest"}], f)

        from training.datasets.vqa import VRSBenchVQAAdapter
        return VRSBenchVQAAdapter(data_dir=tmpdir, split="train")

    def test_sample_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._make_dataset(tmp)
            assert len(ds) == 1
            s = ds[0]
            for key in ("image", "question", "answer", "metadata"):
                assert key in s
            meta = s["metadata"]
            assert meta["dataset"] == "vrsbench"
            assert meta["split"] == "train"
            assert meta["sensor"] == "optical"
            assert meta["question_type"] == "description"
            from PIL import Image
            assert isinstance(s["image"], Image.Image)

    def test_validate_sample_passes(self):
        from training.datasets.vqa import validate_sample
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._make_dataset(tmp)
            validate_sample(ds[0])  # should not raise


class TestRSVQAAdapter:

    def _make_dataset(self, tmpdir):
        os.makedirs(os.path.join(tmpdir, "Images_LR"), exist_ok=True)
        from PIL import Image
        Image.new("RGB", (32, 32)).save(os.path.join(tmpdir, "Images_LR", "5.tif"))
        with open(os.path.join(tmpdir, "LR_split_train_questions.json"), "w") as f:
            json.dump({"questions": [
                {"id": 1, "img_id": 5, "question": "Is there water?",
                 "type": "presence", "active": True},
            ]}, f)
        with open(os.path.join(tmpdir, "LR_split_train_answers.json"), "w") as f:
            json.dump({"answers": [{"id": 1, "question_id": 1, "answer": "yes"}]}, f)

        from training.datasets.vqa import RSVQAAdapter
        return RSVQAAdapter(data_dir=tmpdir, split="train")

    def test_prefix_and_subdir_autodetect(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._make_dataset(tmp)
            assert ds.file_prefix == "LR_split_"
            assert ds.image_subdir == "Images_LR"
            assert len(ds) == 1

    def test_sample_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = self._make_dataset(tmp)
            s = ds[0]
            assert s["question"] == "Is there water?"
            assert s["answer"] == "yes"
            assert s["metadata"]["dataset"] == "rsvqa"
            assert s["metadata"]["question_type"] == "presence"

    def test_inactive_questions_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            os.makedirs(os.path.join(tmp, "Images_LR"), exist_ok=True)
            Image.new("RGB", (16, 16)).save(os.path.join(tmp, "Images_LR", "5.tif"))
            with open(os.path.join(tmp, "LR_split_train_questions.json"), "w") as f:
                json.dump({"questions": [
                    {"id": 1, "img_id": 5, "question": "q1", "active": False},
                    {"id": 2, "img_id": 5, "question": "q2", "active": True},
                ]}, f)
            with open(os.path.join(tmp, "LR_split_train_answers.json"), "w") as f:
                json.dump({"answers": [
                    {"question_id": 1, "answer": "a1"},
                    {"question_id": 2, "answer": "a2"},
                ]}, f)
            from training.datasets.vqa import RSVQAAdapter
            ds = RSVQAAdapter(data_dir=tmp, split="train")
            assert len(ds) == 1
            assert ds[0]["question"] == "q2"

    def test_registry_build(self):
        from training.datasets.vqa import build_vqa_dataset
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp)
            ds = build_vqa_dataset("rsvqa", data_dir=tmp, split="train")
            assert len(ds) == 1

    def test_unknown_dataset_raises(self):
        from training.datasets.vqa import build_vqa_dataset
        with pytest.raises(ValueError):
            build_vqa_dataset("nope", data_dir="/tmp", split="train")


# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — Split derivation
# ═══════════════════════════════════════════════════════════════════════════════

class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"idx": i}


class TestSplitDerivation:

    def test_no_overlap_and_covers_all(self):
        from training.vqa_utils import derive_splits
        ds = _ListDataset(100)
        splits = derive_splits(ds, seed=1, val_fraction=0.1, test_fraction=0.1)
        tr = set(splits["train"].indices)
        va = set(splits["val"].indices)
        te = set(splits["test"].indices)
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        assert len(tr | va | te) == 100

    def test_deterministic(self):
        from training.vqa_utils import derive_splits
        ds = _ListDataset(50)
        a = derive_splits(ds, seed=7)
        b = derive_splits(ds, seed=7)
        assert a["train"].indices == b["train"].indices
        assert a["test"].indices == b["test"].indices


# ═══════════════════════════════════════════════════════════════════════════════
# Test 4 — Dynamic-padding collator
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeTokenizer:
    pad_token_id = 0

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None, max_length=None):
        # Pad to the longest text length in the batch (word count).
        lengths = [len(t.split()) for t in texts]
        L = max(lengths) if lengths else 1
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        for i, t in enumerate(texts):
            for j, _ in enumerate(t.split()):
                ids[i, j] = 1
        out = MagicMock()
        out.input_ids = ids
        return out


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def __call__(self, images=None, text=None, return_tensors=None,
                 padding=None, truncation=None, max_length=None):
        n = len(text)
        L = max(len(t.split()) for t in text)
        return {
            "pixel_values": torch.zeros(n, 3, 8, 8),
            "input_ids": torch.ones(n, L, dtype=torch.long),
            "attention_mask": torch.ones(n, L, dtype=torch.long),
        }


class TestCollator:

    def _batch(self):
        from PIL import Image
        return [
            {"image": Image.new("RGB", (8, 8)), "question": "what is this",
             "answer": "a forest area here",
             "metadata": {"question_type": "desc", "image_id": "1"}},
            {"image": Image.new("RGB", (8, 8)), "question": "count",
             "answer": "two",
             "metadata": {"question_type": "count", "image_id": "2"}},
        ]

    def test_train_collate_has_labels(self):
        from training.vqa_utils import VQACollator
        col = VQACollator(_FakeProcessor(), max_length=32, train=True)
        out = col(self._batch())
        assert "labels" in out
        assert "pixel_values" in out
        # Pad tokens masked to -100.
        assert (out["labels"] == -100).any()

    def test_eval_collate_no_labels_carries_raw(self):
        from training.vqa_utils import VQACollator
        col = VQACollator(_FakeProcessor(), max_length=32, train=False)
        out = col(self._batch())
        assert "labels" not in out
        assert out["raw_answers"] == ["a forest area here", "two"]
        assert len(out["metadata"]) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# Test 5 — Manual LoRA
# ═══════════════════════════════════════════════════════════════════════════════

class TestManualLoRA:

    def test_only_lora_trainable(self):
        from training.vqa_lora import LoRALinear
        lora = LoRALinear(nn.Linear(16, 8), rank=4, alpha=8.0)
        trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
        assert "lora_A" in trainable and "lora_B" in trainable
        frozen = [n for n, p in lora.named_parameters() if not p.requires_grad]
        assert any("weight" in n for n in frozen)

    def test_forward_shape(self):
        from training.vqa_lora import LoRALinear
        lora = LoRALinear(nn.Linear(16, 8), rank=4, alpha=8.0)
        out = lora(torch.randn(3, 16))
        assert out.shape == (3, 8)

    def test_apply_manual_lora_selects_backend(self):
        from training.vqa_lora import apply_vqa_lora, LoRALinear
        model = nn.Sequential()
        model.add_module("query", nn.Linear(16, 16))
        model.add_module("act", nn.ReLU())
        model.add_module("value", nn.Linear(16, 16))
        cfg = {"rank": 4, "alpha": 8, "dropout": 0.0, "target_modules": ["query", "value"]}
        new_model, backend = apply_vqa_lora(model, cfg, prefer_peft=False)
        assert backend == "manual"
        assert isinstance(new_model.query, LoRALinear)
        assert isinstance(new_model.value, LoRALinear)

    def test_apply_manual_lora_zero_match_raises(self):
        from training.vqa_lora import apply_vqa_lora
        model = nn.Sequential(nn.Linear(4, 4))
        with pytest.raises(RuntimeError):
            apply_vqa_lora(model, {"target_modules": ["nonexistent"]}, prefer_peft=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Test 6 — VQA wrapper: base loading, schema, confidence, failure
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeGenOut:
    def __init__(self, sequences, scores):
        self.sequences = sequences
        self.scores = scores


class _FakeBlipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        # Return a 2-token generated sequence with scores over a tiny vocab.
        seq = torch.tensor([[0, 3, 4]])
        vocab = 5
        scores = (
            torch.log_softmax(torch.tensor([[0.1, 0.1, 0.1, 2.0, 0.1]]), dim=-1),
            torch.log_softmax(torch.tensor([[0.1, 0.1, 0.1, 0.1, 2.0]]), dim=-1),
        )
        return _FakeGenOut(seq, scores)


class _FakeProc:
    def __call__(self, images=None, text=None, return_tensors=None):
        d = {"pixel_values": torch.zeros(1, 3, 8, 8),
             "input_ids": torch.ones(1, 3, dtype=torch.long)}
        return _DictToDevice(d)

    def decode(self, ids, skip_special_tokens=True):
        return "forest"

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["forest"]


class _DictToDevice(dict):
    def to(self, device):
        return self


def _make_vqa_with_fakes():
    from models.vqa_model import RemoteSensingVQA
    with patch.object(RemoteSensingVQA, "_load", lambda self: None):
        vqa = RemoteSensingVQA(model_name="Salesforce/blip-vqa-base", device="cpu")
    vqa.model = _FakeBlipModel()
    vqa.processor = _FakeProc()
    vqa._is_blip2 = False
    return vqa


class TestVQAWrapper:

    def test_base_load_path_selected(self):
        """model_source=pretrained routes to base loader."""
        from models.vqa_model import RemoteSensingVQA
        called = {"blip": False}
        with patch.object(RemoteSensingVQA, "_load_blip",
                          lambda self: called.__setitem__("blip", True)):
            RemoteSensingVQA(model_name="Salesforce/blip-vqa-base",
                             device="cpu", model_source="pretrained")
        assert called["blip"] is True

    def test_finetuned_source_tries_finetuned_first(self):
        from models.vqa_model import RemoteSensingVQA
        order = []
        with patch.object(RemoteSensingVQA, "_load_finetuned",
                          lambda self: order.append("ft") or True):
            RemoteSensingVQA(model_name="Salesforce/blip-vqa-base", device="cpu",
                             model_source="finetuned",
                             finetuned_checkpoint="/some/dir")
        assert order == ["ft"]

    def test_finetuned_missing_falls_back_to_base(self):
        from models.vqa_model import RemoteSensingVQA
        events = []
        with patch.object(RemoteSensingVQA, "_load_finetuned", lambda self: False), \
             patch.object(RemoteSensingVQA, "_load_blip",
                          lambda self: events.append("base")):
            RemoteSensingVQA(model_name="Salesforce/blip-vqa-base", device="cpu",
                             model_source="finetuned",
                             finetuned_checkpoint="/missing")
        assert events == ["base"]

    def test_answer_schema(self):
        from PIL import Image
        vqa = _make_vqa_with_fakes()
        out = vqa.answer(Image.new("RGB", (16, 16)), "what is this?")
        assert set(["answer", "confidence", "confidence_is_calibrated", "evidence"]).issubset(out)
        assert out["answer"] == "forest"
        assert out["confidence_is_calibrated"] is False
        ev = out["evidence"]
        assert "sequence_score" in ev and "token_logprobs" in ev
        assert "answer_length" in ev and "decoding" in ev

    def test_confidence_from_score_not_length(self):
        """Confidence must derive from sequence_score, not answer length."""
        from PIL import Image
        vqa = _make_vqa_with_fakes()
        out = vqa.answer(Image.new("RGB", (16, 16)), "q")
        ev = out["evidence"]
        assert ev["sequence_score"] is not None
        # confidence ≈ exp(mean logprob) — a real signal
        assert out["confidence"] == pytest.approx(
            float(np.clip(np.exp(ev["sequence_score"]), 0, 1)), abs=1e-3
        )

    def test_none_image_failure(self):
        vqa = _make_vqa_with_fakes()
        out = vqa.answer(None, "q")
        assert out["confidence"] == 0.0
        assert "evidence" in out

    def test_inference_exception_handled(self):
        from PIL import Image
        vqa = _make_vqa_with_fakes()
        vqa.model.generate = MagicMock(side_effect=RuntimeError("boom"))
        out = vqa.answer(Image.new("RGB", (16, 16)), "q")
        assert "Inference error" in out["answer"]
        assert out["confidence"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Test 7 — Evaluation report shape
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationReport:

    def test_collect_error_examples(self):
        from training.evaluate_vqa import collect_error_examples
        preds = ["yes", "forest"]
        golds = ["no", "forest"]
        metas = [{"image_id": "1", "question_id": "1", "question": "q1",
                  "question_type": "presence"},
                 {"image_id": "2", "question_id": "2", "question": "q2",
                  "question_type": "desc"}]
        errs = collect_error_examples(preds, golds, metas, limit=10)
        assert len(errs) == 1
        assert errs[0]["gold"] == "no"
        assert errs[0]["prediction"] == "yes"

    def test_report_keys(self):
        """The evaluation report must carry the required top-level keys."""
        from training.vqa_utils import aggregate_metrics, dataset_version, git_commit_hash
        metrics = aggregate_metrics(["yes"], ["yes"], categories=["presence"])
        report = {
            "model": "Salesforce/blip-vqa-base",
            "dataset": "rsvqa",
            "split": "test",
            "metrics": metrics,
        }
        assert report["split"] == "test"
        assert "per_category" in report["metrics"]
        # provenance helpers must not raise
        _ = dataset_version([{"name": "rsvqa", "data_dir": "/d", "n_samples": 1}])
        _ = git_commit_hash()  # None or str
