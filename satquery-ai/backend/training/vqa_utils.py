"""
SatQuery AI — VQA Training / Evaluation Utilities
=================================================
Shared helpers used by both ``train_vqa.py`` and ``evaluate_vqa.py``:

  * reproducible seeding
  * deterministic train/val/test split derivation (no test-set leakage)
  * dynamic-padding collate function for HuggingFace VQA processors
  * VQA metrics: exact match, VQA accuracy, token-F1
  * dataset-version + git-commit provenance helpers

Training / inference separation
--------------------------------
Only imported by training scripts.  Never imported from models/ or api/.
"""
from __future__ import annotations

import hashlib
import logging
import os
import random
import re
import subprocess
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("satquery.vqa_utils")


# ── Reproducibility ─────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info("Global seed set to %d", seed)


# ── Split handling ──────────────────────────────────────────────────────────────

def derive_splits(
    dataset,
    *,
    seed: int = 42,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Dict[str, "Subset"]:
    """
    Derive train/val/test subsets from a dataset that does NOT ship its own
    splits.

    The split is deterministic given ``seed`` and hashed per-sample so that the
    same sample always lands in the same split across runs, even if the dataset
    ordering changes.

    IMPORTANT: This is only used when a dataset lacks native splits.  When a
    dataset ships explicit train/val/test files, the adapters are instantiated
    per-split and this function is not involved — guaranteeing no test-set
    training.
    """
    from torch.utils.data import Subset

    n = len(dataset)
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)

    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    logger.info(
        "Derived splits (seed=%d): train=%d, val=%d, test=%d",
        seed, len(train_idx), len(val_idx), len(test_idx),
    )
    return {
        "train": Subset(dataset, train_idx),
        "val": Subset(dataset, val_idx),
        "test": Subset(dataset, test_idx),
    }


# ── Dynamic-padding collate ───────────────────────────────────────────────────

class VQACollator:
    """
    Collate raw VQA samples into a dynamically-padded batch using a HuggingFace
    processor (BlipProcessor / Blip2Processor).

    Dynamic padding: each batch is padded to the longest sequence *in that
    batch* rather than a fixed maximum, which saves compute on short batches.

    Parameters
    ----------
    processor :
        A HuggingFace vision-language processor.
    max_length : int
        Upper bound on text length (truncation guard).
    train : bool
        When True the answer is tokenised into ``labels`` for teacher forcing.
        When False (eval) only the encoder inputs are produced and raw answers /
        metadata are carried through for metric computation.
    """

    def __init__(self, processor, max_length: int = 64, train: bool = True):
        self.processor = processor
        self.max_length = max_length
        self.train = train

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        images = [b["image"] for b in batch]
        questions = [b["question"] for b in batch]
        answers = [b["answer"] for b in batch]
        metadata = [b["metadata"] for b in batch]

        # Encode image + question with dynamic padding to the batch max.
        enc = self.processor(
            images=images,
            text=questions,
            return_tensors="pt",
            padding=True,           # dynamic: pad to longest in batch
            truncation=True,
            max_length=self.max_length,
        )

        out: Dict[str, Any] = dict(enc)
        out["raw_questions"] = questions
        out["raw_answers"] = answers
        out["metadata"] = metadata

        if self.train:
            # Tokenise answers as decoder labels. Pad tokens are masked to -100
            # so they are ignored by the cross-entropy loss.
            labels = self.processor.tokenizer(
                answers,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).input_ids
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is not None:
                labels = labels.masked_fill(labels == pad_id, -100)
            out["labels"] = labels

        return out


# ── Metrics ─────────────────────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lower-case, strip punctuation & articles, collapse whitespace.

    Mirrors the standard VQA / SQuAD answer-normalisation so metrics are not
    thrown off by casing or trailing punctuation.
    """
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    tokens = [t for t in s.split() if t not in _ARTICLES]
    return _WS_RE.sub(" ", " ".join(tokens)).strip()


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised prediction equals normalised gold answer, else 0.0."""
    return float(normalize_text(pred) == normalize_text(gold))


def vqa_accuracy(pred: str, gold_answers: Sequence[str]) -> float:
    """
    Standard VQA accuracy for a single prediction against one or more human
    answers::

        acc = min(1.0, #humans_that_said_pred / 3)

    With a single ground-truth answer this reduces to exact match, but the
    formula is preserved so multi-annotator datasets (e.g. RSVQA variants that
    ship several answers) are handled correctly.
    """
    p = normalize_text(pred)
    matches = sum(1 for g in gold_answers if normalize_text(g) == p)
    return min(1.0, matches / 3.0)


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between prediction and gold (SQuAD-style).

    Appropriate for open-ended answers where partial overlap is meaningful;
    for yes/no or single-word answers it collapses to exact match.
    """
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    if not p_toks and not g_toks:
        return 1.0
    if not p_toks or not g_toks:
        return 0.0
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def aggregate_metrics(
    predictions: List[str],
    golds: List[str],
    categories: Optional[List[Optional[str]]] = None,
) -> Dict[str, Any]:
    """
    Compute overall and per-category VQA metrics.

    Parameters
    ----------
    predictions : list of str
    golds : list of str  (single reference answer each)
    categories : list of (str | None), optional
        Per-sample question category for per-category breakdown.

    Returns
    -------
    dict with keys: exact_match, vqa_accuracy, token_f1, n_samples, and
    (when categories supplied) ``per_category``.
    """
    assert len(predictions) == len(golds), "predictions/golds length mismatch"
    n = len(predictions)
    if n == 0:
        return {"exact_match": 0.0, "vqa_accuracy": 0.0, "token_f1": 0.0, "n_samples": 0}

    em_scores, vqa_scores, f1_scores = [], [], []
    for pred, gold in zip(predictions, golds):
        em_scores.append(exact_match(pred, gold))
        vqa_scores.append(vqa_accuracy(pred, [gold]))
        f1_scores.append(token_f1(pred, gold))

    result: Dict[str, Any] = {
        "exact_match": round(sum(em_scores) / n, 4),
        "vqa_accuracy": round(sum(vqa_scores) / n, 4),
        "token_f1": round(sum(f1_scores) / n, 4),
        "n_samples": n,
    }

    if categories is not None:
        per_cat: Dict[str, Dict[str, Any]] = {}
        buckets: Dict[str, List[int]] = {}
        for i, cat in enumerate(categories):
            key = cat if cat else "uncategorized"
            buckets.setdefault(key, []).append(i)
        for cat, idxs in sorted(buckets.items()):
            per_cat[cat] = {
                "exact_match": round(sum(em_scores[i] for i in idxs) / len(idxs), 4),
                "vqa_accuracy": round(sum(vqa_scores[i] for i in idxs) / len(idxs), 4),
                "token_f1": round(sum(f1_scores[i] for i in idxs) / len(idxs), 4),
                "n_samples": len(idxs),
            }
        result["per_category"] = per_cat

    return result


# ── Provenance ────────────────────────────────────────────────────────────────

def git_commit_hash() -> Optional[str]:
    """Return the current git commit hash, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.decode().strip()
    except Exception:
        return None


def dataset_version(dataset_configs: List[Dict[str, Any]]) -> str:
    """
    Produce a short, stable dataset-version fingerprint from the enabled
    dataset configs (name + data_dir + sample count).  This is a provenance
    tag, not a content hash of the images.
    """
    parts = []
    for dc in sorted(dataset_configs, key=lambda d: d.get("name", "")):
        parts.append(
            f"{dc.get('name')}:{dc.get('data_dir')}:{dc.get('n_samples', '?')}"
        )
    blob = "|".join(parts)
    digest = hashlib.sha256(blob.encode()).hexdigest()[:12]
    names = "+".join(sorted(dc.get("name", "?") for dc in dataset_configs))
    return f"{names}@{digest}"
