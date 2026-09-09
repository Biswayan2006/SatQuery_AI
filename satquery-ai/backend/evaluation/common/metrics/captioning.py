"""
SatQuery AI — Captioning Metrics
================================
Pure-Python (+ optional numpy) implementations of the standard image-captioning
metrics, with NO external NLP dependencies (nltk / pycocoevalcap are NOT in
``requirements.txt``).  Each function accepts a single hypothesis string and one
or more reference strings.

Metrics
-------
  * ``bleu``     — corpus/sentence BLEU-n with a brevity penalty (n=1..4).
  * ``rouge_l``  — ROUGE-L F1 based on the longest common subsequence.
  * ``meteor``   — a WordNet-free METEOR: unigram precision/recall harmonic
                   mean with a chunk-based fragmentation penalty.  (Synonym /
                   stem matching is omitted — documented in EVALUATION.md.)
  * ``cider``    — CIDEr-D style tf-idf-weighted n-gram cosine consensus over
                   the corpus (needs the full set of references for document
                   frequencies), plus a per-sentence fallback.

All functions are deterministic and safe on empty input (return 0.0).  These
are honest approximations of the reference implementations; their limitations
are documented rather than hidden.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np


# ── Tokenisation ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, keep alphanumerics as tokens (matches common caption evals)."""
    if not text:
        return []
    out: List[str] = []
    cur: List[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# ── BLEU ─────────────────────────────────────────────────────────────────────────

def bleu(
    hypothesis: str,
    references: Sequence[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    Sentence-level BLEU with clipped n-gram precision + brevity penalty.

    Returns ``{"bleu_1", "bleu_2", "bleu_3", "bleu_4"}`` where ``bleu_k`` is the
    geometric mean of clipped precisions p1..pk times the brevity penalty.
    Uses a small epsilon floor on zero precisions so higher-order BLEU degrades
    smoothly instead of collapsing to 0 on short captions (method-1 smoothing).
    """
    hyp = _tokenize(hypothesis)
    refs = [_tokenize(r) for r in references if r is not None]
    out = {f"bleu_{k}": 0.0 for k in range(1, max_n + 1)}
    if not hyp or not refs:
        return out

    # Brevity penalty against the closest reference length.
    hyp_len = len(hyp)
    ref_lens = [len(r) for r in refs]
    closest = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
    if hyp_len > closest:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - closest / hyp_len)

    log_precisions: List[float] = []
    for n in range(1, max_n + 1):
        hyp_ng = _ngrams(hyp, n)
        if not hyp_ng:
            # No n-grams of this order: smoothed near-zero precision.
            log_precisions.append(math.log(1e-9))
        else:
            # Clip each hyp n-gram count by the max over references.
            max_ref = Counter()
            for r in refs:
                rc = _ngrams(r, n)
                for g, c in rc.items():
                    if c > max_ref[g]:
                        max_ref[g] = c
            clipped = sum(min(c, max_ref[g]) for g, c in hyp_ng.items())
            total = sum(hyp_ng.values())
            p = clipped / total if total else 0.0
            if p == 0.0:
                p = 1e-9  # method-1 smoothing floor
            log_precisions.append(math.log(p))

        # bleu_n = BP * geomean(p1..pn)
        geo = math.exp(sum(log_precisions) / n)
        out[f"bleu_{n}"] = round(bp * geo, 4)

    return out


# ── ROUGE-L ────────────────────────────────────────────────────────────────────

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: str, references: Sequence[str], beta: float = 1.2) -> float:
    """
    ROUGE-L F1 (LCS-based), taking the best reference.  ``beta`` weights recall
    over precision as in the original ROUGE (default 1.2).  Returns 0.0 on empty.
    """
    hyp = _tokenize(hypothesis)
    if not hyp:
        return 0.0
    best = 0.0
    for r in references:
        ref = _tokenize(r)
        if not ref:
            continue
        lcs = _lcs_length(hyp, ref)
        if lcs == 0:
            continue
        prec = lcs / len(hyp)
        rec = lcs / len(ref)
        denom = rec + beta * beta * prec
        if denom == 0:
            continue
        f = (1 + beta * beta) * prec * rec / denom
        best = max(best, f)
    return round(best, 4)


# ── METEOR (WordNet-free approximation) ─────────────────────────────────────────

def meteor(hypothesis: str, references: Sequence[str]) -> float:
    """
    Simplified METEOR without WordNet/stem synonymy.

    F_mean = P*R / (alpha*P + (1-alpha)*R) with alpha=0.9 (recall-weighted),
    times a fragmentation penalty ``1 - gamma*(chunks/matches)**beta`` with the
    standard gamma=0.5, beta=3.  Best reference is used.  Exact-token matching
    only — see EVALUATION.md for the documented limitation.
    """
    hyp = _tokenize(hypothesis)
    if not hyp:
        return 0.0
    alpha, beta, gamma = 0.9, 3.0, 0.5
    best = 0.0
    for r in references:
        ref = _tokenize(r)
        if not ref:
            continue
        matches, chunks = _meteor_align(hyp, ref)
        if matches == 0:
            continue
        p = matches / len(hyp)
        rec = matches / len(ref)
        denom = alpha * p + (1 - alpha) * rec
        if denom == 0:
            continue
        fmean = p * rec / denom
        frag = chunks / matches
        penalty = gamma * (frag ** beta)
        score = fmean * (1.0 - penalty)
        best = max(best, score)
    return round(best, 4)


def _meteor_align(hyp: List[str], ref: List[str]) -> "tuple[int, int]":
    """
    Greedy exact-match alignment: count matched unigrams and the number of
    contiguous chunks those matches form in the hypothesis order.
    """
    ref_avail = Counter(ref)
    matched_positions: List[int] = []  # indices into hyp that matched
    for i, tok in enumerate(hyp):
        if ref_avail.get(tok, 0) > 0:
            ref_avail[tok] -= 1
            matched_positions.append(i)
    matches = len(matched_positions)
    if matches == 0:
        return 0, 0
    # Count chunks = maximal runs of consecutive matched hyp positions.
    chunks = 1
    for a, b in zip(matched_positions[:-1], matched_positions[1:]):
        if b != a + 1:
            chunks += 1
    return matches, chunks


# ── CIDEr (tf-idf n-gram consensus over the corpus) ─────────────────────────────

def cider(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    n: int = 4,
    sigma: float = 6.0,
) -> Dict[str, float]:
    """
    Corpus CIDEr-D.

    Parameters
    ----------
    hypotheses : one predicted caption per image.
    references : list (per image) of reference captions.

    Document frequencies are computed over the reference corpus, so this needs
    all images at once (corpus-level metric).  Returns
    ``{"cider": mean, "cider_per": [...]}`` averaging n=1..``n`` as in CIDEr-D.
    Returns 0.0 for empty input.
    """
    if not hypotheses or not references or len(hypotheses) != len(references):
        return {"cider": 0.0, "cider_per": []}

    num_images = len(hypotheses)
    # Document frequency of each n-gram: number of images whose reference set
    # contains it (CIDEr counts an n-gram once per image regardless of refs).
    doc_freq: List[Counter] = [Counter() for _ in range(n)]
    ref_tokens = [[_tokenize(r) for r in refs] for refs in references]
    hyp_tokens = [_tokenize(h) for h in hypotheses]

    for img_refs in ref_tokens:
        seen: List[set] = [set() for _ in range(n)]
        for r in img_refs:
            for k in range(1, n + 1):
                for g in _ngrams(r, k):
                    seen[k - 1].add(g)
        for k in range(n):
            for g in seen[k]:
                doc_freq[k][g] += 1

    log_num_images = math.log(max(num_images, 1))

    def _vec(tokens: List[str], k: int) -> "tuple[Dict, float]":
        """tf-idf vector (dict) for order-k n-grams + its L2 norm."""
        counts = _ngrams(tokens, k + 1)
        total = sum(counts.values())
        vec: Dict = {}
        norm_sq = 0.0
        for g, c in counts.items():
            tf = c / total if total else 0.0
            df = doc_freq[k].get(g, 0)
            idf = log_num_images - math.log(max(df, 1))
            w = tf * idf
            vec[g] = w
            norm_sq += w * w
        return vec, math.sqrt(norm_sq)

    per_image: List[float] = []
    for img_idx in range(num_images):
        hyp = hyp_tokens[img_idx]
        img_refs = ref_tokens[img_idx]
        # Average CIDEr over n orders and over references.
        order_scores = []
        for k in range(n):
            hvec, hnorm = _vec(hyp, k)
            if hnorm == 0 or not img_refs:
                order_scores.append(0.0)
                continue
            ref_scores = []
            for r in img_refs:
                rvec, rnorm = _vec(r, k)
                if rnorm == 0:
                    ref_scores.append(0.0)
                    continue
                # cosine similarity of tf-idf vectors
                dot = sum(hvec.get(g, 0.0) * w for g, w in rvec.items())
                cos = dot / (hnorm * rnorm)
                # length-difference gaussian penalty (CIDEr-D)
                delta = len(hyp) - len(r)
                penalty = math.exp(-(delta ** 2) / (2 * sigma * sigma))
                ref_scores.append(cos * penalty)
            order_scores.append(sum(ref_scores) / len(ref_scores))
        # CIDEr-D multiplies the mean-over-orders by 10 by convention.
        per_image.append(10.0 * sum(order_scores) / n)

    mean = float(np.mean(per_image)) if per_image else 0.0
    return {"cider": round(mean, 4), "cider_per": [round(s, 4) for s in per_image]}


# ── Aggregation ─────────────────────────────────────────────────────────────────

def aggregate_caption_metrics(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """
    Compute all captioning metrics over a corpus.

    ``hypotheses[i]`` is scored against ``references[i]`` (a list of ref strings
    for image i).  BLEU/ROUGE-L/METEOR are averaged per-sentence; CIDEr is a
    corpus metric.  Returns rounded scalars; empty input → all zeros.
    """
    n = len(hypotheses)
    if n == 0 or len(references) != n:
        return {
            "bleu_1": 0.0, "bleu_2": 0.0, "bleu_3": 0.0, "bleu_4": 0.0,
            "rouge_l": 0.0, "meteor": 0.0, "cider": 0.0, "n_samples": 0,
        }

    bleu_acc = {f"bleu_{k}": 0.0 for k in range(1, 5)}
    rouge_acc = 0.0
    meteor_acc = 0.0
    for hyp, refs in zip(hypotheses, references):
        b = bleu(hyp, refs, max_n=4)
        for k in range(1, 5):
            bleu_acc[f"bleu_{k}"] += b[f"bleu_{k}"]
        rouge_acc += rouge_l(hyp, refs)
        meteor_acc += meteor(hyp, refs)

    cider_res = cider(list(hypotheses), [list(r) for r in references], n=4)

    result = {k: round(v / n, 4) for k, v in bleu_acc.items()}
    result["rouge_l"] = round(rouge_acc / n, 4)
    result["meteor"] = round(meteor_acc / n, 4)
    result["cider"] = cider_res["cider"]
    result["n_samples"] = n
    return result
