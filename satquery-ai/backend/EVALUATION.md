# SatQuery AI — Reproducible Evaluation Framework

This document describes every evaluation domain, the datasets it expects,
the metrics it computes, the baseline/adapted comparison model, and the
limitations of each metric.  It is intended as the single authoritative
reference for anyone running or interpreting the benchmark.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Framework Design](#2-framework-design)
3. [Configuration](#3-configuration)
4. [Domain: VQA](#4-domain-vqa)
5. [Domain: Captioning](#5-domain-captioning)
6. [Domain: Grounding](#6-domain-grounding)
7. [Domain: Retrieval (RS-CLIP)](#7-domain-retrieval-rs-clip)
8. [Domain: Change Detection](#8-domain-change-detection)
9. [Domain: SAR-Optical Fusion](#9-domain-sar-optical-fusion)
10. [Domain: Task Routing](#10-domain-task-routing)
11. [Domain: Confidence Calibration](#11-domain-confidence-calibration)
12. [Metric Implementations and Limitations](#12-metric-implementations-and-limitations)
13. [Honesty Constraints](#13-honesty-constraints)
14. [Reproducibility Commands](#14-reproducibility-commands)
15. [Output Format](#15-output-format)

---

## 1. Quick Start

```bash
# From satquery-ai/backend/
# Install pytest (once)
venv/Scripts/pip.exe install pytest   # Windows
# or: venv/bin/pip install pytest     # Linux/macOS

# Run offline metric tests (no datasets, no models, no network)
venv/Scripts/python.exe -m pytest tests/test_evaluation_metrics.py -v

# Full test suite (must still pass — no regressions)
venv/Scripts/python.exe -m pytest -q

# Routing smoke-run (offline, uses built-in gold set, no models downloaded)
venv/Scripts/python.exe -m evaluation.routing.run \
    --config evaluation/configs/evaluation.yaml

# VQA smoke-run (downloads BLIP if not cached; skips cleanly without dataset)
venv/Scripts/python.exe -m evaluation.vqa.run \
    --config evaluation/configs/evaluation.yaml --max-samples 4
```

All runners exit 0 and write honest `n/a` / "skipped" entries when data is
absent — they never fabricate numbers.

---

## 2. Framework Design

### Package layout

```
evaluation/
  __init__.py
  common/
    config.py          # YAML load + ${ENV:-default} resolution
    reporting.py       # write_json, append_markdown, comparison_row
    variants.py        # baseline/adapted Variant resolution + honest n/a
    metrics/
      __init__.py      # single import surface (re-exports + new metrics)
      captioning.py    # BLEU, ROUGE-L, METEOR, CIDEr  (new, pure Python)
      detection.py     # Acc@IoU, mIoU, PR@IoU          (new, uses box_iou)
      change_mask.py   # P/R/F1/IoU/OA for binary masks (new, pure numpy)
      routing.py       # accuracy, per-task PRF, confusion (new, no deps)
  vqa/run.py           # wraps training.evaluate_vqa
  captioning/run.py    # wraps models.captioning_model
  grounding/run.py     # wraps models.grounding_model
  retrieval/run.py     # wraps models.rs_clip.encoder + training.evaluate_clip
  change/run.py        # wraps models.change_model (mask + VQA tracks)
  fusion/run.py        # wraps models.sar_fusion_model
  routing/run.py       # wraps agent.task_classifier
  confidence/run.py    # wraps confidence.metrics
  datasets/
    cdvqa.py           # CDVQA bi-temporal change-VQA adapter
  configs/
    evaluation.yaml    # all paths via ${ENV:-default}
  results/             # JSON + Markdown output (gitignored except .gitkeep)
  routing/
    routing_eval_set.json  # 24-entry curated gold set (shipped with repo)
```

### Baseline vs adapted

Each domain compares two variants:

| Variant | Description |
|---------|-------------|
| **Baseline** | Off-the-shelf pretrained model with no RS adaptation |
| **SatQuery adapted** | Domain-adapted model (fine-tuned checkpoint / improved threshold) |

When no adapted checkpoint is configured the SatQuery column shows `n/a`
and a reason note is logged.  **Numbers are never invented.**

### No-data behaviour

Every runner checks whether a dataset path is configured and reachable before
loading.  Missing data → the run still exits 0, writes a JSON with
`"available": false` and a reason string, and appends a skipped row to the
Markdown report.

---

## 3. Configuration

All configuration lives in `evaluation/configs/evaluation.yaml`.  Every
dataset path and checkpoint is supplied via `${ENV_VAR:-default}` expressions
resolved at load time by `evaluation/common/config.py`.  **No path is
hardcoded in source code.**

### Environment variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `VRSBENCH_DIR` | VQA, Captioning, Grounding, Retrieval | Root dir of the VRSBench dataset |
| `RSVQA_DIR` | VQA | Root dir of the RSVQA-LR or RSVQA-HR dataset |
| `LEVIR_CD_DIR` | Change (mask track) | LEVIR-CD root with `A/`, `B/`, `label/` |
| `CDVQA_DIR` | Change (VQA track) | CDVQA root with `questions.json` |
| `BIGEARTHNET_DIR` | Fusion | BigEarthNet root with `BigEarthNet-S2/` |
| `VQA_ADAPTED_CKPT` | VQA | Dir produced by `training/train_vqa.py` |
| `RS_CLIP_CKPT` | Retrieval, Routing | RS-CLIP `.pt` checkpoint from `training/train_clip.py` |
| `SAR_FUSION_CKPT` | Fusion | SAR-fusion checkpoint |
| `MODEL_CACHE_DIR` | All | HuggingFace model cache (default: `./model_cache`) |
| `EVAL_OUTPUT_DIR` | All | Results output dir (default: `./evaluation/results`) |

### CLI overrides

Every runner accepts:

```
--config <path>         Path to YAML config (required)
--output <dir>          Override output_dir
--device auto|cuda|cpu  Compute device
--max-samples <int>     Cap on dataset size (for smoke tests)
```

---

## 4. Domain: VQA

**Runner:** `python -m evaluation.vqa.run --config evaluation/configs/evaluation.yaml`

### What it measures

Whether the VQA model produces correct textual answers to questions about
remote-sensing images.

### Datasets

| Dataset | Split | Notes |
|---------|-------|-------|
| **VRSBench** | test | Human-annotated image-question-answer triples over diverse RS scenes |
| **RSVQA** (LR or HR) | test | Categorical yes/no + numerical questions over Sentinel-2 / aerial imagery |

Both datasets are configured via environment variables.  The runner skips any
dataset whose path is unset or missing.

**Test-set isolation:** adapters are instantiated with `split="test"`.  The
training scripts instantiate with `split="train"` / `split="val"`.  These are
separate objects — no test labels flow into training.

### Models

| Variant | Model | Source |
|---------|-------|--------|
| Baseline | `Salesforce/blip-vqa-base` | HuggingFace pretrained |
| Adapted | Fine-tuned BLIP checkpoint | `$VQA_ADAPTED_CKPT` dir produced by `training/train_vqa.py` |

### Metrics

| Metric | Formula | Range | Better |
|--------|---------|-------|--------|
| **Exact Match** | `normalize(pred) == normalize(gold)` | [0, 1] | ↑ |
| **VQA Accuracy** | `min(1, matches/3)` (multi-annotator formula) | [0, 1] | ↑ |
| **Token F1** | Unigram overlap F1 between pred and gold | [0, 1] | ↑ |

Per-category breakdowns are included when `question_type` metadata is
available (RSVQA provides this; VRSBench may not).

### Confidence evidence

`run_inference` is called with `return_scores=True`.  Each per-sample score is
mapped to a `[0, 1]` confidence via `exp(score)` (clamped; samples with no score
are skipped).  This is **not a calibrated probability** — it is raw model
evidence.  The VQA runner records it as a list of `{confidence, correct}` pairs
under the `confidence_pairs` key of `vqa_results.json`, which the confidence
calibration domain (Section 11) consumes.

---

## 5. Domain: Captioning

**Runner:** `python -m evaluation.captioning.run --config evaluation/configs/evaluation.yaml`

### What it measures

Quality of free-form scene descriptions generated by the captioning model.

### Datasets

| Dataset | Split | Notes |
|---------|-------|-------|
| **VRSBench** | val (captioning split) | Each image has ≥ 1 human reference caption |

### Models

| Variant | Model | Notes |
|---------|-------|-------|
| Baseline | `Salesforce/blip-image-captioning-base` | Pretrained on COCO |
| Adapted | `$CAPTION_ADAPTED_CKPT` | No RS captioning checkpoint exists in the system → reported as `n/a` |

### Metrics

All metrics are computed corpus-level (averaged per image).

| Metric | Range | Better | Notes |
|--------|-------|--------|-------|
| **BLEU-1** | [0, 1] | ↑ | Unigram precision × brevity penalty |
| **BLEU-2** | [0, 1] | ↑ | Bigram; 0 on short hypotheses |
| **BLEU-3** | [0, 1] | ↑ | Trigram |
| **BLEU-4** | [0, 1] | ↑ | 4-gram; often near 0 on short captions |
| **ROUGE-L** | [0, 1] | ↑ | LCS-based F1, best reference |
| **METEOR** | [0, 1] | ↑ | Unigram P/R harmonic mean × fragmentation penalty |
| **CIDEr** | [0, 10] | ↑ | tf-idf n-gram consensus, corpus-level |

See [Section 12](#12-metric-implementations-and-limitations) for limitations.

---

## 6. Domain: Grounding

**Runner:** `python -m evaluation.grounding.run --config evaluation/configs/evaluation.yaml`

### What it measures

Whether the model localises objects in satellite images via text queries.

### Datasets

| Dataset | Split | Notes |
|---------|-------|-------|
| **VRSBench** (referring boxes) | test | Image + text query + reference bounding boxes |

When no referring-expression manifest with boxes is found the runner reports an
explicit `no_data` result rather than inventing samples — it never falls back to
a synthetic probe set.

### Models

| Variant | Model | Notes |
|---------|-------|-------|
| Baseline | `google/owlvit-base-patch32` | Open-vocabulary OWL-ViT, no RS fine-tuning |
| Adapted | `$GROUNDING_ADAPTED_CKPT` | No RS grounding checkpoint → `n/a` |

### Metrics

Boxes are normalised [0, 1] `[x1, y1, x2, y2]`.

| Metric | Description | Better |
|--------|-------------|--------|
| **mIoU** | Mean of max IoU between predicted and reference boxes | ↑ |
| **Acc@0.5** | Fraction of samples with best IoU ≥ 0.5 | ↑ |
| **Acc@0.75** | Fraction of samples with best IoU ≥ 0.75 | ↑ |
| **PR@0.5** | Precision / Recall / F1 at IoU ≥ 0.5 (one-to-one greedy matching) | ↑ |

---

## 7. Domain: Retrieval (RS-CLIP)

**Runner:** `python -m evaluation.retrieval.run --config evaluation/configs/evaluation.yaml`

### What it measures

How well the RS-CLIP image-text encoder retrieves matching captions / images.

### Datasets

| Dataset | Split | Notes |
|---------|-------|-------|
| **VRSBench** (captions) | test | Image–caption pairs; i↔i is ground truth |

### Models

| Variant | Model | Notes |
|---------|-------|-------|
| Baseline | OpenAI CLIP ViT-B/32 | `RSCLIPEncoder.from_pretrained(checkpoint_path=None)` |
| Adapted | RS-CLIP fine-tuned | `$RS_CLIP_CKPT` checkpoint from `training/train_clip.py` |

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **Image→Text R@1** | % of images where the matching text is the top-1 retrieval | ↑ |
| **Image→Text R@5** | % within top-5 | ↑ |
| **Image→Text R@10** | % within top-10 | ↑ |
| **Text→Image R@1** | Symmetric direction | ↑ |
| **Text→Image R@5** | | ↑ |
| **Text→Image R@10** | | ↑ |

---

## 8. Domain: Change Detection

**Runner:** `python -m evaluation.change.run --config evaluation/configs/evaluation.yaml`

Two independent tracks share the same runner.

### Track A — Change-Mask (pixel-level)

#### Dataset

| Dataset | Layout | Notes |
|---------|--------|-------|
| **LEVIR-CD** | `<split>/A/`, `<split>/B/`, `<split>/label/` | Binary PNG change masks |

Configure via `$LEVIR_CD_DIR`.  Adapter: `evaluation/change/run.py::LevirCDAdapter`.

#### Models / variants

| Variant | Threshold | Notes |
|---------|-----------|-------|
| Baseline | 0.35 (ResNet cosine distance) | Pretrained ImageNet ResNet-50 backbone |
| Adapted | `$CHANGE_ADAPTED_THRESHOLD` | Sweep to a tuned threshold; no new backbone training |

#### Metrics

| Metric | Formula | Better |
|--------|---------|--------|
| **Precision** | TP / (TP + FP) over all pixels | ↑ |
| **Recall** | TP / (TP + FN) | ↑ |
| **F1** | 2 × P × R / (P + R) | ↑ |
| **IoU** | TP / (TP + FP + FN) (Jaccard, change class) | ↑ |
| **Overall Accuracy** | (TP + TN) / total pixels | ↑ |

Metrics aggregate across the whole test set (micro-average): TP/FP/FN/TN are
summed over all image pairs before computing ratios.

### Track B — Change VQA

#### Dataset

| Dataset | Layout | Notes |
|---------|--------|-------|
| **CDVQA** | `questions.json`, `images_A/`, `images_B/` | Bi-temporal QA pairs |

Configure via `$CDVQA_DIR`.  Adapter: `evaluation/datasets/cdvqa.py::CDVQAAdapter`.

#### Metrics

Same as VQA domain: Exact Match, VQA Accuracy, Token F1, per-category.

---

## 9. Domain: SAR-Optical Fusion

**Runner:** `python -m evaluation.fusion.run --config evaluation/configs/evaluation.yaml`

### What it measures

Land-cover classification quality when SAR and optical features are fused,
evaluated on multi-label scene classification.

### Dataset

| Dataset | Split | Notes |
|---------|-------|-------|
| **BigEarthNet-S2** | test (MD5 deterministic split) | 43-class multi-label land cover; S1 SAR optionally paired |

Configure via `$BIGEARTHNET_DIR`.  Split: deterministic 80/10/10 via MD5 hash
of patch name — identical across runs and environments.

### Models

| Variant | Notes |
|---------|-------|
| Baseline | `SAROpticalFusionModel` with ImageNet-pretrained encoders, no fusion training — a real but randomly-initialised adapter, flagged `requires_verification` (numbers real, NOT trustworthy) |
| Adapted | `$SAR_FUSION_CKPT` — trained fusion checkpoint; the adapted column is `n/a` when the checkpoint is absent (never fabricated) |

### Metrics

Multi-label classification via `training.evaluate.multilabel_metrics` (sklearn):

| Metric | Description | Better |
|--------|-------------|--------|
| **Macro F1** | Unweighted mean F1 over 43 classes | ↑ |
| **Micro F1** | Globally pooled P/R/F1 | ↑ |
| **Mean AP (mAP)** | Mean Average Precision over classes | ↑ |
| **Macro Precision** | | ↑ |
| **Macro Recall** | | ↑ |

---

## 10. Domain: Task Routing

**Runner:** `python -m evaluation.routing.run --config evaluation/configs/evaluation.yaml`

### What it measures

Whether the `TaskClassifier` routes user queries to the correct downstream
model pipeline.

### Gold set

`evaluation/routing/routing_eval_set.json` — **24 hand-curated entries**
shipped with the repository.  Each entry:

```json
{
  "query": "...",
  "num_images": 1 or 2,
  "modalities": ["optical"] or ["sar", "optical"] etc.,
  "gold_task": "SINGLE_VQA|CAPTIONING|GROUNDING|CHANGE_VQA|CHANGE_DESCRIPTION|SAR_OPTICAL_FUSION"
}
```

Coverage: 4 entries per task type × 6 types = 24 total.

This is **not a held-out benchmark test set from a published dataset** — it is
a curated smoke-test set for the routing layer.  It does not represent the
distribution of real user queries; add more entries before drawing conclusions.

### Models

| Variant | Description |
|---------|-------------|
| Baseline | `TaskClassifier(semantic_router=None)` — keyword + structural heuristics only |
| Adapted | `TaskClassifier(semantic_router=SemanticRouter(RSCLIPEncoder))` — semantic blending; requires `$RS_CLIP_CKPT` and `routing.use_semantic_router: true` |

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **Accuracy** | Fraction of queries routed to the correct task type | ↑ |
| **Macro F1** | Unweighted mean F1 over 6 task types | ↑ |
| **Weighted F1** | Support-weighted F1 | ↑ |
| **Per-task P/R/F1** | Precision, recall, F1 per task type | ↑ |
| **Confusion Matrix** | 6×6 gold×predicted counts | — |

---

## 11. Domain: Confidence Calibration

**Runner:** (run after VQA) `python -m evaluation.confidence.run --config evaluation/configs/evaluation.yaml`

### What it measures

How well the model's reported confidence aligns with empirical correctness
(reliability / calibration).

### Data source

Reads the `confidence_pairs` list embedded in `<output_dir>/vqa_results.json`
(written by the VQA runner; the source domain is configurable via
`CONFIDENCE_SOURCE`, default `vqa`).  Each pair is:

```json
{
  "confidence": 0.73,
  "correct": 1
}
```

`confidence` is the VQA runner's `exp(score)` evidence (Section 4); `correct` is
`1`/`0` from exact match.  Samples whose model score was unavailable are already
excluded upstream (the VQA runner skips them), so no placeholder entries reach
this domain.

When no `confidence_pairs` are available (VQA not run in this `output_dir`, or no
scored samples) the runner reports `no_data` — it never generates synthetic
pairs.

### Models / variants

| Variant | Description |
|---------|-------------|
| Baseline | Raw sequence-score evidence (uncalibrated) |
| Adapted | `n/a` — no post-hoc calibration module configured yet |

### Metrics (all lower-is-better)

| Metric | Description | Range |
|--------|-------------|-------|
| **ECE** | Expected Calibration Error: weighted mean of |confidence − accuracy| per bin | [0, 1] |
| **MCE** | Maximum Calibration Error: worst-bin |confidence − accuracy| | [0, 1] |
| **Brier Score** | Mean squared error: mean((confidence − correct)²) | [0, 1] |

The comparison table negates improvement so "positive = SatQuery is better"
is preserved even though lower ECE/MCE/Brier is better.

---

## 12. Metric Implementations and Limitations

### BLEU

**Implementation:** `evaluation/common/metrics/captioning.py::bleu`

Sentence-BLEU with method-1 smoothing (epsilon floor on zero n-gram counts)
and a brevity penalty against the closest reference length.  Computes BLEU-1
through BLEU-4 as a geometric mean of clipped precisions up to order n.

**Limitations:**
- Uses `1e-9` smoothing instead of the full add-k or geometric-interpolation
  variants used by SacreBLEU.  Numbers may differ slightly from published
  results using sacrebleu.
- Sentence-level averaging (not corpus-level BLEU).  Multi-sentence BLEU
  computed by `aggregate_caption_metrics` averages per-image scores, which
  differs from the true corpus-level BLEU by an amount that grows with
  variance in hypothesis length.

### ROUGE-L

**Implementation:** `evaluation/common/metrics/captioning.py::rouge_l`

LCS-based F1 with β = 1.2 (recall-weighted), taking the best reference.

**Limitations:**
- Uses character-based tokenisation (alphanumerics only), not the
  ROUGE-1.5.5 Porter stemmer.  Scores may differ from the reference
  implementation by 1–3 ROUGE points on some corpora.

### METEOR

**Implementation:** `evaluation/common/metrics/captioning.py::meteor`

F-mean = P×R / (0.9P + 0.1R) with fragmentation penalty
1 − 0.5×(chunks/matches)³.  Greedy exact-token alignment.

**Documented limitation (not hidden):**
METEOR's original design includes synonym matching (WordNet) and stemming.
This implementation uses **exact unigram matching only**.  Neither WordNet
nor a stemmer is in `requirements.txt`; adding external NLP dependencies for
a single metric is out of scope.  Scores will be lower than reference-tool
METEOR by 5–15 points on English captions with synonym-rich references.
Every report generated by this framework includes this warning.

### CIDEr

**Implementation:** `evaluation/common/metrics/captioning.py::cider`

CIDEr-D: tf-idf n-gram cosine similarity with a Gaussian length penalty
(σ = 6.0, δ = |hyp_len − ref_len|), averaged over n=1..4, multiplied by 10.
Document frequencies are computed over the reference corpus passed in.

**Limitations:**
- tf-idf is computed over the evaluation batch only, not a large background
  corpus.  This reduces IDF discriminativeness relative to CIDEr computed
  over the full COCO corpus.
- Per-image score averaging differs from the reference pycocoevalcap
  implementation when references per image are unequal.

### Change-Mask Metrics (P/R/F1/IoU/OA)

**Implementation:** `evaluation/common/metrics/change_mask.py`

Pixel-level binary metrics.  TP/FP/FN/TN are summed across all test-set image
pairs before computing ratios (correct micro-average).

**Limitations:**
- The ResNet-50 change-detection backbone was pre-trained on ImageNet
  classification, not change detection.  Feature-space cosine distance is a
  proxy metric.  Absolute values should not be compared against dedicated
  change-detection models (e.g., ChangeFormer, BIT).

### Detection Metrics (Acc@IoU, mIoU, PR@IoU)

**Implementation:** `evaluation/common/metrics/detection.py`

Built on `confidence.metrics.box_iou`.  PR uses greedy score-descending
one-to-one matching (not COCO's 101-point interpolated AP).

**Limitations:**
- Single-class, single-threshold evaluation.  Not equivalent to COCO mAP
  which averages over 10 IoU thresholds and 80 classes.

### Routing Metrics

**Implementation:** `evaluation/common/metrics/routing.py`

Standard micro/macro precision/recall/F1 over the 6-class routing gold set.

**Limitations:**
- The gold set contains only 24 entries (4 per task type).  Per-task
  statistics based on 4 samples are high-variance.  The set should be
  expanded before drawing strong conclusions.
- The gold set was constructed by project authors and may not reflect
  real user query distributions.

### Calibration Metrics (ECE/MCE/Brier)

**Implementation:** `confidence/metrics.py` (re-exported)

Equal-width bins over [0, 1].

**Limitations:**
- Confidence values are `exp(mean_log_prob)` from beam search decoding —
  a real model signal but not a probability in the frequentist sense.
  A well-calibrated model in this sense means high-log-prob answers tend to be
  correct, which is a weaker claim than probabilistic calibration.
- n_bins = 10 by default; bins with no samples contribute 0 to ECE.

---

## 13. Honesty Constraints

The following invariants are enforced in every runner:

1. **No fabrication.** When a dataset is absent or a checkpoint is unconfigured,
   the SatQuery column is `null` / `"n/a"` in both JSON and Markdown.
   No number is invented.

2. **No test-set training.** Dataset adapters are instantiated per split
   (`split="train"` / `"val"` / `"test"`).  The evaluation runners always
   use `split="test"`.  Adapters use deterministic MD5-based splits, not
   Python's `hash()` (which is non-deterministic across processes).

3. **No calibrated confidence claims.** The VQA runner's `confidence_pairs` are
   raw, uncalibrated `exp(score)` evidence (Section 4), and the confidence
   calibration domain reports `n/a` for any calibrated/adapted variant.  The
   model card and this document state what the confidence field represents.

4. **Explicit degraded flags.** When a model cannot be constructed the runner
   emits a `no_data` report (there is no `MockModel` substitution).  The fusion
   baseline is a real but randomly-initialised adapter: its numbers are flagged
   `requires_verification` and the runner notes they are "real but NOT
   trustworthy — set SAR_FUSION_CKPT for a trained comparison".

5. **Metric limitations documented, not hidden.** Every approximation in
   `evaluation/common/metrics/captioning.py` is documented in this file and
   in the module docstring.

---

## 14. Reproducibility Commands

### Prerequisites

```bash
cd satquery-ai/backend
# Activate venv (Windows)
venv\Scripts\activate
# or Linux/macOS:
source venv/bin/activate
```

### Offline smoke-tests (no data, no models, no network)

```bash
# Unit tests for all pure-Python metrics + config resolver + smoke imports
python -m pytest tests/test_evaluation_metrics.py -v

# Routing runner (uses built-in gold set, only imports agent.task_classifier)
python -m evaluation.routing.run --config evaluation/configs/evaluation.yaml

# Full existing suite — must still pass
python -m pytest -q
```

### With datasets configured

```bash
export VRSBENCH_DIR=/data/VRSBench
export RSVQA_DIR=/data/RSVQA-LR
export LEVIR_CD_DIR=/data/LEVIR-CD
export CDVQA_DIR=/data/CDVQA
export BIGEARTHNET_DIR=/data/BigEarthNet
export MODEL_CACHE_DIR=./model_cache

# Individual domains
python -m evaluation.vqa.run        --config evaluation/configs/evaluation.yaml
python -m evaluation.captioning.run --config evaluation/configs/evaluation.yaml
python -m evaluation.grounding.run  --config evaluation/configs/evaluation.yaml
python -m evaluation.retrieval.run  --config evaluation/configs/evaluation.yaml
python -m evaluation.change.run     --config evaluation/configs/evaluation.yaml
python -m evaluation.fusion.run     --config evaluation/configs/evaluation.yaml
python -m evaluation.routing.run    --config evaluation/configs/evaluation.yaml

# Confidence calibration (reads the confidence_pairs list in vqa_results.json)
python -m evaluation.confidence.run --config evaluation/configs/evaluation.yaml
```

### With adapted checkpoints

```bash
export VQA_ADAPTED_CKPT=checkpoints/vqa/best          # dir from train_vqa.py
export RS_CLIP_CKPT=checkpoints/rs_clip/rs_clip_best.pt
export SAR_FUSION_CKPT=checkpoints/sar_fusion/best.pt
export CHANGE_ADAPTED_THRESHOLD=0.25                   # float threshold

python -m evaluation.vqa.run       --config evaluation/configs/evaluation.yaml
python -m evaluation.retrieval.run --config evaluation/configs/evaluation.yaml
python -m evaluation.change.run    --config evaluation/configs/evaluation.yaml
python -m evaluation.fusion.run    --config evaluation/configs/evaluation.yaml
```

---

## 15. Output Format

### Per-domain JSON (`evaluation/results/<domain>_results.json`)

```json
{
  "domain": "vqa",
  "provenance": {
    "git_commit": "abc123",
    "dataset_version": "vrsbench+rsvqa@4f2c1a8b9d12",
    "datasets": [{ "name": "vrsbench", "data_dir": "...", "n_samples": 500 }]
  },
  "variants": [
    {
      "variant": "BLIP pretrained",
      "kind": "baseline",
      "available": true,
      "metrics": {
        "exact_match": 0.4210,
        "vqa_accuracy": 0.5320,
        "token_f1": 0.6140,
        "n_samples": 500,
        "per_category": { "yes/no": { ... }, "number": { ... } }
      },
      "n_samples": 500,
      "model_name": "Salesforce/blip-vqa-base",
      "datasets": [...]
    },
    {
      "variant": "BLIP fine-tuned",
      "kind": "adapted",
      "available": false,
      "reason": "no adapted checkpoint configured ($VQA_ADAPTED_CKPT is unset)"
    }
  ],
  "comparison_rows": [
    {
      "model": "BLIP",
      "dataset": "vrsbench+rsvqa",
      "metric": "Exact Match",
      "baseline": 0.4210,
      "adapted": null,
      "improvement": null
    }
  ],
  "notes": ["BLIP fine-tuned: no adapted checkpoint configured..."]
}
```

### Markdown report (`evaluation/results/benchmark_report.md`)

Each domain runner appends a section to the shared report:

```markdown
## Visual Question Answering (VQA)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| BLIP  | vrsbench+rsvqa | Exact Match | 0.4210 | n/a | n/a |
| BLIP  | vrsbench+rsvqa | VQA Accuracy | 0.5320 | n/a | n/a |
| BLIP  | vrsbench+rsvqa | Token F1 | 0.6140 | n/a | n/a |

**Notes:**
- BLIP fine-tuned: no adapted checkpoint configured ($VQA_ADAPTED_CKPT is unset)
```

`Improvement` = adapted − baseline for higher-is-better metrics (positive = SatQuery is better).
For lower-is-better metrics (ECE, MCE, Brier) improvement = baseline − adapted.
`n/a` when the adapted variant is unavailable.

---

*Last updated: auto-generated — re-read source code for authoritative metric definitions.*
