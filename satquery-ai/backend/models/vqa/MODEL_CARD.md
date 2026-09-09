# SatQuery AI — Remote-Sensing VQA Model Card

## Model Overview

| Field | Value |
|---|---|
| **Model name** | SatQuery RS-VQA |
| **Version** | 1.0 |
| **Base model** | `Salesforce/blip-vqa-base` (default) or `Salesforce/blip2-opt-2.7b` |
| **Task** | Visual Question Answering on remote-sensing imagery |
| **Fine-tuning method** | LoRA (parameter-efficient) via `peft` or a built-in manual LoRA backend |
| **Framework** | [transformers](https://github.com/huggingface/transformers) |
| **Input** | RGB image + natural-language question |
| **Output** | Free-form text answer + raw generation evidence |

The base BLIP / BLIP-2 VQA support is retained unchanged. Fine-tuning **adapts**
the model to remote-sensing imagery; it does not replace the base capability.

---

## Intended Use

Answering natural-language questions about satellite / aerial imagery:
land-cover presence, object counting, comparison, rural/urban classification,
and scene description.

### In-scope
- Optical remote-sensing VQA (nadir aerial / satellite RGB).
- Yes/no, counting, comparison, and category questions (RSVQA-style).
- Open-ended descriptive questions (VRSBench-style).

### Out-of-scope
- Pixel-level segmentation or detection (use the grounding model).
- Change detection (use the change model).
- SAR-only imagery — not represented in the fine-tuning data.
- Any safety-critical decision without human review.

---

## Training Method

- **Default: LoRA.** Only low-rank adapter weights are trained (typically
  <1% of parameters). Full-model fine-tuning is refused unless BOTH
  `lora.enabled=false` and `training.allow_full_finetune=true` are set — a
  deliberate two-key opt-in.
- **Loss:** the model's native conditional-generation cross-entropy over
  answer tokens (teacher forcing), with pad tokens masked to `-100`.
- **Training features:** dynamic padding, mixed precision (fp16/bf16 on CUDA),
  gradient accumulation, gradient clipping, cosine/linear LR schedule with
  warmup, checkpoint save/resume, per-epoch validation, early stopping, and a
  reproducible global seed.

### Reproduce

```bash
cd satquery-ai/backend
python training/train_vqa.py --config training/configs/vqa_base.yaml
```

Dataset locations are configured in `training/configs/vqa_base.yaml` (or via
environment). They are **never hardcoded** in code.

---

## Training Datasets

| Dataset | Split usage | License / Terms |
|---|---|---|
| [VRSBench VQA](https://arxiv.org/abs/2406.12681) | train + val (test held out) | CC BY 4.0 |
| [RSVQA](https://rsvqa.sylvainlobry.com/) (LR / HR / xBEN) | train + val (test held out) | Research use — see project page |

Additional open-source RS-VQA datasets can be added by writing an adapter and
registering it in `training/datasets/vqa/__init__.py::VQA_DATASET_REGISTRY`.

**No test-set training:** datasets with native splits are loaded per-split and
the test split is never included in the training loader. Datasets lacking
native splits get a deterministic, seeded train/val/test partition.

> Datasets are NOT bundled with this repository.

---

## Evaluation Metrics

Computed on the **test** split by `training/evaluate_vqa.py`:

- **Exact match** — normalised prediction equals normalised gold answer.
- **VQA accuracy** — `min(1, #agreeing_answers / 3)` (reduces to exact match
  for single-reference datasets).
- **Token F1** — SQuAD-style token overlap; meaningful for open-ended answers.
- **Per-question-category metrics** — broken down by `question_type` when the
  dataset provides it (e.g. RSVQA: count / presence / comparison / rural_urban).

The evaluator writes a JSON report:

```json
{
  "model": "...",
  "dataset": "...",
  "split": "test",
  "metrics": { "exact_match": 0.0, "vqa_accuracy": 0.0, "token_f1": 0.0,
               "per_category": { "...": { "...": 0.0 } } },
  "error_examples": [ { "question": "...", "gold": "...", "prediction": "..." } ]
}
```

Error examples (mispredictions) are recorded to support qualitative error
analysis. Actual metric values depend on your fine-tuning run and data.

---

## Confidence & Calibration

**The reported `confidence` is NOT calibrated.** `confidence_is_calibrated` is
always `False`.

Answer length is **not** used as confidence. Instead the wrapper returns the
raw model evidence needed for later calibration, under `evidence`:

| Field | Meaning |
|---|---|
| `sequence_score` | Mean per-token log-probability of the generated answer (or `null`) |
| `token_logprobs` | Per-token log-probabilities (or `null`) |
| `answer_length` | Number of words in the answer |
| `decoding` | Decoding parameters used (`num_beams`, `max_new_tokens`, …) |
| `model_agreement` | Reserved for ensemble / self-consistency signals |

The convenience `confidence` value is derived from `sequence_score` via
`exp(mean_logprob)` when available; it is a rough signal, not a probability of
correctness. Downstream calibration (e.g. temperature scaling on a held-out
set) should use the raw evidence.

---

## Inference / Model Source

The VQA wrapper loads either weights source through configuration — no local
checkpoint path is hardcoded:

```bash
# Base pretrained model
VQA_MODEL_SOURCE=pretrained

# Fine-tuned checkpoint (path from env, never hardcoded)
VQA_MODEL_SOURCE=finetuned
VQA_FINETUNED_CHECKPOINT=./checkpoints/vqa/best
```

A fine-tuned checkpoint directory contains: adapter/model weights, the
processor/tokenizer, `model_version.json` (base model, LoRA config, metrics,
dataset version, git commit), enabling exact reconstruction at inference.

---

## Model Versioning

Each saved checkpoint directory records:

- model / adapter weights (`adapter_weights.pt` or peft `adapter_model.*`)
- processor + tokenizer (`processor`/`tokenizer` files)
- training config (embedded in `model_version.json`)
- metrics (validation metrics at that epoch)
- dataset version fingerprint
- git commit hash (when available)

---

## Known Limitations & Failure Modes

1. **Uncalibrated confidence.** Do not threshold on `confidence` as if it were
   a probability. Use the raw evidence for calibration first.
2. **Optical-only.** No SAR / hyperspectral / nighttime training data;
   answers on such imagery are unreliable.
3. **Counting is hard.** BLIP-family models under- or over-count objects in
   dense scenes; treat exact counts skeptically.
4. **Hallucination on out-of-distribution scenes.** The model may produce a
   fluent but wrong answer for scene types absent from training data.
5. **Short-answer bias.** BLIP-VQA-base tends to emit terse answers; open-ended
   descriptive questions score lower on token-F1.
6. **Resolution loss.** Large tiles are downsampled by the processor; fine
   spatial detail is lost.
7. **Geographic / class bias** inherited from the training datasets (e.g.
   VRSBench / RSVQA scene distributions).

---

## License

- Base model weights: subject to the respective HuggingFace model license
  (BLIP / BLIP-2, Salesforce).
- LoRA adapter weights: governed by the licenses of the training datasets used
  (VRSBench CC BY 4.0; RSVQA per project terms).
- SatQuery AI training / inference code: see the repository root `LICENSE`.

---

*Generated as part of the SatQuery AI RS-VQA fine-tuning pipeline.*
