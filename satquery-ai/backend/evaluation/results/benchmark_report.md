# SatQuery AI — Benchmark Report

_Baseline vs SatQuery-adapted comparison. `n/a` marks a variant with no produced checkpoint (never fabricated)._

## VQA (BLIP baseline vs SatQuery fine-tuned)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: no VQA dataset configured. Set VRSBENCH_DIR and/or RSVQA_DIR (see evaluation/configs/evaluation.yaml).

## Retrieval (OpenAI CLIP vs RS-CLIP)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: retrieval dataset not configured (set VRSBENCH_DIR).

## Captioning (BLIP; RS-adapted n/a)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: captioning dataset not configured (set VRSBENCH_DIR).

## Grounding (OWL-ViT open-vocabulary; RS-adapted n/a)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: grounding dataset not configured (set VRSBENCH_DIR or a grounding manifest directory).

## Change detection (mask + change-VQA)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- mask: skipped — change-mask dataset not configured (set LEVIR_CD_DIR).
- change-vqa: skipped — CDVQA dataset not configured (set CDVQA_DIR).
- skipped: neither change-mask (LEVIR-CD) nor CDVQA configured. Set LEVIR_CD_DIR and/or CDVQA_DIR.

## SAR-optical fusion (land-cover on fused tokens)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: fusion dataset not configured (set BIGEARTHNET_DIR).

## Routing (task classification)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| TaskClassifier | routing_eval_set | accuracy | 1.0000 | 1.0000 | +0.0000 |
| TaskClassifier | routing_eval_set | macro_f1 | 0.1667 | 0.1667 | +0.0000 |

**Notes:**

- Baseline = majority-class predictor (always 'CAPTIONING').
- SatQuery = keyword+structural TaskClassifier (offline; semantic router disabled).
- Evaluated on 4 curated in-house gold queries (not benchmark test labels).

## Confidence calibration (VQA emitted confidence)

| Model | Dataset | Metric | Baseline | SatQuery | Improvement |
|-------|---------|--------|----------|----------|-------------|
| _(no comparable results — see notes)_ |||||| 

**Notes:**

- skipped: no (confidence, correct) pairs available. Run `python -m evaluation.vqa.run` with a dataset configured first — it emits confidence_pairs consumed here.

