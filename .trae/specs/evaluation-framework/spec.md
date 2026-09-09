# SatQuery AI - Reproducible Evaluation Framework

## Overview
- **Summary**: Complete the reproducible `evaluation/` framework that runs each of seven domains (VQA, captioning, grounding, retrieval, change, fusion, routing) plus confidence calibration via `python -m evaluation.<domain>.run`, producing baseline-vs-adapted JSON + Markdown reports with a `Model | Dataset | Metric | Baseline | SatQuery | Improvement` table.
- **Purpose**: Provide quantitative evidence that SatQuery's remote-sensing adaptations (fine-tuned VQA, RS-CLIP, SAR-optical fusion, routing classifier, task-aware confidence) actually improve results over generic pretrained baselines — or honestly report "n/a" where no adapted checkpoint exists.
- **Target Users**: SatQuery AI developers running benchmark evaluations, CI/CD pipelines tracking regressions, and research reviewers needing reproducible numbers.

## Goals
- G1: Every per-domain runner (`vqa`, `captioning`, `grounding`, `retrieval`, `change`, `fusion`, `routing`, `confidence`) is invocable as `python -m evaluation.<domain>.run` with `--config/--max-samples/--output/--device` CLI flags.
- G2: Each runner (where applicable) executes both a baseline variant AND an adapted variant, emitting an honest "n/a" row (never a fabricated number) when the adapted checkpoint is not configured or does not exist on disk.
- G3: Every run writes `<output_dir>/<domain>_results.json` (metrics + provenance + per-variant block) and appends to `<output_dir>/benchmark_report.md` with the standard 6-column comparison table.
- G4: New pure-Python metrics for captioning (BLEU/ROUGE/METEOR/CIDEr approximations), detection (Acc@IoU/mIoU/PR@IoU), change-mask (P/R/F1/IoU/overall), and routing (accuracy/per-task-PRF/confusion) pass offline unit tests against hand-computed values.
- G5: Config + environment only for paths — no hardcoded dataset or checkpoint directories in any Python file.
- G6: Captioning / grounding / change domains whose "adapted" variant does not yet exist in the system are reported as such, never invented.
- G7: Full test suite `tests/test_evaluation_metrics.py` passes; no regressions in the existing Task-4 suite (`pytest -q`).
- G8: Offline smoke runs for `routing.run` and `vqa.run --max-samples 4` succeed against the "no data" honest path (no datasets on disk, no adapted checkpoint).

## Non-Goals
- NG1: Modifying any file under `models/` or `api/` (framework only *reads* existing models).
- NG2: Modifying the Python backend FastAPI routes, controller, or agent logic.
- NG3: Running actual model inference against real datasets during the CI/unit-test phase (that requires user-supplied data + checkpoints).
- NG4: Installing new heavy dependencies (nltk, pycocoevalcap, etc.) — captioning metrics stay dependency-light with documented approximations.
- NG5: Producing real baseline-vs-adapted metric numbers in this task (those require the user to supply dataset paths + produced checkpoints).

## Background & Context
The system has several remote-sensing adaptations: a fine-tunable BLIP VQA path, an RS-CLIP encoder, a SAR-optical fusion adapter, a routing classifier, and a task-aware confidence layer. Scattered eval scripts exist under `training/` (evaluate_vqa.py, evaluate_clip.py, evaluate.py) but are per-model, do not compare baseline vs adapted, and do not cover captioning, grounding, change detection, routing, or confidence.

Already implemented (reused, not rebuilt):
- Common modules: `evaluation/common/config.py` (env-resolving YAML loader), `evaluation/common/reporting.py` (JSON + Markdown writers + comparison table), `evaluation/common/variants.py` (honest baseline/adapted resolution).
- Metrics modules: `evaluation/common/metrics/captioning.py` (bleu/rouge_l/meteor/cider), `change_mask.py`, `detection.py` (built on existing `box_iou`), `routing.py` — all present.
- Config: `evaluation/configs/evaluation.yaml` with per-domain dataset + variant blocks.

Missing, to be built fresh:
- 8 per-domain `run.py` files (vqa, captioning, grounding, retrieval, change, fusion, routing, confidence) each with its own `__init__.py`.
- `evaluation/datasets/cdvqa.py` (bi-temporal change-VQA adapter, honest "no data" path if not configured).
- `evaluation/routing/routing_eval_set.json` (curated gold `{query, num_images, modalities, gold_task}` set).
- `tests/test_evaluation_metrics.py` (offline unit tests against hand-computed values).
- `EVALUATION.md` top-level document.

## Functional Requirements
- **FR-1**: `evaluation.vqa.run` wraps `training/evaluate_vqa.load_eval_model` / `build_test_dataset` / `run_inference`; runs twice (pretrained + finetuned when `vqa_finetuned_checkpoint` set); reuses `aggregate_metrics`; also collects `(confidence, correct)` pairs for the confidence domain.
- **FR-2**: `evaluation.retrieval.run` wraps `training/evaluate_clip.extract_features` / `retrieval_recall`; compares openai-base CLIP vs RS-CLIP adapted encoder when `rs_clip_checkpoint` exists.
- **FR-3**: `evaluation.captioning.run` loads `RemoteSensingCaptioning`, runs over `VRSBenchAdapter`, scores with `aggregate_caption_metrics`; adapted variant = "n/a (no captioning checkpoint)".
- **FR-4**: `evaluation.grounding.run` loads `RemoteSensingGrounding`, runs over a referring-expression set; scores with `aggregate_detection_metrics`; adapted = "n/a".
- **FR-5**: `evaluation.change.run` has two tracks: (a) change-mask metrics via `ChangeDetectionModel.detect_changes` + `change_mask_metrics` against a mask dataset (e.g. LEVIR-CD) when configured; (b) change-VQA via new `CDVQA` adapter + `answer_change_question`. Baseline vs improved-threshold if configured, else single-variant honest report.
- **FR-6**: `evaluation.fusion.run` loads `SAROpticalFusionModel` untrained vs `sar_fusion_checkpoint`; reports `fusion_trained` flag; land-cover classification metrics (via `training.evaluate.multilabel_metrics`) on fused tokens over BigEarthNet paired split; untrained explicitly flagged.
- **FR-7**: `evaluation.routing.run` runs `TaskClassifier.classify` over `routing_eval_set.json`; scores `aggregate_routing_metrics` (accuracy + per-task P/R/F1 + confusion matrix).
- **FR-8**: `evaluation.confidence.run` consumes `(confidence, correct)` pairs emitted by a prior VQA run (or re-runs a mini-VQA); feeds `reliability_stats` → ECE/Brier/reliability curve.
- **FR-9**: `evaluation.datasets.cdvqa.py` implements an adapter for bi-temporal change VQA with the same honest "not configured → empty dataset" pattern as VRSBench.
- **FR-10**: Every `run.py` accepts CLI flags `--config`, `--max-samples`, `--output`, `--device`; resolves device using `config.resolve_device("auto")` by default.
- **FR-11**: All reporting reuses `common/reporting.write_json` and `common/reporting.append_markdown`; all variant decisions reuse `common/variants.resolve_variants` / `unavailable_adapted`.
- **FR-12**: Provenance block (git_commit, dataset_version) attached to every JSON report reusing `training.vqa_utils.{git_commit_hash, dataset_version}` best-effort.
- **FR-13**: Honesty: missing dataset or missing adapted checkpoint never produces a zero metric that could be mistaken for a real result — the table row either shows "n/a" or includes an explicit "skipped: no data" note.

## Non-Functional Requirements
- **NFR-1**: Pure-Python metrics have no new pip dependencies beyond `numpy` (already in requirements).
- **NFR-2**: All metric functions are safe on empty / mismatched input (return zeros + `valid=False` where applicable).
- **NFR-3**: Offline unit tests complete in < 60 seconds on a typical laptop, with no GPU or network calls.
- **NFR-4**: All 8 domain modules import cleanly without triggering model download / inference (`python -c "import evaluation.vqa.run"` must not hang on HF download).
- **NFR-5**: Code style matches existing conventions (dataclasses, logging via `logging.getLogger("satquery.eval.<domain>")`, docstrings with `Examples` sections, no comments-as-decoration).

## Constraints
- **Technical**: No changes to `models/` or `api/`; framework only reads those classes. No new heavy deps. All paths via config + env.
- **Business**: No fabricated adapted numbers. Limitations of pure-Python metric approximations (METEOR without WordNet, CIDEr without pycocoevalcap) documented in EVALUATION.md.
- **Dependencies**: Reuse existing training-side helpers where listed in spec; add no new third-party packages.

## Assumptions
- A-1: The user will supply real dataset paths + produced checkpoints later via env vars / config overrides. The framework runs honestly without them, marking missing data "skipped".
- A-2: `pytest` is available and the existing Task-4 test suite is green before this work begins.
- A-3: Windows venv at `backend/venv/Scripts/python.exe` per the project memory notes.

## Acceptance Criteria

### AC-1: All 8 domain runners exist with correct CLI and module invocation
- **Type**: `rule`
- **Given**: The `evaluation/` package structure.
- **When**: Running `python -c "import evaluation.vqa.run, evaluation.captioning.run, evaluation.grounding.run, evaluation.retrieval.run, evaluation.change.run, evaluation.fusion.run, evaluation.routing.run, evaluation.confidence.run"` from the backend dir.
- **Then**: All imports succeed without error or network downloads.
- **Pass Condition**: No ImportError / ModuleNotFoundError / download hang.
- **Evidence**: `python -c` command output showing success, plus file listing showing each `<domain>/__init__.py` and `<domain>/run.py` present.

### AC-2: Routing eval set exists and is a valid JSON array of gold tuples
- **Type**: `rule`
- **Given**: `evaluation/routing/routing_eval_set.json`.
- **When**: Loading and validating the file contents.
- **Then**: It is a JSON array of >= 15 objects, each with keys `query`, `num_images`, `modalities`, `gold_task`; each `gold_task` is a valid `TaskType` value.
- **Pass Condition**: JSON parses without error; schema check passes on all entries; >= 15 entries cover all 6 TaskTypes.
- **Evidence**: Short script output showing count + TaskType coverage.

### AC-3: CDVQA adapter exists with honest "not configured" path
- **Type**: `rule`
- **Given**: `evaluation/datasets/cdvqa.py` exposing a `CDVQAAdapter` class.
- **When**: Instantiating the adapter with a blank / non-existent data_dir.
- **Then**: `len(adapter) == 0`; iterating yields no samples; no exception raised.
- **Pass Condition**: Constructor succeeds with empty dir and `len == 0`.
- **Evidence**: Unit test in `test_evaluation_metrics.py`.

### AC-4: Captioning metrics match hand-computed values
- **Type**: `rule`
- **Given**: A fixed hypothesis + references pair.
- **When**: Computing `bleu`, `rouge_l`, `meteor` offline.
- **Then**: Values match hand-computed values within tolerance (e.g., identical BLEU-1 on an exact match; identical ROUGE-L on a simple overlap case).
- **Pass Condition**: All assertions in `TestCaptioningMetrics` class pass.
- **Evidence**: pytest green for the class.

### AC-5: Detection and change-mask metrics match hand-computed values
- **Type**: `rule`
- **Given**: Hand-computed box and mask test cases.
- **When**: Computing `acc_at_iou`, `mean_iou`, `pr_at_iou`, `change_mask_metrics`.
- **Then**: Values match known results.
- **Pass Condition**: All assertions in `TestDetectionMetrics` + `TestChangeMaskMetrics` pass.
- **Evidence**: pytest green.

### AC-6: Routing metrics match hand-computed values
- **Type**: `rule`
- **Given**: Known gold/pred label lists.
- **When**: Computing `routing_accuracy`, `per_task_prf`, `confusion_matrix`.
- **Then**: Values match hand-computed.
- **Pass Condition**: All assertions in `TestRoutingMetrics` pass.
- **Evidence**: pytest green.

### AC-7: Comparison table builder produces correct Markdown
- **Type**: `rule`
- **Given**: A list of `comparison_row(...)` objects including adapted=n/a cases.
- **When**: `build_comparison_table(rows)`.
- **Then**: Output is valid Markdown with the correct 6-column header; "n/a" rows correctly display `SatQuery = n/a` and `Improvement = n/a`; numeric rows show `Improvement = adapted - baseline` with proper sign.
- **Pass Condition**: Assertions against the produced string in `TestReporting`.
- **Evidence**: pytest green.

### AC-8: Config env resolution works for ${VAR} and ${VAR:-default}
- **Type**: `rule`
- **Given**: A temp YAML with both placeholder styles and env vars set/unset.
- **When**: `load_config` and `resolve_env` are called.
- **Then**: Unset bare `${VAR}` resolves to ""; `${VAR:-def}` resolves to "def"; set var resolves to its value.
- **Pass Condition**: `TestConfigEnv` assertions pass.
- **Evidence**: pytest green.

### AC-9: Honest routing.run smoke test succeeds with no semantic router
- **Type**: `rule`
- **Given**: An environment with no dataset paths and no adapted checkpoints.
- **When**: Running `venv/Scripts/python.exe -m evaluation.routing.run --config evaluation/configs/evaluation.yaml`.
- **Then**: The script exits with code 0; writes `routing_results.json` with honest results; appends to `benchmark_report.md` with a (possibly sparse) comparison table.
- **Pass Condition**: Exit code 0 + both files exist.
- **Evidence**: Shell command output + file listing.

### AC-10: Honest vqa.run smoke test succeeds with --max-samples 4
- **Type**: `rule`
- **Given**: Same environment as AC-9.
- **When**: Running `venv/Scripts/python.exe -m evaluation.vqa.run --config evaluation/configs/evaluation.yaml --max-samples 4`.
- **Then**: Script exits 0; detects "no dataset configured"; writes a JSON report marked "skipped: no data"; appends honest notes to benchmark_report.md.
- **Pass Condition**: Exit code 0 + JSON contains a "skipped" / "notes" block + no fabricated metric rows.
- **Evidence**: Shell command output + JSON inspection.

### AC-11: Full pytest suite green (no regressions)
- **Type**: `rule`
- **Given**: The completed implementation.
- **When**: Running `venv/Scripts/python.exe -m pytest -q` from the backend dir.
- **Then**: All tests pass (Task-4 suite + new evaluation test file).
- **Pass Condition**: Exit code 0; summary shows `passed == N_total`.
- **Evidence**: pytest terminal summary.

### AC-12: EVALUATION.md documentation completeness
- **Type**: `rubric`
- **Dimension**: Documentation of datasets, splits, metrics, methodology, limitations, reproducibility commands.
- **Scale**: 1-5
- **Anchors**: 1 = Missing or placeholder doc; 3 = Present but misses limitations or commands; 5 = All sections present, honest approximations documented, copy-paste-ready commands for each domain + provenance notes.
- **Pass Threshold**: >= 4
- **Evidence**: Review of `EVALUATION.md` file content.

## Open Questions
- [ ] None remaining — the scope is fully bounded by the user's detailed spec; any missing adapted checkpoint or dataset is handled by the honest "n/a" / "skipped" path rather than a question to the user.
