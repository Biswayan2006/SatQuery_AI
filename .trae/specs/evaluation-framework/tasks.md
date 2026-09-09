# SatQuery AI - Evaluation Framework Implementation Plan

## Task 1: Create package scaffolding (all 8 domain __init__.py files)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: None
- **Description**:
  - Create `evaluation/vqa/__init__.py`, `evaluation/captioning/__init__.py`, `evaluation/grounding/__init__.py`, `evaluation/retrieval/__init__.py`, `evaluation/change/__init__.py`, `evaluation/fusion/__init__.py`, `evaluation/routing/__init__.py`, `evaluation/confidence/__init__.py`, `evaluation/datasets/__init__.py`.
  - Each `__init__.py` is a minimal docstring-only file matching the style of existing `evaluation/common/__init__.py`.
- **Acceptance Criteria Addressed**: AC-1 (imports succeed)
- **Test Requirements**:
  - `rule` TR-1.1: `python -c "import evaluation.vqa.run"` — but because run.py does not exist yet, validate only the package inits exist; the actual import test occurs in Task 9.
- **Notes**: Run this first so Task 3–Task 8 files have parent packages.

## Task 2: Create routing_eval_set.json + CDVQA adapter
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - Create `evaluation/routing/routing_eval_set.json` with >= 15 gold entries covering all 6 `TaskType` values (SINGLE_VQA, CAPTIONING, GROUNDING, CHANGE_VQA, CHANGE_DESCRIPTION, SAR_OPTICAL_FUSION). Each entry: `{"query": str, "num_images": int, "modalities": list[str], "gold_task": str}`.
  - Create `evaluation/datasets/cdvqa.py` implementing `CDVQAAdapter(Dataset)`:
    - Constructor signature mirrors `VRSBenchAdapter`: `(data_dir, split="test", image_size=224, max_samples=None, ...)`.
    - Auto-detects a layout like `data_dir/{images_A, images_B, questions.json}` with `{image_id_A, image_id_B, question, answer}` tuples.
    - If `data_dir` is not configured or files are missing, returns an empty dataset (`len == 0`) silently.
  - Add `CDVQAAdapter` re-export in `evaluation/datasets/__init__.py`.
- **Acceptance Criteria Addressed**: AC-2 (routing eval set), AC-3 (CDVQA adapter empty-path)
- **Test Requirements**:
  - `rule` TR-2.1: Routing JSON schema validation script — >= 15 entries, all 6 TaskTypes covered, each entry has all 4 keys.
  - `rule` TR-2.2: `CDVQAAdapter(data_dir="nonexistent", split="test")` — `len(adapter) == 0` and `list(adapter) == []` without raising.
- **Notes**: Both outputs are needed before `routing/run.py` (Task 7) and `change/run.py` (Task 6) can be verified.

## Task 3: Create VQA runner (evaluation/vqa/run.py)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - Create `evaluation/vqa/run.py` with:
    - `argparse` accepting `--config`, `--max-samples`, `--output`, `--device`, `--batch-size` (default 8).
    - `load_config` + `resolve_device` via common modules.
    - Resolve variants via `resolve_variants(cfg, "vqa", baseline_label="BLIP pretrained", adapted_label="BLIP fine-tuned")`.
    - For each available variant:
      - Call `training.evaluate_vqa.load_eval_model(cfg, "pretrained"|"finetuned", variant.checkpoint, device)`.
      - If the variant is unavailable, collect an empty metrics dict with `available=False`.
      - Attempt `build_test_dataset(cfg)`. If dataset is empty/missing datasets, log "skipped: no data" and produce empty metrics rather than failing.
      - Reuse `VQACollator`, `run_inference`, `aggregate_metrics`, `collect_error_examples` from `training/evaluate_vqa.py`.
      - Also collect `(confidence, correct)` pairs for each sample: use `1.0 if pred_normalised == gold_normalised else 0.0` for correctness; derive confidence from `vqa_accuracy` per-answer, or use a uniform `0.8` placeholder flagged `confidence_is_calibrated=False` and saved into a shared `confidence_pairs.jsonl` sidecar in the output dir for the confidence domain to consume.
    - Build comparison rows via `comparison_row(...)` for exact_match / vqa_accuracy / token_f1 per dataset.
    - Write JSON via `write_json(output_dir, "vqa", report)` and Markdown via `append_markdown(output_dir, "vqa", "VQA", rows, notes)`.
- **Acceptance Criteria Addressed**: FR-1, FR-10, FR-11, FR-12, FR-13; AC-1, AC-10
- **Test Requirements**:
  - `rule` TR-3.1: Module imports cleanly without triggering model download (`python -c "import evaluation.vqa.run"`) — guard heavy imports inside functions.
  - `rule` TR-3.2: Smoke test AC-10 passes when datasets unconfigured.
- **Notes**: All model/training imports must be lazy (inside `run_variant(...)`) so the unit-test import check never pulls torch/HF.

## Task 4: Create Retrieval runner (evaluation/retrieval/run.py)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - Create `evaluation/retrieval/run.py` wrapping `training.evaluate_clip.extract_features` / `retrieval_recall` / `build_label_features`.
  - Variants: baseline = OpenCLIP base weights (`RSCLIPEncoder.from_pretrained(checkpoint_path=None)`); adapted = `RSCLIPEncoder.from_pretrained(checkpoint_path=rs_clip_ckpt)`.
  - Load dataset via `VRSBenchAdapter` (or enabled datasets in cfg) with a basic collate.
  - Extract features with each variant; compute `image_to_text` R@1/R@5/R@10 + `text_to_image` R@1/R@5/R@10 using `retrieval_recall(...)`.
  - Build comparison rows, write JSON + Markdown; adapted = n/a if ckpt missing.
- **Acceptance Criteria Addressed**: FR-2, FR-10, FR-11, FR-12, FR-13; AC-1
- **Test Requirements**:
  - `rule` TR-4.1: Module imports cleanly (`python -c "import evaluation.retrieval.run"` with lazy imports inside funcs).
- **Notes**: Reuse `RSCLIPEncoder` inference interface rather than touching training code.

## Task 5: Create Captioning + Grounding runners
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - **Captioning** (`evaluation/captioning/run.py`):
    - Load `RemoteSensingCaptioning` (once, reused for variant loop since no adapted ckpt).
    - Resolve variants with `unavailable_adapted("RS captioning adapted", "no remote-sensing captioning checkpoint exists in system")`.
    - Dataset via `VRSBenchAdapter`; for each sample produce a caption then score against `sample["captions"]` (all refs) using `aggregate_caption_metrics`.
    - Rows: BLEU-1..4, ROUGE-L, METEOR, CIDEr. SatQuery col = n/a.
  - **Grounding** (`evaluation/grounding/run.py`):
    - Load `RemoteSensingGrounding`.
    - Adapted variant unavailable via `unavailable_adapted(...)`.
    - Dataset: for each VRSBench image, construct a referring-expression query from its caption (or use a small fixed query set like `["building", "road", "water body", "vegetation area"]`) when no human grounding annotations exist; collect pred boxes + empty-box refs (score honestly: the comparison table says "n/a for refs unavailable" in notes, or if boxes can be derived, Acc@0.5 and mIoU vs a coarse proxy mask). To avoid fabrication, the honest path is to report metrics for a hand-curated small "grounding probe" set if VRSBench ships no boxes; otherwise mark the whole dataset row "skipped: no grounding annotations".
    - Rows: `acc@0.5`, `m_iou`, plus notes.
- **Acceptance Criteria Addressed**: FR-3, FR-4, FR-10, FR-11, FR-13; AC-1
- **Test Requirements**:
  - `rule` TR-5.1: Both modules import cleanly with lazy model loading.

## Task 6: Create Change + Fusion runners
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1, Task 2 (CDVQA adapter needed for change)
- **Description**:
  - **Change** (`evaluation/change/run.py`):
    - Two tracks: (a) change-mask dataset (LEVIR-CD) — load via a minimal `LevirCDAdapter` (on the fly inside the module, honest empty if not configured); run `ChangeDetectionModel.detect_changes`; compute pixel-level `change_mask_metrics`. (b) change-VQA via `CDVQAAdapter` + `ChangeDetectionModel.answer_change_question`; score with VQA `aggregate_metrics` (re-exported via `vqa_metrics_module()`).
    - Variants: baseline threshold from cfg; adapted = improved threshold if `variants.adapted_threshold` is set, otherwise single-variant report.
  - **Fusion** (`evaluation/fusion/run.py`):
    - Instantiate `SAROpticalFusionModel(fusion_checkpoint=None)` (baseline, fusion_trained=False) and then with `sar_fusion_checkpoint` (adapted).
    - Dataset: BigEarthNet paired split via `BigEarthNetAdapter(use_sar=True)`.
    - For each sample run `fuse_and_analyze`; compute land-cover multilabel metrics on the auxiliary classifier logits (via `training.evaluate.multilabel_metrics`) if the adapter exposes them; otherwise score deterministically and flag `requires_verification` in notes.
    - Always write `fusion_trained` status into JSON provenance.
- **Acceptance Criteria Addressed**: FR-5, FR-6, FR-10, FR-11, FR-12, FR-13; AC-1
- **Test Requirements**:
  - `rule` TR-6.1: Both modules import cleanly with lazy model loading.

## Task 7: Create Routing + Confidence runners
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Task 1, Task 2 (routing_eval_set.json needed)
- **Description**:
  - **Routing** (`evaluation/routing/run.py`):
    - Load `routing_eval_set.json` from cfg (`routing.eval_set` path); resolve to relative-to-backend.
    - Instantiate `TaskClassifier()` (keyword-only mode — no semantic router loaded by default unless `cfg.routing.use_semantic_router` is True and encoder is available).
    - Collect gold/pred lists.
    - Score with `aggregate_routing_metrics(...)`.
    - Build comparison rows: accuracy, macro_f1, weighted_f1 (single-variant table, "baseline" = keyword-only; "satquery" col = "n/a" unless a semantic-router variant is configured, which also resolves honestly via `unavailable_adapted` if no RS-CLIP ckpt).
  - **Confidence** (`evaluation/confidence/run.py`):
    - Consume `(confidence, correct)` pairs from a prior run's `confidence_pairs.jsonl` sidecar, or re-run a mini VQA/ synthetic collection.
    - Feed into `reliability_stats(confidences, correct, n_bins=cfg.confidence.n_bins)` → ECE, MCE, Brier, per-bin curve.
    - Build comparison rows: ECE, MCE, Brier (lower-is-better → flip sign for improvement column via `higher_is_better=False`). Variant: baseline = "before calibration" (same pairs, raw conf); SatQuery = "task-aware confidence" (same pairs if only one pass exists but honestly mark `n/a` when no post-calibrated variant is configured).
- **Acceptance Criteria Addressed**: FR-7, FR-8, FR-10, FR-11, FR-13; AC-1, AC-9
- **Test Requirements**:
  - `rule` TR-7.1: Both modules import cleanly.
  - `rule` TR-7.2: Smoke test AC-9 (routing.run without datasets) exits 0 and writes both files.

## Task 8: Write tests/test_evaluation_metrics.py
- **Status**: `pending`
- **Priority**: high
- **Depends On**: All run.py files exist (import check needed)
- **Description**:
  - Create a single file `tests/test_evaluation_metrics.py` covering:
    1. `TestCaptioningMetrics`: exact-match BLEU-1 = 1.0 case; hand-computed ROUGE-L overlap-of-3 case with 5-token ref; METEOR on a short exact match.
    2. `TestDetectionMetrics`: hand-built boxes with known IoU — two perfect boxes produce acc_at_iou(0.5)==1.0, mean_iou==1.0; disjoint boxes produce 0.0. PR@0.5 counts TP/FP/FN correctly.
    3. `TestChangeMaskMetrics`: hand-built 4x4 pred/gold arrays (8 TP, 2 FP, 4 FN, 2 TN) → verify P/R/F1/IoU/overall match formulas exactly.
    4. `TestRoutingMetrics`: gold=[A,A,B,B,C], pred=[A,B,A,B,C] → accuracy 3/5=0.6, per-task F1 for each, confusion matrix diagonal values; verify the sums.
    5. `TestReporting`: build a 2-row comparison table with one adapted=n/a and one numeric improvement; assert rendered string contains correct columns and correct n/a display; verify `higher_is_better=False` flips the improvement sign.
    6. `TestConfigEnv`: set+unset env vars against a tiny inline YAML string (or temp file); assert `${VAR}`, `${VAR:-default}` resolution is correct; assert `is_configured` correctly rejects unresolved placeholders.
    7. `TestModuleImports`: `importlib.import_module` each of the 8 run modules; assert no exception and no torch/HF import happens at top level (detect by checking if `sys.modules` contains `transformers` before and after — it should NOT be added until a fn is called; alternatively just verify import succeeds).
    8. `TestCDVQAAdapterEmpty`: instantiate with nonexistent dir; assert len 0.
    9. `TestRoutingEvalSetSchema`: load `evaluation/routing/routing_eval_set.json`; validate counts and TaskType coverage.
- **Acceptance Criteria Addressed**: AC-4, AC-5, AC-6, AC-7, AC-8; indirectly AC-1, AC-2, AC-3
- **Test Requirements**:
  - `rule` TR-8.1: `venv/Scripts/python.exe -m pytest tests/test_evaluation_metrics.py -v` passes all tests.

## Task 9: Write EVALUATION.md
- **Status**: `pending`
- **Priority**: medium
- **Depends On**: All run.py exist
- **Description**:
  - Create `backend/EVALUATION.md` with sections:
    1. **Overview** — purpose + 8 domains.
    2. **Datasets** — supported dataset names, expected layouts, which adapters serve them, split policies (MD5 deterministic for BEN, native for VRSBench/RSVQA).
    3. **Methodology** — baseline vs adapted resolution policy, honesty rules (no fabricated adapted numbers), provenance fields.
    4. **Metrics per domain** — tables listing each metric used with formulas / references.
    5. **Limitations** — METEOR lacks WordNet, CIDEr is pure-Python approximation, captioning/grounding/change have no adapted variant in this system version, routing is keyword-only by default.
    6. **Reproducibility commands** — copy-paste blocks for each domain (`python -m evaluation.<domain>.run --config ...` with env-var prefixes), plus the smoke tests and the full pytest line.
    7. **Checkpoints** — naming convention + which env vars supply them.
- **Acceptance Criteria Addressed**: AC-12
- **Test Requirements**:
  - `rubric` TR-9.1: Documentation completeness per AC-12; scale 1-5; threshold >= 4; evidence = file content review.

## Task 10: Verification (run tests + smoke + full suite)
- **Status**: `pending`
- **Priority**: high
- **Depends On**: Tasks 1–9 completed
- **Description**:
  - Run `venv/Scripts/python.exe -m pytest tests/test_evaluation_metrics.py -v` and record results.
  - Run the two reproducibility smoke commands (AC-9 + AC-10) from `backend/` with env vars unset so honest "no data" path is exercised.
  - Run `venv/Scripts/python.exe -m pytest -q` to confirm no Task-4 regressions.
- **Acceptance Criteria Addressed**: AC-9, AC-10, AC-11
- **Test Requirements**:
  - `rule` TR-10.1: New test file (AC-11 target subset) passes.
  - `rule` TR-10.2: Smoke commands exit 0 and produce both files.
  - `rule` TR-10.3: Full `pytest -q` passes (no regressions).
