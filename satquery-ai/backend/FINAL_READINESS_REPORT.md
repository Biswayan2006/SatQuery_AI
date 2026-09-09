# SatQuery AI — FINAL Production & SIH-Readiness Report

**Audience:** Internal team + ISRO/SAC (Smart India Hackathon) technical evaluation
**Scope:** Complete end-to-end verification of the SatQuery AI backend (`satquery-ai/backend/`)
**Method:** Read-only. Two independent evidence streams, cross-validated:
1. **Static audit** — 8 parallel read-only code reviewers (Read/Grep/Glob only), every verdict anchored to `file:line`.
2. **Live audit** — the running server exercised over HTTP end-to-end (upload → analyze → report), all responses, timings, failure behaviour and concurrency recorded to `_audit_results.json`.

**Constraint honoured:** No application code was modified during this audit. All numbers below are **actually measured or read from disk** — none are estimated or fabricated. Where a number does not exist (e.g. accuracy on an RS benchmark), the report says so explicitly.

**Git commit at audit time:** `68386f85b04eed842856474d936b926899d1cc54`
**Host:** Windows 11, CPU-only (no CUDA device present), Python 3.13 venv.

---

## 0. Bottom-Line Verdict

SatQuery AI is a **functionally complete, architecturally sound, and unusually honest agentic remote-sensing analysis pipeline.** It runs end-to-end on a CPU-only box, dynamically routes each query to task-specific models and deterministic geospatial tools, exposes an auditable execution trace **without** leaking chain-of-thought, degrades gracefully, and never silently fabricates a confident answer.

It is **NOT** yet a *validated, accuracy-proven, production-hardened* system. The three headline gaps a judge will find:

| # | Reality | Impact |
|---|---------|--------|
| **1** | **Every model runs BASE / pretrained / heuristic weights.** No fine-tuned checkpoint of any kind ships or exists on disk. The RS-adaptation, LoRA, and fusion-training code is real scaffolding that has **never been run**. | Any "remote-sensing fine-tuned" claim is *documented-only*. |
| **2** | **No measured accuracy on any RS benchmark.** 7 of 8 evaluation domains report `no_data` (datasets not configured). The only real number is task-routing on **4 single-class samples**, which trivially ties a majority-class baseline. | There is no quantitative evidence that any answer is *correct*. |
| **3** | **Several correctness/robustness defects** — the geospatial co-registration path is dead in the live API, change-region geo-coordinates are mis-scaled off the 256-px model grid, the concurrency limiter did not serialize concurrent inference at runtime, and timeouts are unenforced. | Geographic outputs are unreliable for non-256-px imagery; the box can be overloaded. |

**Readiness call:** ✅ **Demo-ready and defensible** (the honesty is a genuine strength for a technical panel). ⚠️ **Not production-accurate** — treat every model answer as an *unverified* draft, exactly as the system itself already flags.

**Status legend used throughout:**
`Implemented` = real, wired, reachable · `Partial` = works but with a material gap/caveat · `Documented-only` = code/plan exists but inactive in the shipped build · `Not-implemented` = absent · `Mock-only` = only the degraded-mode stub runs.

---

## 1. Requirement Matrix

Eight dimensions. Every `Evidence` cell is a `file:line` relative to `satquery-ai/backend/`, verified by static reading; runtime confirmations are tagged **[live]**.

### 1.1 Single-image capabilities

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Optical VQA (image + question → answer) | `/analyze` → controller classifies `SINGLE_VQA` → `RemoteSensingVQA.answer()` runs BLIP-2 `generate()`, confidence from mean token log-prob | `api/routes.py:239-259`; `agent/controller.py:594-614`; `models/vqa_model.py:208-231,354-387` · **[live]** test 1: 200, 12.7 s, real answer | **Implemented** |
| VQA uses an RS-*specific* model | Loads **base** `Salesforce/blip2-opt-2.7b`; `vqa_model_source="pretrained"`, checkpoint path empty; finetune loader never fed an artifact | `config.py:46,64,68`; `models/vqa_model.py:83-95`; no VQA `.pt` on disk | **Documented-only** |
| Captioning (image → caption) | `RemoteSensingCaptioning.generate_caption()` runs BLIP-2 (beams=5) w/ "A satellite image showing" prefix; GIT fallback | `agent/controller.py:616-635`; `models/captioning_model.py:166-246` · **[live]** test 2: 200, 23.2 s | **Implemented** |
| Grounding (text-referring → boxes + overlay) | `RemoteSensingGrounding.ground()` runs OWL-ViT @ score 0.2, draws boxes to base64 | `agent/controller.py:637-672`; `models/grounding_model.py:66-130,145-180` · **[live]** test 3: 200, overlay returned | **Implemented** (⚠ uses full query string as prompt; declared NMS never applied) |
| Multispectral (>3 bands preserved/used, e.g. NDVI) | All bands kept in `numpy_array`; NDVI/NDWI/NDBI tools compute on full array — **but** models only ever see a 3-band composite and API never carries `sensor`/`band_map`, so only 3-band(RGB)/4-band(RGBN) resolve | `utils/image_utils.py:46-58,96-136`; `tools/spectral.py:57-98`; `tools/raster.py:130-152`; `api/schemas.py:34-37` · **[live]** NDVI on 4-band `ms.tif`: mean 0.1725 | **Partial** |
| Multispectral evidence conditions the answer | Tool evidence is computed **after** inference and returned as a separate field — never injected into the model prompt | `agent/controller.py:396,408,1020` | **Partial** |
| Lone-SAR single image handled distinctly | Detected as `modality=sar` → real backscatter stats; but no SAR-specific neural path (routes to captioning or errors demanding a 2nd image) | `agent/input_validator.py:190-213`; `agent/task_classifier.py:199-207`; `tools/sar.py:60-119` | **Partial** |

### 1.2 Multi-temporal (bi-temporal change)

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Image-pair validation | API caps 1–2 images; handlers guard `len<2`; a real `check_compatibility()` exists but is **dead code** ("validate_pair" is only a plan label) | `api/schemas.py:35`; `agent/controller.py:675-679,726-730`; `agent/input_validator.py:84-119` (no callers) | **Partial** |
| Geospatial alignment (reproject/resample to common grid) | Full rasterio reproject+resample pipeline exists — **but is unreachable via the API** (opens `image_data['_path']`, which routes never set → `rasterio.open('')` raises → pixel-resize fallback) | `utils/geospatial_aligner.py:306-562`; `api/routes.py:240-244` (no `_path`) · **[live]** test 4/5 warning: *"Rasterio resampling failed (: No such file or directory); fell back to pixel alignment"* — **confirms dead path** | **Partial** |
| Change detection algorithm | **HEURISTIC**, not trained: Siamese ImageNet ResNet-50 (layer3) → per-pixel cosine distance → fixed threshold 0.35 → connected components. No checkpoint, no training, config backbone id ignored | `models/change_model.py:63-88,122-151,215-255`; `evaluation/change/run.py:239` (project admits it) | **Partial** |
| Change VQA / description | No trained change-language model. CHANGE_VQA = rule-based templating over %/regions; CHANGE_DESCRIPTION = two **independent** per-image BLIP-2 captions + templated % sentence | `models/change_model.py:299-334`; `agent/controller.py:725-805` · **[live]** test 5 answer stitches 2 captions + "0.6% changed" | **Partial** |
| Change visualization (map PNG) | `create_change_map` renders distance field as color-coded PNG + colorbar; returned as base64 `change_map` | `utils/visualization.py:19-89`; `agent/controller.py:789-794` · **[live]** test 4/5: 15,144-byte PNG | **Implemented** |
| Change statistics (px, m²/km², %) | `ChangeAreaTool` real + honest when resolution missing — **but runs only for CHANGE_DESCRIPTION, not CHANGE_VQA** (VQA path omits `binary_mask`) | `models/change_model.py:194-200`; `agent/controller.py:451-455`; `tools/change.py:44-98` · **[live]** test 4 `tool_evidence:null`; test 5 area=37,700 m² | **Partial** |
| Geographic localization of change | Real affine+pyproj → WGS84 for GeoTIFF; PNG honestly returns `None`. **Scale bug:** regions normalised on fixed 256×256 model grid but converted with the original/aligned-grid transform → lat/lon & area off by 256/width unless source is 256 px | `utils/geo_utils.py:194-299`; `models/change_model.py:129,144,257-297` · **[live]** test 5 geo_bbox looked correct **only because fixtures were 256×256** | **Partial** |

### 1.3 Optical + SAR fusion

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| SAR+optical compatibility validation | `check_compatibility` whitelists {sar,optical}/{sar,multispectral}; `_exec_sar_fusion` requires exactly 1 SAR + 1 optical, else honest degraded result; declared `align_images` step is **not** executed for fusion | `agent/input_validator.py:84-119`; `agent/controller.py:807-844` · **[live]** test 6: refused ("needs one SAR image… none provided"), degraded, conf 0.0 | **Partial** |
| SAR preprocessing | Real dB (10·log10), VV/VH cross-pol ratio, per-band standardization, BigEarthNet-S1 norm, calibration honesty — **but NO speckle filtering** (no Lee/Frost/median/multilook) | `models/sar_fusion/sar_preprocess.py:126-246`; `tools/sar.py:41-119` | **Partial** |
| Optical preprocessing (fusion path) | `encode_optical` via BLIP processor + `normalize_to_rgb` (real) — dormant under default (never called under BLIP-2) | `models/sar_fusion_model.py:320-327,498-529` | **Implemented** (dormant) |
| **Actual trained feature fusion** | Genuine cross-modal net exists (SAR encoder + multi-head cross-attention adapter) but **weights are random** (`fusion_trained=False`, checkpoint empty, none on disk); and under shipped BLIP-2 config injection is disabled so the adapter **never runs** — a deterministic stats template answers | `models/sar_fusion_model.py:79-201,239,278-316,381-409`; `config.py:78`; `.env` VQA=blip2-opt-2.7b; `vqa_model.py:258-261` | **Documented-only** |
| Fusion-conditioned reasoning | On the (non-default) BLIP path, fused tokens are truly injected; under default BLIP-2 the answer is a hand-written deterministic template from SAR/optical stats. Does **not** silently fall back to single-image VQA | `vqa_model.py:233-314`; `models/sar_fusion_model.py:341-399,573-584` | **Partial** |
| Fusion visualization | `create_fusion_visualization` 3-panel [Optical\|SAR\|Fusion-heat]; heat panel needs the adapter's gate_map, so degrades to 2 panels under default | `utils/visualization.py:143-196`; `models/sar_fusion_model.py:586-600` | **Implemented** (heat inactive by default) |

### 1.4 Agentic orchestration

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Query interpretation (intent) | `IntentExtractor.extract` → concept/operation/flags; optional CLIP blend (0.6 kw / 0.4 semantic) | `agent/query_intent.py:165-249`; `agent/controller.py:134,214` · **[live]** intent surfaced per test | **Implemented** |
| 6-task classification | `TaskClassifier` → one of 6 `TaskType` via keyword+structural scoring + priority tie-break | `agent/task_classifier.py:31-37,117-239`; `agent/controller.py:207` | **Implemented** |
| Semantic routing (CLIP) enabled by default | Real router (task-desc text embeddings, cosine, 0.55/0.25/0.20 blend) — **but runs BASE OpenAI ViT-B-32, not RS-CLIP**; enablement conditional; silently degrades to keyword on any failure; whether it ran is **not observable** in the response | `models/rs_clip/semantic_router.py:112-198`; `agent/controller.py:138,509-524`; `config.py:54` | **Partial** |
| Rule-based fallback | `_keyword_classify` additive scoring + deterministic tie-break; reached when router unavailable/raises | `agent/task_classifier.py:211-251` | **Implemented** |
| Execution planning | Per-task `ExecutionPlan` (models+steps+params) + intent/band-gated tool-step injection; `to_dict` exposes plan pre-execution | `agent/controller.py:231-365` · **[live]** `exec.plan` present every test | **Implemented** |
| Deterministic tool dispatch | `ToolPlanner.run_for_task` data-aware (NDVI iff nir+red; SAR stats iff modality=sar; geo iff has_geo); 10 real numpy tools | `agent/tool_planner.py:64-200`; `tools/registry.py:31-44` | **Implemented** |
| Model selection per task | Base dict assigns specialists per task; lazy-loaded w/ mock fallback. Caveat: `models_used` is partly declarative (CHANGE_VQA lists VQA but only Change model runs) | `agent/controller.py:246-287,674-704` · **[live]** test 4 `models_used` lists RemoteSensingVQA though it did not run | **Implemented** (audit trail overstates) |
| Evidence integration | Model output + tool evidence both in response; **but model runs before tools**, so tool numbers never condition the answer (only change-mask flows model→tools) | `agent/controller.py:396,408,453-456,1020` | **Partial** |
| Execution summary WITHOUT chain-of-thought | Step {name,status,kind,duration,scalar-summary}; classifier free-text `reason` is computed then **discarded**; no prompts/logits/reasoning text | `agent/execution_trace.py:36-118`; `api/schemas.py:197-204`; `agent/controller.py:207,981-993` | **Implemented** ✅ |

### 1.5 Remote-sensing adaptation & training

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Fine-tuned model exists on disk | **None.** All checkpoint paths empty; exhaustive glob finds only base HF caches (blip2-opt-2.7b, blip-vqa-base, blip-image-captioning-base, owlvit-base); no `checkpoints/`, no `model_version.json` | `config.py:54,64,68,78`; `models/{rs_clip/encoder.py:138,vqa_model.py:76,sar_fusion_model.py:239}` | **Not-implemented** |
| Training data documented | Datasets/splits/licenses documented in model cards + EVALUATION.md + YAMLs; real torch adapters — but no data bundled | `models/*/MODEL_CARD.md`; `EVALUATION.md:122-133`; `training/datasets/*` | **Implemented** |
| Checkpoint process documented & reproducible | Full non-stub training scripts (real `loss.backward()`/`optimizer.step()`/save) + YAMLs + reproduce commands — but **never executed** | `training/{train_clip,train_vqa,train_fusion}.py`; `EVALUATION.md:643-703` | **Implemented** (unexecuted) |
| Evaluation performed with real numbers | Harness runs & emits honest JSON; only `routing` = `ok` (4 single-class samples); 7 domains `no_data` | `evaluation/results/*.json` | **Partial** |
| Baseline-vs-adapted comparison | Only routing populated: +0.0000 (ties majority class); all model-quality comparisons empty | `evaluation/results/benchmark_report.md:71-78` | **Partial** |

### 1.6 Confidence framework

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| NOT answer-length based | Confidence from real signals: `exp(mean token log-prob)` (VQA/caption), detection score+IoU (grounding), threshold-margin (change); `answer_length` stored but never read | `confidence/confidence_service.py:436-452`; `models/vqa_model.py:363-414` | **Implemented** ✅ |
| Task-specific logic | `assess()` dispatches to 5 distinct extractors (vqa/captioning/grounding/change/fusion) | `confidence/confidence_service.py:174-341` | **Implemented** |
| Calibration where available, else "uncalibrated" | Full temperature-scaling machinery exists (LBFGS + grid-search fallback) — **but no `calibration.json` exists**; fit path called only by unit tests; `./calibration` created empty → **every task is `uncalibrated` at runtime** | `confidence/calibration.py:110-284`; `confidence/confidence_service.py:369-377`; `config.py:120,208-217` · **[live]** every test `uncalibrated`/`unavailable` | **Partial** |
| Uncertainty bands + abstention | `final≥0.70`→low, `≥0.45`→medium, else high; `requires_verification = final<0.45`; visible "⚠ Low confidence" caveat prepended | `confidence/confidence_service.py:390-433`; `config.py:112-117`; `agent/controller.py:418-426` · **[live]** low-conf tests carried the caveat | **Implemented** |
| Provenance (`confidence_type` + components) | `{calibrated,uncalibrated,unavailable}` + `{model,evidence,consistency}` surfaced to client | `confidence/confidence_service.py:57-118`; `api/schemas.py:214-241` · **[live]** all present | **Implemented** |

### 1.7 Geospatial correctness

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| CRS read & used | rasterio `str(src.crs)` at load+upload; geolocation tools reproject to WGS84 via pyproj; branch on projected vs geographic | `utils/image_utils.py:52`; `tools/geolocation.py:48-59,173-193` · **[live]** GeoTIFF uploads reported EPSG:32643 | **Implemented** |
| Pair alignment/reprojection | Real pipeline exists but **unreachable** (missing `_path`) → pixel-resize fallback | `utils/geospatial_aligner.py:306-562`; `api/routes.py:240-244` · **[live]** confirmed fallback | **Partial** |
| Pixel → lat/lon | Correct affine+pyproj, reachable via tools; **but change-region enrichment mis-scaled** on 256-grid | `tools/geolocation.py:40-119`; `models/change_model.py:43,129-144` | **Partial** |
| Scene/region bounds (W/S/E/N) | `ImageBoundsTool` corners→WGS84; `GeoBBox` schema carries per-region bounds; `geo_utils.get_image_bounds` is dead alt | `tools/geolocation.py:122-156`; `api/schemas.py:128-137` | **Implemented** |
| Ground resolution (m) | `GroundResolutionTool` from affine (projected=linear, geographic=haversine) | `tools/geolocation.py:159-216` | **Implemented** |
| Real NDVI/NDWI/NDBI w/ band-gating | Genuine normalized-difference math, NaN-guarded, honest `unsupported` when band absent; rarely satisfiable via API (no band_map; NDBI/SWIR effectively never) | `tools/spectral.py:37-122`; `tools/raster.py:130-152` · **[live]** NDVI real on 4-band | **Partial** |

### 1.8 Deployment & production-readiness

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| Docker (torch pin matches base image, no conflicting reinstall) | Base `pytorch/pytorch:2.6.0-cuda12.4`; pins `torch==2.6.0`+`torchvision==0.21.0` (exact pair) → no reinstall; healthcheck+GDAL/opencv present; **compose declares no GPU; image runs as root** | `Dockerfile:1,23-38`; `requirements.txt:9-10`; `docker-compose.yml:4-30` | **Partial** |
| Env config, no hardcoded paths | pydantic-settings; all paths relative env-overridable; **but default `secret_key='dev-secret-change-me'`; docs enabled by default** | `config.py:15-40,25,208-225` | **Implemented** |
| Health endpoint | Real models/GPU/concurrency stats — **but `degradation_status` hard-coded `"unknown"`; `timed_out_requests` always empty** | `api/routes.py:331-359` · **[live]** health `"degraded"` (partial-load), degradation `"unknown"` | **Partial** |
| Model loading (lazy singleton, warm, timeouts) | Thread-safe singleton + lazy loaders + GPU precheck + mock fallback; **background thread only registers (no pre-warm); `model_load_timeout_seconds` unused** | `models/registry.py:37-207,507`; `main.py:47-56` · **[live]** 2 loaded at start → 5 after | **Partial** |
| Structured logging + request IDs, no sensitive data | JSON logs, `X-Request-ID` middleware; query truncated to 50 chars; keys never logged | `utils/logging.py`; `middleware/security.py:115-137`; `api/routes.py:176` | **Implemented** ✅ |
| Security (Pillow validation, traversal, size, API-key, CORS, rate-limit) | All real middleware; Pillow (not libmagic); traversal blocked; streaming size cap — **but docs bypass key; rate-limit keys on client.host in-memory; TrustedHost `*`** | `utils/file_validation.py:55-224`; `middleware/security.py:22-208` · **[live]** 5/5 failure scenarios clean, **no stack-trace leak** | **Implemented** |
| TTL cleanup | `CleanupService` mtime sweeps + admin `/cleanup`; auto only in prod/opt-in | `services/cleanup.py:26-291` | **Implemented** |
| Concurrency (semaphore/queue) | Real async per-model+global queue, seeded from `max_concurrent_inference` — **but companion timeout never enforced; and see §4: runtime did NOT serialize** | `utils/concurrency.py:18-233`; `models/registry.py:285-371` · **[live]** 2 concurrent requests fully overlapped | **Implemented** (not enforced at runtime) |
| Graceful degradation (mock w/o faking) | `MockModel` → explicit "[Degraded Mode]", conf 0.01, `is_degraded`+`requires_verification`; event-driven `DegradationMonitor` — **but /health ignores it; 2 recorders dead** | `models/_mock.py:30-127`; `agent/controller.py:412-426`; `services/degradation_monitor.py` | **Implemented** |

---

## 2. Final Test — Actual Results

10 mandated tests + 2 auxiliary, run live against the server. Fixtures were **synthetic 256×256 images** (color-block optical, grayscale speckle "SAR", and georeferenced EPSG:32643 GeoTIFFs). **Because inputs are synthetic, answer *quality* is not an accuracy measurement** — these tests validate pipeline function, the execution trace, confidence behaviour, tool execution, geospatial math, error handling, and honest degradation.

| # | Test | Result | Wall | Task | Confidence | Key observation |
|---|------|--------|------|------|-----------|-----------------|
| 1 | Single-image VQA | ✅ 200 | 12.7 s | SINGLE_VQA | 0.208 uncalibrated, high, ⚠verify | Real BLIP-2 answer; correctly low-confidence + verification flag |
| 1b | VQA (repeat/warm) | ✅ 200 | 12.6 s | SINGLE_VQA | 0.208 | Warm inference ≈ cold here (VQA pre-loaded) |
| 2 | Captioning | ✅ 200 | 23.2 s | CAPTIONING | 0.225 uncalibrated, high | Includes BLIP-2 cold-load into caption slot; `spectral_statistics` tool ran |
| 3 | Grounding | ✅ 200 | 4.2 s | GROUNDING | 0.0 **unavailable** | OWL-ViT found nothing (searched the *whole query sentence*); returned overlay, honestly flagged |
| 4 | Change VQA | ✅ 200 | 1.3 s | CHANGE_VQA | 0.896 uncalibrated, low | "0.6% changed"; change_map returned; geo_bbox lat/lon from GeoTIFF transform; **no tool_evidence** (VQA path skips tools) |
| 5 | Change description | ✅ 200 | 37.6 s | CHANGE_DESCRIPTION | 0.896 | 2 independent captions + "0.6% changed"; change_area=**37,700 m²** (flagged approximate), 1 connected region w/ geo-coords |
| 6 | SAR-optical fusion | ✅ 200 | 0.2 s | SAR_OPTICAL_FUSION | 0.0 **unavailable**, degraded | **Refused** — synthetic 8-bit SAR PNG classified `optical`; honest message, no fabrication (adapter is untrained anyway) |
| — | NDVI (4-band aux) | ✅ 200 | 27.7 s | SINGLE_VQA | 0.204 | **NDVI tool real**: mean 0.1725, range −0.579…0.68, formula `(NIR−RED)/(NIR+RED)` |
| 7 | PDF report | ✅ 200 | — | — | — | **2,375 bytes, valid `%PDF-`** |
| 8 | Health | ✅ 200 | — | — | — | `"degraded"` (lazy partial-load), device cpu, GPU absent |
| 9 | Model registry | ✅ 200 | — | — | — | 6 models listed, all `is_mock=false` (no mocks this run) |
| 10 | Failure scenarios | ✅ 5/5 | — | — | — | See below — **zero stack-trace leaks** |

**Test 10 — failure handling (all return a clean structured error envelope, no traceback):**

| Scenario | HTTP | Leak? |
|---|---|---|
| Non-image bytes with `.png` name | 400 `ERR_400` "File content is not a valid image" | **No** |
| Unknown image_id | 404 `ERR_404` "…not found. Upload it first" | **No** |
| 3 images (>max 2) | 422 validation "at most 2 items" | **No** |
| Empty image_ids | 422 validation "at least 1 item" | **No** |
| Empty query | 422 validation "at least 1 character" | **No** |

**Cross-validation wins** (static prediction ⇒ confirmed live):
- Change VQA skips the area/region tools (static: `answer_change_question` omits `binary_mask`) → **test 4 `tool_evidence:null`, test 5 populated.** ✔
- Co-registration is dead via API (static: `_path` never set) → **test 4/5 warning "Rasterio resampling failed (: No such file or directory)".** ✔
- SAR modality detection is fragile for 8-bit PNG (static) → **upload classified `sar_scene.png` as `optical`, fusion refused.** ✔

---

## 3. Performance (measured, CPU-only)

| Metric | Measured value | Notes |
|---|---|---|
| **Device** | CPU only | `gpu_status.available=false`, 0 CUDA devices |
| **Warm VQA inference** | **12.6–12.7 s** | opt 256×256, 150 tokens, 4 beams (tests 1/1b) |
| **Captioning (first call)** | **23.2 s** | incl. BLIP-2 cold-load into caption slot; 200 tokens, 5 beams |
| **Change VQA** | **1.3 s** | ResNet-50 heuristic + rule template (no BLIP invoked) |
| **Change description** | **37.6 s** | 2× BLIP-2 caption generations + change detection — heaviest path |
| **Grounding** | **4.2 s** | OWL-ViT (cold-loaded here) |
| **NDVI VQA (4-band)** | **27.7 s** | VQA on false-color composite (variance vs test 1 attributable to memory pressure after 5 models loaded) |
| **Model load time** | **Not reliably reportable from API** | `/health` `load_duration` is a computation bug (≈0/negative). Derived from wall-clock: BLIP-2 cold-load contributes ~10 s to first caption call |
| **GPU memory** | **N/A** | CPU-only host |
| **CPU memory (RSS)** | **≈ 31,259,096 KB ≈ 29.8 GiB (~30 GB)** working set | Measured via `tasklist` on the uvicorn PID with **5/6 models resident**. Registry `estimated_memory_mb` sums to ~5 GB — **6× too low**; two independent BLIP-2 instances (VQA + captioning) dominate |
| **Upload limit** | **50 MB**/file; ext ∈ {tif,tiff,png,jpg,jpeg}; ≤2 images/analyze | `config.py` `max_image_size_mb=50`; `api/schemas.py:35` |
| **Concurrency (configured)** | `max_concurrent_inference=1`, timeout 300 s | `config.py` |
| **Concurrency (observed)** | **NOT serialized** — 2 simultaneous VQA requests fully overlapped (each 17.5 s vs 12.6 s solo) | `_audit_results.json` performance block — see §4 CRITICAL/HIGH finding |

---

## 4. Accuracy — Actual Benchmark Results

> **Per instruction, only actually-measured numbers appear here. Nothing is estimated or fabricated.**

| Domain | Baseline | SatQuery (adapted) | Improvement | Status |
|---|---|---|---|---|
| **Task routing** — accuracy | 1.0000 (majority class) | 1.0000 | **+0.0000** | Measured, **degenerate** |
| **Task routing** — macro-F1 | 0.1667 | 0.1667 | **+0.0000** | Measured, **degenerate** |
| VQA | — | — | — | **no_data** (VRSBench/RSVQA not configured) |
| Retrieval (CLIP) | — | — | — | **no_data** |
| Captioning | — | — | — | **no_data** |
| Grounding | — | — | — | **no_data** |
| Change (mask + VQA) | — | — | — | **no_data** (LEVIR-CD/CDVQA not configured) |
| SAR-optical fusion | — | — | — | **no_data** (BigEarthNet not configured) |
| Confidence calibration | — | — | — | **no_data** (no (conf,correct) pairs) |

**Critical honesty note on the one real number:** routing was scored on **4 in-house gold queries that are ALL class `CAPTIONING`**. A "always predict CAPTIONING" baseline therefore also scores 1.0000, and macro-F1 = 0.1667 = 1/6 (five of six task classes have zero support). This demonstrates only that the classifier does not misfire on four captioning prompts — **it is not evidence of routing quality across the six tasks.** (The shipped `routing_eval_set.json` actually holds 33 entries across all 6 tasks; the committed run exercised only the first 4. `EVALUATION.md` also states 24 entries — a doc inaccuracy.)

**There is currently no measured accuracy for any model answer (VQA, captioning, grounding, change, fusion) on any remote-sensing benchmark.** Producing one requires configuring the documented datasets and running the (real, but never-run) `evaluation/*` harness.

---

## 5. Final Risk Register

### 🔴 CRITICAL

| Risk | Consequence | Recommendation |
|---|---|---|
| **No accuracy evidence for any model output** (7/8 eval domains `no_data`; only degenerate 4-sample routing) | Cannot substantiate that *any* answer is correct; a judge asking "how accurate is your VQA?" gets no number | Configure ≥1 dataset per domain (RSVQA, VRSBench, LEVIR-CD, BigEarthNet) and run `evaluation/*` to produce **real baseline numbers** before finals. If not feasible, present the system explicitly as an *unvalidated pipeline* |
| **SAR-optical fusion is untrained AND disabled by default** | The optical+SAR mandate is met only by a deterministic stats template + honest placeholder; the learned adapter never runs under the shipped config and has random weights regardless | Either (a) train & ship a fusion checkpoint and switch `VQA_MODEL_NAME` to the injection-supported model, or (b) present fusion as "deterministic multi-modal evidence" and stop implying learned fusion |

### 🟠 HIGH

| Risk | Consequence | Recommendation |
|---|---|---|
| **Geospatial co-registration path is dead via the API** (`_path` never set → `rasterio.open('')` raises → pixel-resize fallback) — **confirmed live** | Two real GeoTIFFs are *not* reprojected/resampled; alignment is a naive resize; `geographic_coordinates_available=false` | Set `image_data['_path']` in `api/routes.py` when loading GeoTIFFs so the real aligner runs |
| **Change-region geo-coordinates & area are mis-scaled** (regions normalised on 256×256 grid, converted with original/aligned transform) | Change lat/lon and m²/km² are wrong by 256/width for any non-256-px image — i.e. essentially all real imagery | Convert region coords using the actual aligned-grid dimensions, not the fixed 256 model grid |
| **Concurrency limiter did not serialize inference at runtime** (2 requests fully overlapped, each slowed 12.6→17.5 s) | On CPU, N concurrent requests thrash cores and multiply peak RAM (each BLIP-2 activation set) → latency spikes / OOM risk on a ~30 GB-resident server | Verify the async semaphore actually gates the `generate()` call; consider a hard global inference lock for the CPU demo box |
| **~30 GB RAM resident with 5/6 models**; registry estimates 6× too low | A machine with <32 GB will OOM once VQA+captioning+others are all loaded; concurrent load makes it worse | Size the host at ≥32–48 GB; unload idle models; document real footprint (not the 5 GB estimate) |
| **All models are base/pretrained/heuristic — no fine-tuning active** | "Remote-sensing fine-tuned" is not true of the shipped build | Either train & wire a checkpoint, or describe models precisely as base BLIP-2 / OWL-ViT / ImageNet-ResNet + deterministic RS tooling |

### 🟡 MEDIUM

| Risk | Consequence | Recommendation |
|---|---|---|
| **Confidence is never calibrated at runtime** (no `calibration.json`) | Every `confidence_type` is `uncalibrated`/`unavailable`; numbers must not be read as probabilities | Fit temperature scaling from a validation run and ship `calibration.json`, or keep labeling clearly (already done) |
| **`/health` degradation is a stub** (`"unknown"`, `timed_out_requests` always empty) | Ops dashboards can't see real degradation; "degraded" shows even for normal lazy-load | Wire `DegradationMonitor` into `/health`; compute status from actual model-load ratio |
| **`models_used` overstates invocations** (CHANGE_VQA lists VQA though only Change runs) | Audit trail claims a model that never executed — an evaluator may call this out | Populate `models_used` from actual invocations, not the declarative plan |
| **Grounding uses the entire query sentence as the detection prompt; no NMS** | Referring expressions ("the water body") search for the literal sentence → misses / duplicate boxes | Feed the extracted concept (`intent.concept`) to OWL-ViT; apply the declared NMS threshold |
| **NDVI/NDWI/NDBI unsatisfiable for real >4-band imagery** (no `sensor`/`band_map` plumbing) | The exact Sentinel-2/Landsat stacks the docs cite return `unsupported` | Add `sensor`/`band_map` to `AnalysisRequest` + presets so multispectral bands resolve |
| **Timeouts unenforced** (`inference_timeout_seconds`, `model_load_timeout_seconds`) | A stuck inference/load hangs the worker with no cancellation | Wrap inference in `asyncio.wait_for`; enforce load timeout |
| **`load_duration` in `/health` is a bug** (≈0/negative) | Load-time telemetry is meaningless | Compute duration as `end−start`, not against a stored timestamp |

### 🟢 LOW

| Risk | Consequence | Recommendation |
|---|---|---|
| Docs (`/docs`,`/redoc`,`/openapi.json`) exposed & bypass API key by default | Full schema public in prod | Disable docs when `is_production` |
| `secret_key='dev-secret-change-me'` default, no startup guard | Latent risk (currently only derives a dev key; not used for crypto) | Reject the default in production startup |
| Prod `docker-compose.yml` declares no GPU; image runs as root | `compose up` runs the CUDA image CPU-only; container is root | Add `deploy.resources…devices` / `gpus`; add non-root `USER` |
| Vestigial `python-magic==0.4.27` in requirements + `test_docker.py` import | `import magic` would fail in-container (libmagic not installed) — affects only the test script | Remove `python-magic` from requirements & `test_docker.py` (validator is Pillow) |
| Rate-limiter keys on `client.host`, in-memory | Behind a proxy all clients collapse to one IP | Honour `X-Forwarded-For`; move to shared store if multi-worker |
| Static `/uploads`,`/reports` unauthenticated when API key disabled (default) | UUID-named files publicly served | Gate static mounts, or accept UUID-obscurity for the demo |
| Dead code (`check_compatibility`, `geo_utils.get_image_bounds`, `estimate_ground_resolution`, unused degradation recorders) | Reader confusion | Remove or wire up |

---

## 6. Judge Questions — 30 Likely ISRO/SAC Questions (answered ONLY from what is implemented)

**Architecture & orchestration**

1. **Is this one general VLM or specialist models?** — Specialist. A deterministic agentic controller classifies each query into one of 6 tasks and dispatches to task-specific models + geospatial tools. It does **not** send everything to one generic VLM. (`agent/controller.py`)
2. **Which exact models run?** — VQA & captioning: `Salesforce/blip2-opt-2.7b` (base); grounding: `google/owlvit-base-patch32` (base); change: torchvision ResNet-50 (ImageNet, truncated) as a heuristic feature differ; routing/intent: OpenCLIP ViT-B-32 `openai`; fusion: an untrained custom adapter.
3. **Are these fine-tuned on remote-sensing data?** — **No.** All run base/pretrained/heuristic weights. Fine-tuning code, LoRA, dataset adapters and training scripts exist but have never been run; no checkpoint exists on disk.
4. **How does the agent decide the task?** — Keyword+structural scoring with deterministic priority tie-break, optionally blended with a CLIP semantic router (base CLIP). Falls back to keyword-only if the router is unavailable. (`agent/task_classifier.py`)
5. **Do you expose the model's chain-of-thought?** — **No.** The execution summary contains only step name/status/kind/duration/scalar-stats/plan/intent. The classifier's free-text rationale is computed then discarded. Verified. (`agent/execution_trace.py`, `api/schemas.py:197-204`)

**Single-image**

6. **How is VQA answered?** — BLIP-2 `generate()` on the RGB render; confidence = `exp(mean per-token log-prob)`. (`models/vqa_model.py`)
7. **How is grounding done?** — OWL-ViT open-vocabulary detection at score threshold 0.2, boxes drawn on the image. Honest limitation: it currently uses the full query string as the prompt and does not apply NMS.
8. **How do you handle multispectral (>3 bands)?** — All bands are preserved and fed to deterministic index tools (NDVI/NDWI/NDBI); the neural model still receives a 3-band composite. Indices resolve for 3-band (RGB) and 4-band (RGBN) today; >4-band stacks need `band_map` plumbing that isn't wired yet.
9. **Can you compute NDVI?** — Yes, real `(NIR−RED)/(NIR+RED)` with divide-by-zero guarding; **measured live** on a 4-band image (mean 0.1725). Returns an honest `unsupported` when NIR is absent rather than faking it.
10. **Does the spectral evidence change the model's answer?** — No; tool numbers are returned alongside the answer but are not injected into the model prompt (model runs before tools). This is an integration gap we can close.

**Multi-temporal**

11. **Is change detection a trained model?** — **No, it's a heuristic:** Siamese ImageNet ResNet-50 features → per-pixel cosine distance → fixed 0.35 threshold → connected components. There is no change checkpoint or change training.
12. **How is the change *answer* produced?** — CHANGE_VQA is rule-based templating over change %/regions; CHANGE_DESCRIPTION concatenates two independent per-image BLIP-2 captions plus a templated percentage. Captions are not change-aware.
13. **Do you quantify change area in m²/km²?** — Yes (`ChangeAreaTool`), and it honestly flags the figure approximate when resolution is uncertain (**live: 37,700 m²**). Caveat: it runs for CHANGE_DESCRIPTION but not CHANGE_VQA, and area is mis-scaled off the 256-px grid for non-256 imagery.
14. **Are the two images co-registered?** — A full rasterio reproject+resample aligner exists, **but it is not reachable through the API** (a path field is never populated), so pair alignment currently degrades to a pixel resize. This is a known fix.
15. **Do change regions get real coordinates?** — For GeoTIFFs, yes via affine+pyproj — but there is a **scale bug**: coordinates are computed on the fixed 256-px model grid, so they are only correct for 256-px sources. Fixing the grid dimension resolves it.

**Optical + SAR**

16. **Does your optical+SAR fusion actually fuse learned features?** — Under the shipped configuration, **no** — a deterministic statistics template answers, because visual-token injection is disabled for BLIP-2 and the fusion adapter is randomly initialized. A genuine cross-attention fusion network exists in code but is untrained and dormant.
17. **How do you preprocess SAR?** — Real dB conversion (`10·log10`), VV/VH cross-pol ratio, standardization, BigEarthNet-S1 normalization, calibration honesty. **No speckle filtering** is implemented (only log-scaling).
18. **How do you detect SAR vs optical?** — A band-count/dtype/value heuristic. Honest limitation: an 8-bit single-band PNG SAR image is misclassified as optical (**observed live**); float/uint16/2-band GeoTIFFs are detected correctly.
19. **What if the SAR path can't run?** — It refuses with a clear message and `unavailable` confidence rather than fabricating a fusion result (**observed live**).
20. **Do you provide a fusion visualization?** — Yes, a 3-panel Optical\|SAR\|Fusion-heat image; the heat panel only appears when the (currently dormant) adapter runs, else it degrades to 2 panels.

**Confidence & trust**

21. **Is your confidence just answer length?** — No; it's derived from token log-probs / detection scores / threshold margins. `answer_length` is explicitly not used. (`confidence/confidence_service.py:436-452`)
22. **Is the confidence calibrated?** — **No.** Temperature-scaling machinery exists but no calibration file ships, so every output is `uncalibrated` (or `unavailable`). We label this honestly and tell clients not to treat it as a probability.
23. **What does a low confidence do?** — Below 0.45 the answer is flagged `requires_verification` and gets a visible "⚠ Low confidence" caveat (**observed live** on VQA/captioning/grounding).
24. **Does HTTP 200 mean the answer is trustworthy?** — Not necessarily. A 200 can be a degraded/mock result; clients must check `is_degraded` / `requires_verification`. Mocks return conf 0.01 and never a confident fabrication.
25. **What happens when a model fails to load?** — The registry substitutes an explicit `MockModel` ("[Degraded Mode]", conf 0.01, `requires_verification=True`); the response is flagged `is_degraded`. It never silently returns a fake confident answer.

**Geospatial & correctness**

26. **What CRS/projections do you support?** — Any rasterio-readable CRS on GeoTIFF input; geolocation tools reproject to WGS84 via pyproj. Plain PNG/JPEG has no geo and returns `unsupported` honestly (**observed**: EPSG:32643 read correctly).
27. **Do you report ground resolution?** — Yes, from the affine transform (linear units for projected CRS, haversine for geographic).

**Deployment**

28. **Is it production-hardened?** — Partly: structured JSON logging with request IDs, per-IP rate limiting, optional API-key auth, Pillow content validation, path-traversal guards, streaming size caps, TTL cleanup, and a clean structured error envelope (no stack-trace leaks — **verified across 5 failure cases**). Gaps: docs exposed by default, dev secret default, `/health` degradation stub, unenforced timeouts, compose has no GPU.
29. **What hardware does it need?** — CPU-only works (this audit ran on CPU). Realistic footprint: **~30 GB RAM** with the main models resident; a GPU is optional (the code detects CUDA but the compose file doesn't request one). Warm inference is ~12 s (VQA) to ~38 s (change description) on CPU.
30. **Top items to fix before production?** — (1) Produce real accuracy numbers on ≥1 RS dataset per task; (2) train/ship checkpoints or restate models as base+deterministic-tooling; (3) fix the co-registration `_path` gap and the 256-grid geo scale bug; (4) enforce the concurrency limit and inference timeout; (5) ship confidence calibration; (6) disable docs / replace the dev secret in prod.

---

## 7. Provenance & Reproducibility

- **Static audit:** 8 parallel read-only agents (`Read`/`Grep`/`Glob` only), 233 tool calls, every verdict `file:line`-anchored. No files modified.
- **Live audit:** `_audit_run.py` drove the running server over HTTP; raw results in `_audit_results.json`, console log in `_audit_run.log`. Synthetic fixtures in `_audit_fixtures/` (color-block optical PNGs, grayscale speckle "SAR", EPSG:32643 GeoTIFF pair + a 4-band RGBN GeoTIFF).
- **Accuracy source of truth:** `evaluation/results/*.json` + `evaluation/results/benchmark_report.md` (git commit `68386f8…`).
- **Memory measurement:** Windows `tasklist` working-set on the uvicorn PID with 5/6 models loaded.

**Scratch/evidence artifacts created by this audit** (untracked, not application code): `_audit_run.py`, `_audit_fixtures/`, `_audit_results.json`, `_audit_run.log`. These can be kept as evidence or deleted; they do not affect the app.

**No application code was changed during this audit.** Recommended fixes above are pending your explicit go-ahead.
