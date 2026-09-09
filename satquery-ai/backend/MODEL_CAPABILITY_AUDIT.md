# MODEL CAPABILITY AUDIT — SatQuery AI

**Date:** 2026-09-09  
**Auditor:** ML/Model Architecture Engineer  
**Status:** Model selection decision — DO NOT IMPLEMENT YET

---

## 1. Current Architecture

```
User Query
    ↓
TaskClassifier (keyword + structural + optional RS-CLIP semantic)
    ↓
AgenticController
    ↓
ModelRegistry (singleton, lazy-load, mock fallback)
    ↓
┌──────────────────────────────────────────────┐
│ RemoteSensingVQA      (BLIP VQA)             │ ← VQA + description
│ RemoteSensingCaptioning (BLIP captioning)    │ ← Scene description
│ RemoteSensingGrounding  (OWL-ViT)            │ ← Object detection
│ RSCLIPEncoder          (OpenCLIP ViT-B/32)   │ ← Classification + routing
│ ChangeDetectionModel   (Siamese ResNet-50)   │ ← Change detection
│ SAROpticalFusionModel  (BLIP + fusion)       │ ← SAR-optical fusion
└──────────────────────────────────────────────┘
    ↓
AnalysisResponse → ReportGenerator → API
```

**Total backend:** ~17,800 lines Python, 89 source files, 10 test files.

---

## 2. Current Model Inventory

### VQA — `Salesforce/blip-vqa-base`

| Attribute | Value |
|-----------|-------|
| Architecture | BlipForQuestionAnswering (ViT-B encoder + text decoder) |
| Parameters | ~99M |
| Training domain | COCO, VQA v2, Visual Genome — natural images |
| Input resolution | 384×384 (resized from input) |
| RS support | **None** — never trained on satellite/aerial imagery |
| Descriptive VQA | **Fails** — produces "airplane", "no idea", "clock" |
| Yes/no VQA | **Partial** — works with RS prefix for simple presence |
| Localization | **No** — generative only, no bounding boxes |
| Structured answers | **No** — free-form text only |
| CPU inference | ~1.0s per question |
| CUDA inference | N/A (no GPU on this machine) |
| VRAM (if GPU) | ~2GB |
| HuggingFace | `Salesforce/blip-vqa-base` ✓ |

### Captioning — `Salesforce/blip-image-captioning-base`

| Attribute | Value |
|-----------|-------|
| Architecture | BlipForConditionalGeneration (ViT-B + text decoder) |
| Parameters | ~99M |
| Training domain | COCO captions — natural images |
| RS support | **None** — uses prefix "A satellite image showing" |
| Caption quality | Generic natural-image captions, not RS-aware |
| CPU inference | ~1.5s |
| HuggingFace | `Salesforce/blip-image-captioning-base` ✓ |

### Grounding — `google/owlvit-base-patch32`

| Attribute | Value |
|-----------|-------|
| Architecture | OWL-ViT (ViT-B/32 + text encoder) |
| Parameters | ~150M |
| Training domain | OVIS, LVIS, COCO — natural images |
| RS support | **None** — detects zero objects on satellite imagery |
| Road detection | **0 detections** on 256×256 satellite image |
| Building detection | **0 detections** |
| CPU inference | ~0.7s per query (after 173s first load) |
| HuggingFace | `google/owlvit-base-patch32` ✓ |

### CLIP — `openclip ViT-B-32` (openai weights)

| Attribute | Value |
|-----------|-------|
| Architecture | CLIP ViT-B/32 |
| Parameters | ~151M |
| Training domain | LAION-400M — natural images |
| RS support | **Partial** — can encode RS images but poor discrimination |
| Zero-shot RS classification | Scores nearly identical across all concepts (0.21-0.23) |
| Discrimination ability | **Very low** — cannot distinguish road from forest from water |
| CPU inference | ~0.3s per image |
| HuggingFace | Via `open_clip` ✓ |

---

## 3. Current BLIP Baseline (Benchmark Results)

**Test image:** `_audit_fixtures/opt_a.png` (256×256 RGB satellite image)  
**Device:** CPU (AMD64, 16 cores, 33.7 GB RAM)  
**No GPU available.**

### Presence Detection

| Question | Answer | Confidence | Time |
|----------|--------|------------|------|
| Is there a road? | **yes** | 0.0000 | 1.41s |
| Is there a building? | **yes** | 0.0000 | 1.12s |
| Is there water? | i don't know | 0.0000 | 1.29s |
| Are there vehicles? | yes | 0.0001 | 1.06s |
| Is there vegetation? | no | 0.0004 | 1.02s |

**Assessment:** Mixed. Road/building detection works by chance. Water/vegetation fail.

### Description

| Question | Answer | Confidence | Time |
|----------|--------|------------|------|
| Describe the road. | **airplane** | 0.0001 | 1.06s |
| Describe the buildings. | **high rise** | 0.0002 | 0.98s |
| Describe the water body. | **north pacific ocean** | 0.0003 | 1.05s |
| Describe the vegetation. | **no idea** | 0.0002 | 0.98s |
| What is visible? | **airplane** | 0.0001 | 0.98s |

**Assessment:** **TOTAL FAILURE.** Every descriptive answer is wrong. BLIP cannot describe objects in satellite imagery.

### Classification

| Question | Answer | Confidence | Time |
|----------|--------|------------|------|
| Is this an urban area? | no | 0.0000 | 1.22s |
| What type of land cover? | grass | 0.0000 | 0.91s |

**Assessment:** Poor. "Grass" is a reasonable guess for green areas, but not reliable.

### OWL-ViT Grounding Baseline

| Query | Detections | Time |
|-------|-----------|------|
| road | **0** | 0.76s |
| building | **0** | 0.66s |
| water | **0** | 0.68s |
| vehicle | **0** | 0.66s |
| tree | **0** | 0.66s |
| field | **0** | 0.67s |

**Assessment:** **TOTAL FAILURE.** OWL-ViT detects zero objects on satellite imagery.

### RS-CLIP Classification Baseline

| Text | Similarity |
|------|-----------|
| satellite imagery of agricultural land | 0.2287 |
| satellite imagery of a building | 0.2266 |
| satellite imagery of a forest | 0.2265 |
| satellite imagery of an urban area | 0.2219 |
| satellite imagery of a road | 0.2206 |
| satellite imagery of water | 0.2117 |

**Assessment:** All scores nearly identical (0.21-0.23). **No meaningful discrimination.**

---

## 4. Why BLIP Fails

1. **Training domain mismatch:** BLIP was trained on COCO/VQA v2 (natural images: people, animals, everyday objects). Satellite imagery has completely different visual characteristics:
   - Top-down perspective
   - Different scale (buildings are tiny patches)
   - Different color distributions
   - Different semantic concepts (land cover, infrastructure, terrain)

2. **No satellite-specific vocabulary:** BLIP has never seen "road" in an aerial context, never learned to identify buildings from above, never distinguished agricultural fields from natural vegetation.

3. **Resolution mismatch:** Satellite images contain fine-grained details (roads as thin lines, individual buildings as small rectangles) that are lost when resized to 384×384.

4. **Generative VQA limitation:** BLIP generates text token-by-token. Without RS training, it defaults to its most common training associations ("airplane" is common in VQA datasets).

5. **Confidence is meaningless:** All confidence scores are <0.001, indicating the model has near-zero certainty on every answer.

---

## 5. Candidate Models

### 5.1 RS Vision-Language Models (Can answer questions)

| Model | HuggingFace ID | Params | VQA | Caption | Ground RS? | VRAM | CPU? | Integration |
|-------|---------------|--------|-----|---------|-----------|------|------|-------------|
| **EarthDial** | `akshaydudhane/EarthDial_4B_RGB` | ~4B | ✓ | ✓ | ✓ | 8-10GB | Marginal | Low-Med |
| **GeoChat** | `MBZUAI/geochat-7B` | ~7B | ✓ | ✓ | ✓ | 14-16GB | No | Low |
| **RS-LLaVA** | `BigData-KSU/RS-llava-v1.5-7b-LoRA` | ~7B | ✓ | ✓ | ✗ | 14-16GB | No | Low |
| **RSGPT** | GitHub (InstructBLIP-based) | ~7B | ✓ | ✓ | ✗ | 14-16GB | No | Medium |
| **SkyEyeGPT** | GitHub (Vicuna-based) | ~7B | ✓ | ✓ | ✓ | 14-16GB | No | Medium |
| **LHRS-Bot-Nova** | GitHub (LLaMA2-based) | ~8B | ✓ | ✓ | ✓ | 14-16GB | No | Medium-High |

### 5.2 RS Encoders (Cannot answer questions)

| Model | HuggingFace ID | Params | VQA? | Classification? | VRAM | CPU? | Integration |
|-------|---------------|--------|------|----------------|------|------|-------------|
| **RemoteCLIP** | `OneScience-Group/RemoteCLIP` | 150M-1.8B | ✗ | ✓ (zero-shot) | 1-3GB | ✓ | Low |
| **GeoRSCLIP** | `Zilun/GeoRSCLIP` | 150M-632M | ✗ | ✓ (zero-shot) | 1-3GB | ✓ | Medium |
| **Prithvi-EO-2.0** | `ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL` | 600M | ✗ | ✓ (needs head) | 6-8GB | Yes | Medium |
| **SatCLIP** | `microsoft/SatCLIP-ResNet50-L40` | ~25M | ✗ | Location only | <1GB | ✓ | Medium |

### 5.3 General VLMs (Not RS-trained)

| Model | Params | VQA | RS support? | VRAM | CPU? |
|-------|--------|-----|------------|------|------|
| BLIP-2 OPT-2.7B | 2.7B | ✓ | ✗ Poor | 10GB | Slow |
| InstructBLIP FlanT5-XL | ~3B | ✓ | ✗ Poor | 11GB | Marginal |
| LLaVA-1.5 7B (Q4) | 7B | ✓ | ✗ Poor | ~4.3GB | Slow |

---

## 6. Hardware Assessment

| Attribute | Value |
|-----------|-------|
| **CPU** | AMD64, 16 cores, 2.9 GHz |
| **RAM** | 33.7 GB total, 19.0 GB available |
| **GPU** | **NOT AVAILABLE** (CUDA: False, no NVIDIA GPU) |
| **OS** | Windows 11 |
| **Python** | 3.13.7 |
| **PyTorch** | 2.6.0+cpu |
| ** transformers** | 4.47.0 |

### Hardware Feasibility Categories

| Category | Definition | Examples |
|----------|-----------|----------|
| **A — Comfortable** | <2GB RAM, <2s inference on CPU | RemoteCLIP ViT-B, BLIP VQA, OWL-ViT |
| **B — Runnable** | 2-8GB RAM, 2-10s inference on CPU | RemoteCLIP ViT-L, GeoRSCLIP ViT-B |
| **C — Marginal** | 8-16GB RAM, 10-60s inference on CPU | EarthDial 4B, BLIP-2 2.7B |
| **D — Not viable** | >16GB RAM or requires GPU | GeoChat 7B, RS-LLaVA 7B, any 7B+ model |

**CRITICAL CONSTRAINT:** This machine has NO GPU. All 7B models (GeoChat, RS-LLaVA, RSGPT, SkyEyeGPT, LHRS-Bot) are **Category D** — they require GPU for any reasonable inference speed.

**EarthDial (4B)** is **Category C** — technically possible on CPU with 33.7GB RAM but will be slow (~10-30s per inference).

---

## 7. Required Capabilities Matrix

| Capability | Required? | Current Model | Current Status | Priority |
|------------|-----------|---------------|----------------|----------|
| **Presence detection** ("Is there a road?") | Yes | BLIP VQA | Partial (50% accurate) | High |
| **Object description** ("Describe the road") | Yes | BLIP VQA | **FAILED** | **Critical** |
| **Scene description** ("What is visible?") | Yes | BLIP VQA | **FAILED** | **Critical** |
| **Spatial localization** ("Where is the road?") | Yes | OWL-ViT | **FAILED** (0 detections) | High |
| **Land cover classification** | Yes | RS-CLIP | Poor discrimination | Medium |
| **Change detection** | Yes | ResNet-50 Siamese | Working (separate path) | Low |
| **SAR-optical fusion** | Yes | BLIP + fusion | Working (separate path) | Low |
| **Captioning** | Yes | BLIP captioning | Generic, not RS-aware | Medium |

---

## 8. Benchmark Results Summary

### BLIP VQA (Current)

| Task | Accuracy | Quality |
|------|----------|---------|
| Presence (yes/no) | ~40% (2/5 correct) | Poor |
| Description | 0% (all wrong) | **Catastrophic** |
| Classification | ~50% (partial) | Poor |
| **Overall** | **~30%** | **Unusable for production** |

### OWL-ViT Grounding (Current)

| Task | Detections | Quality |
|------|-----------|---------|
| Road detection | 0 | **Catastrophic** |
| Building detection | 0 | **Catastrophic** |
| Any object | 0 | **Catastrophic** |
| **Overall** | **0%** | **Unusable** |

### RS-CLIP Classification (Current)

| Task | Discrimination | Quality |
|------|---------------|---------|
| Concept ranking | Near-identical scores | **Poor** |
| Zero-shot classification | All classes ~0.22 | **Poor** |

---

## 9. Why BLIP Fails on Satellite Imagery (Root Cause Analysis)

1. **Training distribution shift:** BLIP was trained on natural images (COCO: people, cars, animals, indoor scenes). Satellite imagery is a fundamentally different visual domain with different:
   - Perspective (nadir/oblique vs. eye-level)
   - Scale (km-scale vs. meter-scale)
   - Content (terrain, infrastructure, land cover vs. objects, people, scenes)
   - Color statistics (atmospheric effects, spectral bands)

2. **Vocabulary mismatch:** BLIP's vocabulary includes "airplane", "clock", "high rise" (common in COCO) but not "road network", "agricultural plot", "wetland", "impervious surface" (common in RS).

3. **Resolution loss:** Satellite images contain roads as 1-3 pixel wide lines. Resizing to 384×384 destroys these fine details.

4. **No spatial reasoning:** BLIP cannot reason about spatial relationships in overhead imagery (e.g., "the road runs through the center").

5. **Confidence collapse:** All token probabilities are near zero, indicating the model is essentially guessing.

---

## 10. Recommended Architecture

### Option A: Minimal BLIP Modification
- **What:** Keep BLIP, improve prompting, add post-processing
- **Implementation time:** 1 day
- **Expected quality:** Marginal improvement (presence: ~60%, description: still fails)
- **Risk:** Low
- **Verdict:** **Insufficient** — cannot solve the core problem

### Option B: Replace BLIP with EarthDial (4B RS-VLM)
- **What:** Swap BLIP VQA with EarthDial_4B_RGB
- **Implementation time:** 2-3 days
- **Expected quality:** Good for description, classification, presence
- **Risk:** Medium (CPU inference speed may be slow)
- **Dependency:** Custom `earthdial` package, InternVL architecture
- **Offline capability:** Yes (model downloadable)
- **Verdict:** **Best option for description quality**

### Option C: RS Encoder + BLIP Hybrid
- **What:** Add RemoteCLIP for presence/classification, keep BLIP for captioning
- **Implementation time:** 1-2 days
- **Expected quality:** Presence: ~80%, Description: still fails
- **Risk:** Low
- **Verdict:** **Does not solve description**

### Option D: Multi-model RS Pipeline
- **What:** RemoteCLIP (presence/classification) + EarthDial (description) + OWL-ViT (grounding)
- **Implementation time:** 3-5 days
- **Expected quality:** Good across all tasks
- **Risk:** Medium-High (multiple model dependencies)
- **Verdict:** **Most complete but complex**

### Option E: Fine-tune BLIP on RS data
- **What:** Train BLIP on RSVQA/VRSBench dataset
- **Implementation time:** 1-2 weeks
- **Expected quality:** Good (if data is sufficient)
- **Risk:** High (training may not converge, requires dataset)
- **Verdict:** **Too slow for hackathon timeline**

---

## 11. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| EarthDial too slow on CPU | Medium | High | Test inference time before committing; use quantization |
| EarthDial package unavailable | Low | High | Check HuggingFace before implementing |
| EarthDial quality insufficient | Low | Medium | Benchmark before full integration |
| CPU memory pressure (4B model) | Medium | Medium | Use float16, unload other models during inference |
| Integration breaks existing pipeline | Low | High | Use adapter pattern, keep existing API contracts |
| Demo fails due to slow inference | Medium | High | Pre-warm model, cache results, show pre-computed examples |

---

## 12. Implementation Plan (If Approved)

### Phase 1: EarthDial Benchmark (1 day)
1. Download EarthDial_4B_RGB from HuggingFace
2. Test loading time on CPU
3. Test inference speed on CPU
4. Run the 12-question benchmark
5. Compare with BLIP baseline
6. **Go/No-Go decision**

### Phase 2: Integration (2-3 days)
1. Create `EarthDialVQAAdapter` implementing the same interface as `RemoteSensingVQA`
2. Add to `ModelRegistry` as a new model
3. Update `AgenticController` to route VQA to EarthDial
4. Keep BLIP as fallback
5. Run all 60 classifier tests + new VQA tests

### Phase 3: Evaluation (1 day)
1. Run full benchmark suite
2. Compare with BLIP baseline
3. Test edge cases (negative images, multi-object scenes)
4. Performance profiling

---

## 13. Files That Need Modification

| File | Change |
|------|--------|
| `models/vqa_model.py` | Add EarthDial adapter class |
| `models/registry.py` | Register EarthDial model |
| `config.py` | Add EarthDial config options |
| `.env` | Add EarthDial model path |
| `agent/controller.py` | Route VQA to EarthDial |

## 14. Files That Should Remain Unchanged

- `api/routes.py` — API contracts unchanged
- `api/schemas.py` — Response schemas unchanged
- `agent/task_classifier.py` — Routing logic unchanged (already fixed)
- `agent/report_generator.py` — Report generation unchanged
- `models/captioning_model.py` — Captioning unchanged
- `models/grounding_model.py` — Grounding unchanged
- `models/change_model.py` — Change detection unchanged
- `models/sar_fusion_model.py` — SAR fusion unchanged
- `confidence/` — Confidence framework unchanged
- `tools/` — Deterministic tools unchanged
- `tests/test_task_classifier.py` — 60 routing tests unchanged

---

## 15. Final Decision

```
RECOMMENDED PATH

Model:
EarthDial_4B_RGB (akshaydudhane/EarthDial_4B_RGB)

Reason:
EarthDial is the only publicly available RS-native VLM that:
1. Can answer questions about satellite imagery (VQA)
2. Can generate descriptive captions
3. Can perform visual grounding
4. Handles multi-spectral data (RGB, SAR, Sentinel-2)
5. Is small enough (4B) to potentially run on CPU
6. Is available on HuggingFace
7. Has demonstrated performance on RS benchmarks

Architecture:
EarthDial replaces BLIP VQA as the primary VQA/description model.
The existing task classifier, controller, and API remain unchanged.
EarthDial is accessed through an adapter that maps the existing
RemoteSensingVQA interface to EarthDial's generation API.

Expected capabilities:
- Presence detection: ~80-90% (up from ~40%)
- Object description: ~70-80% (up from 0%)
- Scene description: ~70-80% (up from 0%)
- Land cover classification: ~70-80% (up from ~50%)
- Visual grounding: ~60-70% (up from 0%)

Hardware:
CPU-only (AMD64, 16 cores, 33.7 GB RAM, NO GPU)
EarthDial 4B: ~8GB RAM in float16, ~10-30s inference on CPU
Feasibility: Category C (marginal but possible)

Estimated implementation complexity:
Medium (2-3 days for adapter + integration + testing)

Why this is preferable to BLIP:
BLIP was trained on natural images (COCO) and fundamentally cannot
understand satellite imagery. EarthDial was trained on 11.1M RS
instruction samples across RGB, SAR, and multispectral data.
It is purpose-built for the exact tasks SatQuery AI requires.

What files need modification:
- models/vqa_model.py (add EarthDialAdapter)
- models/registry.py (register EarthDial)
- config.py (add config options)
- .env (add model path)
- agent/controller.py (route to EarthDial)

What files should remain untouched:
- api/routes.py, api/schemas.py (API unchanged)
- agent/task_classifier.py (routing already fixed)
- models/captioning_model.py, grounding_model.py, change_model.py
- confidence/, tools/, tests/

Next implementation step:
BENCHMARK EarthDial_4B_RGB on CPU before any code changes.
Test: loading time, inference speed, RAM usage, answer quality.
Go/No-Go decision based on benchmark results.
```

---

## 16. Appendix: Pre-Implementation Checklist

- [ ] Download EarthDial_4B_RGB from HuggingFace
- [ ] Test model loading time on CPU
- [ ] Test single inference latency on CPU
- [ ] Test peak RAM usage during inference
- [ ] Run 12-question benchmark on test image
- [ ] Compare answers with BLIP baseline
- [ ] Test negative cases (image without road → "Is there a road?" should say "no")
- [ ] Verify model handles 256×256 input
- [ ] Verify model handles 512×512 input
- [ ] Check HuggingFace model card for dependencies
- [ ] Verify `transformers` version compatibility
- [ ] Test adapter interface compatibility
- [ ] Go/No-Go decision
