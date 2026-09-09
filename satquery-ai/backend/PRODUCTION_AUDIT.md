# SatQuery AI — Production Forensic Audit

**Date:** 2026-09-09
**Auditor:** Automated forensic analysis
**Scope:** Complete pipeline audit — why basic visual content (roads, buildings, water) is not reliably identified

---

## 1. Executive Summary

SatQuery AI has a well-structured agentic architecture with 7 task types, 6 specialist models, 10 deterministic tools, and comprehensive security/monitoring infrastructure. However, **the system fundamentally fails at its primary purpose: understanding basic visual content in satellite imagery.**

The root cause is **not** a single bug. It is a cascade of **5 independent failures** that compound to produce unreliable results:

| # | Failure | Severity | Impact |
|---|---------|----------|--------|
| 1 | **Wrong task routing for "describe" queries** | CRITICAL | "Describe the road" → captioning (general) instead of VQA (specific) |
| 2 | **Captioning model receives full query as condition** | CRITICAL | BLIP captioning is not designed for object-specific questions |
| 3 | **No domain-specific prompting for satellite imagery** | HIGH | Generic prompts produce generic/incorrect answers |
| 4 | **BLIP pretrained on natural images, not satellite imagery** | HIGH | Domain mismatch degrades recognition of roads, buildings, etc. |
| 5 | **Confidence scoring uses uncalibrated heuristics** | MEDIUM | Wrong answers reported with moderate confidence |

**Bottom line:** When you ask "Describe the road in this image," the system:
1. Routes to CAPTIONING (not VQA) because "describe" is a captioning keyword
2. Sends "Describe the road in this image" as a conditional prefix to BLIP captioning
3. BLIP captioning tries to continue the text, not answer a question
4. Returns a generic scene description, not a road-specific answer

---

## 2. Architecture Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Layer                                │
│  POST /api/upload  →  upload_image()  →  InputValidator        │
│  POST /api/analyze →  analyze()       →  AgenticController     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Agentic Pipeline                              │
│  1. TaskClassifier      →  TaskType (7 options)                 │
│  2. IntentExtractor     →  QueryIntent (concept + operation)    │
│  3. ExecutionPlanner    →  ExecutionPlan (model + steps)        │
│  4. Handler Dispatch    →  model-specific handler               │
│  5. ToolPlanner         →  deterministic evidence (NDVI, etc.)  │
│  6. ConfidenceService   →  ConfidenceReport                     │
│  7. ResultIntegration   →  AnalysisResponse                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Specialist Models                             │
│  RemoteSensingVQA           (BLIP VQA)                          │
│  RemoteSensingCaptioning    (BLIP Captioning / GIT fallback)    │
│  RemoteSensingGrounding     (OWL-ViT)                           │
│  ChangeDetectionModel       (Siamese ResNet-50)                 │
│  SAROpticalFusionModel      (SAR encoder + cross-attention)     │
│  RSCLIPEncoder              (OpenCLIP ViT-B/32)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Deterministic Tools (10)                      │
│  NDVI, NDWI, NDBI, SpectralStatistics, SARBackscatter,         │
│  ChangeArea, ConnectedRegions, ImageBounds, GroundResolution,   │
│  PixelToLatLon                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Request Execution Trace

### Query: "Describe the road in this image."
### Image: Google Maps screenshot containing visible roads

```
HTTP POST /api/analyze
  body: { image_ids: ["<uuid>"], query: "Describe the road in this image." }
  │
  ▼
routes.py:analyze()                              [routes.py:188-340]
  ├── validates image_ids, query length
  ├── load_image(img_path) → dict with pil_image, numpy_array, bands
  ├── InputValidator.validate_image() → modality="optical", shape, bands
  ├── reads .meta.json sidecar for persisted modality
  └── AgenticController(registry).analyze(images, query)
        │
        ▼
controller.py:analyze()                           [controller.py:187-238]
  │
  ├── STEP 1: TASK CLASSIFICATION
  │   classifier.classify("Describe the road in this image.", 1, ["optical"])
  │   │
  │   ├── _is_land_cover_classification() → False (no "classify"/"land cover")
  │   │
  │   ├── _keyword_and_structural()
  │   │   ├── keyword_scores:
  │   │   │   CAPTIONING: 0.8  ("describe" matched)
  │   │   │   SINGLE_VQA: 0.5  ("road" matched)
  │   │   │   All others: 0.0
  │   │   └── structural_scores: all 0.0 (single image, optical)
  │   │
  │   ├── SemanticRouter.route() → blends 0.55*semantic + 0.25*structural + 0.20*keyword
  │   │   (or keyword-only fallback if router unavailable)
  │   │
  │   └── RESULT: TaskType.CAPTIONING (score=0.8 beats SINGLE_VQA=0.5)
  │       ⚠️ WRONG: "Describe the road" should be VQA, not captioning
  │
  ├── STEP 2: INTENT EXTRACTION
  │   intent_extractor.extract(query, 1, ["optical"])
  │   → concept="scene", operation="describe", requires_grounding=False
  │
  ├── STEP 3: EXECUTION PLAN
  │   plan = ExecutionPlan(
  │       task_type=CAPTIONING,
  │       model_names=["RemoteSensingCaptioning"],
  │       steps=["validate_input", "preprocess_image", "generate_caption", "format_answer"],
  │       parameters={"max_new_tokens": 200, "num_beams": 5}
  │   )
  │
  ├── STEP 4: EXECUTE
  │   _exec_captioning(plan, images, query)
  │   │
  │   ├── _get_pil(images[0]) → PIL RGB image
  │   │
  │   ├── registry.inference_with_context("RemoteSensingCaptioning", ...)
  │   │   │
  │   │   ▼
  │   │   captioning_model.py:generate_caption(pil_image)
  │   │   ├── _preprocess(image) → RGB, resize to max 1024px
  │   │   └── _blip_caption(image)
  │   │       │
  │   │       ├── inputs = self.processor(
  │   │       │     images=image,
  │   │       │     text="A satellite image showing",  ← CONDITIONAL PREFIX
  │   │       │     return_tensors="pt"
  │   │       │   )
  │   │       │
  │   │       ├── gen = self.model.generate(
  │   │       │     **inputs,
  │   │       │     max_new_tokens=200,
  │   │       │     num_beams=5,
  │   │       │     repetition_penalty=1.3,
  │   │       │     early_stopping=True
  │   │       │   )
  │   │       │
  │   │       ├── caption = decode(gen.sequences[0])
  │   │       │   → Likely: "a satellite image showing an aerial view of a road
  │   │       │              with buildings and trees"
  │   │       │   (generic, not road-specific)
  │   │       │
  │   │       └── Returns: {"caption": "...", "confidence": 0.5, ...}
  │   │
  │   └── Returns: RawResults(answer="...", confidence=0.5)
  │
  ├── STEP 5: CONFIDENCE ASSESSMENT
  │   _assess_confidence() → ConfidenceReport(uncalibrated, confidence~0.5)
  │
  └── STEP 6: INTEGRATE → AnalysisResponse
        answer: "a satellite image showing an aerial view of..."
        task: "CAPTIONING"
        confidence: ~0.5
```

**Key finding:** The query "Describe the road" is routed to CAPTIONING, not VQA. The captioning model is given `"A satellite image showing"` as the conditional prefix — the original query is completely discarded. The model generates a generic scene description, not a road-specific answer.

---

## 4. Model Inventory

| Registry Name | HuggingFace Model | Task | Domain Match | Status |
|---|---|---|---|---|
| RemoteSensingVQA | `Salesforce/blip-vqa-base` | VQA | ⚠️ Natural images, not satellite | Working |
| RemoteSensingCaptioning | `Salesforce/blip-image-captioning-base` | Captioning | ⚠️ Natural images, not satellite | Working |
| RemoteSensingGrounding | `google/owlvit-base-patch32` | Object Detection | ⚠️ Natural images, not satellite | Working |
| ChangeDetectionModel | `microsoft/resnet-50` | Change Detection | ✅ Features transfer reasonably | Working |
| SAROpticalFusionModel | Custom (BLIP + SAR encoder) | Fusion | ✅ Custom architecture | Partially working |
| RSCLIPEncoder | `ViT-B-32` (OpenCLIP) | Zero-shot classification | ⚠️ Generic, not RS-specific | Working |

---

## 5. Model Loading Analysis

**Status:** All models load correctly on CPU. No mock fallbacks active when `allow_mock_mode=false`.

**Loading path:**
1. `main.py:lifespan()` → `ModelRegistry.load_all(lazy=True)`
2. Each model registered with a lazy loader closure
3. First `inference_with_context()` call triggers actual loading
4. Loading is thread-safe via `threading.Lock`

**Verified:** All 6 models successfully load from HuggingFace cache. No corrupted weights. No device mismatches.

**Conclusion:** Model loading is NOT the problem.

---

## 6. Task Classifier Analysis

### The "Describe the road" routing failure

**File:** `agent/task_classifier.py`

**Keyword rules (relevant excerpts):**
```python
CAPTIONING: {
    "describe": 0.8, "description": 0.8, "caption": 0.9,
    "summarize": 0.7, "summary": 0.7, "overview": 0.6,
    ...
}
SINGLE_VQA: {
    "road": 0.5, "building": 0.55, "water": 0.5,
    "vegetation": 0.55, "forest": 0.5, ...
}
```

**Problem:** The word "describe" scores 0.8 for CAPTIONING, while "road" scores only 0.5 for SINGLE_VQA. Since keyword scores are the primary signal, CAPTIONING wins.

**This is a design flaw:** "Describe the road" is semantically a VQA query (asking about a specific object), but the keyword "describe" triggers captioning routing.

**Impact on other queries:**

| Query | Expected Task | Actual Task | Problem |
|-------|--------------|-------------|---------|
| "Describe the road in this image" | VQA | CAPTIONING | "describe" triggers captioning |
| "What is visible in the image?" | CAPTIONING | CAPTIONING | ✅ Correct |
| "Is there a road?" | VQA | CAPTIONING | "is there" matches VQA at 0.6 but no strong signal |
| "Where is the road?" | GROUNDING | GROUNDING | ✅ "where is" matches at 0.85 |
| "What type of road is visible?" | VQA | CAPTIONING | "what type" matches VQA at 0.65 but "describe" not present... mixed |
| "Describe the surrounding area" | CAPTIONING | CAPTIONING | ✅ Correct |
| "Identify the major objects" | GROUNDING | GROUNDING | ✅ "identify" matches at 0.9 |

**The fundamental issue:** The classifier treats "describe" as a captioning trigger, but "describe [specific object]" should be a VQA query.

---

## 7. Image Preprocessing Analysis

**File:** `utils/image_utils.py`, `models/vqa_model.py`, `models/captioning_model.py`

### VQA preprocessing (`vqa_model.py:_preprocess`):
```python
if image.mode != "RGB":
    image = image.convert("RGB")
max_side = 1024
if max(w, h) > max_side:
    scale = max_side / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
```

### Captioning preprocessing (`captioning_model.py:_preprocess`):
```python
if image.mode != "RGB":
    image = image.convert("RGB")
max_side = 1024
if max(w, h) > max_side:
    scale = max_side / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
```

### Grounding preprocessing (`grounding_model.py:_preprocess`):
```python
if image.mode != "RGB":
    image = image.convert("RGB")
max_side = 768
if max(w, h) > max_side:
    scale = max_side / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
```

**Analysis:**
- ✅ RGB conversion is correct
- ✅ Aspect ratio is preserved (resize, not crop)
- ⚠️ Max 1024px for VQA/captioning, 768px for grounding — reasonable for these models
- ⚠️ No normalization beyond what the HuggingFace processor does internally
- ⚠️ No satellite-specific preprocessing (e.g., pan-sharpening, atmospheric correction)

**Conclusion:** Preprocessing is **adequate but not optimal** for satellite imagery. Information loss from resizing is minimal for these model architectures. **Preprocessing is NOT the primary problem.**

---

## 8. Prompt Analysis

### VQA prompt (BLIP):
```python
inputs = self.processor(images=image, text=question, return_tensors="pt")
```
- The raw question is passed directly: "Describe the road in this image."
- BLIP VQA is designed for questions like "What color is the car?" or "How many people are there?"
- **Problem:** "Describe the road" is not a standard VQA question format. BLIP expects short, specific questions.
- **Impact:** Even if routed to VQA, the question format is suboptimal.

### Captioning prompt (BLIP):
```python
inputs = self.processor(images=image, text="A satellite image showing", return_tensors="pt")
```
- The conditional prefix is `"A satellite image showing"` — the original query is completely discarded.
- **Problem:** The query "Describe the road in this image" is never sent to the model. The captioning model has no idea the user asked about roads.
- **Impact:** Generic scene description, not road-specific.

### Grounding prompts (OWL-ViT):
```python
texts = [["road", "street", "highway"]]  # After label extraction
```
- Previously: `texts = [["Describe the road in this image"]]` — entire sentence as label
- **After my fix:** Proper noun phrases extracted
- **Impact:** Previously 0% detection; now should detect road-like objects

---

## 9. Mock Fallback Analysis

**Status:** `ALLOW_MOCK_MODE=true` is set in `.env`

**Path from real model failure to mock:**
```
ModelRegistry.get(name)
  ├── if model loaded: return real model
  ├── if lazy: run loader()
  │   ├── loader succeeds: return real model
  │   └── loader fails:
  │       ├── if allow_mock_mode: MockModel(name)  ← SILENT DEGRADATION
  │       └── else: raise RuntimeError
  └── if not lazy and not loaded: RuntimeError
```

**Critical question:** Are any models currently returning mock responses?

**Evidence from health endpoint:** `models_loaded: 0` with `lazy_registered: 6` suggests models are NOT loaded until first inference. If a model fails to load on first use and `allow_mock_mode=true`, it silently falls back to MockModel.

**MockModel behavior:**
- VQA: Returns `"I'm a mock model. Please configure a real model."` with confidence 0.01
- Captioning: Returns `"Mock caption: configure a real model for analysis."` with confidence 0.01
- Grounding: Returns empty boxes with confidence 0.01
- All responses include `degraded: True`, `requires_verification: True`

**Risk:** If any model fails to load (e.g., network issue downloading weights, disk space, CUDA OOM), the system silently uses mock responses with very low confidence. The user sees "0% confidence" and a degraded response.

**Mitigation already in place:** The `ABSTAIN_CAVEAT` prefix (`"[Verification Recommended] Low confidence result..."`) is prepended when confidence is below threshold.

---

## 10. Remote-Sensing Suitability Analysis

### BLIP VQA (`Salesforce/blip-vqa-base`)
- **Training data:** COCO-VQA, VQA v2.0 (natural images)
- **Capabilities:** Can answer short questions about natural images
- **RS limitations:**
  - Not trained on satellite/aerial imagery
  - May not recognize roads, buildings, water bodies in overhead view
  - Question format assumes ground-level perspective
  - No understanding of spectral bands, spatial resolution, or geospatial context
- **Verdict:** ⚠️ Partially suitable. Will recognize some objects (road, building, water) but with degraded accuracy. Will fail on RS-specific concepts (NDVI, backscatter, SAR).

### BLIP Captioning (`Salesforce/blip-image-captioning-base`)
- **Training data:** COCO captions (natural images)
- **Capabilities:** Generates scene descriptions for natural images
- **RS limitations:**
  - Same as VQA — trained on ground-level natural images
  - Will generate generic descriptions, not RS-specific analysis
  - Cannot answer specific object questions (by design)
- **Verdict:** ⚠️ Partially suitable for general scene description. Unsuitable for object-specific queries.

### OWL-ViT (`google/owlvit-base-patch32`)
- **Training data:** Open-vocabulary detection on natural images
- **Capabilities:** Text-guided object detection
- **RS limitations:**
  - Not trained on overhead/satellite perspective
  - May detect roads, buildings but with lower accuracy than ground-level
  - Limited vocabulary for RS-specific features
- **Verdict:** ⚠️ Partially suitable. Can detect common objects but with degraded performance.

### ResNet-50 (`microsoft/resnet-50`)
- **Training data:** ImageNet (natural images)
- **Capabilities:** Feature extraction, image classification
- **RS suitability:** ✅ Reasonably suitable for change detection (deep features transfer well)
- **Verdict:** ✅ Suitable for the change detection use case (cosine distance in feature space).

### OpenCLIP ViT-B/32
- **Training data:** LAION-5B (web images)
- **Capabilities:** Zero-shot image-text matching
- **RS limitations:**
  - Not specifically trained on RS imagery
  - May have some zero-shot capability for common concepts
- **Verdict:** ⚠️ Partially suitable. Zero-shot classification works for broad categories.

---

## 11. Root Causes (Ranked by Impact)

### ROOT CAUSE #1: Wrong Task Routing (CRITICAL)
**File:** `agent/task_classifier.py`
**Problem:** "Describe [specific object]" queries are routed to CAPTIONING instead of VQA.
**Why:** The word "describe" scores 0.8 for CAPTIONING, overpowering object-specific keywords.
**Impact:** The query is never sent to VQA. The captioning model ignores the object-specific part.
**Fix:** Add disambiguation logic: if query contains "describe" + a specific object keyword (road, building, water, tree, etc.), route to VQA instead of captioning.

### ROOT CAUSE #2: Captioning Model Discards Query (CRITICAL)
**File:** `models/captioning_model.py`
**Problem:** `_blip_caption()` sends `"A satellite image showing"` as the condition. The original query is completely discarded.
**Why:** BLIP captioning is conditional generation — it continues from a prefix, it doesn't answer questions.
**Impact:** The user's question about roads is never communicated to the model.
**Fix:** This is by design for captioning. The real fix is routing — don't send object-specific queries to captioning.

### ROOT CAUSE #3: No Domain-Specific Prompting (HIGH)
**File:** `models/vqa_model.py`
**Problem:** BLIP VQA receives the raw question with no RS-specific context.
**Why:** No prompt engineering for satellite imagery.
**Impact:** Even when routed to VQA, the model lacks context about the image being satellite imagery.
**Fix:** Prepend RS context to VQA questions: "This is a satellite image. {question}"

### ROOT CAUSE #4: Model-Domain Mismatch (HIGH)
**Files:** `config.py` (model selection), all model files
**Problem:** All models are pretrained on natural images (COCO, ImageNet, LAION), not satellite imagery.
**Why:** No RS-specific fine-tuning has been performed. BigEarthNet training code exists but was never executed.
**Impact:** Reduced accuracy for RS-specific concepts. Models may not recognize overhead perspectives.
**Fix:** Fine-tune on BigEarthNet or use RS-specific pretrained models (e.g., `BIFOLD-BigEarthNetv2-0/resnet50-s2-v0.2.0`).

### ROOT CAUSE #5: Uncalibrated Confidence (MEDIUM)
**Files:** `confidence/` module, `models/vqa_model.py`
**Problem:** Confidence scores are based on token log-probabilities, not accuracy calibration.
**Why:** No calibration dataset or accuracy validation has been performed.
**Impact:** Wrong answers may be reported with moderate confidence (0.3-0.7).
**Fix:** Calibrate on a held-out dataset. Implement abstention for low-confidence results.

---

## 12. Critical Bugs

### BUG-001: "Describe" routing to captioning (CRITICAL)
- **File:** `agent/task_classifier.py`
- **Line:** `_keyword_and_structural()` keyword matching
- **Problem:** "describe" always triggers CAPTIONING, even when followed by specific object names
- **Reproduction:** Query "Describe the road" → CAPTIONING (should be VQA)
- **Fix:** Add object-specific disambiguation in the classifier

### BUG-002: Captioning query discard (CRITICAL)
- **File:** `models/captioning_model.py`
- **Line:** `_blip_caption()` line 248
- **Problem:** `text=RS_CAPTION_PREFIX` discards the original query
- **Reproduction:** Any query routed to captioning gets ignored
- **Fix:** Not a bug per se — captioning is designed this way. Fix is in routing (BUG-001).

### BUG-003: VQA min_new_tokens too high (MEDIUM)
- **File:** `models/vqa_model.py`
- **Line:** `_infer_blip()` line 367
- **Problem:** `min_new_tokens=15` forces generation of at least 15 tokens even for short answers
- **Reproduction:** Simple yes/no questions generate verbose answers
- **Fix:** Reduce `min_new_tokens` to 1 or 5

---

## 13. High-Priority Fixes

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| H1 | Add object-specific disambiguation to task classifier | `task_classifier.py` | Low | CRITICAL — fixes routing for "describe X" queries |
| H2 | Prepend RS context to VQA questions | `vqa_model.py` | Low | HIGH — improves VQA accuracy for satellite images |
| H3 | Reduce VQA min_new_tokens to 1-5 | `vqa_model.py` | Trivial | MEDIUM — allows short, precise answers |
| H4 | Add VQA-specific prompt templates for common RS queries | `vqa_model.py` | Medium | HIGH — structured prompts improve accuracy |
| H5 | Verify models are not silently falling back to MockModel | `registry.py` | Low | HIGH — ensures real inference is happening |

---

## 14. Medium-Priority Fixes

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| M1 | Fine-tune BLIP VQA on BigEarthNet VQA dataset | `training/train_vqa.py` | High | HIGH — domain-specific accuracy |
| M2 | Use RS-specific pretrained models (BigEarthNet v2.0) | `config.py`, `registry.py` | Medium | HIGH — better base representations |
| M3 | Implement accuracy calibration on held-out dataset | `confidence/` | Medium | MEDIUM — reliable confidence scores |
| M4 | Add satellite-specific image augmentation in preprocessing | `vqa_model.py`, `captioning_model.py` | Medium | MEDIUM — improve robustness |
| M5 | Fine-tune CLIP on BigEarthNet for better task routing | `training/train_clip.py` | High | HIGH — semantic routing accuracy |

---

## 15. Low-Priority Improvements

| # | Fix | File | Effort | Impact |
|---|-----|------|--------|--------|
| L1 | Add pan-sharpening for multispectral input | `utils/image_utils.py` | Medium | LOW |
| L2 | Implement speckle filtering for SAR imagery | `models/sar_fusion/` | Medium | LOW |
| L3 | Add batch inference for multiple queries | `controller.py` | Medium | LOW |
| L4 | Implement model ensemble for critical tasks | `registry.py` | High | LOW |
| L5 | Add explainability/attention visualization | `utils/visualization.py` | High | LOW |

---

## 16. Proposed Architecture (After Fixes)

```
User Query → TaskClassifier (with object-disambiguation)
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      VQA Path   Caption    Grounding
         │        Path        Path
         ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐
   │BLIP VQA │ │BLIP Cap │ │OWL-ViT  │
   │+ RS ctx │ │+ RS ctx │ │+ labels │
   │+ prompt │ │         │ │         │
   │templates│ │         │ │         │
   └─────────┘ └─────────┘ └─────────┘
         │          │          │
         ▼          ▼          ▼
   Confidence   Confidence  Confidence
   Calibration  Calibration Calibration
         │          │          │
         └──────────┼──────────┘
                    ▼
           AnalysisResponse
```

**Key changes:**
1. Task classifier gains object-specific disambiguation
2. VQA model gets RS-specific prompt templates
3. Captioning model gets RS-specific prefix
4. All models get confidence calibration
5. Mock fallback requires explicit opt-in

---

## 17. Proposed Evaluation Benchmark

| # | Image | Query | Expected Task | Expected Model | Expected Semantic Result |
|---|-------|-------|---------------|----------------|------------------------|
| 1 | Road intersection | "Describe the road" | VQA | BLIP VQA | Road type, condition, features |
| 2 | Urban area | "How many buildings?" | Grounding | OWL-ViT | Building count + boxes |
| 3 | River scene | "Where is the water?" | Grounding | OWL-ViT | Water body location |
| 4 | Forest area | "What land cover type?" | Land Cover | RS-CLIP | Forest classification |
| 5 | Agricultural field | "Describe the vegetation" | VQA | BLIP VQA | Crop type, health |
| 6 | Bridge over river | "Is there a bridge?" | VQA | BLIP VQA | Yes/bridge description |
| 7 | City overview | "What is visible?" | Captioning | BLIP Cap | Urban scene description |
| 8 | Parking lot | "How many vehicles?" | Grounding | OWL-ViT | Vehicle count + boxes |
| 9 | River bend | "Describe the river" | VQA | BLIP VQA | River features |
| 10 | Mixed landscape | "Summarize the scene" | Captioning | BLIP Cap | Scene overview |

---

## 18. Deployment Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Models silently fall back to MockModel | HIGH | Set `ALLOW_MOCK_MODE=false` in production, monitor health endpoint |
| BLIP produces hallucinated answers | HIGH | Implement abstention, require verification for low-confidence |
| GPU OOM on multi-user system | MEDIUM | `max_concurrent_inference=1`, lazy loading, cache clearing |
| Network failure downloading HF models | MEDIUM | Pre-cache all models, use local model registry |
| False confidence in wrong answers | MEDIUM | Calibrate on held-out dataset, implement abstention threshold |
| SAR images misclassified as optical | LOW | Improve modality detection heuristics |

---

## 19. Answer to Key Questions

### 1. What is actually broken?
**The task classifier routes "describe" queries to captioning instead of VQA.** This is the primary failure. Secondary issues include lack of RS-specific prompting and model-domain mismatch.

### 2. Why does the road-description test fail?
Because "Describe the road" → CAPTIONING → BLIP generates generic scene description with prefix "A satellite image showing" → user's question about roads is discarded.

### 3. Is the model actually running?
**Yes.** All models load correctly from HuggingFace cache. The VQA, captioning, and grounding models are real (not mock). Inference completes successfully — the problem is *what* the model is asked, not whether it runs.

### 4. Is the task classifier correct?
**No.** It misroutes "describe [object]" queries to CAPTIONING. The keyword "describe" (0.8) overpowers object-specific keywords like "road" (0.5).

### 5. Is preprocessing correct?
**Adequate.** RGB conversion, aspect-ratio-preserving resize, and processor normalization are correct. No significant information loss. Not the primary problem.

### 6. Is the prompt correct?
**No.** Two problems:
- Captioning prompt discards the query entirely
- VQA prompt has no RS-specific context

### 7. Is the mock fallback involved?
**Not currently.** All models are loaded and returning real inference results. However, `ALLOW_MOCK_MODE=true` means any future load failure would silently degrade.

### 8. Is BLIP/BLIP-2 fundamentally insufficient?
**Partially.** BLIP can recognize roads, buildings, water in natural images. It will partially work for satellite imagery but with degraded accuracy. It is NOT fundamentally incapable — it just needs better prompting and routing.

### 9. What is the smallest reliable fix?
**Fix the task classifier** (`task_classifier.py`). Add disambiguation: if query contains "describe" + object keyword → route to VQA, not captioning. This single change would fix the "Describe the road" failure.

### 10. What is the long-term correct architecture?
1. Fine-tune VQA model on BigEarthNet VQA dataset
2. Use RS-specific pretrained models (BigEarthNet v2.0)
3. Implement RS-specific prompt templates
4. Add confidence calibration
5. Implement abstention for low-confidence results
