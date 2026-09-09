# VQA Routing Fix Report

## 1. Original Failure

**Query:** `"Describe the road in this image."`  
**Expected:** Routed to `SINGLE_VQA`, answered by BLIP VQA model  
**Actual (before fix):** Routed to `CAPTIONING`, user's question discarded, generic caption returned

**Root cause chain:**
1. Task classifier scored `"describe"` → CAPTIONING=0.8, `"road"` → VQA=0.5
2. CAPTIONING won (0.8 > 0.5)
3. Captioning model received prefix `"A satellite image showing"` — user's road question was discarded
4. Generic scene description returned instead of road-specific answer

## 2. Root Cause

**Primary:** Task classifier had no logic to distinguish "describe + entity" (object-specific) from "describe the image" (generic scene description). The keyword `"describe"` always scored CAPTIONING highest.

**Secondary:** VQA model used raw question without remote-sensing context, and `min_new_tokens=15` forced verbose answers for yes/no questions.

## 3. Files Changed

| File | Change |
|------|--------|
| `agent/task_classifier.py` | Added `_RS_ENTITY_LEXICON` (60+ RS terms), `_disambiguate_entity_description()` method, `"identify"` keyword to GROUNDING, changed default fallback from CAPTIONING to VQA for single images, added VQA before CAPTIONING in priority list |
| `models/vqa_model.py` | Added `_build_prompt()` with RS context prefix, changed `min_new_tokens` from 15→1, `max_new_tokens` from 100→50, `num_beams` from 1→4 in BLIP path |
| `agent/controller.py` | Updated plan parameters to match new VQA defaults, added `VQA_DEBUG` structured logging |
| `.env` | Changed `ALLOW_MOCK_MODE=true` → `ALLOW_MOCK_MODE=false` |
| `tests/test_task_classifier.py` | **NEW** — 60 parametrized tests across 8 test classes |
| `_test_classifier.py` | Temporary smoke test (can be deleted) |
| `test_direct_blip_vqa.py` | Standalone BLIP diagnostic (can be deleted) |
| `_quick_blip_test.py` | Quick BLIP test (can be deleted) |
| `_quick_blip_test_real.py` | Real image BLIP test (can be deleted) |
| `_quick_blip_ablation.py` | Prefix ablation test (can be deleted) |
| `_e2e_road_test.py` | End-to-end pipeline test (can be deleted) |

## 4. Routing Logic Before

```python
# Keyword scores
"describe" → CAPTIONING: 0.8
"road"     → VQA: 0.5

# No disambiguation — CAPTIONING always wins when "describe" is present
# Default fallback for no keywords: CAPTIONING: 0.4
# Priority: [..., CAPTIONING, SINGLE_VQA]  # CAPTIONING beats VQA on ties
```

## 5. Routing Logic After

```python
# Same keyword scoring, PLUS:
# 1. _disambiguate_entity_description() detects "describe + entity"
#    → boosts VQA to 0.9, suppresses CAPTIONING to 0.3
# 2. Default fallback for no keywords: VQA: 0.3 (was CAPTIONING: 0.4)
# 3. Priority: [..., SINGLE_VQA, CAPTIONING]  # VQA beats CAPTIONING on ties
# 4. "identify" added to GROUNDING keywords
```

## 6. Prompt Before

```python
# BLIP VQA:
inputs = self.processor(images=image, text=question, return_tensors="pt")
# No RS context — BLIP defaults to natural-scene interpretation
```

## 7. Prompt After

```python
# BLIP VQA:
prompted_q = self._build_prompt(question)
# = "This is an overhead satellite image. {question}"
inputs = self.processor(images=image, text=prompted_q, return_tensors="pt")
# BLIP receives domain context for satellite imagery
```

## 8. Generation Settings Before/After

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `min_new_tokens` | 15 | 1 | Allows "yes"/"no" answers without forced verbosity |
| `max_new_tokens` | 100 | 50 | VQA answers are short; reduces garbage tokens |
| `num_beams` (BLIP) | 1 | 4 | Better answer quality with beam search |
| `num_beams` (BLIP2) | 4 | 4 | Unchanged |

## 9. Mock-Mode Behavior Before/After

| | Before | After |
|---|--------|-------|
| `.env` | `ALLOW_MOCK_MODE=true` | `ALLOW_MOCK_MODE=false` |
| Model load failure | Silent mock fallback | `RuntimeError` raised |
| API response | Fake analysis returned | 503 error with clear message |
| Health status | "healthy" (lying) | "failed" (honest) |

## 10. Direct BLIP Test Results

**Test image:** `_audit_fixtures/opt_a.png` (256×256 RGB satellite image)

| Question | Answer (with RS prefix) | Answer (without prefix) |
|----------|------------------------|------------------------|
| "Is there a road?" | **yes** | no |
| "Describe the road." | airplane | black and white |
| "What is visible in the image?" | airplane | clock |
| "Is there a building?" | **yes** | no |
| "Is there water?" | i don't know | — |

**Finding:** RS prefix helps yes/no questions (road=yes, building=yes) but descriptive questions still fail. BLIP cannot reliably describe objects in satellite imagery.

## 11. End-to-End Test Results

```
Query: "Describe the road in this image."
Task:  SINGLE_VQA (confidence: 0.90)
Prompt: "This is an overhead satellite image. Describe the road in this image."
Answer: "no idea"
Confidence: 0.0001 (UNCALIBRATED)
```

- Routing: **PASS** (correctly routed to VQA, not captioning)
- Prompt: **PASS** (includes RS context)
- Intent preserved: **PASS** (user's question reaches VQA model)
- Answer quality: **FAIL** ("no idea" — model-domain limitation)

## 12. Actual Model Outputs

| Input | Raw BLIP Output | Cleaned Answer |
|-------|----------------|----------------|
| "Is there a road?" | "yes" | "yes" |
| "Describe the road." | "no idea" | "no idea" |
| "What is visible?" | "airplane" | "airplane" |
| "Is there a building?" | "yes" | "yes" |
| "Is there water?" | "i don ' t know" | "i don't know" |

## 13. Whether Road Recognition Works

**Partially.** BLIP VQA can answer "Is there a road?" with "yes" on this satellite image (with RS prefix). But it cannot describe, locate, or provide meaningful details about the road.

**Yes/no detection: WORKS** (with RS prefix)  
**Descriptive understanding: FAILS** (model-domain limitation)  
**Spatial localization: N/A** (would need GROUNDING model)

## 14. Remaining Limitations

1. **BLIP VQA is a general-purpose model** trained on COCO/VQA natural images, not satellite imagery. It cannot reliably:
   - Describe road conditions, width, or type
   - Identify building types or infrastructure
   - Distinguish water bodies from other features
   - Provide meaningful scene descriptions

2. **Confidence is uncalibrated** — token log-probabilities are not meaningful probability estimates. The system correctly flags `confidence_is_calibrated=False`.

3. **Model loading is slow** (~135s first load, ~5s cached on CPU).

4. **Test image is small** (256×256) — real satellite images may behave differently.

## 15. Recommended Next Step

**Immediate (hackathon demo):**
- The routing fix is complete and testable
- Use yes/no questions ("Is there a road?") for reliable demo interactions
- Avoid "Describe the road" type questions in the demo

**Post-hackathon (production):**
- Fine-tune BLIP VQA on BigEarthNet VQA dataset for satellite-specific understanding
- Or replace with a satellite-native model (e.g., SatCLIP, GeoVLM)
- Add confidence calibration against a validation set
- Implement structured VQA output (parsed labels + descriptions)

---

## Test Results Summary

| Test | Status |
|------|--------|
| Task classifier routing (60 tests) | ✅ 60/60 pass |
| Direct BLIP VQA (synthetic image) | ✅ Yes/no works |
| Direct BLIP VQA (real satellite) | ⚠️ Yes/no works, descriptions fail |
| Ablation (with/without RS prefix) | ✅ Prefix helps yes/no |
| End-to-end pipeline | ✅ Routing correct, model limitation identified |
| Mock mode enforcement | ✅ ALLOW_MOCK_MODE=false |
