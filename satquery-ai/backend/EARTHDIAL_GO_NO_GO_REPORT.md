# EARTHDIAL GO/NO-GO REPORT

**Date:** 2026-09-09  
**Model:** `akshaydudhane/EarthDial_4B_RGB`  
**Verdict:** **NO-GO**

---

## 1. Environment

| Attribute | Value |
|-----------|-------|
| CPU | AMD64, 16 cores, 2.9 GHz |
| RAM | 33.7 GB total, 19.0 GB available |
| OS | Windows 11 |
| Python | 3.13.7 |
| PyTorch | 2.6.0+**cpu** |
| transformers | 4.47.0 |
| GPU (physical) | NVIDIA RTX 3050 Laptop, 4 GB VRAM |

---

## 2. CUDA/GPU Verification

**nvidia-smi output:**
```
GPU  Name: NVIDIA GeForce RTX 3050 ... WDDM
VRAM: 434 MiB / 4096 MiB used
Driver: 610.88, CUDA UMD: 13.3
```

**PyTorch CUDA check:**
```
torch.cuda.is_available() = False
torch version = 2.6.0+cpu
```

**Diagnosis:** The GPU EXISTS and is functional (detected by nvidia-smi, 4GB VRAM). However, the current Python environment has **CPU-only PyTorch** installed (`torch 2.6.0+cpu`). To use the GPU, PyTorch must be reinstalled with CUDA support (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu121`).

**Impact on EarthDial:** Even with CUDA-enabled PyTorch, a 4B model in BF16 requires ~8GB VRAM. The RTX 3050 has only 4GB. EarthDial would NOT fit in GPU memory without aggressive quantization.

---

## 3. Model Availability

| Check | Result |
|-------|--------|
| HuggingFace repo exists? | **YES** — `akshaydudhane/EarthDial_4B_RGB` |
| Downloads | 825 |
| Likes | 2 |
| Model size | 4B params, BF16 |
| Files | config.json, 2× safetensors, tokenizer, inference.py |
| Custom code required? | **YES** — `auto_map` references external files |
| Model card / documentation? | **NO** — "No model card" on HuggingFace |

---

## 4. Verified Model Architecture

| Component | Value |
|-----------|-------|
| Architecture | `InternVLChatModel` |
| LLM backbone | `Phi-3-mini-128k-instruct` (32 layers, 3072 hidden, 32 heads) |
| Vision encoder | `InternViT-6B` variant (24 layers, 1024 hidden, image_size=448) |
| Total params | ~4B (3.8B LLM + vision encoder + MLP connector) |
| Image size | 448×448 (forced) |
| Tensor type | BF16 |
| License | Not specified in repo |

---

## 5. Verified Capabilities

| Capability | Claimed | Verified | Evidence |
|------------|---------|----------|----------|
| VQA | ✓ | **UNVERIFIED** | Model cannot load — no benchmark possible |
| Captioning | ✓ | **UNVERIFIED** | Model cannot load |
| Grounding | ✓ | **UNVERIFIED** | Model cannot load |
| RGB input | ✓ | **UNVERIFIED** | Model cannot load |
| SAR input | ✓ | **UNVERIFIED** | Model cannot load |
| Sentinel-2 | ✓ | **UNVERIFIED** | Model cannot load |

**None of EarthDial's capabilities can be verified because the model cannot be loaded.**

---

## 6. Claims From Previous Audit — Verification

| Claim | Evidence | Verified? |
|-------|----------|-----------|
| VQA ✓ | Model cannot load | **UNVERIFIED** |
| Captioning ✓ | Model cannot load | **UNVERIFIED** |
| Grounding ✓ | Model cannot load | **UNVERIFIED** |
| RGB ✓ | Model cannot load | **UNVERIFIED** |
| SAR ✓ | Model cannot load | **UNVERIFIED** |
| Sentinel-2 ✓ | Model cannot load | **UNVERIFIED** |
| ~4B parameters | Config confirms Phi-3 (3.8B) + InternViT | **YES** |
| CPU feasible | Cannot test — model won't load | **UNVERIFIED** |
| 8–10 GB RAM | Cannot test — model won't load | **UNVERIFIED** |
| 10–30s CPU inference | Cannot test — model won't load | **UNVERIFIED** |

---

## 7. CPU Benchmark

**NOT POSSIBLE.** The model cannot be loaded.

```
Attempt: AutoConfig.from_pretrained('akshaydudhane/EarthDial_4B_RGB', trust_remote_code=True)
Result:  OSError: akshaydudhane/EarthDial_4B_RGB does not appear to have a file
         named configuration_internvl_chat.py
```

The `config.json` specifies:
```json
"auto_map": {
    "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
    "AutoModel": "modeling_internvl_chat.InternVLChatModel"
}
```

These files (`configuration_internvl_chat.py`, `modeling_internvl_chat.py`) do **NOT** exist in the HuggingFace repo. They are expected to come from the `earthdial` Python package, which:
- Is **NOT** on PyPI
- Has **NO** public GitHub repository (404)
- Is **NOT** installed in the environment

**Without these files, the model cannot be loaded by transformers.**

---

## 8. GPU Benchmark

**NOT POSSIBLE.** The model cannot be loaded.

Additionally, even if it could load:
- RTX 3050 VRAM: 4 GB
- EarthDial BF16 size: ~8 GB
- **EarthDial would NOT fit in GPU memory**

---

## 9. Memory Benchmark

**NOT POSSIBLE.** The model cannot be loaded.

---

## 10. 12-Question Benchmark

**NOT POSSIBLE.** The model cannot be loaded.

---

## 11. BLIP vs EarthDial

**NOT POSSIBLE.** EarthDial cannot be loaded for comparison.

BLIP baseline (already measured):

| Question | BLIP Answer |
|----------|-------------|
| Is there a road? | yes |
| Describe the road. | airplane |
| Where is the road? | (N/A — BLIP cannot localize) |
| Is there a building? | yes |
| Describe the buildings. | high rise |
| Is there water? | i don't know |
| Describe the water body. | north pacific ocean |
| Is there vegetation? | no |
| Describe the vegetation. | no idea |
| What is visible? | airplane |
| Is this an urban area? | no |
| What type of land cover? | grass |

---

## 12. Road Description Test

**NOT POSSIBLE.** The model cannot be loaded.

---

## 13. Negative Tests

**NOT POSSIBLE.** The model cannot be loaded.

---

## 14. Resolution Tests

**NOT POSSIBLE.** The model cannot be loaded.

---

## 15. Multispectral/SAR Verification

**NOT POSSIBLE.** The model cannot be loaded.

---

## 16. Grounding Verification

**NOT POSSIBLE.** The model cannot be loaded.

---

## 17. Integration Requirements

Since the model cannot be loaded, integration requirements cannot be fully assessed. However, the following blockers are identified:

| Requirement | Status |
|-------------|--------|
| `earthdial` Python package | **NOT AVAILABLE** (not on PyPI, GitHub 404) |
| `configuration_internvl_chat.py` | **NOT IN REPO** |
| `modeling_internvl_chat.py` | **NOT IN REPO** |
| `earthdial.train.dataset.build_transform` | **UNAVAILABLE** |
| Custom tokenizer | Available in repo ✓ |
| Model weights | Available in repo ✓ |

**The model is fundamentally incomplete as a standalone HuggingFace model.**

---

## 18. Risks

| Risk | Probability | Impact |
|------|------------|--------|
| Model cannot load | **CONFIRMED** | **CRITICAL** — blocks everything |
| Package unavailable | **CONFIRMED** | **CRITICAL** — no workaround |
| GPU insufficient | **CONFIRMED** | HIGH — 4GB < 8GB required |
| Unknown quality | **CONFIRMED** | HIGH — cannot benchmark |
| No documentation | **CONFIRMED** | MEDIUM — cannot verify claims |

---

## 19. GO / NO-GO Decision

### GO Criteria Assessment

| Criterion | Required | Actual | Pass? |
|-----------|----------|--------|-------|
| 1. Model can load | YES | **NO** — missing custom code files | **FAIL** |
| 2. Processes RGB satellite image | YES | Cannot test | **FAIL** |
| 3. Meaningful "Describe the road" answer | YES | Cannot test | **FAIL** |
| 4. Materially better than BLIP | YES | Cannot test | **FAIL** |
| 5. No hallucination | YES | Cannot test | **FAIL** |
| 6. Acceptable latency | YES | Cannot test | **FAIL** |
| 7. Memory compatible | YES | Cannot test (likely fails on 4GB GPU) | **FAIL** |

### NO-GO Conditions Triggered

- ✗ Model cannot load
- ✗ Custom code package unavailable
- ✗ Cannot process any images
- ✗ No benchmark possible

---

## 20. Recommended Next Step

### Immediate: Switch to InternVL2-4B

The base architecture (`OpenGVLab/InternVL2-4B`) IS loadable:
- Same architecture as EarthDial (InternVLChatModel + Phi-3-mini)
- All custom modeling files present in repo
- 12,727 downloads, 57 likes, MIT license
- Can be loaded with `trust_remote_code=True`

**Caveat:** InternVL2-4B is NOT trained on remote sensing data. It may perform similarly to BLIP on satellite imagery. But at minimum it CAN be loaded and benchmarked.

### Alternative: Other RS Models

| Model | Loadable? | RS-trained? | VQA? | Notes |
|-------|-----------|------------|------|-------|
| EarthDial | **NO** | Yes | Yes | Missing custom code |
| GeoChat | Need to verify | Yes | Yes | 7B, needs GPU |
| RS-LLaVA | Need to verify | Yes | Yes | 7B, needs GPU |
| InternVL2-4B | **YES** | No | Yes | Can benchmark |
| RemoteCLIP | **YES** | Yes | No (encoder only) | Cannot answer questions |

### Recommended Action

1. **Benchmark InternVL2-4B** on the same 12-question test to establish whether a 4B VLM (even without RS training) outperforms BLIP
2. If InternVL2-4B works → consider it as a stepping stone
3. If InternVL2-4B also fails on RS → the problem is deeper than model choice, and fine-tuning or a purpose-built RS VLM is needed

---

## 21. Appendix: Why EarthDial Failed

The EarthDial HuggingFace repo is **incomplete**. It contains:
- ✗ Config (references missing files)
- ✗ Model weights (safetensors)
- ✗ Tokenizer
- ✗ Inference script
- ✗ Missing: `configuration_internvl_chat.py`
- ✗ Missing: `modeling_internvl_chat.py`
- ✗ Missing: `earthdial` package

The inference script imports:
```python
from earthdial.model.internvl_chat import InternVLChatModel
from earthdial.train.dataset import build_transform
```

Neither `earthdial.model` nor `earthdial.train` are available. The GitHub repo returns 404. PyPI has no `earthdial` package.

This appears to be a research model where the authors uploaded weights but not the complete codebase.

---

```
========================================
EARTHDIAL DECISION
========================================

Decision:
NO-GO

Road description:
NOT TESTED (model cannot load)

VQA:
NOT TESTED (model cannot load)

Scene description:
NOT TESTED (model cannot load)

Grounding:
NOT TESTED (model cannot load)

Remote-sensing support:
UNVERIFIED (claims cannot be tested)

CPU:
NOT VIABLE (model cannot load)

GPU:
NOT VIABLE (4GB VRAM < 8GB required)

Peak RAM:
UNMEASURED

Average inference:
UNMEASURED

Better than BLIP:
UNVERIFIED

Primary reason:
EarthDial REQUIRES a custom `earthdial` Python package that does not exist
on PyPI and has no public GitHub repository. The HuggingFace repo is missing
critical code files (configuration_internvl_chat.py, modeling_internvl_chat.py).
The model CANNOT be loaded by transformers.

Next action:
Benchmark OpenGVLab/InternVL2-4B (same architecture, loadable) to determine
whether a 4B VLM outperforms BLIP on satellite imagery, even without RS training.
========================================
```
