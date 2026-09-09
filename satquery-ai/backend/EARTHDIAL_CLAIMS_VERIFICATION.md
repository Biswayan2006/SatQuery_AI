# EARTHDIAL_CLAIMS_VERIFICATION.md

**Date:** 2026-09-09  
**Model:** `akshaydudhane/EarthDial_4B_RGB`

---

## Claims Verification Table

| Claim | Evidence | Verified? |
|-------|----------|-----------|
| VQA ✓ | Model cannot load — `configuration_internvl_chat.py` missing from repo, `earthdial` package not on PyPI/GitHub | **UNVERIFIED** |
| Captioning ✓ | Model cannot load | **UNVERIFIED** |
| Grounding ✓ | Model cannot load | **UNVERIFIED** |
| RGB ✓ | Model cannot load | **UNVERIFIED** |
| SAR ✓ | Model cannot load | **UNVERIFIED** |
| Sentinel-2 ✓ | Model cannot load | **UNVERIFIED** |
| ~4B parameters | Config confirms Phi-3-mini (3.8B) + InternViT vision encoder | **YES** |
| CPU feasible | Model cannot load — cannot test | **UNVERIFIED** |
| 8–10 GB RAM | Model cannot load — cannot measure | **UNVERIFIED** |
| 10–30s CPU inference | Model cannot load — cannot measure | **UNVERIFIED** |

---

## Root Cause of Failure

The HuggingFace repo `akshaydudhane/EarthDial_4B_RGB` is **incomplete**:

1. `config.json` specifies `auto_map` pointing to:
   - `configuration_internvl_chat.InternVLChatConfig`
   - `modeling_internvl_chat.InternVLChatModel`

2. These Python files do **NOT** exist in the repo (only 11 files total, none are `.py` modeling files except `inference.py`)

3. The `inference.py` script imports from a package called `earthdial`:
   ```python
   from earthdial.model.internvl_chat import InternVLChatModel
   from earthdial.train.dataset import build_transform
   ```

4. The `earthdial` package:
   - Is **NOT** on PyPI (`pip show earthdial` → not found)
   - Has **NO** public GitHub repository (https://github.com/akshaydudhane/EarthDial → 404)
   - Is **NOT** installed in the environment

5. Attempting to load via `AutoConfig.from_pretrained('akshaydudhane/EarthDial_4B_RGB', trust_remote_code=True)` produces:
   ```
   OSError: akshaydudhane/EarthDial_4B_RGB does not appear to have a file
   named configuration_internvl_chat.py
   ```

**Conclusion:** All capability claims are UNVERIFIED because the model cannot be loaded. The claims cannot be confirmed or denied without the missing code.
