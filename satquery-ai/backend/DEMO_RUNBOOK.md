# SatQuery AI — Hackathon Demo Runbook

> **DO NOT** modify any code after reading this runbook. This document reflects the **verified** state of the system as of the final smoke test. Do not change model architecture, routing logic, demo queries, or enable mock mode.

---

## 1. Start Backend

```powershell
cd D:\PROJECTS\SIH2026\satquery-ai\backend
.\venv\Scripts\Activate.ps1
python main.py
```

**Expected:** Uvicorn server starts on `http://127.0.0.1:8000`. Models load lazily (first inference triggers load). Wait for the "Application startup complete" log line before querying.

---

## 2. Start Frontend

Open a second terminal:

```powershell
cd D:\PROJECTS\SIH2026\satquery-ai\frontend
npm run dev
```

**Expected:** Next.js dev server starts on `http://localhost:3000`. Open this URL in the browser.

---

## 3. Verify Backend

```powershell
Invoke-RestMethod -Uri http://localhost:8000/api/health
```

**Expected:**
- `status: healthy`
- `models_loaded: 0` (lazy load; will increase after first inference)
- `upload_dir_writable: True`

---

## 4. Demo Image

Use the **verified Sentinel optical image** of London (1470x951 pixels, 4-band GeoTIFF converted to PNG).

**File:** Any optical satellite image with clear urban features works. For consistent results, use the image from `_audit_fixtures/opt_a.png` (256x256 synthetic) or upload a real Sentinel/Landsat scene.

**Upload via UI:** Drag-and-drop onto the upload zone in the "Analyse Data" card. Wait for the green "Optical Image" preview tile to appear with dimensions and file size.

**Upload via API (alternative):**
```powershell
$form = @{ file = Get-Item "path\to\your\image.png" }
Invoke-RestMethod -Uri "http://localhost:8000/api/upload" -Method POST -Form $form
```

Note the `image_id` from the response — use it for API-based queries.

---

## 5. Demo Query 1 — "Describe this image"

| Field | Value |
|-------|-------|
| **Query** | `Describe this image` |
| **Expected route** | `CAPTIONING` |
| **Model used** | `RemoteSensingCaptioning` (BLIP, beams=5) |
| **Expected answer** | Scene description starting with "a satellite image showing…" |
| **Tool evidence** | `spectral_statistics` — band-wise mean, std, min, max, percentiles for blue/green/red/nir |
| **Inference time** | ~6-8 seconds on CPU |
| **Confidence** | ~0.22 (low — flagged with `[Verification Recommended]` prefix) |

**What to tell judges:** The classifier detected an open-ended scene description intent. It dispatched to the captioning specialist with an RS-aware prefix. Simultaneously, the deterministic tool layer computed per-band spectral statistics — numbers the model must not fabricate. The confidence framework correctly flagged this as requiring manual verification.

---

## 6. Demo Query 2 — "Is this a rural or urban area?"

| Field | Value |
|-------|-------|
| **Query** | `Is this a rural or urban area?` |
| **Expected route** | `SINGLE_VQA` |
| **Model used** | `RemoteSensingVQA` (BLIP, beams=4) |
| **Expected answer** | `urban` |
| **Tool evidence** | `null` (no spectral index matches rural/urban keyword intent) |
| **Inference time** | ~4-5 seconds on CPU |
| **Confidence** | ~0.21 (low — flagged with `[Verification Recommended]` prefix) |

**What to tell judges:** The classifier recognized a binary classification question about settlement density. It routed to the VQA specialist with an RS context prefix that primes the model for satellite imagery. The answer "urban" is concise and correct for the London scene.

---

## 7. Demo Query 3 — "Classify the land cover types"

| Field | Value |
|-------|-------|
| **Query** | `Classify the land cover types` |
| **Expected route** | `LAND_COVER_CLASSIFICATION` |
| **Model used** | `RSCLIPEncoder` (OpenCLIP ViT-B/32, zero-shot) |
| **Expected answer** | `dense urban built-up area (RS-CLIP score: 0.63)` with top candidate classes |
| **Tool evidence** | `null` (RS-CLIP handles classification end-to-end) |
| **Inference time** | ~0.3-0.5 seconds on CPU (fastest query) |
| **Task confidence** | 0.99 (high — classifier is certain about the task type) |

**What to tell judges:** The classifier detected a land cover classification intent. Instead of a generative model, it dispatched to RS-CLIP — a remote-sensing specialist trained on satellite data. RS-CLIP performs zero-shot classification against predefined land cover classes using CLIP's contrastive vision-language matching. The result includes per-class similarity scores, providing transparent evidence.

---

## 8. What to Show Judges

### Agentic Routing
- **TaskClassifier** reads the query, extracts entity keywords from a 60+ term lexicon, and selects one of 7 task types with a confidence score.
- Show the **Execution Trace** panel (terminal-style log) — it reveals the classifier decision, step sequence, and per-stage timing.

### Specialist Models
- Each task type maps to a dedicated model: BLIP VQA for questions, BLIP Captioning for scene descriptions, RS-CLIP for land cover classification.
- Show the **Model Registry** panel — it displays loaded/loaded counts and device info.

### Deterministic Tool Evidence
- After model inference, the **tool planner** runs spectral analysis (NDVI, NDWI, NDBI, band statistics) on the actual pixel data.
- These are **numbers the model cannot fabricate** — they come from numpy operations on the raw raster.
- Show the **Spectral Tool Evidence** section in the result card.

### Confidence Framework
- Each response includes `confidence`, `uncertainty`, `requires_verification`, and `is_degraded` fields.
- Low-confidence answers are prefixed with `[Verification Recommended]` — the system abstains rather than asserts.
- Show the confidence bar in the result card.

### Execution Trace
- The **Audit Trail** panel shows every step: classifier decision → model inference → tool execution → completion.
- Timing is per-step, showing where CPU time goes.

### PDF Report
- Click **Download Report** to generate a session PDF with the image, query, answer, confidence, and evidence.
- The PDF is generated server-side by `fpdf2`.

---

## 9. What NOT to Demonstrate

These features are **unreliable** or **untrained** and will produce poor results:

| Feature | Why it fails | Safe alternative |
|---------|-------------|-----------------|
| **OWL-ViT satellite grounding/counting** | Returns 0 detections on satellite imagery (validated in benchmark) | Skip counting queries entirely |
| **SAR-Optical Fusion** | Adapter is untrained (placeholder only) | Only demonstrate `Is this a SAR image?` → "yes" as a binary check |
| **Complex change detection** | Requires two georeferenced images; single-image queries give meaningless results | Skip unless both images are prepared and validated |
| **Open-ended BLIP VQA** (e.g., "Describe the buildings") | BLIP hallucinates or gives generic answers on satellite data | Stick to yes/no or binary classification queries |
| **GeoTIFF coordinate display** | Some images lack geo metadata; the warning is shown but may confuse judges | Use pre-validated images |

### Known Limitations to Acknowledge If Asked
- BLIP is a general vision-language model, not RS-specific — confidence scores are honest about this.
- No GPU available (CPU-only inference) — latency is 4-8 seconds per query instead of <1 second.
- 6 registered models, 2 actively used in the demo flow (BLIP VQA, BLIP Captioning, RS-CLIP).

---

## 10. Emergency Troubleshooting

### Backend won't start
```powershell
# Check if port 8000 is already in use
netstat -ano | Select-String ":8000"
# Kill the existing process (replace PID)
taskkill /PID <PID> /F
# Retry
python main.py
```

### Frontend can't reach backend
```powershell
# Verify backend is responding
Invoke-RestMethod -Uri http://localhost:8000/api/health
# Check CORS: frontend must be on localhost:3000 or localhost:3001
# (already configured in .env: CORS_ORIGINS=http://localhost:3000,http://localhost:3001)
```

### Models stuck loading
```powershell
# Check model registry status
Invoke-RestMethod -Uri http://localhost:8000/api/models
# If models show "registered" but not "loaded", trigger by sending any analysis request
# First inference will load the model (~10-15 seconds on CPU)
```

### Upload fails
```powershell
# Verify uploads directory is writable
Test-Path "D:\PROJECTS\SIH2026\satquery-ai\backend\uploads"
# Check disk space
Get-PSDrive C | Select-Object Used, Free
```

### PDF generation fails
```powershell
# Verify reports directory exists
Test-Path "D:\PROJECTS\SIH2026\satquery-ai\backend\reports"
# Check fpdf2 is installed
& "D:\PROJECTS\SIH2026\satquery-ai\backend\venv\Scripts\python.exe" -c "import fpdf; print(fpdf.__version__)"
```

### Nuclear option — full restart
```powershell
# Kill all Python and Node processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force
# Clear temp files if needed
Remove-Item -Recurse -Force "D:\PROJECTS\SIH2026\satquery-ai\backend\temp\*" -ErrorAction SilentlyContinue
# Restart
cd D:\PROJECTS\SIH2026\satquery-ai\backend; .\venv\Scripts\Activate.ps1; python main.py
# (new terminal)
cd D:\PROJECTS\SIH2026\satquery-ai\frontend; npm run dev
```

---

## Smoke Test Results (verified at freeze time)

| Check | Result |
|-------|--------|
| Backend health endpoint | ✅ `healthy`, 6 models registered |
| Mock mode disabled | ✅ `ALLOW_MOCK_MODE=false` |
| Frontend build | ✅ `npm run build` passes, 8 pages generated |
| Upload verified image | ✅ Optical 1470x951 accepted, `modality: optical` |
| Query 1: "Describe this image" | ✅ CAPTIONING → BLIP → spectral_statistics evidence |
| Query 2: "Is this a rural or urban area?" | ✅ SINGLE_VQA → BLIP → "urban" |
| Query 3: "Classify the land cover types" | ✅ LAND_COVER_CLASSIFICATION → RS-CLIP → "dense urban built-up area" |
| Tool evidence rendered | ✅ Spectral band statistics displayed in UI |
| PDF generation | ✅ Session PDF generated and downloadable |
| Example queries in UI | ✅ All 3 verified queries present in CommandBar suggestions |

**FINAL STATUS = DEMO FROZEN**
