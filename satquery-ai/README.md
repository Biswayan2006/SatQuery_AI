# SatQuery AI

**An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries**

> Developed for the ISRO/SAC Smart India Hackathon 2026 Challenge

---

## Overview

SatQuery AI is an agentic vision-language system that lets users query satellite and aerial imagery in plain English. Instead of relying on a single generic model, it automatically selects and executes the right specialist model based on the query and input images — returning evidence-grounded answers, visual outputs, and downloadable reports.

Most existing remote sensing AI tools are built for a single predefined task. SatQuery AI breaks that pattern with a query-driven agentic framework that handles:

- Single optical/multispectral or SAR image analysis
- Bi-temporal image pairs for change detection and change-based Q&A
- Co-registered optical–SAR pairs for cross-modal fusion analysis

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Visual Question Answering** | Ask natural language questions about a satellite image |
| **Scene Captioning** | Generate detailed land-cover descriptions |
| **Text-Guided Grounding** | Locate objects and regions described in a query |
| **Change Detection** | Detect, quantify, and describe changes between two dates |
| **SAR–Optical Fusion** | Extract complementary information from radar + optical pairs |
| **Agentic Orchestration** | Automatic task routing — no manual model selection needed |
| **Execution Trace** | Auditable log of selected task, models used, and parameters |
| **PDF Reports** | Downloadable analysis reports with visual evidence |
| **BigEarthNet Adaptation** | CLIP fine-tuned on BigEarthNet for remote sensing representations |

---

## Architecture

```
User Query + Image(s)
        │
        ▼
  Input Validator
  (modality detection, format & compatibility check)
        │
        ▼
  Task Classifier
  (VQA / Captioning / Grounding / Change VQA / Change Description / SAR Fusion)
        │
        ▼
  Agentic Controller
  (plan → execute → integrate)
        │
   ┌────┴──────────┐
   ▼               ▼
Specialist      Report
 Models        Generator
```

### Specialist Models

| Module | Base Model | Task |
|--------|-----------|------|
| VQA | `Salesforce/blip-vqa-base` | Remote sensing Q&A |
| Captioning | `Salesforce/blip-image-captioning-base` | Scene description |
| Grounding | `google/owlvit-base-patch32` | Object localization |
| Change Detection | Siamese ResNet-50 features | Temporal change analysis |
| SAR Fusion | Dual ResNet-50 + MLP fusion | Cross-modal analysis |
| RS Adaptation | OpenCLIP fine-tuned on BigEarthNet | Image-text alignment |

---

## Supported Inputs

| Input Type | Description |
|-----------|-------------|
| Single image | Optical, multispectral, or SAR — for VQA, captioning, grounding |
| Bi-temporal pair | Two images of the same area at different times — for change analysis |
| SAR–Optical pair | Co-registered radar + optical — for fusion analysis |

**Formats:** GeoTIFF / TIFF (with geospatial metadata), PNG, JPEG

---

## Representative Queries

```
"Describe the land-cover and major objects visible in this image."
"Highlight the water body referred to in the query."
"What changed between these two dates, and where did the change occur?"
"Use the optical and SAR images together to identify built-up and water-covered regions."
"Has the built-up area increased, decreased, or remained unchanged?"
```

---

## Project Structure

```
satquery-ai/
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Pydantic settings
│   ├── requirements.txt
│   ├── agent/
│   │   ├── controller.py        # Agentic orchestration (plan→execute→integrate)
│   │   ├── task_classifier.py   # Query → TaskType classification
│   │   ├── input_validator.py   # Image validation & modality detection
│   │   └── report_generator.py  # PDF report generation
│   ├── models/
│   │   ├── registry.py          # Thread-safe model registry
│   │   ├── vqa_model.py         # BLIP VQA
│   │   ├── captioning_model.py  # BLIP captioning
│   │   ├── grounding_model.py   # OWL-ViT grounding
│   │   ├── change_model.py      # Siamese change detection
│   │   ├── sar_fusion_model.py  # SAR-optical fusion
│   │   └── _mock.py             # Graceful fallback during model loading
│   ├── training/
│   │   ├── bigearthnet_dataset.py  # BigEarthNet PyTorch dataset
│   │   ├── finetune_clip.py        # OpenCLIP fine-tuning script
│   │   └── evaluate.py             # Benchmark evaluation
│   └── utils/
│       ├── image_utils.py       # GeoTIFF/PIL loading, RGB composites
│       ├── visualization.py     # Change maps, overlays, fusion viz
│       └── geo_utils.py         # Geospatial utilities
├── frontend/
│   └── src/
│       ├── app/page.tsx         # Main three-panel UI
│       ├── components/          # ImageUpload, QueryInput, ResultDisplay, etc.
│       └── hooks/useAnalysis.ts # API hooks
├── notebooks/
│   ├── 01_bigearthnet_exploration.ipynb
│   ├── 02_clip_finetuning.ipynb
│   └── 03_benchmark_evaluation.ipynb
├── docker-compose.yml
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.10–3.13
- Node.js 18+
- 8 GB RAM minimum (16 GB recommended)
- NVIDIA GPU optional (CUDA 12.1) — CPU works but is slower

### Option 1 — Docker (recommended)

```bash
git clone https://github.com/yourorg/satquery-ai
cd satquery-ai
cp .env.example .env
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2 — Local Development

**Backend** (Terminal 1):

```bash
cd satquery-ai/backend
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# Install PyTorch first (CPU)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# Or CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Copy env config
cp ../.env.example .env

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Frontend** (Terminal 2):

```bash
cd satquery-ai/frontend
npm install
npm run dev
```

> **Note:** Models download from HuggingFace on first run (~3 GB total). The server starts immediately and returns mock responses until models are ready. Check `GET /api/health` — `models_loaded: 5` means all models are ready.

---

## API Reference

### `POST /api/upload`
Upload a single image.
```json
// Response
{
  "image_id": "uuid",
  "modality": "optical",
  "shape": [512, 512, 3],
  "valid": true,
  "message": "Image validated"
}
```

### `POST /api/analyze`
Run agentic analysis on 1 or 2 uploaded images.
```json
// Request
{
  "image_ids": ["uuid1", "uuid2"],
  "query": "What changed between these two images?"
}

// Response
{
  "task": "CHANGE_VQA",
  "answer": "Approximately 23% of the scene changed...",
  "confidence": 0.84,
  "change_map": "<base64 PNG>",
  "execution_summary": {
    "selected_task": "CHANGE_VQA",
    "models_used": ["ChangeDetectionModel", "RemoteSensingVQA"],
    "processing_time_ms": 1840.2
  }
}
```

### `GET /api/report/{session_id}`
Download the PDF analysis report.

### `GET /api/health`
Returns service status and number of models loaded.

### `GET /api/models`
Lists all registered models and their load status.

---

## BigEarthNet Fine-tuning

```bash
cd satquery-ai/backend
python training/finetune_clip.py \
  --data-dir /path/to/BigEarthNet \
  --output-dir ./checkpoints \
  --epochs 10 \
  --batch-size 64 \
  --model-name ViT-B-32
```

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `VQA_MODEL_NAME` | `Salesforce/blip-vqa-base` | HuggingFace model ID for VQA |
| `CAPTIONING_MODEL_NAME` | `Salesforce/blip-image-captioning-base` | HuggingFace model ID for captioning |
| `GROUNDING_MODEL_NAME` | `google/owlvit-base-patch32` | HuggingFace model ID for grounding |
| `MODEL_CACHE_DIR` | `./model_cache` | Where to cache downloaded models |
| `UPLOAD_DIR` | `./uploads` | Where uploaded images are stored |
| `MAX_IMAGE_SIZE_MB` | `50` | Maximum upload size |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed frontend origins |

---

## Evaluation Benchmarks

| Benchmark | Task | Split |
|-----------|------|-------|
| RSVQA | Single-image VQA | Test |
| VRSBench | Captioning & Grounding | Test |
| CDVQA | Change-based VQA | Test |
| ISRO/SAC dataset | Cartosat-2S + RISAT SAR pairs | Evaluation set |

---

## License

MIT License — see [LICENSE](LICENSE) for details.
