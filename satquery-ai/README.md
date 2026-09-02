# SatQuery AI

**Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis**

SatQuery AI is an intelligent system that enables natural language querying of satellite and aerial imagery. It combines state-of-the-art vision-language models with specialized remote sensing modules to support tasks like scene captioning, visual question answering, change detection, and SAR-optical fusion analysis.

---

## Features

- **Visual Question Answering (VQA)**: Ask natural language questions about satellite images
- **Scene Captioning**: Generate detailed descriptions of remote sensing scenes
- **Text-Guided Grounding**: Locate objects and regions described in natural language
- **Change Detection**: Identify and quantify changes between multi-temporal image pairs
- **SAR-Optical Fusion**: Combine synthetic aperture radar and optical imagery for enriched analysis
- **Agentic Orchestration**: Automatic task routing to the best specialist model
- **PDF Report Generation**: Downloadable analysis reports with visual evidence
- **BigEarthNet Fine-tuning**: Scripts to adapt CLIP for remote sensing scene understanding

---

## Architecture

```
User Query + Image(s)
        │
        ▼
  Input Validator ──→ modality detection, format check
        │
        ▼
  Task Classifier ──→ VQA / Captioning / Grounding / Change / SAR-Fusion
        │
        ▼
  Agentic Controller ──→ plan → execute → integrate
        │
   ┌────┴────┐
   ▼         ▼
Specialist  Report
 Models    Generator
```

### Specialist Models

| Module | Base Model | Task |
|--------|-----------|------|
| VQA | BLIP-2 / InstructBLIP | Remote sensing Q&A |
| Captioning | BLIP-2 / GIT | Scene description |
| Grounding | OWL-ViT / GroundingDINO | Object localization |
| Change Detection | Siamese ResNet features | Temporal change analysis |
| SAR Fusion | Feature concatenation + VQA | Multi-modal analysis |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU recommended (CUDA 11.8+)
- 16 GB RAM minimum

### Run with Docker

```bash
git clone https://github.com/yourorg/satquery-ai
cd satquery-ai
cp .env.example .env
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Run Locally (Development)

**Backend:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## API Reference

### POST /api/upload
Upload an image for analysis.

```json
Request: multipart/form-data  { file: <image> }
Response: { "image_id": "uuid", "modality": "optical", "shape": [512, 512, 3], "valid": true }
```

### POST /api/analyze
Run agentic analysis on uploaded image(s).

```json
{
  "image_ids": ["uuid1"],
  "query": "What land cover types are visible in this image?"
}
```

### GET /api/report/{session_id}
Download PDF analysis report.

---

## BigEarthNet Fine-tuning

```bash
cd backend
python training/finetune_clip.py \
  --data-dir /path/to/BigEarthNet \
  --output-dir ./checkpoints \
  --epochs 10 \
  --batch-size 64 \
  --model-name ViT-B-32
```

---

## Supported Image Formats

- GeoTIFF (`.tif`, `.tiff`) — with full geospatial metadata support
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- Multi-band images (Sentinel-1, Sentinel-2, Landsat)

---

## Environment Variables

See `.env.example` for all configuration options.

---

## License

MIT License — see LICENSE for details.
