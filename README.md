# 🛰️ SatQuery AI

### **An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis**

<p align="center">

**Ask questions. Analyze satellite imagery. Detect change. Understand the Earth.**

</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge\&logo=fastapi\&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?style=for-the-badge\&logo=next.js\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=for-the-badge\&logo=huggingface\&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

<p align="center">

🏆 **Developed for the ISRO/SAC Smart India Hackathon 2026 Challenge**

</p>

---

## 🌍 What is SatQuery AI?

**SatQuery AI** is an **agentic vision-language system** that allows users to analyze satellite and aerial imagery using **plain natural-language queries**.

Instead of forcing users to manually select different AI models for different tasks, SatQuery AI acts as an intelligent orchestration layer:

```text
                  ┌─────────────────────────┐
                  │      USER QUERY         │
                  │  "What changed here?"   │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │    INPUT VALIDATOR      │
                  │  Modality & Validation  │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │     TASK CLASSIFIER     │
                  │  What does the user     │
                  │       actually need?    │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   AGENTIC CONTROLLER    │
                  │     PLAN → EXECUTE      │
                  │        → INTEGRATE      │
                  └────────────┬────────────┘
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        🧠 VQA Model     🛰️ Change Model   🎯 Grounding
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │   EVIDENCE-GROUNDED    │
                  │        ANSWER           │
                  │   + VISUAL EVIDENCE     │
                  │   + EXECUTION TRACE     │
                  │   + PDF REPORT          │
                  └─────────────────────────┘
```

### 💡 The Core Idea

Traditional remote-sensing AI systems are usually built around **one predefined task**.

SatQuery AI instead provides a **query-driven interface** where the user simply asks what they want to know.

> **One interface → Multiple specialist models → One intelligent analysis pipeline**

---

# ✨ Key Features

| Feature                          | Description                                                      |
| -------------------------------- | ---------------------------------------------------------------- |
| 🧠 **Visual Question Answering** | Ask natural-language questions about satellite imagery           |
| 📝 **Scene Captioning**          | Generate detailed descriptions of land-cover and visible objects |
| 🎯 **Text-Guided Grounding**     | Locate objects or regions described by the user                  |
| 🔄 **Change Detection**          | Detect and quantify changes between two dates                    |
| 💬 **Change-Based Q&A**          | Ask natural-language questions about temporal changes            |
| 🛰️ **SAR–Optical Fusion**       | Combine complementary information from radar and optical imagery |
| 🤖 **Agentic Orchestration**     | Automatically determine which specialist models are required     |
| 🔍 **Execution Trace**           | Maintain an auditable record of tasks, models and parameters     |
| 📄 **PDF Reports**               | Generate downloadable analysis reports with visual evidence      |
| 🌍 **Remote-Sensing Adaptation** | CLIP adaptation using BigEarthNet representations                |

---

# 🚀 Why SatQuery AI?

### Traditional Approach

```text
User
 │
 ├── Select VQA model
 ├── Select change detection model
 ├── Select SAR model
 ├── Configure parameters
 ├── Run analysis
 └── Interpret output
```

### SatQuery AI

```text
User
 │
 │  "Has the built-up area increased?"
 ▼
┌──────────────────────────┐
│      SatQuery AI         │
│                          │
│  Understands the query  │
│           ↓              │
│  Selects required model  │
│           ↓              │
│  Executes analysis       │
│           ↓              │
│  Integrates evidence     │
└────────────┬─────────────┘
             ▼
      Human-readable answer
             +
      Visual evidence
             +
      Confidence
             +
      Execution trace
```

---

# 🧠 AI Capabilities

## 1. 🔎 Visual Question Answering

Ask questions directly about a satellite image.

**Example:**

```text
"What major objects are visible in this image?"
```

The system analyzes the image and produces a natural-language response.

---

## 2. 📝 Scene Captioning

Generate a detailed description of the observed scene.

**Example:**

```text
"Describe the land-cover and major objects visible in this image."
```

Possible output:

```text
The scene contains a densely built-up region,
a large water body, surrounding vegetation,
and several linear transportation structures.
```

---

## 3. 🎯 Text-Guided Grounding

Users can describe an object or region in natural language.

```text
"Highlight the water body referred to in the query."
```

The grounding model identifies the relevant region and produces a visual localization.

---

## 4. 🔄 Change Detection

Given two images captured at different times:

```text
Image — T1                 Image — T2
    │                          │
    └──────────┬───────────────┘
               ▼
       Change Detection
               │
               ▼
          Change Map
               │
               ▼
      Quantified Changes
```

Example query:

```text
"What changed between these two dates,
and where did the change occur?"
```

The system can identify:

* Changed regions
* Approximate change percentage
* Spatial location
* Visual change map
* Natural-language explanation

---

## 5. 🛰️ SAR–Optical Fusion

Satellite sensors provide different types of information.

**Optical imagery** can provide:

* Color
* Vegetation information
* Visible structures
* Land-cover appearance

**SAR imagery** can provide:

* Radar backscatter
* Information under cloudy conditions
* Structural characteristics
* Complementary surface information

SatQuery AI combines these modalities for richer analysis.

Example:

```text
"Use the optical and SAR images together
to identify built-up and water-covered regions."
```

---

# 🤖 Agentic Architecture

SatQuery AI follows a **Plan → Execute → Integrate** architecture.

```text
                         USER
                           │
                           ▼
                ┌─────────────────────┐
                │   Input Validator   │
                │                     │
                │ • File validation   │
                │ • Modality detection│
                │ • Compatibility     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Task Classifier   │
                │                     │
                │ VQA                 │
                │ Captioning          │
                │ Grounding           │
                │ Change VQA          │
                │ Change Description   │
                │ SAR Fusion          │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Agentic Controller  │
                │                     │
                │ PLAN                │
                │   ↓                 │
                │ EXECUTE             │
                │   ↓                 │
                │ INTEGRATE           │
                └──────────┬──────────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
        🧠 VQA        🔄 CHANGE       🎯 GROUND
        MODEL          MODEL           MODEL
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Result Integrator  │
                └──────────┬──────────┘
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Answer      Visual Evidence   Report
```

---

# 🧩 Specialist Models

| Module              | Base Model / Architecture               | Purpose                             |
| ------------------- | --------------------------------------- | ----------------------------------- |
| 🧠 VQA              | `Salesforce/blip-vqa-base`              | Visual question answering           |
| 📝 Captioning       | `Salesforce/blip-image-captioning-base` | Scene description                   |
| 🎯 Grounding        | `google/owlvit-base-patch32`            | Text-guided object localization     |
| 🔄 Change Detection | Siamese ResNet-50                       | Temporal feature comparison         |
| 🛰️ SAR Fusion      | Dual ResNet-50 + MLP                    | Optical + SAR fusion                |
| 🌍 RS Adaptation    | OpenCLIP + BigEarthNet                  | Remote-sensing image-text alignment |

---

# 📡 Supported Inputs

SatQuery AI supports multiple remote-sensing input configurations.

### 🖼️ Single Image

```text
Optical / Multispectral / SAR
            │
            ▼
     VQA / Captioning
       / Grounding
```

### 🔄 Bi-Temporal Pair

```text
Image T1 ──────┐
               ├──► Change Analysis
Image T2 ──────┘
```

### 🛰️ SAR + Optical Pair

```text
Optical Image ──┐
                ├──► Fusion Analysis
SAR Image ──────┘
```

### Supported Formats

* GeoTIFF
* TIFF
* PNG
* JPEG

GeoTIFF/TIFF files can retain geospatial metadata for downstream geospatial processing.

---

# 💬 Example Queries

### Single Image

```text
"Describe the land-cover and major objects visible in this image."
```

### Grounding

```text
"Highlight the water body referred to in the query."
```

### Change Detection

```text
"What changed between these two dates,
and where did the change occur?"
```

### SAR + Optical

```text
"Use the optical and SAR images together
to identify built-up and water-covered regions."
```

### Change-Based Q&A

```text
"Has the built-up area increased,
decreased, or remained unchanged?"
```

---

# 📊 Example Analysis Response

```json
{
  "task": "CHANGE_VQA",
  "answer": "Approximately 23% of the scene changed...",
  "confidence": 0.84,
  "change_map": "<base64 PNG>",
  "execution_summary": {
    "selected_task": "CHANGE_VQA",
    "models_used": [
      "ChangeDetectionModel",
      "RemoteSensingVQA"
    ],
    "processing_time_ms": 1840.2
  }
}
```

The output is designed to be:

* ✅ Human-readable
* ✅ Evidence-grounded
* ✅ Auditable
* ✅ Machine-readable
* ✅ Exportable

---

# 📁 Project Structure

```text
satquery-ai/
│
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── requirements.txt
│   │
│   ├── agent/
│   │   ├── controller.py
│   │   ├── task_classifier.py
│   │   ├── input_validator.py
│   │   └── report_generator.py
│   │
│   ├── models/
│   │   ├── registry.py
│   │   ├── vqa_model.py
│   │   ├── captioning_model.py
│   │   ├── grounding_model.py
│   │   ├── change_model.py
│   │   ├── sar_fusion_model.py
│   │   └── _mock.py
│   │
│   ├── training/
│   │   ├── bigearthnet_dataset.py
│   │   ├── finetune_clip.py
│   │   └── evaluate.py
│   │
│   └── utils/
│       ├── image_utils.py
│       ├── visualization.py
│       └── geo_utils.py
│
├── frontend/
│   └── src/
│       ├── app/
│       │   └── page.tsx
│       ├── components/
│       │   ├── ImageUpload/
│       │   ├── QueryInput/
│       │   └── ResultDisplay/
│       └── hooks/
│           └── useAnalysis.ts
│
├── notebooks/
│   ├── 01_bigearthnet_exploration.ipynb
│   ├── 02_clip_finetuning.ipynb
│   └── 03_benchmark_evaluation.ipynb
│
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# ⚡ Quick Start

## Prerequisites

| Requirement |  Minimum |     Recommended |
| ----------- | -------: | --------------: |
| Python      |     3.10 |           3.11+ |
| Node.js     |      18+ |             20+ |
| RAM         |     8 GB |          16 GB+ |
| GPU         | Optional | NVIDIA CUDA GPU |
| Storage     |        — |          ~5 GB+ |

> **GPU is optional.** The system can run on CPU, although inference will be considerably slower.

---

## 🐳 Option 1 — Docker

```bash
git clone https://github.com/Biswayan2006/SatQuery_AI

cd satquery-ai

cp .env.example .env

docker compose up --build
```

Once running:

| Service     | URL                          |
| ----------- | ---------------------------- |
| 🌐 Frontend | `http://localhost:3000`      |
| ⚙️ Backend  | `http://localhost:8000`      |
| 📚 API Docs | `http://localhost:8000/docs` |

---

# 🛠️ Option 2 — Local Development

## Backend

```bash
cd satquery-ai/backend

python -m venv venv
```

### Windows

```powershell
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

### Install PyTorch — CPU

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

### Or CUDA 12.1

```bash
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

```bash
cp ../.env.example .env
```

### Start the backend

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Frontend

Open another terminal:

```bash
cd satquery-ai/frontend

npm install

npm run dev
```

---

# 🤗 Model Loading

On the first run, the required Hugging Face models are downloaded and cached locally.

```text
First launch
     │
     ▼
Download models
     │
     ▼
Cache locally
     │
     ▼
Load specialist models
     │
     ▼
Ready for inference
```

Approximately **3 GB of model data** may be downloaded depending on the configured models.

During model initialization, the backend can return **mock responses** so that the application remains usable while models are loading.

Check:

```http
GET /api/health
```

When:

```json
{
  "models_loaded": 5
}
```

the configured specialist models are ready.

---

# 🔌 API Reference

## `POST /api/upload`

Upload and validate an image.

### Response

```json
{
  "image_id": "uuid",
  "modality": "optical",
  "shape": [512, 512, 3],
  "valid": true,
  "message": "Image validated"
}
```

---

## `POST /api/analyze`

Run agentic analysis on one or two uploaded images.

### Request

```json
{
  "image_ids": [
    "uuid1",
    "uuid2"
  ],
  "query": "What changed between these two images?"
}
```

### Response

```json
{
  "task": "CHANGE_VQA",
  "answer": "Approximately 23% of the scene changed...",
  "confidence": 0.84,
  "change_map": "<base64 PNG>",
  "execution_summary": {
    "selected_task": "CHANGE_VQA",
    "models_used": [
      "ChangeDetectionModel",
      "RemoteSensingVQA"
    ],
    "processing_time_ms": 1840.2
  }
}
```

---

## `GET /api/report/{session_id}`

Generate and download the PDF analysis report.

---

## `GET /api/health`

Returns:

* Service status
* Model loading status
* Number of loaded models

---

## `GET /api/models`

Lists registered specialist models and their current loading status.

---

# 🧪 BigEarthNet Fine-Tuning

SatQuery AI includes infrastructure for adapting CLIP representations to remote-sensing imagery using **BigEarthNet**.

```bash
cd satquery-ai/backend

python training/finetune_clip.py \
  --data-dir /path/to/BigEarthNet \
  --output-dir ./checkpoints \
  --epochs 10 \
  --batch-size 64 \
  --model-name ViT-B-32
```

### Training Pipeline

```text
BigEarthNet
     │
     ▼
Dataset Loader
     │
     ▼
Remote-Sensing Image/Text Pairs
     │
     ▼
OpenCLIP
     │
     ▼
Fine-Tuning
     │
     ▼
Remote-Sensing Representation
     │
     ▼
Evaluation
```

---

# ⚙️ Environment Variables

Create a `.env` file from `.env.example`.

| Variable                | Default                                 | Description              |
| ----------------------- | --------------------------------------- | ------------------------ |
| `DEVICE`                | `auto`                                  | `auto`, `cuda`, or `cpu` |
| `VQA_MODEL_NAME`        | `Salesforce/blip-vqa-base`              | VQA model                |
| `CAPTIONING_MODEL_NAME` | `Salesforce/blip-image-captioning-base` | Captioning model         |
| `GROUNDING_MODEL_NAME`  | `google/owlvit-base-patch32`            | Grounding model          |
| `MODEL_CACHE_DIR`       | `./model_cache`                         | Model cache location     |
| `UPLOAD_DIR`            | `./uploads`                             | Uploaded image storage   |
| `MAX_IMAGE_SIZE_MB`     | `50`                                    | Maximum image size       |
| `CORS_ORIGINS`          | `http://localhost:3000`                 | Allowed frontend origin  |

---

# 📈 Evaluation Benchmarks

SatQuery AI is designed to support evaluation across multiple remote-sensing benchmarks.

| Benchmark            | Task                          | Split      |
| -------------------- | ----------------------------- | ---------- |
| **RSVQA**            | Single-image VQA              | Test       |
| **VRSBench**         | Captioning & Grounding        | Test       |
| **CDVQA**            | Change-based VQA              | Test       |
| **ISRO/SAC Dataset** | Cartosat-2S + RISAT SAR pairs | Evaluation |

---

# 🗺️ Intended Applications

SatQuery AI can serve as an experimental platform for applications such as:

* 🌆 Urban expansion monitoring
* 🌊 Water-body monitoring
* 🌾 Agricultural land analysis
* 🏗️ Infrastructure monitoring
* 🌳 Land-cover analysis
* 🛰️ Multi-modal satellite analysis
* 🔄 Temporal change assessment
* 🗺️ Geospatial intelligence workflows
* 📊 Evidence-based remote-sensing analysis

---

# 🔬 Research & Innovation

The project combines several research directions into one interactive system:

```text
             ┌────────────────────────┐
             │  Vision-Language Models│
             └────────────┬───────────┘
                          │
             ┌────────────▼───────────┐
             │ Remote Sensing AI      │
             └────────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Optical Vision     SAR Analysis    Temporal Analysis
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                ┌───────────────────┐
                │ Agentic Reasoning │
                └─────────┬─────────┘
                          ▼
                ┌───────────────────┐
                │ Natural Language   │
                │ Earth Observation  │
                └───────────────────┘
```

The central goal is to make complex remote-sensing analysis **accessible through natural language**, while retaining the ability to inspect the underlying models and execution process.

---

# 🛣️ Roadmap

### ✅ Current

* [x] Multi-modal image input
* [x] Input validation
* [x] Task classification
* [x] Specialist model registry
* [x] VQA pipeline
* [x] Captioning pipeline
* [x] Grounding pipeline
* [x] Change detection pipeline
* [x] SAR–optical fusion architecture
* [x] Execution trace
* [x] PDF report generation
* [x] BigEarthNet training infrastructure

### 🚧 Future

* [ ] Advanced multispectral band handling
* [ ] Stronger remote-sensing-specific VLM
* [ ] Improved change-detection models
* [ ] Geo-referenced result visualization
* [ ] Interactive map integration
* [ ] More robust SAR interpretation
* [ ] Model confidence calibration
* [ ] Larger benchmark evaluation
* [ ] ISRO/SAC dataset evaluation
* [ ] Production-scale inference optimization

---

# 👥 Team

### **SatQuery AI**

Built for the **Smart India Hackathon 2026 — ISRO/SAC Challenge**.

> **Turning satellite imagery into something you can simply ask questions about.**

---

# 📜 License

This project is released under the **MIT License**.

See [`LICENSE`](LICENSE) for details.

---

<p align="center">

### 🛰️ SatQuery AI

**Query the Earth. Understand the Change.**

Built with ❤️ for **Smart India Hackathon 2026**

</p>
