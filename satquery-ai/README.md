# SatQuery AI

SatQuery AI is an agentic vision-language API for multimodal remote-sensing image analysis. It accepts PNG, JPEG, TIFF, and GeoTIFF inputs, routes queries to specialist models and deterministic tools, and can produce visual evidence and PDF reports.

## Features

- Image upload with content validation and modality detection
- VQA, captioning, grounding, change analysis, and SAR-optical fusion
- Agentic task routing with execution traces and confidence reporting
- Geospatial alignment and geographic change regions where metadata permits
- Model registry with lazy/background loading and explicit readiness states
- PDF reports, health checks, structured errors, rate limiting, and API-key auth

## Architecture

HTTP request -> FastAPI -> routes -> AgenticController -> task classifier -> model registry/tools -> AnalysisResponse -> PDF report.

The backend uses one Uvicorn worker by default. Large models are loaded once per process and inference concurrency is bounded; extra workers can duplicate model memory and are not recommended for GPU deployments.

## Supported Inputs

PNG, JPEG, TIFF, and GeoTIFF. Single images support optical, multispectral, and SAR workflows. Two images support bi-temporal change analysis and compatible SAR-optical fusion. Uploaded data and reports belong in external runtime volumes, not Git.

## Supported Tasks

VQA, image captioning, text-guided grounding, change detection, change VQA/change description, and SAR-optical fusion. The classifier may select a task automatically, or `task_hint` can be supplied.

## Models

Model identifiers and cache paths are environment-configurable. The registry reports `loading`, `ready`, or `failed` through `/api/models` and `/api/health`. Model weights are downloaded from Hugging Face or mounted from an external cache. `ALLOW_MOCK_MODE=false` is the production default; failed real models never silently become production inference.

## Remote-Sensing Adaptation

The repository includes RS-CLIP, SAR-fusion, VQA fine-tuning, training configs, evaluation runners, and notebooks. Datasets and checkpoints are external inputs configured by path and excluded from Git and Docker builds.

## API

With API-key authentication enabled, send `X-API-Key` on non-health requests.

- `POST /api/upload`: multipart upload under configured size and format limits
- `POST /api/analyze`: analyze one or two uploaded `image_ids`
- `GET /api/report/{session_id}`: download a generated PDF report
- `GET /api/health`: readiness, storage, GPU, registry, and queue status
- `GET /api/models`: model IDs, tasks, devices, source, and readiness
- `/docs` and `/redoc`: OpenAPI documentation

Errors use `{ "error": { "code": "...", "message": "..." }, "request_id": "..." }`.

## Installation

Copy `.env.example` to `.env` and replace every placeholder. For local development:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Optional: copy backend/.env.example to backend/.env for local paths/auth defaults
uvicorn main:app --host 0.0.0.0 --port 8000
```

Install `requirements-dev.txt` for tests. Install `requirements-training.txt` or `requirements-evaluation.txt` only for those workflows. Keep datasets, checkpoints, uploads, reports, and model caches outside the repository.

## Environment Variables

Important settings include `ENVIRONMENT`, `SECRET_KEY`, `API_KEY_ENABLED`, `API_KEYS`, `ADMIN_API_KEYS`, `CORS_ORIGINS`, `UPLOAD_DIR`, `REPORTS_DIR`, `TEMP_DIR`, `MODEL_CACHE_DIR`, `MAX_IMAGE_SIZE_MB`, `DEVICE`, `ALLOW_MOCK_MODE`, `MAX_CONCURRENT_INFERENCE`, and the model-name variables. See `.env.example` for the complete template. List settings use JSON array syntax with pydantic-settings.

## Local Development

Use `backend/Dockerfile.dev` or a local Python environment. Development may set `API_KEY_ENABLED=false` and `ALLOW_MOCK_MODE=true`; those settings must not be copied to production.

## Docker

The production image uses `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, a single Uvicorn worker, a non-root user, and mounted runtime volumes. From the repository root:

```powershell
Copy-Item .env.example .env
# Edit .env and replace secrets/placeholders
docker compose up --build -d
curl http://localhost:8000/api/health
```

The frontend is optional; the backend is the deployment unit for API-only installations.

## DigitalOcean Deployment

Use a Droplet for CPU workloads or a GPU Droplet for BLIP/OWL-ViT workloads. App Platform is suitable only after validating memory, startup time, persistent model-cache behavior, and latency; it is not the default for persistent GPU inference. See [DEPLOYMENT.md](backend/DEPLOYMENT.md) for firewall, volumes, HTTPS, restart, health-check, and backup guidance.

## GPU Requirements

CPU mode is supported but slow. GPU memory requirements depend on selected models; BLIP-2 and fusion configurations require materially more memory than change detection or CLIP. Use one worker and `MAX_CONCURRENT_INFERENCE=1` unless load testing proves a larger safe value.

## Model Downloads

Models download into `MODEL_CACHE_DIR` on first load. In deployment, mount a persistent volume at `/app/data/cache`; do not bake weights into Git or the image. Private Hugging Face models use `HUGGINGFACE_TOKEN` supplied only through the deployment environment.

For local runs, `backend/.env.example` uses `./model_cache`, so downloaded weights remain in `backend/model_cache` across restarts. This directory is ignored by Git.

## Evaluation

Evaluation runners live under `backend/evaluation` and require external datasets. Install evaluation requirements and configure dataset paths; no benchmark dataset is bundled.

## Project Structure

`backend/` contains the FastAPI runtime, tests, training, and evaluation code. `frontend/` contains the Next.js client. `notebooks/` contains research workflows. `backend/requirements*.txt` separates runtime, development, training, and evaluation dependencies.

## Security

Use long random secrets, API keys, restricted CORS origins, HTTPS, upload limits, named runtime volumes, and a reverse proxy with appropriate request and inference timeouts. Do not commit `.env`, credentials, model files, datasets, uploads, reports, or logs. Review [SECURITY.md](backend/SECURITY.md) before exposing the service publicly.

## Limitations

Model quality and latency depend on selected weights, device, and remote-sensing adaptation. Geographic outputs require valid CRS/transform metadata. Full inference and clean Docker smoke tests require model downloads, compatible hardware, and a working Docker daemon.

## Troubleshooting

Check `/api/health` and `/api/models` first. `loading` means startup model initialization is still running; `failed` means a model or cache/device configuration needs attention. A `503` health response means no registered model is usable. Inspect structured container logs without exposing uploaded content or secrets.

## License

See [LICENSE](LICENSE) if present in the repository.
