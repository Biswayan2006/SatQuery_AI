# Backend Deployment Reference

The repository root [DEPLOYMENT.md](../DEPLOYMENT.md) is the authoritative DigitalOcean guide. This file records backend-specific choices.

## Runtime

The production image is based on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`, installs `requirements.txt`, runs as a non-root user, and starts one Uvicorn worker:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Do not use `--reload` in production. Do not increase workers for GPU inference without measuring duplicated model memory.

## Volumes

Mount persistent volumes at `/app/data/uploads`, `/app/data/reports`, and `/app/data/cache`. Set `UPLOAD_DIR`, `REPORTS_DIR`, `TEMP_DIR`, and `MODEL_CACHE_DIR` through the environment. Never mount a host source tree over `/app` in production.

## Required production configuration

Set `ENVIRONMENT=production`, `SECRET_KEY`, `API_KEY_ENABLED=true`, JSON-array `API_KEYS` and `ADMIN_API_KEYS`, exact `CORS_ORIGINS`, `TRUSTED_HOSTS`, `DEVICE`, and `ALLOW_MOCK_MODE=false`. Do not place secrets in Compose files, Dockerfiles, source code, or documentation.

## Health and model readiness

`GET /api/health` checks storage, GPU information, registry state, and inference queue. During background loading it reports `loading`; failed models are visible as `failed`; HTTP 503 is returned when every registered model has failed. `GET /api/models` provides per-model source, task, device, and readiness.

## Development

Use `Dockerfile.dev` or `requirements-dev.txt` only for development. Development reload and explicit mock mode must never be copied into the production environment.
