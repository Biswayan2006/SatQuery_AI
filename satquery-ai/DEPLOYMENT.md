# DigitalOcean Deployment

## Recommended topology

Use a DigitalOcean Droplet for CPU inference or a GPU Droplet for the larger VQA/grounding/fusion models:

Internet -> HTTPS reverse proxy -> one SatQuery FastAPI container -> persistent data/model-cache volumes.

The GPU deployment should use one Uvicorn worker and `MAX_CONCURRENT_INFERENCE=1` initially. Multiple workers can duplicate model weights in GPU memory.

## Requirements

Use at least 8 vCPUs/16 GB RAM for a serious CPU deployment; select more RAM for the chosen model set. GPU deployments must provide enough VRAM for the largest enabled model and local disk for model cache plus runtime data. Validate these numbers with the actual model identifiers before sizing the Droplet.

## Deploy

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin
sudo usermod -aG docker "$USER"
# Start a new login session after the group change
git clone <repository-url> satquery-ai
cd satquery-ai
cp .env.example .env
# Edit .env: secrets, API keys, CORS, DEVICE, and model settings
ufw allow OpenSSH
ufw allow 80/tcp
ufw allow 443/tcp
docker compose up --build -d
docker compose ps
curl -f http://127.0.0.1:8000/api/health
```

Mount or retain the named `uploads_data`, `reports_data`, and `model_cache` volumes. Back up reports only when required by policy; treat uploads as sensitive user data. Do not copy model caches into Git or the image.

## HTTPS and health checks

Terminate TLS with Caddy or Nginx and proxy to `127.0.0.1:8000`. Set a request body limit at least as large as `MAX_IMAGE_SIZE_MB`, and use long upstream read/send timeouts for inference. Configure the proxy and DigitalOcean health check to call `/api/health`; `200` means operational or still starting, while `503` means every registered model failed.

## Restart and operations

The Compose service uses `restart: unless-stopped`. Inspect `docker compose logs --tail=200 backend`, monitor disk/GPU/RAM, and alert on repeated restarts, failed model status, or cleanup errors. Keep Docker and the host patched. App Platform is not the default for persistent GPU inference; consider it only after validating its resource, timeout, and persistent-storage limits.

## Smoke test

Set `SATQUERY_API_URL` and run:

```bash
python scripts/smoke_test.py path/to/small-valid-image.png
```

The upload/analyze/report steps require a usable model registry and a valid API key when authentication is enabled. Without model weights, those steps are `NOT TESTED`, not passing.
