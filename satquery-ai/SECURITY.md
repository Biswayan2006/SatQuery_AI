# Security

## Required production settings

- Set `ENVIRONMENT=production`, a long random `SECRET_KEY`, `API_KEY_ENABLED=true`, and non-placeholder `API_KEYS`/`ADMIN_API_KEYS`.
- Set `CORS_ORIGINS` to exact trusted origins. Do not use `*` with credentials.
- Keep `ALLOW_MOCK_MODE=false` in production.
- Put the API behind HTTPS and a reverse proxy. Restrict firewall ingress to SSH and the proxy ports.

## Uploads and runtime data

Uploads use UUID storage names, extension and content checks, size limits, and canonical path containment. Keep `/app/data/uploads`, `/app/data/reports`, `/app/data/temp`, and `/app/data/cache` on private mounted volumes. Do not serve these volumes directly from the host.

## Secrets and artifacts

Never commit `.env`, API keys, Hugging Face tokens, model weights, datasets, reports, uploads, logs, or virtual environments. Rotate credentials if they are ever exposed. Container logs omit API keys and uploaded image contents, but operational logs should still be access-controlled.

## Operational controls

Use the configured request rate limit and inference queue. Keep one worker for GPU deployments. Review health status, failed model loads, and container restarts. Run dependency and image vulnerability scans as part of CI before public deployment.
