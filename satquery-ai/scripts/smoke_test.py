"""Small deployment smoke test; requires httpx and a valid image for upload steps."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx


API_URL = os.getenv("SATQUERY_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("SATQUERY_API_KEY", "")


def headers() -> dict[str, str]:
    return {"X-API-Key": API_KEY} if API_KEY else {}


def main() -> int:
    image = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    with httpx.Client(base_url=API_URL, headers=headers(), timeout=300.0) as client:
        health = client.get("/api/health")
        print(f"health: {health.status_code} {health.json().get('status', 'unknown')}")
        models = client.get("/api/models")
        print(f"models: {models.status_code}")
        if image is None:
            print("upload/analyze/report: NOT TESTED - provide a valid image path")
            return 0 if health.status_code in (200, 503) and models.is_success else 1

        with image.open("rb") as handle:
            upload = client.post("/api/upload", files={"file": (image.name, handle)})
        print(f"upload: {upload.status_code}")
        upload.raise_for_status()
        image_id = upload.json()["image_id"]

        analysis = client.post(
            "/api/analyze",
            json={"image_ids": [image_id], "query": "Describe this image."},
        )
        print(f"analysis: {analysis.status_code}")
        analysis.raise_for_status()
        session_id = analysis.json()["session_id"]

        report = client.get(f"/api/report/{session_id}")
        print(f"report: {report.status_code}")
        report.raise_for_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
