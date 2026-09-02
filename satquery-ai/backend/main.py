"""
SatQuery AI — FastAPI Application Entry Point
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from config import get_settings
from api.routes import router
from models.registry import ModelRegistry

# ── Logging ──────────────────────────────────────────────────────────────────
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger("satquery")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models in background on startup; clean up on shutdown."""
    logger.info("Starting SatQuery AI backend…")
    registry = ModelRegistry.get_instance()

    # Store registry immediately so health/upload endpoints work right away
    app.state.registry = registry
    app.state.settings = settings

    # Load models in a background thread so uvicorn starts accepting
    # requests immediately (models will be mock until loaded)
    import threading

    def _load_models():
        try:
            registry.load_all(settings)
            logger.info("Model registry initialized — device: %s", settings.resolved_device)
        except Exception as exc:
            logger.warning("Model pre-loading failed: %s", exc)

    t = threading.Thread(target=_load_models, daemon=True)
    t.start()
    logger.info("Backend ready — models loading in background…")

    yield

    logger.info("Shutting down SatQuery AI backend…")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (uploaded images, generated reports) ────────────────────────
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")
app.mount("/reports", StaticFiles(directory=settings.reports_dir), name="reports")

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api")


# ── Root / Health (top-level shortcut) ───────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"service": "SatQuery AI", "status": "running", "version": "1.0.0"})


# ── Run guard ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=(settings.environment == "development"),
        log_level=settings.log_level.lower(),
    )
