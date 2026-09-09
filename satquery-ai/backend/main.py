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
from middleware.security import setup_security_middleware
from middleware.exception_handler import setup_exception_handlers
from middleware.logging_middleware import setup_logging_middleware
from services.cleanup import setup_cleanup_scheduler

# ── Logging ──────────────────────────────────────────────────────────────────
settings = get_settings()

# Set up structured logging
from utils.logging import setup_logging
perf_logger = setup_logging()
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

    # Preload real models in a background thread so the API remains responsive
    # while health reports the actual loading/failed/ready state.
    import threading

    def _load_models():
        try:
            registry.load_all(settings, lazy=settings.lazy_load_models)
            logger.info(
                "Model registry initialized — device: %s, status: %s",
                settings.resolved_device,
                registry.get_status(),
            )
        except Exception as exc:
            logger.warning("Model pre-loading failed: %s", exc)

    t = threading.Thread(target=_load_models, daemon=True)
    t.start()
    logger.info("Backend accepting requests — models loading in background")
    
    # Set up cleanup scheduler
    setup_cleanup_scheduler(app)

    yield

    logger.info("Shutting down SatQuery AI backend…")
    
    # Clean up any temporary files on shutdown
    try:
        from services.cleanup import get_cleanup_service
        service = get_cleanup_service()
        service.cleanup_temp_files()
        service.stop_scheduled_cleanup()
    except Exception as e:
        logger.warning(f"Failed to run cleanup on shutdown: {e}")


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SatQuery AI",
    description=(
        "Agentic Vision-Language Assistant for Multimodal Remote Sensing Image Analysis"
    ),
    version="1.0.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.redoc_enabled else None,
    lifespan=lifespan,
)

# ── Security Middleware ─────────────────────────────────────────────────────
setup_security_middleware(app, settings)

# ── Logging Middleware ──────────────────────────────────────────────────────
setup_logging_middleware(app)

# ── Exception Handlers ──────────────────────────────────────────────────────
setup_exception_handlers(app)

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
