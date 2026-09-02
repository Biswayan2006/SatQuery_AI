"""
SatQuery AI — API Routes
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from api.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    HealthResponse,
    ImageUploadResponse,
    ModelInfo,
    ModelsListResponse,
    ReportResponse,
)
from config import Settings, get_settings

logger = logging.getLogger("satquery.routes")
router = APIRouter()


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_registry(request: Request):
    return request.app.state.registry


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=ImageUploadResponse,
    summary="Upload a satellite image",
    status_code=status.HTTP_201_CREATED,
)
async def upload_image(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Upload a single satellite image (GeoTIFF, PNG, JPEG).
    Returns an image_id that can be used in /analyze.
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {settings.allowed_extensions}",
        )

    # Check file size (read into memory in chunks)
    image_id = str(uuid.uuid4())
    dest_path = os.path.join(settings.upload_dir, f"{image_id}{suffix}")

    total_bytes = 0
    async with aiofiles.open(dest_path, "wb") as out_file:
        while chunk := await file.read(65536):
            total_bytes += len(chunk)
            if total_bytes > settings.max_image_bytes:
                await out_file.close()
                os.remove(dest_path)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum size of {settings.max_image_size_mb} MB",
                )
            await out_file.write(chunk)

    # Validate with InputValidator
    from agent.input_validator import InputValidator

    validator = InputValidator()
    result = validator.validate_image(dest_path)

    return ImageUploadResponse(
        image_id=image_id,
        modality=result.modality,
        shape=result.shape,
        bands=result.bands,
        valid=result.valid,
        message=result.message,
        filename=file.filename or "",
        file_size_kb=round(total_bytes / 1024, 2),
        is_geotiff=result.is_geotiff,
        crs=result.crs,
    )


# ── Analyze ───────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Run agentic analysis on uploaded image(s)",
)
async def analyze(
    request_body: AnalysisRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    registry=Depends(get_registry),
):
    """
    Accept 1–2 previously uploaded image IDs and a natural-language query.
    Runs the agentic pipeline and returns structured analysis results.
    """
    from agent.controller import AgenticController
    from agent.input_validator import InputValidator
    from utils.image_utils import load_image

    validator = InputValidator()
    images_data = []

    for img_id in request_body.image_ids:
        # Find the file on disk (any allowed extension)
        img_path = None
        for ext in settings.allowed_extensions:
            candidate = os.path.join(settings.upload_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image '{img_id}' not found. Upload it first via /api/upload.",
            )

        img_data = load_image(img_path)
        validation = validator.validate_image(img_path)
        img_data["image_id"] = img_id
        img_data["modality"] = validation.modality
        images_data.append(img_data)

    controller = AgenticController(registry=registry)

    try:
        response = await controller.analyze(
            images=images_data,
            query=request_body.query,
            task_hint=request_body.task_hint,
        )
    except Exception as exc:
        logger.exception("Analysis pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(exc)}",
        )

    # Persist report for later download
    try:
        from agent.report_generator import ReportGenerator

        gen = ReportGenerator()
        pdf_bytes = gen.generate_pdf(
            analysis_response=response,
            images_data=images_data,
            session_id=response.session_id,
        )
        report_path = os.path.join(settings.reports_dir, f"{response.session_id}.pdf")
        async with aiofiles.open(report_path, "wb") as f:
            await f.write(pdf_bytes)
    except Exception as exc:
        logger.warning("PDF generation failed (non-fatal): %s", exc)

    return response


# ── Report Download ───────────────────────────────────────────────────────────

@router.get(
    "/report/{session_id}",
    summary="Download PDF analysis report",
    response_class=FileResponse,
)
async def download_report(
    session_id: str,
    settings: Settings = Depends(get_settings),
):
    report_path = os.path.join(settings.reports_dir, f"{session_id}.pdf")
    if not os.path.exists(report_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found. Run an analysis first.",
        )
    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"satquery_report_{session_id}.pdf",
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Service health check")
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    registry = get_registry(request)
    available = registry.list_available()
    loaded_count = sum(1 for m in available if m.get("loaded"))

    upload_writable = os.access(settings.upload_dir, os.W_OK)

    return HealthResponse(
        status="healthy" if upload_writable else "degraded",
        version="1.0.0",
        device=settings.resolved_device,
        models_loaded=loaded_count,
        upload_dir_writable=upload_writable,
    )


# ── Models ────────────────────────────────────────────────────────────────────

@router.get("/models", response_model=ModelsListResponse, summary="List available models")
async def list_models(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    registry = get_registry(request)
    available = registry.list_available()

    model_infos: List[ModelInfo] = []
    for m in available:
        model_infos.append(
            ModelInfo(
                name=m["name"],
                loaded=m["loaded"],
                model_id=m.get("model_id", ""),
                task=m.get("task", ""),
                device=m.get("device"),
            )
        )

    return ModelsListResponse(
        models=model_infos,
        device=settings.resolved_device,
        total_loaded=sum(1 for m in model_infos if m.loaded),
    )
