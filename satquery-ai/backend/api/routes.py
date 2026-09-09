"""
SatQuery AI — API Routes
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
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
from services.cleanup import cleanup_endpoint

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
    modality: str = Form("auto"),
    settings: Settings = Depends(get_settings),
    request: Request = None,
):
    """
    Upload a single satellite image (GeoTIFF, PNG, JPEG).
    Returns an image_id that can be used in /analyze.
    """
    # Get request ID for logging
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"
    
    if modality not in {"auto", "optical", "sar", "multispectral"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Modality must be auto, optical, sar, or multispectral",
        )

    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided",
        )
    
    logger.info("Upload request %s", request_id)
    
    # Validate extension and sanitize filename
    from utils.file_validation import FileValidator
    file_validator = FileValidator(settings)
    
    is_valid, error, extension = file_validator.validate_extension(file.filename)
    if not is_valid:
        logger.warning(f"Upload {request_id}: invalid extension - {error}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=error,
        )
    
    sanitized_name = file_validator.sanitize_filename(file.filename)
    
    # Check file size during upload
    image_id = str(uuid.uuid4())
    dest_path = os.path.join(settings.upload_dir, f"{image_id}{extension}")
    
    # Check path safety
    is_safe, safety_error = file_validator.check_path_traversal(dest_path, settings.upload_dir)
    if not is_safe:
        logger.warning(f"Upload {request_id}: path traversal attempt - {safety_error}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path",
        )
    
    total_bytes = 0
    try:
        async with aiofiles.open(dest_path, "wb") as out_file:
            while chunk := await file.read(65536):
                total_bytes += len(chunk)
                if total_bytes > settings.max_image_bytes:
                    await out_file.close()
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    logger.warning(f"Upload {request_id}: file too large - {total_bytes} bytes")
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum size of {settings.max_image_size_mb} MB",
                    )
                await out_file.write(chunk)
    except HTTPException:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
    except Exception as e:
        logger.error(f"Upload {request_id}: failed to write file - {str(e)}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )
    
    # Validate file integrity and content
    is_valid, error, _ = file_validator.validate_upload(dest_path, file.filename)
    if not is_valid:
        logger.warning(f"Upload {request_id}: file validation failed - {error}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File validation failed",
        )
    
    # Validate with InputValidator
    from agent.input_validator import InputValidator
    import json

    validator = InputValidator()
    result = validator.validate_image(dest_path)
    if modality != "auto":
        result.modality = modality
        result.message = f"Image validated with user-selected modality: {modality}"
    
    # Save upload metadata sidecar for persistent modality retrieval
    meta_path = os.path.join(settings.upload_dir, f"{image_id}.meta.json")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "image_id": image_id,
                "modality": result.modality,
                "filename": sanitized_name,
                "is_geotiff": result.is_geotiff,
                "crs": result.crs,
            }, f)
    except Exception as exc:
        logger.warning(f"Failed to write sidecar metadata for {image_id}: {exc}")

    logger.info(f"Upload {request_id}: successful - {image_id}, {total_bytes} bytes (modality: {result.modality})")

    return ImageUploadResponse(
        image_id=image_id,
        modality=result.modality,
        shape=result.shape,
        bands=result.bands,
        valid=result.valid,
        message=result.message,
        filename=sanitized_name,
        file_size_kb=round(total_bytes / 1024, 2),
        is_geotiff=result.is_geotiff,
        crs=result.crs,
        modality_source="user" if modality != "auto" else "detected",
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
    # Get request ID for logging
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(f"Analysis request {request_id}: {len(request_body.image_ids)} images, query: '{request_body.query[:50]}...'")
    
    from agent.controller import AgenticController
    from agent.input_validator import InputValidator
    from utils.image_utils import load_image
    from utils.file_validation import FileValidator

    validator = InputValidator()
    file_validator = FileValidator(settings)
    images_data = []

    # Validate image IDs
    if not request_body.image_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one image ID is required",
        )
    
    if len(request_body.image_ids) > 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 2 images allowed per analysis",
        )
    
    # Validate query
    if not request_body.query or not request_body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty",
        )
    
    # Limit query length
    if len(request_body.query) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query too long (max 1000 characters)",
        )

    for img_id in request_body.image_ids:
        # Find the file on disk (any allowed extension)
        img_path = None
        for ext in settings.allowed_extensions:
            candidate = os.path.join(settings.upload_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            logger.warning(f"Analysis {request_id}: image '{img_id}' not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image '{img_id}' not found. Upload it first via /api/upload.",
            )
        
        # Check path safety
        is_safe, safety_error = file_validator.check_path_traversal(img_path, settings.upload_dir)
        if not is_safe:
            logger.warning(f"Analysis {request_id}: path traversal attempt for image '{img_id}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image path",
            )

        try:
            img_data = load_image(img_path)
            validation = validator.validate_image(img_path)
            
            # Check for saved upload metadata sidecar
            meta_path = os.path.join(settings.upload_dir, f"{img_id}.meta.json")
            modality = validation.modality
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        saved_meta = json.load(f)
                        if saved_meta.get("modality") and saved_meta["modality"] != "unknown":
                            modality = saved_meta["modality"]
                except Exception as meta_exc:
                    logger.debug(f"Failed to read metadata sidecar for {img_id}: {meta_exc}")

            img_data["image_id"] = img_id
            img_data["modality"] = modality
            images_data.append(img_data)
        except Exception as exc:
            logger.error(f"Analysis {request_id}: failed to load image '{img_id}' - {str(exc)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load image '{img_id}': {str(exc)}",
            )

    controller = AgenticController(registry=registry)

    try:
        response = await controller.analyze(
            images=images_data,
            query=request_body.query,
            task_hint=request_body.task_hint,
        )
    except Exception as exc:
        logger.exception(f"Analysis {request_id}: pipeline failed - {str(exc)}")
        # Don't expose internal error details in production
        if settings.is_production:
            error_detail = "Analysis failed due to an internal error"
        else:
            error_detail = f"Analysis failed: {str(exc)}"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
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
        
        # Check path safety
        is_safe, safety_error = file_validator.check_path_traversal(report_path, settings.reports_dir)
        if not is_safe:
            logger.warning(f"Analysis {request_id}: unsafe report path for session '{response.session_id}'")
        else:
            async with aiofiles.open(report_path, "wb") as f:
                await f.write(pdf_bytes)
            logger.info(f"Analysis {request_id}: report saved - {response.session_id}")
    except Exception as exc:
        logger.warning(f"Analysis {request_id}: PDF generation failed (non-fatal) - {str(exc)}")

    logger.info(f"Analysis {request_id}: completed successfully - session {response.session_id}")
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
    reports_writable = os.access(settings.reports_dir, os.W_OK)
    model_cache_writable = os.access(settings.model_cache_dir, os.W_OK)
    
    # Get GPU information
    from utils.gpu_monitor import get_gpu_monitor
    gpu_monitor = get_gpu_monitor()
    gpu_stats = gpu_monitor.get_all_memory_stats()
    
    # Get model registry status
    model_registry_status = registry.get_status()
    
    # Get concurrency statistics
    from utils.concurrency import get_concurrency_manager
    concurrency_manager = get_concurrency_manager()
    concurrency_stats = concurrency_manager.get_all_stats()
    
    # Check for active requests approaching timeout
    from utils.concurrency import TimeoutManager
    timeout_manager = TimeoutManager(timeout_seconds=settings.inference_timeout_seconds)
    timed_out_requests = timeout_manager.check_timeouts()
    
    # Degradation monitoring can be unavailable while its background state is
    # being initialized; it must not block the health endpoint.
    degradation_status = {"system_status": "unknown"}
    
    # Lazy registration is ready for on-demand loading; it is not an active
    # startup download and should not appear as a stuck/degraded service.
    all_lazy_registered = bool(available) and all(
        model.get("loader_registered") and not model.get("loaded")
        for model in available
    )
    status = "healthy" if loaded_count == len(available) or all_lazy_registered else "loading"
    degradation_reasons = []
    
    # Check storage
    if not upload_writable:
        status = "degraded"
        degradation_reasons.append("upload_dir_not_writable")
    
    # Check models
    failed_count = sum(1 for model in available if model.get("status") == "failed")
    if failed_count == len(available) and available:
        status = "failed"
        degradation_reasons.append("all_models_failed")
    elif loaded_count < len(available):
        if all_lazy_registered and settings.lazy_load_models:
            degradation_reasons.append(f"models_available_on_demand_{loaded_count}/{len(available)}")
        else:
            status = "degraded" if failed_count else "loading"
            degradation_reasons.append(f"models_partially_loaded_{loaded_count}/{len(available)}")
    
    # Check timeouts
    if timed_out_requests:
        status = "degraded"
        degradation_reasons.append(f"timed_out_requests_{len(timed_out_requests)}")
    
    # Check concurrency queue pressure
    global_queue = concurrency_stats.get("global", {})
    queue_pressure = global_queue.get("queue_length", 0) > 3
    high_wait_time = global_queue.get("avg_wait_time_seconds", 0) > 10
    
    if queue_pressure:
        status = "degraded"
        degradation_reasons.append("high_queue_pressure")
    if high_wait_time:
        status = "degraded"
        degradation_reasons.append("high_wait_time")
    
    # Check degradation monitor status (this overrides other statuses)
    degradation_system_status = degradation_status.get("system_status", "healthy")
    if degradation_system_status == "critical":
        status = "critical"
        degradation_reasons.append("system_critical_incidents")
    elif degradation_system_status == "degraded" and status != "critical":
        status = "degraded"
        degradation_reasons.append("system_degraded_incidents")
    elif degradation_system_status == "warning" and status == "healthy":
        status = "warning"
        degradation_reasons.append("system_warning_incidents")
    
    health_response = HealthResponse(
        status=status,
        version="1.0.0",
        device=settings.resolved_device,
        models_loaded=loaded_count,
        upload_dir_writable=upload_writable,
        storage_status={
            "uploads": upload_writable,
            "reports": reports_writable,
            "model_cache": model_cache_writable,
        },
        gpu_status=gpu_stats,
        model_registry_status=model_registry_status,
        concurrency_status=concurrency_stats,
        degradation_status=degradation_status,
        degradation_reasons=degradation_reasons if degradation_reasons else None,
        queue_pressure=queue_pressure,
        timed_out_requests=list(timed_out_requests.keys()) if timed_out_requests else None,
    )
    if status == "failed":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content=health_response.model_dump())
    return health_response


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
                status=m.get("status", "loading"),
                version=m.get("version", "unknown"),
                model_id=m.get("model_id", ""),
                task=m.get("task", ""),
                device=m.get("device"),
                source=m.get("source"),
            )
        )

    return ModelsListResponse(
        models=model_infos,
        device=settings.resolved_device,
        total_loaded=sum(1 for m in model_infos if m.loaded),
    )


# ── Model Management ─────────────────────────────────────────────────────────

@router.get(
    "/models/{model_name}/status",
    summary="Get detailed status for a specific model",
)
async def get_model_status(
    model_name: str,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Get detailed status and statistics for a specific model."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    registry = get_registry(request)
    model_stats = registry.get_model_stats(model_name)
    
    if not model_stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_name}' not found",
        )
    
    # Add current GPU memory info
    from utils.gpu_monitor import get_gpu_monitor
    gpu_monitor = get_gpu_monitor()
    current_memory = gpu_monitor.get_current_memory(0)
    
    response = {
        **model_stats,
        "current_gpu_memory_mb": current_memory["allocated_mb"] if current_memory else None,
        "gpu_available": gpu_monitor.has_gpu,
        "request_id": request_id,
    }
    
    return response


@router.post(
    "/models/{model_name}/reload",
    summary="Reload a specific model",
)
async def reload_model(
    model_name: str,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Reload a specific model (admin only in production)."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Check permissions in production
    if settings.is_production:
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        admin_keys = settings.admin_api_keys
        
        if not api_key or api_key not in admin_keys:
            logger.warning(f"Model reload {request_id}: unauthorized attempt for '{model_name}'")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for model reload",
            )
    
    registry = get_registry(request)
    success, message = registry.reload_model(model_name, request_id=request_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message,
        )
    
    return {
        "status": "success",
        "message": message,
        "model": model_name,
        "request_id": request_id,
    }


@router.post(
    "/models/{model_name}/unload",
    summary="Unload a specific model from memory",
)
async def unload_model(
    model_name: str,
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Unload a specific model from memory (admin only in production)."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Check permissions in production
    if settings.is_production:
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        admin_keys = settings.admin_api_keys
        
        if not api_key or api_key not in admin_keys:
            logger.warning(f"Model unload {request_id}: unauthorized attempt for '{model_name}'")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for model unload",
            )
    
    registry = get_registry(request)
    success = registry.unload_model(model_name)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to unload model '{model_name}'",
        )
    
    return {
        "status": "success",
        "message": f"Model '{model_name}' unloaded from memory",
        "model": model_name,
        "request_id": request_id,
    }


# ── Cleanup ──────────────────────────────────────────────────────────────────

@router.post(
    "/cleanup",
    summary="Trigger manual cleanup of old files",
)
async def trigger_cleanup(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """
    Manually trigger cleanup of old uploaded files, reports, and temporary files.
    Requires API key authentication in production.
    """
    # Get request ID for logging
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.info(f"Cleanup request {request_id}: manual trigger")
    
    # In production, restrict to admin users
    if settings.is_production:
        # Check for admin API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        admin_keys = settings.admin_api_keys
        
        if not api_key or api_key not in admin_keys:
            logger.warning(f"Cleanup {request_id}: unauthorized attempt")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for cleanup",
            )
    
    # Run cleanup
    from services.cleanup import get_cleanup_service
    service = get_cleanup_service()
    stats = service.run_full_cleanup()
    
    return {
        "status": "success",
        "message": f"Cleanup completed: {stats['total_removed']} files removed",
        "stats": stats,
        "request_id": request_id,
    }
# ── Degradation Monitoring ───────────────────────────────────────────────────

@router.get(
    "/degradation/status",
    summary="Get system degradation status",
)
async def get_degradation_status(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Get detailed system degradation status and incident history."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    from services.degradation_monitor import get_degradation_monitor
    monitor = get_degradation_monitor()
    
    status = monitor.get_system_status()
    
    return {
        "status": status,
        "request_id": request_id,
    }


@router.get(
    "/degradation/history",
    summary="Get degradation incident history",
)
async def get_degradation_history(
    request: Request,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of incidents to return"),
    severity: Optional[str] = Query(None, description="Filter by severity: info, warning, error, critical"),
    component: Optional[str] = Query(None, description="Filter by component name"),
    include_resolved: bool = Query(True, description="Include resolved incidents"),
    settings: Settings = Depends(get_settings),
):
    """Get historical degradation incidents with filtering."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    from services.degradation_monitor import get_degradation_monitor, DegradationSeverity
    monitor = get_degradation_monitor()
    
    # Parse severity filter
    severity_enum = None
    if severity:
        try:
            severity_enum = DegradationSeverity(severity)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid severity: {severity}. Must be one of: info, warning, error, critical",
            )
    
    incidents = monitor.get_incident_history(
        limit=limit,
        severity=severity_enum,
        component=component,
        include_resolved=include_resolved,
    )
    
    return {
        "incidents": incidents,
        "total_count": len(incidents),
        "limit": limit,
        "has_more": len(incidents) >= limit,
        "request_id": request_id,
    }


@router.post(
    "/degradation/{incident_id}/recover",
    summary="Mark a degradation incident as recovered",
)
async def mark_incident_recovered(
    incident_id: str,
    request: Request,
    message: str = Body(default="Manually marked as recovered", description="Recovery message"),
    settings: Settings = Depends(get_settings),
):
    """Manually mark a degradation incident as recovered."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # In production, restrict to admin users
    if settings.is_production:
        # Check for admin API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        admin_keys = os.getenv("ADMIN_API_KEYS", "").split(",")
        
        if not api_key or api_key not in admin_keys:
            logger.warning(f"Degradation recovery {request_id}: unauthorized attempt")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for marking incidents as recovered",
            )
    
    from services.degradation_monitor import get_degradation_monitor
    monitor = get_degradation_monitor()
    
    success = monitor.record_recovery(incident_id, message)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Incident '{incident_id}' not found or already recovered",
        )
    
    return {
        "success": True,
        "incident_id": incident_id,
        "message": f"Incident marked as recovered: {message}",
        "request_id": request_id,
    }


@router.post(
    "/degradation/cleanup",
    summary="Clean up old degradation incidents",
)
async def cleanup_degradation_history(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Clean up degradation incidents older than retention period."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    # In production, restrict to admin users
    if settings.is_production:
        # Check for admin API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        admin_keys = os.getenv("ADMIN_API_KEYS", "").split(",")
        
        if not api_key or api_key not in admin_keys:
            logger.warning(f"Degradation cleanup {request_id}: unauthorized attempt")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for cleaning up degradation history",
            )
    
    from services.degradation_monitor import get_degradation_monitor
    monitor = get_degradation_monitor()
    
    monitor.cleanup_old_events()
    
    return {
        "success": True,
        "message": "Degradation history cleanup completed",
        "request_id": request_id,
    }