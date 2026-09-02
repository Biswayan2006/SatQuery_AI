"""
SatQuery AI — Pydantic API Schemas
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class ImageUploadResponse(BaseModel):
    image_id: str = Field(..., description="Unique identifier for the uploaded image")
    modality: str = Field(..., description="Detected modality: optical | sar | multispectral | unknown")
    shape: List[int] = Field(..., description="Image dimensions [height, width, bands]")
    bands: int = Field(..., description="Number of image bands/channels")
    valid: bool = Field(..., description="Whether the image passed validation")
    message: str = Field(..., description="Validation message or error description")
    filename: str = Field(default="", description="Original filename")
    file_size_kb: float = Field(default=0.0, description="File size in kilobytes")
    is_geotiff: bool = Field(default=False, description="Whether the image is a GeoTIFF")
    crs: Optional[str] = Field(default=None, description="Coordinate reference system if GeoTIFF")


# ── Analysis ─────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    image_ids: List[str] = Field(..., min_length=1, max_length=2, description="1 or 2 uploaded image IDs")
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    task_hint: Optional[str] = Field(
        default=None,
        description="Optional task hint: vqa | captioning | grounding | change | sar_fusion",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "image_ids": ["abc123"],
            "query": "What land cover types are visible in this satellite image?",
        }
    }}


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float


class ExecutionSummary(BaseModel):
    selected_task: str = Field(..., description="Task type that was executed")
    task_confidence: float = Field(..., description="Classifier confidence for task selection")
    models_used: List[str] = Field(..., description="Names of models invoked")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    steps: List[str] = Field(default_factory=list, description="Ordered execution steps")


class AnalysisResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for report download")
    task: str = Field(..., description="Task type that was performed")
    answer: str = Field(..., description="Natural language answer to the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Answer confidence score")

    # Visual outputs (base64-encoded PNG)
    visual_evidence: Optional[str] = Field(
        default=None, description="Base64-encoded annotated image"
    )
    change_map: Optional[str] = Field(
        default=None, description="Base64-encoded change detection map"
    )
    fusion_map: Optional[str] = Field(
        default=None, description="Base64-encoded SAR-optical fusion visualization"
    )

    # Structured outputs
    grounding_boxes: Optional[List[BoundingBox]] = Field(
        default=None, description="Detected bounding boxes for grounding task"
    )
    change_percentage: Optional[float] = Field(
        default=None, description="Percentage of image area that changed (change detection)"
    )

    execution_summary: ExecutionSummary = Field(..., description="Agentic execution details")

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "sess_abc123",
            "task": "SINGLE_VQA",
            "answer": "The image shows a dense urban area with residential buildings and roads.",
            "confidence": 0.87,
            "execution_summary": {
                "selected_task": "SINGLE_VQA",
                "task_confidence": 0.92,
                "models_used": ["RemoteSensingVQA"],
                "parameters": {"max_new_tokens": 100},
                "processing_time_ms": 1240.5,
                "steps": ["validate_input", "classify_task", "run_vqa"],
            },
        }
    }}


# ── Report ────────────────────────────────────────────────────────────────────

class ReportResponse(BaseModel):
    session_id: str
    download_url: str
    filename: str


# ── Models ────────────────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    loaded: bool
    model_id: str
    task: str
    device: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    device: str
    total_loaded: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: int
    upload_dir_writable: bool
