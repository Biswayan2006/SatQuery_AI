"""
SatQuery AI — Pydantic API Schemas

Backward-compatibility note
----------------------------
All new fields added in the geospatial correctness layer are ``Optional``
with sensible defaults (``None`` or ``False``).  Existing clients that
deserialise ``AnalysisResponse`` / ``ExecutionSummary`` without the new
fields will continue to work unchanged.
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
    modality_source: str = Field(default="detected", description="Whether modality was detected or user-selected")


# ── Analysis ─────────────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    image_ids: List[str] = Field(..., min_length=1, max_length=2, description="1 or 2 uploaded image IDs")
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    task_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional task override.  Accepted values (case-insensitive): "
            "SINGLE_VQA | VQA | CAPTIONING | GROUNDING | CHANGE_VQA | CHANGE | "
            "CHANGE_DESCRIPTION | SAR_OPTICAL_FUSION | SAR_FUSION"
        ),
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


# ── Geospatial alignment metadata ─────────────────────────────────────────────

class AlignmentInfo(BaseModel):
    """
    Metadata describing the image-pair alignment performed before change
    detection.  All fields are optional for backward compatibility.
    """
    performed: bool = Field(
        default=False,
        description="Whether spatial alignment was performed on the image pair",
    )
    method: Optional[str] = Field(
        default=None,
        description=(
            "Alignment method used: "
            "reproject_and_resample | resample_to_common_extent | "
            "pixel_resize_to_smaller | none"
        ),
    )
    source_crs: Optional[str] = Field(
        default=None,
        description="CRS of the source image before reprojection (if applicable)",
    )
    target_crs: Optional[str] = Field(
        default=None,
        description="CRS of the aligned output images",
    )
    target_resolution: Optional[float] = Field(
        default=None,
        description="Pixel resolution of the aligned output (in target CRS units)",
    )
    common_bounds: Optional[List[float]] = Field(
        default=None,
        description="Spatial intersection [west, south, east, north] in target CRS",
    )
    common_bounds_wgs84: Optional[List[float]] = Field(
        default=None,
        description="Spatial intersection [west, south, east, north] in WGS84",
    )
    output_grid: Optional[Dict[str, int]] = Field(
        default=None,
        description="Pixel dimensions of the aligned grid: {width, height}",
    )
    overlap_fraction: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of the smaller image's footprint that overlaps the larger "
            "(1.0 = fully contained, 0.0 = no overlap)"
        ),
    )
    geographic_coordinates_available: bool = Field(
        default=False,
        description=(
            "True when CRS + affine transform are available and geographic "
            "coordinates can be derived for detected change regions"
        ),
    )
    note: Optional[str] = Field(
        default=None,
        description="Human-readable note about alignment decisions or fallbacks",
    )


# ── Geographic bounding box ───────────────────────────────────────────────────

class GeoBBox(BaseModel):
    """
    Geographic bounding box in WGS84 (EPSG:4326) for a detected change region.
    """
    min_lat: float = Field(..., description="Minimum latitude (south edge)")
    max_lat: float = Field(..., description="Maximum latitude (north edge)")
    min_lon: float = Field(..., description="Minimum longitude (west edge)")
    max_lon: float = Field(..., description="Maximum longitude (east edge)")
    center_lat: float = Field(..., description="Centroid latitude")
    center_lon: float = Field(..., description="Centroid longitude")


# ── Change region ─────────────────────────────────────────────────────────────

class ChangeRegion(BaseModel):
    """
    A single connected change region detected by the change detection model.
    Pixel coordinates are normalised [0, 1].
    Geographic coordinates are populated when CRS metadata is available.
    """
    x1: float = Field(..., description="Normalised left pixel coordinate [0, 1]")
    y1: float = Field(..., description="Normalised top pixel coordinate [0, 1]")
    x2: float = Field(..., description="Normalised right pixel coordinate [0, 1]")
    y2: float = Field(..., description="Normalised bottom pixel coordinate [0, 1]")
    area_pct: float = Field(..., description="Percentage of the aligned image area covered by this region")
    pixel_bbox: Optional[List[float]] = Field(
        default=None,
        description="Absolute pixel coordinates [x1, y1, x2, y2] in the aligned image",
    )
    geo_bbox: Optional[GeoBBox] = Field(
        default=None,
        description="Geographic bounding box in WGS84; None when CRS unavailable",
    )


# ── Execution summary ─────────────────────────────────────────────────────────

class ExecutionSummary(BaseModel):
    selected_task: str = Field(..., description="Task type that was executed")
    task_confidence: float = Field(..., description="Classifier confidence for task selection")
    models_used: List[str] = Field(..., description="Names of models invoked")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    steps: List[str] = Field(default_factory=list, description="Ordered execution steps")
    # Geospatial alignment metadata (None for single-image tasks)
    alignment: Optional[AlignmentInfo] = Field(
        default=None,
        description="Spatial alignment metadata for bi-temporal or cross-modal tasks",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-fatal warnings generated during execution (e.g. mock model in use)",
    )
    # ── Structured agentic planning (backward-compatible extensions) ──────────
    plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "The explicit execution plan produced before execution: "
            "{task, intent, models, steps, parameters}."
        ),
    )
    intent: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured query understanding: {concept, operation, "
            "requires_comparison, requires_grounding, requires_fusion}. "
            "Only the flags that apply are present."
        ),
    )
    step_trace: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Per-step execution metadata (concise, no chain-of-thought): each "
            "entry is {step, status, kind, duration_ms, output_summary?, "
            "parameters?, reason?}."
        ),
    )


class AnalysisResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for report download")
    task: str = Field(..., description="Task type that was performed")
    answer: str = Field(..., description="Natural language answer to the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Final answer confidence score")
    is_georeferenced: bool = Field(
        default=False,
        description=(
            "Whether the input imagery had CRS and affine georeferencing. "
            "False indicates that CV or pixel fallback alignment was used."
        ),
    )

    # ── Task-aware confidence framework (backward-compatible extensions) ──────
    confidence_type: str = Field(
        default="uncalibrated",
        description=(
            "How to read `confidence`: 'calibrated' (temperature-scaled on a "
            "validation split), 'uncalibrated' (real signals combined but not "
            "calibrated — do NOT treat as a probability), or 'unavailable' "
            "(no usable confidence signal; result requires verification)"
        ),
    )
    confidence_components: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Per-source confidence signals that fed the final value, e.g. "
            "{'model': 0.84, 'evidence': 0.79, 'consistency': 0.83}. "
            "Only present when at least one signal was available."
        ),
    )
    uncertainty: str = Field(
        default="unknown",
        description="Qualitative uncertainty band: low | medium | high | unknown",
    )
    requires_verification: bool = Field(
        default=False,
        description=(
            "True when confidence is below the abstention threshold or "
            "unavailable — the result should be verified rather than trusted."
        ),
    )

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
    # Enriched change regions with optional geographic coordinates
    change_regions: Optional[List[ChangeRegion]] = Field(
        default=None,
        description=(
            "Detected change regions with pixel and (when available) "
            "geographic bounding boxes"
        ),
    )

    # Deterministic tool evidence (spectral indices, changed-area, geolocation …)
    tool_evidence: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Structured results from the deterministic remote-sensing tool layer "
            "(NDVI/NDWI/NDBI, SAR statistics, change-area, connected regions, "
            "geolocation). Each entry is a tool result: {'tool', 'status', ...}. "
            "Tools that cannot run on the given data report status='unsupported' "
            "with a reason — these are evidence, not errors, and are never fabricated."
        ),
    )

    execution_summary: ExecutionSummary = Field(..., description="Agentic execution details")

    # Degraded-mode flag (Rule 7 / audit P0-2)
    is_degraded: bool = Field(
        default=False,
        description=(
            "True when one or more models fell back to the mock implementation "
            "due to a load failure.  Results should be treated as placeholders."
        ),
    )

    model_config = {"json_schema_extra": {
        "example": {
            "session_id": "sess_abc123",
            "task": "SINGLE_VQA",
            "answer": "The image shows a dense urban area with residential buildings and roads.",
            "confidence": 0.82,
            "confidence_type": "calibrated",
            "confidence_components": {"model": 0.84, "evidence": 0.79, "consistency": 0.83},
            "uncertainty": "low",
            "requires_verification": False,
            "is_degraded": False,
            "execution_summary": {
                "selected_task": "SINGLE_VQA",
                "task_confidence": 0.92,
                "models_used": ["RemoteSensingVQA"],
                "parameters": {"max_new_tokens": 100},
                "processing_time_ms": 1240.5,
                "steps": ["validate_input", "classify_task", "run_vqa"],
                "alignment": None,
                "warnings": [],
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
    status: str = "loading"
    version: str = "unknown"
    model_id: str
    task: str
    device: Optional[str] = None
    source: Optional[str] = None


class ModelsListResponse(BaseModel):
    models: List[ModelInfo]
    device: str
    total_loaded: int


# ── GPU Information ──────────────────────────────────────────────────────────

class GPUDeviceInfo(BaseModel):
    index: int
    name: str
    total_memory_mb: float
    allocated_mb: float
    free_mb: float
    utilization_percent: float


class GPUInfo(BaseModel):
    available: bool
    device_count: int
    devices: List[GPUDeviceInfo] = []


# ── Model Status ──────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name: str
    loaded: bool
    loader_registered: bool = False
    model_id: str
    task: str
    device: str
    estimated_memory_mb: Optional[float] = None
    load_time: Optional[float] = None
    load_duration: Optional[float] = None
    gpu_memory_after_load_mb: Optional[float] = None
    is_mock: bool = False
    mock_reason: Optional[str] = None


class ModelRegistryStatus(BaseModel):
    total_models: int
    loaded_models: int
    lazy_registered: int
    gpu: GPUInfo
    models: List[ModelStatus]


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: int
    upload_dir_writable: bool
    storage_status: Dict[str, bool] = {}
    gpu_status: Optional[GPUInfo] = None
    model_registry_status: Optional[Dict] = None
    concurrency_status: Optional[Dict] = None
    degradation_status: Optional[Dict] = None
    degradation_reasons: Optional[List[str]] = None
    queue_pressure: bool = False
    timed_out_requests: Optional[List[str]] = None
