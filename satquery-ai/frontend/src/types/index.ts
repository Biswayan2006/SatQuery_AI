// ── Upload ────────────────────────────────────────────────────────────────────

export interface ImageUploadResponse {
  image_id: string;
  modality: "optical" | "sar" | "multispectral" | "unknown";
  shape: number[];
  bands: number;
  valid: boolean;
  message: string;
  filename: string;
  file_size_kb: number;
  is_geotiff: boolean;
  crs: string | null;
  modality_source?: "detected" | "user";
}

// ── Analysis ──────────────────────────────────────────────────────────────────

export interface AnalysisRequest {
  image_ids: string[];
  query: string;
  task_hint?: string;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score: number;
}

export interface ChangeRegion {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  area_pct: number;
  pixel_bbox: number[] | null;
  geo_bbox: { lat_min: number; lon_min: number; lat_max: number; lon_max: number } | null;
}

export interface ToolEvidence {
  tool: string;
  status: string;
  output?: Record<string, unknown>;
  reason?: string;
}

export interface ExecutionSummary {
  selected_task: string;
  task_confidence: number;
  models_used: string[];
  parameters: Record<string, unknown>;
  processing_time_ms: number;
  steps: string[];
  alignment?: Record<string, unknown> | null;
  warnings?: string[] | null;
  plan?: Record<string, unknown>;
  intent?: Record<string, unknown> | null;
  step_trace?: Array<Record<string, unknown>>;
}

export interface AnalysisResponse {
  session_id: string;
  task: string;
  answer: string;
  confidence: number;
  confidence_type: string;
  confidence_components?: Record<string, unknown>;
  uncertainty: string;
  requires_verification: boolean;
  is_degraded: boolean;
  is_georeferenced: boolean;
  visual_evidence: string | null;
  change_map: string | null;
  fusion_map: string | null;
  grounding_boxes: BoundingBox[] | null;
  change_percentage: number | null;
  change_regions: ChangeRegion[] | null;
  tool_evidence: ToolEvidence[] | null;
  execution_summary: ExecutionSummary;
}

// ── Models ────────────────────────────────────────────────────────────────────

export interface ModelInfo {
  name: string;
  loaded: boolean;
  model_id: string;
  task: string;
  device: string | null;
}

export interface ModelsListResponse {
  models: ModelInfo[];
  device: string;
  total_loaded: number;
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_loaded: number;
  upload_dir_writable: boolean;
}

// ── UI State ──────────────────────────────────────────────────────────────────

export type TaskType =
  | "SINGLE_VQA"
  | "LAND_COVER_CLASSIFICATION"
  | "CAPTIONING"
  | "GROUNDING"
  | "CHANGE_VQA"
  | "CHANGE_DESCRIPTION"
  | "SAR_OPTICAL_FUSION"
  | "GEOLOCATION";

export type UploadedImage = {
  file: File;
  previewUrl: string;
  uploadResponse: ImageUploadResponse | null;
  uploading: boolean;
  error: string | null;
};

export type AnalysisState = {
  loading: boolean;
  result: AnalysisResponse | null;
  error: string | null;
};
