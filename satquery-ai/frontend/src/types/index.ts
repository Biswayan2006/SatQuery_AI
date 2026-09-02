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

export interface ExecutionSummary {
  selected_task: string;
  task_confidence: number;
  models_used: string[];
  parameters: Record<string, unknown>;
  processing_time_ms: number;
  steps: string[];
}

export interface AnalysisResponse {
  session_id: string;
  task: string;
  answer: string;
  confidence: number;
  visual_evidence: string | null;  // base64 PNG
  change_map: string | null;        // base64 PNG
  fusion_map: string | null;        // base64 PNG
  grounding_boxes: BoundingBox[] | null;
  change_percentage: number | null;
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
  | "CAPTIONING"
  | "GROUNDING"
  | "CHANGE_VQA"
  | "CHANGE_DESCRIPTION"
  | "SAR_OPTICAL_FUSION";

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
