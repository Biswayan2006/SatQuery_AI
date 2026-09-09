"""
SatQuery AI — Application Configuration
Uses pydantic-settings to load from environment / .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Server ───────────────────────────────────────────────────────────────
    environment: str = "development"
    secret_key: str = ""
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    api_prefix: str = "/api"
    docs_enabled: bool = True
    redoc_enabled: bool = True

    # ── Storage ──────────────────────────────────────────────────────────────
    # Compose overrides these with /app/data/*; relative defaults keep local
    # Windows/Linux runs self-contained under the backend directory.
    data_dir: str = "./data"
    upload_dir: str = "./uploads"
    model_cache_dir: str = "./model_cache"
    reports_dir: str = "./reports"
    temp_dir: str = "./temp"
    max_image_size_mb: int = 50
    upload_ttl_hours: int = 24  # Hours before uploaded files are cleaned up
    report_ttl_hours: int = 72  # Hours before reports are cleaned up
    temp_ttl_hours: int = 1  # Hours before temporary files are cleaned up

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ── Model Names ──────────────────────────────────────────────────────────
    vqa_model_name: str = "Salesforce/blip-vqa-base"
    # Keep captioning separate from the large VQA checkpoint.
    captioning_model_name: str = "Salesforce/blip-image-captioning-base"
    clip_model_name: str = "ViT-B-32"
    change_model_name: str = "microsoft/resnet-50"
    grounding_model_name: str = "google/owlvit-base-patch32"
    # Path to a fine-tuned RS-CLIP checkpoint produced by training/train_clip.py
    # Leave empty to use base OpenCLIP weights (semantic routing still works,
    # but without remote-sensing domain adaptation).
    rs_clip_checkpoint: str = ""

    # ── VQA model source ───────────────────────────────────────────────────────
    # Selects which weights the VQA wrapper loads at startup:
    #   "pretrained" — use the base HuggingFace model (``vqa_model_name``)
    #   "finetuned"  — load the base model and apply the fine-tuned adapter /
    #                  checkpoint at ``vqa_finetuned_checkpoint``
    # The local checkpoint path is NEVER hardcoded — it is read from the
    # environment / .env so different deployments can point at their own
    # fine-tuned artifacts.
    vqa_model_source: str = "pretrained"  # "pretrained" | "finetuned"
    # Directory holding a fine-tuned VQA checkpoint produced by
    # training/train_vqa.py (contains adapter/ or model weights + processor +
    # model_version.json). Only used when vqa_model_source == "finetuned".
    vqa_finetuned_checkpoint: str = ""
    # Max new tokens generated per VQA answer.
    vqa_max_new_tokens: int = 100

    # ── SAR-optical fusion ───────────────────────────────────────────────────────
    # Path to a trained fusion-adapter checkpoint produced by
    # training/train_fusion.py (contains the SAR encoder + cross-modal fusion
    # adapter state).  Read from the environment / .env — never hardcoded.
    # Absent → the fusion adapter stays randomly initialised (fusion_trained=False)
    # and any fused answer is flagged as untrained / requiring verification.
    sar_fusion_checkpoint: str = ""

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:3001"

    # ── HuggingFace ──────────────────────────────────────────────────────────
    huggingface_token: str = ""

    # ── File Types ───────────────────────────────────────────────────────────
    allowed_extensions: List[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]
    allowed_mime_types: List[str] = ["image/tiff", "image/geotiff", "image/png", "image/jpeg"]
    max_filename_length: int = 255
    disallowed_filenames: List[str] = ["..", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]

    # ── Security ─────────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60  # Requests per minute per IP
    api_key_enabled: bool = False  # Enable API key authentication
    api_keys: List[str] = []  # List of valid API keys
    cors_max_age: int = 600  # CORS preflight cache in seconds
    admin_api_keys: List[str] = []
    trusted_hosts: str = "*"
    allow_mock_mode: bool = False
    
    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_file: str = ""  # Empty for stdout only, or path to log file
    request_id_header: str = "X-Request-ID"

    # ── Model Concurrency ───────────────────────────────────────────────────
    max_concurrent_inference: int = 1  # Max concurrent GPU inference requests
    inference_timeout_seconds: int = 300  # Timeout for model inference
    model_load_timeout_seconds: int = 600  # Timeout for model loading
    lazy_load_models: bool = True  # Load heavy models only when their task is requested
    
    # ── Confidence framework ─────────────────────────────────────────────────
    # Task-aware confidence + abstention.  Thresholds are env-overridable.
    confidence_enabled: bool = True
    # final confidence < abstain → mark "requires verification"
    confidence_abstain_threshold: float = 0.45
    # final ≥ low_band → uncertainty "low"; ≥ medium_band → "medium"; else "high"
    confidence_low_band: float = 0.70
    confidence_medium_band: float = 0.45
    # Directory holding an optional calibration.json (temperature scalers fit on
    # a VALIDATION split).  Absent file → every task reported as "uncalibrated".
    calibration_dir: str = "./calibration"

    # ── Derived properties ───────────────────────────────────────────────────

    @property
    def resolved_device(self) -> str:
        """Return the actual torch device string."""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def trusted_hosts_list(self) -> List[str]:
        return [host.strip() for host in self.trusted_hosts.split(",") if host.strip()]

    @property
    def max_image_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v

    @field_validator("vqa_model_source")
    @classmethod
    def validate_vqa_model_source(cls, v: str) -> str:
        allowed = {"pretrained", "finetuned"}
        vl = v.lower().strip()
        if vl not in allowed:
            raise ValueError(f"vqa_model_source must be one of {allowed}")
        return vl

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "production", "testing"}
        vl = v.lower().strip()
        if vl not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return vl

    @field_validator("upload_ttl_hours", "report_ttl_hours", "temp_ttl_hours")
    @classmethod
    def validate_ttl_hours(cls, v: int) -> int:
        if v < 0:
            raise ValueError("TTL hours must be >= 0 (0 disables cleanup)")
        return v

    @field_validator("max_concurrent_inference")
    @classmethod
    def validate_max_concurrent_inference(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_concurrent_inference must be >= 1")
        return v

    @field_validator("rate_limit_per_minute")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")
        return v

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"

    @property
    def default_api_key(self) -> str:
        """Generate a default API key for development if none provided."""
        if self.api_key_enabled and not self.api_keys:
            # Generate a deterministic dev key
            import hashlib
            key_hash = hashlib.sha256(self.secret_key.encode()).hexdigest()[:32]
            return f"dev_{key_hash}"
        return ""

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for path in [
            self.upload_dir,
            self.model_cache_dir,
            self.reports_dir,
            self.calibration_dir,
            self.temp_dir,
        ]:
            Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, value: str) -> str:
        if value:
            return value
        if os.getenv("ENVIRONMENT", "development").lower() == "production":
            raise ValueError("SECRET_KEY must be set in production")
        return "development-only-secret"

    @model_validator(mode="after")
    def validate_api_key_configuration(self):
        """Validate auth after all environment fields have been parsed."""
        if self.environment == "production" and not self.api_key_enabled:
            raise ValueError("API_KEY_ENABLED must be true in production")
        if self.environment == "production" and self.api_key_enabled and not self.api_keys:
            raise ValueError("API_KEYS must be configured when API_KEY_ENABLED=true in production")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
