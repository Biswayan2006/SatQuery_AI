"""
SatQuery AI — Application Configuration
Uses pydantic-settings to load from environment / .env file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import field_validator
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
    secret_key: str = "dev-secret-change-me"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    # ── Storage ──────────────────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    model_cache_dir: str = "./model_cache"
    reports_dir: str = "./reports"
    max_image_size_mb: int = 50

    # ── Device ───────────────────────────────────────────────────────────────
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # ── Model Names ──────────────────────────────────────────────────────────
    vqa_model_name: str = "Salesforce/blip2-opt-2.7b"
    captioning_model_name: str = "Salesforce/blip2-opt-2.7b"
    clip_model_name: str = "ViT-B-32"
    change_model_name: str = "microsoft/resnet-50"
    grounding_model_name: str = "google/owlvit-base-patch32"

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:3001"

    # ── HuggingFace ──────────────────────────────────────────────────────────
    huggingface_token: str = ""

    # ── File Types ───────────────────────────────────────────────────────────
    allowed_extensions: List[str] = [".tif", ".tiff", ".png", ".jpg", ".jpeg"]

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"

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
    def max_image_bytes(self) -> int:
        return self.max_image_size_mb * 1024 * 1024

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}")
        return v

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        for path in [self.upload_dir, self.model_cache_dir, self.reports_dir]:
            os.makedirs(path, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings
