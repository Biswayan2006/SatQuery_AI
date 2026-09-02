"""
SatQuery AI — Model Registry
Singleton registry for lazy-loading and accessing specialist models.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger("satquery.registry")


class ModelRegistry:
    """
    Thread-safe singleton model registry.

    Models are registered with a name and loaded lazily on first access.
    Each model entry tracks whether it has been successfully loaded.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton ─────────────────────────────────────────────────────────────

    def __new__(cls) -> "ModelRegistry":
        with cls._lock:
            if cls._instance is None:
                obj = super().__new__(cls)
                obj._models: Dict[str, Any] = {}
                obj._metadata: Dict[str, Dict] = {}
                cls._instance = obj
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        return cls()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, model_instance: Any, metadata: Optional[Dict] = None) -> None:
        """Register a pre-instantiated model."""
        self._models[name] = model_instance
        self._metadata[name] = metadata or {}
        self._metadata[name]["loaded"] = True
        logger.info("Model registered: %s", name)

    def register_stub(self, name: str, metadata: Dict) -> None:
        """Register a placeholder entry (not yet loaded)."""
        self._models[name] = None
        self._metadata[name] = {**metadata, "loaded": False}

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Any:
        """
        Retrieve a model by name.
        If not registered yet (still loading), return a mock fallback
        so requests don't hard-fail during startup downloads.
        """
        if name not in self._models:
            logger.warning("Model '%s' not registered yet — returning mock fallback", name)
            from models._mock import MockModel
            return MockModel(name)
        model = self._models[name]
        if model is None:
            logger.warning("Model '%s' registered but not loaded yet — returning mock fallback", name)
            from models._mock import MockModel
            return MockModel(name)
        return model

    def is_loaded(self, name: str) -> bool:
        return self._models.get(name) is not None

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_available(self) -> List[Dict]:
        result = []
        for name, meta in self._metadata.items():
            entry = {
                "name": name,
                "loaded": self.is_loaded(name),
                **meta,
            }
            entry.pop("loaded", None)  # re-add computed value
            entry["loaded"] = self.is_loaded(name)
            result.append(entry)
        return result

    # ── Bulk loading ──────────────────────────────────────────────────────────

    def load_all(self, config) -> None:
        """
        Attempt to load all specialist models.
        Failures are logged but do not raise — the system degrades gracefully.
        """
        loaders = [
            ("RemoteSensingVQA", self._load_vqa, config),
            ("RemoteSensingCaptioning", self._load_captioning, config),
            ("RemoteSensingGrounding", self._load_grounding, config),
            ("ChangeDetectionModel", self._load_change, config),
            ("SAROpticalFusionModel", self._load_sar_fusion, config),
        ]

        for name, loader, cfg in loaders:
            try:
                logger.info("Loading model: %s …", name)
                loader(cfg)
                logger.info("Model loaded: %s ✓", name)
            except Exception as exc:
                logger.warning("Failed to load %s: %s — will use mock fallback", name, exc)
                self._register_mock(name)

    # ── Individual loaders ────────────────────────────────────────────────────

    def _load_vqa(self, config) -> None:
        from models.vqa_model import RemoteSensingVQA
        model = RemoteSensingVQA(
            model_name=config.vqa_model_name,
            device=config.resolved_device,
            cache_dir=config.model_cache_dir,
        )
        self.register("RemoteSensingVQA", model, {
            "model_id": config.vqa_model_name,
            "task": "Visual Question Answering",
            "device": config.resolved_device,
        })

    def _load_captioning(self, config) -> None:
        from models.captioning_model import RemoteSensingCaptioning
        model = RemoteSensingCaptioning(
            model_name=config.captioning_model_name,
            device=config.resolved_device,
            cache_dir=config.model_cache_dir,
        )
        self.register("RemoteSensingCaptioning", model, {
            "model_id": config.captioning_model_name,
            "task": "Image Captioning",
            "device": config.resolved_device,
        })

    def _load_grounding(self, config) -> None:
        from models.grounding_model import RemoteSensingGrounding
        model = RemoteSensingGrounding(
            model_name=config.grounding_model_name,
            device=config.resolved_device,
            cache_dir=config.model_cache_dir,
        )
        self.register("RemoteSensingGrounding", model, {
            "model_id": config.grounding_model_name,
            "task": "Text-Guided Grounding",
            "device": config.resolved_device,
        })

    def _load_change(self, config) -> None:
        from models.change_model import ChangeDetectionModel
        model = ChangeDetectionModel(
            backbone_name=config.change_model_name,
            device=config.resolved_device,
            cache_dir=config.model_cache_dir,
        )
        self.register("ChangeDetectionModel", model, {
            "model_id": config.change_model_name,
            "task": "Change Detection",
            "device": config.resolved_device,
        })

    def _load_sar_fusion(self, config) -> None:
        from models.sar_fusion_model import SAROpticalFusionModel
        # SAR fusion uses the VQA model internally; pass the same name
        model = SAROpticalFusionModel(
            device=config.resolved_device,
            cache_dir=config.model_cache_dir,
            vqa_model_name=config.vqa_model_name,
        )
        self.register("SAROpticalFusionModel", model, {
            "model_id": "sar-optical-fusion",
            "task": "SAR-Optical Fusion",
            "device": config.resolved_device,
        })

    def _register_mock(self, name: str) -> None:
        """Register a mock fallback model so the system stays functional."""
        from models._mock import MockModel
        mock = MockModel(name)
        self.register(name, mock, {
            "model_id": f"mock-{name.lower()}",
            "task": "Mock (model load failed)",
            "device": "cpu",
        })
