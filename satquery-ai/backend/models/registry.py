"""
SatQuery AI — Model Registry
Singleton registry for lazy-loading and accessing specialist models.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
import gc
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from middleware.logging_middleware import ModelUsageLogger
from services.degradation_monitor import (
    record_gpu_memory_exhaustion,
    record_model_load_failure,
    record_model_inference_failure,
)
from utils.concurrency import get_concurrency_manager

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
                obj._loaders: Dict[str, Any] = {}  # Lazy loaders
                obj._config = None
                obj._loading_lock = threading.Lock()
                cls._instance = obj
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        return cls()

    # ── Registration ─────────────────────────────────────────────────────────

    def register(self, name: str, model_instance: Any, metadata: Optional[Dict] = None) -> None:
        """Register a pre-instantiated model."""
        with self._loading_lock:
            self._models[name] = model_instance
            self._metadata[name] = metadata or {}
            self._metadata[name]["loaded"] = True
            self._metadata[name]["load_time"] = time.time()
            logger.info("Model registered: %s", name)

    def register_lazy(self, name: str, loader: Callable, metadata: Dict) -> None:
        """Register a model for lazy loading."""
        with self._loading_lock:
            self._models[name] = None
            self._loaders[name] = loader
            self._metadata[name] = {**metadata, "loaded": False, "loader_registered": True}
            logger.info("Model registered for lazy loading: %s", name)

    def register_stub(self, name: str, metadata: Dict) -> None:
        """Register a placeholder entry (not yet loaded)."""
        with self._loading_lock:
            self._models[name] = None
            self._metadata[name] = {**metadata, "loaded": False}
            logger.info("Model stub registered: %s", name)

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, name: str, force_load: bool = False, request_id: str = None, 
            session_id: str = None) -> Any:
        """
        Retrieve a model by name, loading it if necessary.
        
        Args:
            name: Model name
            force_load: Force loading even if already loaded (e.g., for reloading)
            request_id: Request ID for logging
            session_id: Session ID for logging
            
        Returns:
            Model instance or mock fallback
        """
        # Check if model exists
        if name not in self._models:
            logger.error(
                "Model '%s' is not registered",
                extra={"model": name, "request_id": request_id, "session_id": session_id}
            )
            return self._fallback_or_raise(name, "model is not registered")
        
        # Check if already loaded and not forcing reload
        model = self._models[name]
        if model is not None and not force_load:
            return model
        
        # Try to load the model
        with self._loading_lock:
            # Double-check after acquiring lock
            model = self._models[name]
            if model is not None and not force_load:
                return model
            
            # Try to load from loader if available
            if name in self._loaders:
                try:
                    logger.info(
                        f"Loading model '{name}' lazily",
                        extra={"model": name, "request_id": request_id, "session_id": session_id}
                    )
                    
                    # Check GPU memory before loading
                    from utils.gpu_monitor import get_gpu_monitor
                    gpu_monitor = get_gpu_monitor()
                    
                    model_metadata = self._metadata[name]
                    model_name = model_metadata.get("model_id", name)
                    
                    is_sufficient, message, stats = gpu_monitor.check_memory_sufficient(
                        model_name=model_name,
                        batch_size=1,
                        device_id=0
                    )
                    
                    if not is_sufficient:
                        logger.warning(
                            f"Insufficient GPU memory to load model '{name}': {message}",
                            extra={
                                "model": name, 
                                "request_id": request_id, 
                                "session_id": session_id,
                                "gpu_stats": stats
                            }
                        )
                        
                        # Record GPU memory exhaustion
                        required_mb = stats.get("required_memory_mb", 0)
                        available_mb = stats.get("available_memory_mb", 0)
                        record_gpu_memory_exhaustion(
                            device_id=0,
                            required_mb=required_mb,
                            available_mb=available_mb,
                            model_name=model_name,
                        )
                        
                        # Try to clear cache and check again
                        gpu_monitor.clear_cache()
                        is_sufficient, message, stats = gpu_monitor.check_memory_sufficient(
                            model_name=model_name,
                            batch_size=1,
                            device_id=0
                        )
                        
                        if not is_sufficient:
                            logger.error(
                                f"Still insufficient GPU memory after cache clear: {message}",
                                extra={
                                    "model": name, 
                                    "request_id": request_id, 
                                    "session_id": session_id,
                                    "gpu_stats": stats
                                }
                            )
                            return self._fallback_or_raise(name, "insufficient GPU memory")
                    
                    # Load the model
                    with ModelUsageLogger(name, session_id, request_id):
                        loader = self._loaders[name]
                        model = loader()
                    
                    # Register the loaded model
                    self._models[name] = model
                    self._metadata[name]["loaded"] = True
                    self._metadata[name]["load_time"] = time.time()
                    self._metadata[name]["load_duration"] = time.time() - self._metadata[name].get("load_start_time", time.time())
                    
                    # Update GPU stats
                    gpu_stats = gpu_monitor.get_current_memory(0)
                    if gpu_stats:
                        self._metadata[name]["gpu_memory_after_load_mb"] = gpu_stats["allocated_mb"]
                    
                    logger.info(
                        f"Model '{name}' loaded successfully",
                        extra={
                            "model": name, 
                            "request_id": request_id, 
                            "session_id": session_id,
                            "gpu_memory_mb": gpu_stats["allocated_mb"] if gpu_stats else None
                        }
                    )
                    
                    return model
                    
                except Exception as e:
                    logger.error(
                        f"Failed to load model '{name}': {e}",
                        extra={
                            "model": name, 
                            "request_id": request_id, 
                            "session_id": session_id
                        },
                        exc_info=True
                    )
                    
                    # Record model load failure
                    gpu_memory = None
                    if "memory" in str(e).lower() or "cuda" in str(e).lower():
                        from utils.gpu_monitor import get_gpu_monitor
                        gpu_monitor = get_gpu_monitor()
                        gpu_stats = gpu_monitor.get_current_memory(0)
                        if gpu_stats:
                            gpu_memory = gpu_stats.get("allocated_mb")
                    
                    record_model_load_failure(
                        model_name=name,
                        error=str(e),
                        gpu_memory_mb=gpu_memory,
                    )
                    
                    return self._fallback_or_raise(name, "model failed to load")
            
            # No loader available, check if we have a mock
            if model is None:
                logger.error(
                    "Model '%s' registered but not loaded",
                    extra={"model": name, "request_id": request_id, "session_id": session_id}
                )
                return self._fallback_or_raise(name, "model is not loaded")
            
            return model
    
    async def get_for_inference(self, name: str, request_id: str = None, 
                               session_id: str = None) -> Any:
        """
        Get a model for inference with concurrency control.
        
        This should be used when performing inference to ensure proper
        concurrency management.
        
        Args:
            name: Model name
            request_id: Request ID for logging
            session_id: Session ID for logging
            
        Returns:
            Model instance (loaded via concurrency-controlled path)
        """
        # Get the concurrency manager
        concurrency_manager = get_concurrency_manager()
        
        # Use concurrency-controlled context
        async with concurrency_manager.inference_context(name, request_id or "unknown") as concurrency_stats:
            model = self.get(name, request_id=request_id, session_id=session_id)
            
            # Add concurrency stats to model metadata if we have a real model
            if hasattr(model, 'metadata'):
                model.metadata = model.metadata or {}
                model.metadata.update({
                    "concurrency_stats": concurrency_stats,
                    "concurrency_controlled": True,
                })
            
            return model
    
    async def inference_with_context(self, name: str, inference_func: Callable, 
                                     request_id: str = None, session_id: str = None, 
                                     **kwargs) -> Any:
        """
        Execute inference with full concurrency and timeout control.
        
        This is the recommended way to run model inference in production.
        
        Args:
            name: Model name
            inference_func: Function that performs the actual inference (takes model as first arg)
            request_id: Request ID
            session_id: Session ID
            **kwargs: Additional arguments to pass to inference_func
            
        Returns:
            Inference result
        """
        concurrency_manager = get_concurrency_manager()
        
        # Use the concurrency-controlled context
        async with concurrency_manager.inference_context(name, request_id or "unknown") as stats:
            # Get the model
            model = self.get(name, request_id=request_id, session_id=session_id)
            
            # Log inference start with concurrency info
            logger.info(
                f"Starting concurrency-controlled inference for model {name}",
                extra={
                    "request_id": request_id,
                    "model": name,
                    "session_id": session_id,
                    "concurrency_stats": stats,
                    "action": "inference_start_concurrent",
                }
            )
            
            # Execute the inference
            try:
                start_time = time.time()
                try:
                    import torch
                    inference_context = torch.inference_mode()
                except (ImportError, AttributeError):
                    inference_context = nullcontext()

                with inference_context:
                    result = inference_func(model, **kwargs)
                
                # If result is awaitable (async), await it
                if asyncio.iscoroutine(result):
                    result = await result
                
                inference_time = time.time() - start_time
                
                logger.info(
                    f"Concurrency-controlled inference completed for model {name} "
                    f"in {inference_time:.2f}s",
                    extra={
                        "request_id": request_id,
                        "model": name,
                        "session_id": session_id,
                        "inference_time": inference_time,
                        "wait_time": stats.get("wait_time", 0),
                        "total_time": time.time() - stats.get("start_time", start_time),
                        "action": "inference_complete_concurrent",
                    }
                )
                
                return result
                
            except Exception as e:
                logger.error(
                    f"Concurrency-controlled inference failed for model {name}: {e}",
                    extra={
                        "request_id": request_id,
                        "model": name,
                        "session_id": session_id,
                        "error": str(e),
                        "concurrency_stats": stats,
                        "action": "inference_error_concurrent",
                    },
                    exc_info=True,
                )
                
                # Record model inference failure
                record_model_inference_failure(
                    model_name=name,
                    request_id=request_id or "unknown",
                    error=str(e),
                    duration=time.time() - start_time,
                )
                
                raise
            finally:
                try:
                    del model
                except UnboundLocalError:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, AttributeError):
                    pass
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is loaded."""
        return self._models.get(name) is not None

    # ── Status and Statistics ────────────────────────────────────────────────

    def list_available(self) -> List[Dict]:
        """List all registered models with their status."""
        result = []
        for name, meta in self._metadata.items():
            entry = {
                "name": name,
                "loaded": self.is_loaded(name),
                "status": (
                    "ready" if self.is_loaded(name)
                    else meta.get("status", "registered" if meta.get("loader_registered") else "loading")
                ),
                "version": meta.get("version", "unknown"),
                "loader_registered": meta.get("loader_registered", False),
                "model_id": meta.get("model_id", ""),
                "task": meta.get("task", ""),
                "device": meta.get("device", ""),
                "source": meta.get("source", "huggingface"),
                "estimated_memory_mb": meta.get("estimated_memory_mb"),
                "is_mock": meta.get("is_mock", False),
                "mock_reason": meta.get("mock_reason"),
            }
            
            # Add load time if loaded
            if self.is_loaded(name):
                entry["load_time"] = meta.get("load_time")
                entry["load_duration"] = meta.get("load_duration")
                entry["gpu_memory_after_load_mb"] = meta.get("gpu_memory_after_load_mb")
            
            result.append(entry)
        return result
    
    def get_status(self) -> Dict:
        """Get overall registry status."""
        models = self.list_available()
        loaded_count = sum(1 for m in models if m["loaded"])
        
        # Get GPU memory info
        from utils.gpu_monitor import get_gpu_monitor
        gpu_monitor = get_gpu_monitor()
        gpu_stats = gpu_monitor.get_all_memory_stats()
        
        return {
            "total_models": len(models),
            "loaded_models": loaded_count,
            "lazy_registered": sum(1 for m in models if m.get("loader_registered", False)),
            "gpu": gpu_stats,
            "models": models,
        }
    
    def get_model_stats(self, name: str) -> Optional[Dict]:
        """Get detailed statistics for a specific model."""
        if name not in self._metadata:
            return None
        
        meta = self._metadata[name]
        stats = {
            "name": name,
            "loaded": self.is_loaded(name),
            **meta,
        }
        
        # Add current GPU memory if loaded
        if self.is_loaded(name):
            from utils.gpu_monitor import get_gpu_monitor
            gpu_monitor = get_gpu_monitor()
            current_memory = gpu_monitor.get_current_memory(0)
            if current_memory:
                stats["current_gpu_memory_mb"] = current_memory["allocated_mb"]
        
        return stats
    
    def reload_model(self, name: str, request_id: str = None, session_id: str = None) -> Tuple[bool, str]:
        """Reload a model, clearing it from memory first."""
        if name not in self._models:
            return False, f"Model '{name}' not registered"
        
        logger.info(
            f"Reloading model '{name}'",
            extra={"model": name, "request_id": request_id, "session_id": session_id}
        )
        
        # Clear the model from memory
        with self._loading_lock:
            self._models[name] = None
            self._metadata[name]["loaded"] = False
            
            # Clear GPU cache
            from utils.gpu_monitor import get_gpu_monitor
            gpu_monitor = get_gpu_monitor()
            gpu_monitor.clear_cache()
        
        # Load it again
        try:
            model = self.get(name, force_load=True, request_id=request_id, session_id=session_id)
            if isinstance(model, type(self._create_mock_model(name))):
                return False, f"Failed to reload model '{name}' - fell back to mock"
            return True, f"Model '{name}' reloaded successfully"
        except Exception as e:
            return False, f"Failed to reload model '{name}': {str(e)}"
    
    def unload_model(self, name: str) -> bool:
        """Unload a model from memory (but keep registration)."""
        if name not in self._models:
            return False
        
        with self._loading_lock:
            if self._models[name] is not None:
                # Try to properly cleanup if the model has a cleanup method
                model = self._models[name]
                if hasattr(model, 'cleanup'):
                    try:
                        model.cleanup()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup model '{name}': {e}")
                
                self._models[name] = None
                self._metadata[name]["loaded"] = False
                self._metadata[name]["unload_time"] = time.time()
                
                # Clear GPU cache
                from utils.gpu_monitor import get_gpu_monitor
                gpu_monitor = get_gpu_monitor()
                gpu_monitor.clear_cache()
                
                logger.info(f"Model '{name}' unloaded from memory")
                return True
        
        return False

    # ── Bulk loading ──────────────────────────────────────────────────────────

    def load_all(self, config, lazy: bool = True) -> None:
        """
        Register all specialist models, optionally loading them.
        
        Args:
            config: Configuration object
            lazy: If True, register for lazy loading; if False, load immediately
        """
        self._config = config
        registrations = [
            ("RemoteSensingVQA", self._create_vqa_loader(config), {
                "model_id": config.vqa_model_name,
                "task": "Visual Question Answering",
                "device": config.resolved_device,
                "model_source": getattr(config, "vqa_model_source", "pretrained"),
                "estimated_memory_mb": 1500,  # BLIP base estimate
            }),
            ("RemoteSensingCaptioning", self._create_captioning_loader(config), {
                "model_id": config.captioning_model_name,
                "task": "Image Captioning",
                "device": config.resolved_device,
                "estimated_memory_mb": 1500,
            }),
            ("RemoteSensingGrounding", self._create_grounding_loader(config), {
                "model_id": config.grounding_model_name,
                "task": "Text-Guided Grounding",
                "device": config.resolved_device,
                "estimated_memory_mb": 1000,
            }),
            ("ChangeDetectionModel", self._create_change_loader(config), {
                "model_id": config.change_model_name,
                "task": "Change Detection",
                "device": config.resolved_device,
                "estimated_memory_mb": 500,
            }),
            ("SAROpticalFusionModel", self._create_sar_fusion_loader(config), {
                "model_id": "sar-optical-fusion",
                "task": "SAR-Optical Fusion",
                "device": config.resolved_device,
                "estimated_memory_mb": 2000,  # VQA + fusion adapter
            }),
            ("RSCLIPEncoder", self._create_rs_clip_loader(config), {
                "model_id": getattr(config, "clip_model_name", "ViT-B-32"),
                "task": "Remote-Sensing CLIP Encoder",
                "device": config.resolved_device,
                "estimated_memory_mb": 500,
            }),
        ]

        for name, loader, metadata in registrations:
            self.register_lazy(name, loader, metadata)

        if lazy:
            return

        for name, loader, metadata in registrations:
            try:
                logger.info("Loading model: %s …", name)
                model = loader()
                self.register(name, model, metadata)
                logger.info("Model loaded: %s ✓", name)
            except Exception as exc:
                logger.error("Failed to register/load %s: %s", name, exc)
                self._register_failed(name, str(exc), metadata)

    # ── Individual loader creators ────────────────────────────────────────────

    def _create_vqa_loader(self, config):
        """Create a loader function for VQA model."""
        def load_vqa():
            from models.vqa_model import RemoteSensingVQA
            model = RemoteSensingVQA(
                model_name=config.vqa_model_name,
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
                max_new_tokens=getattr(config, "vqa_max_new_tokens", 150),
                model_source=getattr(config, "vqa_model_source", "pretrained"),
                finetuned_checkpoint=getattr(config, "vqa_finetuned_checkpoint", "") or None,
            )
            return model
        return load_vqa

    def _create_captioning_loader(self, config):
        """Create a loader function for captioning model."""
        def load_captioning():
            from models.captioning_model import RemoteSensingCaptioning
            model = RemoteSensingCaptioning(
                model_name=config.captioning_model_name,
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
            )
            return model
        return load_captioning

    def _create_grounding_loader(self, config):
        """Create a loader function for grounding model."""
        def load_grounding():
            from models.grounding_model import RemoteSensingGrounding
            model = RemoteSensingGrounding(
                model_name=config.grounding_model_name,
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
            )
            return model
        return load_grounding

    def _create_change_loader(self, config):
        """Create a loader function for change detection model."""
        def load_change():
            from models.change_model import ChangeDetectionModel
            model = ChangeDetectionModel(
                backbone_name=config.change_model_name,
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
            )
            return model
        return load_change

    def _create_sar_fusion_loader(self, config):
        """Create a loader function for SAR fusion model."""
        def load_sar_fusion():
            from models.sar_fusion_model import SAROpticalFusionModel
            model = SAROpticalFusionModel(
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
                vqa_model_name=config.vqa_model_name,
                fusion_checkpoint=getattr(config, "sar_fusion_checkpoint", "") or None,
            )
            return model
        return load_sar_fusion

    def _create_rs_clip_loader(self, config):
        """Create a loader function for RS-CLIP encoder."""
        def load_rs_clip():
            from models.rs_clip.encoder import RSCLIPEncoder
            
            checkpoint_path = getattr(config, "rs_clip_checkpoint", None)
            model_name = getattr(config, "clip_model_name", "ViT-B-32")

            encoder = RSCLIPEncoder.from_pretrained(
                model_name=model_name,
                pretrained="openai",
                checkpoint_path=checkpoint_path,
                device=config.resolved_device,
                cache_dir=config.model_cache_dir,
            )
            return encoder
        return load_rs_clip

    def _create_mock_model(self, name: str):
        """Create a mock model instance."""
        from models._mock import MockModel
        return MockModel(name)

    def _register_mock(self, name: str) -> None:
        """Register a mock fallback model so the system stays functional."""
        mock = self._create_mock_model(name)
        self.register(name, mock, {
            "model_id": f"mock-{name.lower()}",
            "task": "Mock (model load failed)",
            "device": "cpu",
            "is_mock": True,
            "mock_reason": "Model failed to load",
        })

    def _register_failed(self, name: str, reason: str, metadata: Optional[Dict] = None) -> None:
        """Keep failed models visible without making mock inference look real."""
        with self._loading_lock:
            self._models[name] = None
            self._metadata[name] = {
                **(metadata or {}),
                "loaded": False,
                "status": "failed",
                "load_error": reason,
                "is_mock": False,
            }

    def _fallback_or_raise(self, name: str, reason: str) -> Any:
        """Use mocks only when explicitly enabled; otherwise fail honestly."""
        if self._config is not None and getattr(self._config, "allow_mock_mode", False):
            self._register_mock(name)
            return self._models[name]
        self._metadata.setdefault(name, {})["status"] = "failed"
        self._metadata.setdefault(name, {})["load_error"] = reason
        raise RuntimeError(f"Model '{name}' unavailable: {reason}")
