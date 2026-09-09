"""
SatQuery AI — Concurrency Management
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import AsyncIterator, Deque, Dict, Optional, Tuple

from config import get_settings

logger = logging.getLogger("satquery.concurrency")


class InferenceQueue:
    """Queue for managing concurrent GPU inference requests."""
    
    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self.current_tasks = 0
        self.queue: Deque[asyncio.Future] = deque()
        self.lock = asyncio.Lock()
        self.wait_times: Dict[str, float] = {}
        self.completion_times: Dict[str, float] = {}
        
        logger.info(f"Inference queue initialized with max_concurrent={max_concurrent}")
    
    async def acquire(self, request_id: str, model_name: str) -> Tuple[bool, float]:
        """
        Acquire a slot for inference.
        Returns: (acquired, wait_time)
        """
        start_time = time.time()
        
        async with self.lock:
            if self.current_tasks < self.max_concurrent:
                # Slot available immediately
                self.current_tasks += 1
                logger.info(
                    f"Inference slot acquired immediately for request {request_id}, model {model_name}",
                    extra={
                        "request_id": request_id,
                        "model": model_name,
                        "current_tasks": self.current_tasks,
                        "max_concurrent": self.max_concurrent,
                        "wait_time": 0.0,
                    }
                )
                return True, 0.0
            
            # Need to wait in queue
            future = asyncio.Future()
            self.queue.append((future, request_id, model_name))
            
            logger.info(
                f"Inference request {request_id} for model {model_name} queued (queue length: {len(self.queue)})",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "queue_length": len(self.queue),
                    "current_tasks": self.current_tasks,
                }
            )
        
        # Wait for slot to become available
        await future
        wait_time = time.time() - start_time
        self.wait_times[request_id] = wait_time
        
        logger.info(
            f"Inference slot acquired after {wait_time:.2f}s for request {request_id}",
            extra={
                "request_id": request_id,
                "model": model_name,
                "wait_time": wait_time,
                "current_tasks": self.current_tasks,
            }
        )
        
        return True, wait_time
    
    async def release(self, request_id: str, model_name: str):
        """Release an inference slot."""
        async with self.lock:
            self.current_tasks -= 1
            self.completion_times[request_id] = time.time()
            
            # Notify next in queue if any
            if self.queue:
                next_future, next_request_id, next_model_name = self.queue.popleft()
                self.current_tasks += 1
                next_future.set_result(True)
                
                logger.info(
                    f"Inference slot released by request {request_id}, assigned to queued request {next_request_id}",
                    extra={
                        "releasing_request": request_id,
                        "releasing_model": model_name,
                        "next_request": next_request_id,
                        "next_model": next_model_name,
                        "current_tasks": self.current_tasks,
                        "queue_length": len(self.queue),
                    }
                )
            else:
                logger.info(
                    f"Inference slot released by request {request_id}",
                    extra={
                        "request_id": request_id,
                        "model": model_name,
                        "current_tasks": self.current_tasks,
                    }
                )
    
    def get_stats(self) -> Dict:
        """Get queue statistics."""
        avg_wait_time = 0.0
        if self.wait_times:
            avg_wait_time = sum(self.wait_times.values()) / len(self.wait_times)
        
        return {
            "max_concurrent": self.max_concurrent,
            "current_tasks": self.current_tasks,
            "queue_length": len(self.queue),
            "total_waiting": len(self.queue),
            "avg_wait_time_seconds": avg_wait_time,
            "total_processed": len(self.completion_times),
        }


class ModelSemaphore:
    """Semaphore for controlling concurrent access to specific models."""
    
    def __init__(self):
        self.settings = get_settings()
        self.queues: Dict[str, InferenceQueue] = {}
        self.global_queue = InferenceQueue(max_concurrent=self.settings.max_concurrent_inference)
        self.lock = asyncio.Lock()
        
        # Model-specific limits (can be extended per model)
        self.model_limits = {
            "RemoteSensingVQA": 1,  # BLIP models are large, limit to 1
            "RemoteSensingCaptioning": 1,
            "SAROpticalFusionModel": 1,
            "RSCLIPEncoder": 2,  # CLIP is smaller, allow more concurrent
            "default": self.settings.max_concurrent_inference,
        }
    
    def get_queue_for_model(self, model_name: str) -> InferenceQueue:
        """Get or create inference queue for a specific model."""
        if model_name not in self.queues:
            limit = self.model_limits.get(model_name, self.model_limits["default"])
            self.queues[model_name] = InferenceQueue(max_concurrent=limit)
            logger.info(f"Created inference queue for model {model_name} with limit {limit}")
        
        return self.queues[model_name]
    
    @asynccontextmanager
    async def inference_context(self, model_name: str, request_id: str) -> AsyncIterator[Dict]:
        """
        Context manager for model inference with concurrency control.
        
        Usage:
            async with concurrency_manager.inference_context("RemoteSensingVQA", request_id) as stats:
                # Perform inference
                result = await model.inference(...)
        """
        queue = self.get_queue_for_model(model_name)
        start_time = time.time()
        
        try:
            # Acquire slot
            acquired, wait_time = await queue.acquire(request_id, model_name)
            
            if not acquired:
                raise RuntimeError(f"Failed to acquire inference slot for model {model_name}")
            
            inference_start = time.time()
            stats = {
                "model": model_name,
                "request_id": request_id,
                "wait_time": wait_time,
                "inference_start": inference_start,
                "queue_length": queue.queue_length if hasattr(queue, 'queue_length') else 0,
            }
            
            logger.info(
                f"Starting inference for model {model_name}, request {request_id} after {wait_time:.2f}s wait",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "wait_time": wait_time,
                    "action": "inference_start",
                }
            )
            
            yield stats
            
        except Exception as e:
            logger.error(
                f"Inference failed for model {model_name}, request {request_id}: {e}",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "error": str(e),
                    "action": "inference_error",
                },
                exc_info=True,
            )
            raise
            
        finally:
            # Release slot
            inference_time = time.time() - inference_start if 'inference_start' in locals() else 0
            total_time = time.time() - start_time
            
            logger.info(
                f"Inference completed for model {model_name}, request {request_id} "
                f"(inference: {inference_time:.2f}s, total: {total_time:.2f}s)",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "inference_time": inference_time,
                    "total_time": total_time,
                    "wait_time": wait_time if 'wait_time' in locals() else 0,
                    "action": "inference_complete",
                }
            )
            
            await queue.release(request_id, model_name)
    
    async def global_inference_context(self, request_id: str) -> AsyncIterator[Dict]:
        """
        Context manager for global inference concurrency control.
        Use this when you don't know which specific model will be used.
        """
        start_time = time.time()
        
        try:
            # Acquire global slot
            acquired, wait_time = await self.global_queue.acquire(request_id, "global")
            
            if not acquired:
                raise RuntimeError("Failed to acquire global inference slot")
            
            stats = {
                "request_id": request_id,
                "wait_time": wait_time,
                "start_time": time.time(),
                "global_queue": True,
            }
            
            logger.info(
                f"Starting global inference for request {request_id} after {wait_time:.2f}s wait",
                extra={
                    "request_id": request_id,
                    "wait_time": wait_time,
                    "action": "global_inference_start",
                }
            )
            
            yield stats
            
        except Exception as e:
            logger.error(
                f"Global inference failed for request {request_id}: {e}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "action": "global_inference_error",
                },
                exc_info=True,
            )
            raise
            
        finally:
            # Release global slot
            total_time = time.time() - start_time
            
            logger.info(
                f"Global inference completed for request {request_id} (total: {total_time:.2f}s)",
                extra={
                    "request_id": request_id,
                    "total_time": total_time,
                    "wait_time": wait_time if 'wait_time' in locals() else 0,
                    "action": "global_inference_complete",
                }
            )
            
            await self.global_queue.release(request_id, "global")
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all queues."""
        stats = {
            "global": self.global_queue.get_stats(),
            "models": {},
            "settings": {
                "max_concurrent_inference": self.settings.max_concurrent_inference,
                "inference_timeout_seconds": self.settings.inference_timeout_seconds,
            },
        }
        
        for model_name, queue in self.queues.items():
            stats["models"][model_name] = queue.get_stats()
        
        return stats


# Singleton instance
_concurrency_manager: Optional[ModelSemaphore] = None

def get_concurrency_manager() -> ModelSemaphore:
    """Get or create the concurrency manager singleton."""
    global _concurrency_manager
    if _concurrency_manager is None:
        _concurrency_manager = ModelSemaphore()
    return _concurrency_manager


class TimeoutManager:
    """Manage inference timeouts."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.active_inferences: Dict[str, float] = {}
        self.lock = asyncio.Lock()
    
    async def start_inference(self, request_id: str, model_name: str):
        """Start tracking an inference."""
        async with self.lock:
            self.active_inferences[request_id] = time.time()
            logger.info(
                f"Started timeout tracking for request {request_id}, model {model_name}",
                extra={
                    "request_id": request_id,
                    "model": model_name,
                    "timeout_seconds": self.timeout_seconds,
                }
            )
    
    async def complete_inference(self, request_id: str):
        """Complete tracking an inference."""
        async with self.lock:
            if request_id in self.active_inferences:
                duration = time.time() - self.active_inferences[request_id]
                del self.active_inferences[request_id]
                
                logger.info(
                    f"Completed timeout tracking for request {request_id} (duration: {duration:.2f}s)",
                    extra={
                        "request_id": request_id,
                        "duration": duration,
                        "within_timeout": duration < self.timeout_seconds,
                    }
                )
    
    def check_timeouts(self) -> Dict[str, float]:
        """Check for timed out inferences."""
        current_time = time.time()
        timed_out = {}
        
        for request_id, start_time in self.active_inferences.items():
            if current_time - start_time > self.timeout_seconds:
                timed_out[request_id] = current_time - start_time
        
        if timed_out:
            logger.warning(
                f"Found {len(timed_out)} timed out inferences",
                extra={
                    "timed_out_requests": list(timed_out.keys()),
                    "timeouts": timed_out,
                }
            )
        
        return timed_out
    
    @asynccontextmanager
    async def timeout_context(self, request_id: str, model_name: str) -> AsyncIterator:
        """
        Context manager for timeout tracking.
        """
        await self.start_inference(request_id, model_name)
        
        try:
            yield
            
            # Check if we're approaching timeout
            current_time = time.time()
            start_time = self.active_inferences.get(request_id)
            if start_time and current_time - start_time > self.timeout_seconds * 0.8:
                logger.warning(
                    f"Inference for request {request_id} is approaching timeout "
                    f"({current_time - start_time:.1f}s / {self.timeout_seconds}s)",
                    extra={
                        "request_id": request_id,
                        "model": model_name,
                        "elapsed_time": current_time - start_time,
                        "timeout_limit": self.timeout_seconds,
                        "percentage": (current_time - start_time) / self.timeout_seconds * 100,
                    }
                )
                
        finally:
            await self.complete_inference(request_id)