"""
SatQuery AI — Logging Middleware
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config import get_settings
from utils.logging import get_perf_logger

logger = logging.getLogger("satquery.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request logging and performance tracking."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.perf_logger = get_perf_logger()
        self.settings = get_settings()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Start timer
        start_time = time.time()
        
        # Get request ID
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", ""),
            },
        )
        
        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
            
            # Log successful completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "duration": time.time() - start_time,
                },
            )
            
        except Exception as exc:
            # Log error
            duration = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(exc)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                    "duration": duration,
                },
                exc_info=True,
            )
            raise
        
        # Log performance
        duration = time.time() - start_time
        self.perf_logger.log_request(
            request_id=request_id,
            endpoint=request.url.path,
            duration=duration,
            status=response.status_code,
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response


class ModelUsageLogger:
    """Context manager for logging model usage."""
    
    def __init__(self, model_name: str, session_id: str = None, request_id: str = None):
        self.model_name = model_name
        self.session_id = session_id
        self.request_id = request_id
        self.start_time = None
        self.logger = logging.getLogger("satquery.models")
        self.perf_logger = get_perf_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        
        self.logger.info(
            f"Model {self.model_name} starting inference",
            extra={
                "model": self.model_name,
                "session_id": self.session_id,
                "request_id": self.request_id,
                "action": "inference_start",
            },
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        extra = {
            "model": self.model_name,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "duration": duration,
            "action": "inference_complete",
        }
        
        if exc_type:
            extra["error"] = str(exc_val)
            extra["success"] = False
            self.logger.error(
                f"Model {self.model_name} inference failed after {duration:.3f}s: {exc_val}",
                extra=extra,
                exc_info=True,
            )
        else:
            extra["success"] = True
            self.logger.info(
                f"Model {self.model_name} inference completed in {duration:.3f}s",
                extra=extra,
            )
            
            # Log performance
            self.perf_logger.log_inference(
                model=self.model_name,
                duration=duration,
            )
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


class TaskLogger:
    """Context manager for logging task execution."""
    
    def __init__(self, task_name: str, session_id: str, request_id: str = None):
        self.task_name = task_name
        self.session_id = session_id
        self.request_id = request_id
        self.start_time = None
        self.logger = logging.getLogger("satquery.tasks")
        self.perf_logger = get_perf_logger()
    
    def __enter__(self):
        self.start_time = time.time()
        
        self.logger.info(
            f"Task {self.task_name} starting for session {self.session_id}",
            extra={
                "task": self.task_name,
                "session_id": self.session_id,
                "request_id": self.request_id,
                "action": "task_start",
            },
        )
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        extra = {
            "task": self.task_name,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "duration": duration,
            "action": "task_complete",
        }
        
        if exc_type:
            extra["error"] = str(exc_val)
            extra["success"] = False
            self.logger.error(
                f"Task {self.task_name} failed after {duration:.3f}s: {exc_val}",
                extra=extra,
                exc_info=True,
            )
        else:
            extra["success"] = True
            self.logger.info(
                f"Task {self.task_name} completed in {duration:.3f}s",
                extra=extra,
            )
            
            # Log performance
            self.perf_logger.log_task(
                task=self.task_name,
                session_id=self.session_id,
                duration=duration,
                success=True,
            )
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


def setup_logging_middleware(app: ASGIApp):
    """Set up logging middleware."""
    app.add_middleware(LoggingMiddleware)
    logger.info("Logging middleware configured")