"""
SatQuery AI — Structured Logging
"""
from __future__ import annotations

import json
import logging
import logging.config
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import get_settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields from record
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        
        if hasattr(record, "task"):
            log_data["task"] = record.task
        
        if hasattr(record, "model"):
            log_data["model"] = record.model
        
        if hasattr(record, "tool"):
            log_data["tool"] = record.tool
        
        if hasattr(record, "duration"):
            log_data["duration"] = record.duration
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }
        
        # Add any extra attributes
        for key, value in record.__dict__.items():
            if key not in self.default_fields and not key.startswith("_"):
                if isinstance(value, (str, int, float, bool, type(None))):
                    log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False)
    
    @property
    def default_fields(self):
        return {
            "timestamp", "level", "logger", "message", "module", 
            "function", "line", "request_id", "session_id", "task",
            "model", "tool", "duration", "exception"
        }


class RequestContextFilter(logging.Filter):
    """Add request context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # These will be set by middleware
        record.request_id = getattr(record, "request_id", None)
        record.session_id = getattr(record, "session_id", None)
        return True


class PerformanceLogger:
    """Utility for logging performance metrics."""
    
    def __init__(self, name: str = "performance"):
        self.logger = logging.getLogger(name)
    
    def log_inference(self, model: str, duration: float, input_size: Optional[int] = None):
        """Log model inference performance."""
        extra = {
            "model": model,
            "duration": duration,
            "metric": "inference_time",
            "unit": "seconds",
        }
        if input_size:
            extra["input_size"] = input_size
        
        self.logger.info(f"Model inference: {model} took {duration:.3f}s", extra=extra)
    
    def log_request(self, request_id: str, endpoint: str, duration: float, status: int):
        """Log request performance."""
        extra = {
            "request_id": request_id,
            "endpoint": endpoint,
            "duration": duration,
            "status": status,
            "metric": "request_time",
            "unit": "seconds",
        }
        
        self.logger.info(
            f"Request {request_id}: {endpoint} completed in {duration:.3f}s (status {status})",
            extra=extra,
        )
    
    def log_task(self, task: str, session_id: str, duration: float, success: bool):
        """Log task execution performance."""
        extra = {
            "task": task,
            "session_id": session_id,
            "duration": duration,
            "success": success,
            "metric": "task_time",
            "unit": "seconds",
        }
        
        status = "success" if success else "failed"
        self.logger.info(
            f"Task {task} for session {session_id} {status} in {duration:.3f}s",
            extra=extra,
        )


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None, level: int = logging.INFO):
        self.name = name
        self.logger = logger or logging.getLogger("timer")
        self.level = level
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        extra = {
            "timer": self.name,
            "duration": duration,
            "metric": "execution_time",
            "unit": "seconds",
        }
        
        if exc_type:
            extra["error"] = str(exc_val)
            self.logger.log(
                self.level,
                f"Timer {self.name} failed after {duration:.3f}s: {exc_val}",
                extra=extra,
            )
        else:
            self.logger.log(
                self.level,
                f"Timer {self.name} completed in {duration:.3f}s",
                extra=extra,
            )
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


def setup_logging():
    """Set up structured logging based on configuration."""
    settings = get_settings()
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Create logs directory if logging to file
    if settings.log_file:
        os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
    
    # Configure logging
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "request_context": {
                "()": RequestContextFilter,
            },
        },
        "formatters": {
            "text": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
            },
            "json": {
                "()": StructuredFormatter,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "text" if settings.log_format == "text" else "json",
                "stream": "ext://sys.stdout",
                "filters": ["request_context"],
            },
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
        },
        "loggers": {
            "satquery": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn": {
                "level": logging.WARNING,
                "handlers": ["console"],
                "propagate": False,
            },
            "fastapi": {
                "level": logging.WARNING,
                "handlers": ["console"],
                "propagate": False,
            },
        },
    }
    
    # Add file handler if log_file is specified
    if settings.log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": settings.log_file,
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
            "filters": ["request_context"],
        }
        config["root"]["handlers"].append("file")
        config["loggers"]["satquery"]["handlers"].append("file")
    
    logging.config.dictConfig(config)
    
    # Create performance logger
    perf_logger = PerformanceLogger()
    
    logger = logging.getLogger("satquery.logging")
    logger.info(
        f"Logging configured: level={settings.log_level}, format={settings.log_format}",
        extra={
            "log_file": settings.log_file,
            "environment": settings.environment,
        },
    )
    
    return perf_logger


# Global performance logger instance
_perf_logger: Optional[PerformanceLogger] = None

def get_perf_logger() -> PerformanceLogger:
    """Get or create the performance logger singleton."""
    global _perf_logger
    if _perf_logger is None:
        _perf_logger = PerformanceLogger()
    return _perf_logger