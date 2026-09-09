"""
SatQuery AI — Global Exception Handler
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings

logger = logging.getLogger("satquery.exception")


class DetailedHTTPException(StarletteHTTPException):
    """Extended HTTP exception with additional details."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        additional_info: Dict[str, Any] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code or f"ERR_{status_code}"
        self.additional_info = additional_info or {}


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    settings = get_settings()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log the error
    logger.warning(
        f"HTTP exception {request_id}: {exc.status_code} - {exc.detail} - {request.url}"
    )
    
    # Prepare response
    response_data = {
        "error": {
            "code": getattr(exc, "error_code", f"ERR_{exc.status_code}"),
            "message": str(exc.detail),
        },
        "status_code": exc.status_code,
    }
    
    # Add additional info if available
    if hasattr(exc, "additional_info"):
        response_data["details"] = exc.additional_info
    
    # Include request ID
    response_data["request_id"] = request_id
    
    # Don't include stack trace in production
    if settings.is_development:
        response_data["debug"] = {
            "path": str(request.url),
            "method": request.method,
        }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response_data,
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    settings = get_settings()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log validation error
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error.get("loc"),
            "msg": error.get("msg"),
            "type": error.get("type"),
        })
    
    logger.warning(
        f"Validation error {request_id}: {len(error_details)} errors - {request.url}"
    )
    
    # Prepare response
    response_data = {
        "error": {
            "code": "ERR_VALIDATION",
            "message": "Validation error",
        },
        "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
        "request_id": request_id,
        "validation_errors": error_details[:5],  # Limit to 5 errors
    }
    
    if settings.is_development:
        response_data["debug"] = {
            "path": str(request.url),
            "method": request.method,
            "all_errors": error_details,
        }
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=response_data,
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    settings = get_settings()
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Log the full exception with traceback
    logger.error(
        f"Unhandled exception {request_id}: {type(exc).__name__} - {str(exc)}",
        exc_info=exc,
    )
    
    # Prepare response
    response_data = {
        "error": {
            "code": "ERR_INTERNAL",
            "message": "Internal server error",
        },
        "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        "request_id": request_id,
    }
    
    # Include debug info in development
    if settings.is_development:
        response_data["debug"] = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc().split("\n")[-10:],  # Last 10 lines
        }
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response_data,
    )


def setup_exception_handlers(app: FastAPI):
    """Set up exception handlers for the application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
    
    logger.info("Exception handlers configured")