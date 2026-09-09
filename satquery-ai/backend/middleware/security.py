"""
SatQuery AI — Security Middleware
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config import get_settings

logger = logging.getLogger("satquery.security")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.cleanup_interval = 60  # Clean up old entries every 60 seconds
        self.last_cleanup = time.time()
        
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks
        if request.url.path in ["/api/health", "/", "/health"]:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean up old requests periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_requests(current_time)
            self.last_cleanup = current_time
        
        # Check rate limit
        window_start = current_time - 60  # 1 minute window
        client_requests = [t for t in self.requests[client_ip] if t > window_start]
        
        if len(client_requests) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for IP: %s", client_ip)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        client_requests.append(current_time)
        self.requests[client_ip] = client_requests
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self.requests_per_minute - len(client_requests))
        response.headers["X-RateLimit-Reset"] = str(int(window_start + 60))
        
        return response
    
    def _cleanup_old_requests(self, current_time: float):
        """Remove requests older than 1 minute from all IPs."""
        window_start = current_time - 60
        for ip in list(self.requests.keys()):
            self.requests[ip] = [t for t in self.requests[ip] if t > window_start]
            if not self.requests[ip]:
                del self.requests[ip]


class APIKeyMiddleware(BaseHTTPMiddleware):
    """API key authentication middleware."""
    
    def __init__(self, app, api_keys: List[str], enabled: bool = False):
        super().__init__(app)
        self.api_keys = set(api_keys)
        self.enabled = enabled
        
    async def dispatch(self, request: Request, call_next):
        # Skip API key check for health and docs endpoints
        if request.url.path in ["/api/health", "/", "/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        if not self.enabled:
            return await call_next(request)
        
        # Check for API key in header or query parameter
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        if not api_key:
            logger.warning("API key missing for request to %s", request.url.path)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": "ApiKey"}
            )
        
        if api_key not in self.api_keys:
            logger.warning("Invalid API key provided for request to %s", request.url.path)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid API key"
            )
        
        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add request ID to all requests."""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
        
    async def dispatch(self, request: Request, call_next):
        import uuid
        
        # Get request ID from header or generate new one
        request_id = request.headers.get(self.header_name) or str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy (adjust for your needs)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
        )
        response.headers["Content-Security-Policy"] = csp
        
        return response


class FileUploadValidationMiddleware(BaseHTTPMiddleware):
    """Validate file uploads for security."""
    
    def __init__(self, app, settings):
        super().__init__(app)
        self.settings = settings
        
    async def dispatch(self, request: Request, call_next):
        # Only validate upload endpoint
        if request.url.path == "/api/upload" and request.method == "POST":
            # Check content type
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("multipart/form-data"):
                logger.warning("Invalid content type for upload: %s", content_type)
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="File uploads must use multipart/form-data"
                )
        
        return await call_next(request)


def setup_security_middleware(app: FastAPI, settings):
    """Set up all security middleware."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        max_age=settings.cors_max_age,
    )
    
    # Add trusted host middleware (in production)
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.trusted_hosts_list,
        )
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Add request ID middleware
    app.add_middleware(RequestIDMiddleware, header_name=settings.request_id_header)
    
    # Add rate limiting middleware
    app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
    
    # Add API key middleware if enabled
    if settings.api_key_enabled:
        api_keys = settings.api_keys or []
        if not api_keys and settings.is_development:
            # Generate default dev key
            dev_key = settings.default_api_key
            api_keys = [dev_key]
            logger.info("Generated development API key: %s", dev_key)
        
        app.add_middleware(APIKeyMiddleware, api_keys=api_keys, enabled=settings.api_key_enabled)
    
    # Add file upload validation middleware
    app.add_middleware(FileUploadValidationMiddleware, settings=settings)
    
    logger.info("Security middleware configured")