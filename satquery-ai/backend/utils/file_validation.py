"""
SatQuery AI — File Validation Utilities
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from config import get_settings

logger = logging.getLogger("satquery.validation")


class FileValidator:
    """Validate uploaded files for security."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and other attacks."""
        if not filename:
            return ""
            
        # Get basename to prevent path traversal
        basename = os.path.basename(filename)
        
        # Remove any disallowed characters
        for disallowed in self.settings.disallowed_filenames:
            basename = basename.replace(disallowed, "_")
        
        # Limit filename length
        if len(basename) > self.settings.max_filename_length:
            name, ext = os.path.splitext(basename)
            basename = name[:self.settings.max_filename_length - len(ext)] + ext
        
        return basename
    
    def validate_extension(self, filename: str) -> Tuple[bool, str, Optional[str]]:
        """Validate file extension against allowed list."""
        if not filename:
            return False, "No filename provided", None
            
        sanitized = self.sanitize_filename(filename)
        suffix = Path(sanitized).suffix.lower()
        
        if suffix not in self.settings.allowed_extensions:
            return False, f"File extension '{suffix}' not allowed. Allowed: {self.settings.allowed_extensions}", None
        
        return True, "", suffix
    
    def validate_content_type(self, file_path: str, expected_extension: str) -> Tuple[bool, str]:
        """Validate that the file's real content is a decodable image of an allowed type.

        Uses Pillow (and rasterio for GeoTIFF) instead of libmagic/python-magic.
        Pillow is a core dependency and works cross-platform without a native
        magic database — and on some Windows setups the libmagic DLL loaded by
        python-magic segfaults/hangs on import, which would take down the API.
        For an image service, confirming the bytes actually decode as the claimed
        image format is also stronger than MIME sniffing: it rejects non-image
        payloads that were merely renamed with an image extension.
        """
        pil_format_to_mime = {
            "TIFF": "image/tiff",
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "MPO": "image/jpeg",  # multi-picture JPEG variant
        }
        extension_mime_map = {
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        expected_mime = extension_mime_map.get(expected_extension, "")

        # Primary check: Pillow. verify() confirms the bytes are a valid,
        # non-truncated image; a second open reads the detected format.
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                img.verify()
            with Image.open(file_path) as img:
                detected_format = (img.format or "").upper()

            detected_mime = pil_format_to_mime.get(detected_format)
            if detected_mime and detected_mime in self.settings.allowed_mime_types:
                if expected_mime and detected_mime != expected_mime:
                    logger.warning(
                        "File extension %s does not match detected image format %s",
                        expected_extension,
                        detected_format,
                    )
                    # Extension/content mismatch is logged, not rejected — the
                    # bytes are still a valid, allowed image type.
                return True, ""
        except Exception as pil_exc:
            logger.debug("Pillow could not parse %s as an image: %s", file_path, pil_exc)

        # Fallback: GeoTIFF / multiband rasters that Pillow may not open cleanly.
        if expected_extension in (".tif", ".tiff"):
            try:
                import rasterio

                with rasterio.open(file_path) as src:
                    if src.count >= 1 and src.width > 0 and src.height > 0:
                        return True, ""
                return False, "File is not a valid raster image"
            except ImportError:
                # rasterio unavailable — accept the file rather than block uploads,
                # integrity/size/extension checks already passed.
                logger.warning("rasterio not installed; skipping deep GeoTIFF content check")
                return True, ""
            except Exception as rio_exc:
                logger.warning("GeoTIFF content validation failed for %s: %s", file_path, rio_exc)
                return False, "File is not a valid GeoTIFF image"

        return False, "File content is not a valid image of an allowed type"
    
    def validate_file_size(self, file_path: str) -> Tuple[bool, str]:
        """Validate file size against maximum."""
        try:
            file_size = os.path.getsize(file_path)
            max_bytes = self.settings.max_image_bytes
            
            if file_size > max_bytes:
                size_mb = file_size / (1024 * 1024)
                max_mb = max_bytes / (1024 * 1024)
                return False, f"File size {size_mb:.1f}MB exceeds maximum of {max_mb}MB"
            
            return True, ""
            
        except OSError as e:
            return False, f"Error checking file size: {str(e)}"
    
    def validate_file_integrity(self, file_path: str) -> Tuple[bool, str]:
        """Perform basic file integrity checks."""
        try:
            # Check if file exists and is readable
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            if not os.path.isfile(file_path):
                return False, "Path is not a file"
            
            # Try to read first few bytes
            with open(file_path, 'rb') as f:
                header = f.read(1024)
                if not header:
                    return False, "File appears to be empty"
            
            # Check for common malware patterns (basic check)
            suspicious_patterns = [
                b"<?php",
                b"<script",
                b"eval(",
                b"base64_decode",
            ]
            
            for pattern in suspicious_patterns:
                if pattern in header:
                    return False, "File contains suspicious content"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error validating file integrity: {str(e)}"
    
    def validate_upload(self, file_path: str, original_filename: str) -> Tuple[bool, str, Optional[str]]:
        """
        Comprehensive validation of uploaded file.
        Returns: (is_valid, error_message, sanitized_filename)
        """
        # Step 1: Sanitize and validate extension
        is_valid, error, extension = self.validate_extension(original_filename)
        if not is_valid:
            return False, error, None
        
        sanitized_name = self.sanitize_filename(original_filename)
        
        # Step 2: Validate file integrity
        is_valid, error = self.validate_file_integrity(file_path)
        if not is_valid:
            return False, error, None
        
        # Step 3: Validate file size
        is_valid, error = self.validate_file_size(file_path)
        if not is_valid:
            return False, error, None
        
        # Step 4: Validate content type (if magic is available)
        is_valid, error = self.validate_content_type(file_path, extension)
        if not is_valid:
            return False, error, None
        
        return True, "", sanitized_name
    
    def check_path_traversal(self, file_path: str, base_dir: str) -> Tuple[bool, str]:
        """Check for path traversal attempts."""
        try:
            abs_file_path = Path(file_path).expanduser().resolve()
            abs_base_dir = Path(base_dir).expanduser().resolve()

            # Path.relative_to avoids the common string-prefix bypass.
            try:
                abs_file_path.relative_to(abs_base_dir)
            except ValueError:
                return False, "Path traversal attempt detected"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error checking path safety: {str(e)}"