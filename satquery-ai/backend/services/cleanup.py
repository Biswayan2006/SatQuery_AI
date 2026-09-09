"""
SatQuery AI — TTL Cleanup Service
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# import schedule  # Will be imported conditionally
from config import get_settings

logger = logging.getLogger("satquery.cleanup")


class CleanupService:
    """Service for cleaning up old files based on TTL configuration."""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.running = False
        
    def cleanup_uploads(self) -> Tuple[int, List[str]]:
        """Clean up old uploaded files."""
        if self.settings.upload_ttl_hours <= 0:
            return 0, []
            
        upload_dir = Path(self.settings.upload_dir)
        if not upload_dir.exists():
            return 0, []
            
        cutoff_time = time.time() - (self.settings.upload_ttl_hours * 3600)
        removed_files = []
        removed_count = 0
        
        for file_path in upload_dir.glob("*"):
            if not file_path.is_file():
                continue
                
            # Check file age
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove upload file {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old upload files")
            
        return removed_count, removed_files
    
    def cleanup_reports(self) -> Tuple[int, List[str]]:
        """Clean up old report files."""
        if self.settings.report_ttl_hours <= 0:
            return 0, []
            
        reports_dir = Path(self.settings.reports_dir)
        if not reports_dir.exists():
            return 0, []
            
        cutoff_time = time.time() - (self.settings.report_ttl_hours * 3600)
        removed_files = []
        removed_count = 0
        
        for file_path in reports_dir.glob("*.pdf"):
            if not file_path.is_file():
                continue
                
            # Check file age
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove report file {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old report files")
            
        return removed_count, removed_files
    
    def cleanup_temp_files(self) -> Tuple[int, List[str]]:
        """Clean up temporary files."""
        if self.settings.temp_ttl_hours <= 0:
            return 0, []
            
        temp_dir = Path(self.settings.temp_dir)
        if not temp_dir.exists():
            return 0, []
            
        cutoff_time = time.time() - (self.settings.temp_ttl_hours * 3600)
        removed_files = []
        removed_count = 0
        
        # Clean up all files in temp directory
        for file_path in temp_dir.rglob("*"):
            if not file_path.is_file():
                continue
                
            # Check file age
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")
        
        # Remove empty directories
        for dir_path in sorted(temp_dir.rglob("*"), reverse=True):
            if dir_path.is_dir() and not any(dir_path.iterdir()):
                try:
                    dir_path.rmdir()
                except OSError:
                    pass  # Directory not empty
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old temporary files")
            
        return removed_count, removed_files
    
    def cleanup_model_cache(self, max_age_days: int = 30) -> Tuple[int, List[str]]:
        """Clean up old model cache files (optional)."""
        model_cache_dir = Path(self.settings.model_cache_dir)
        if not model_cache_dir.exists():
            return 0, []
            
        cutoff_time = time.time() - (max_age_days * 86400)
        removed_files = []
        removed_count = 0
        
        # This is a more conservative cleanup - only remove very old cache files
        # that aren't likely to be used
        for file_path in model_cache_dir.rglob("*.bin"):
            if not file_path.is_file():
                continue
                
            # Check file age
            try:
                file_mtime = file_path.stat().st_mtime
                if file_mtime < cutoff_time:
                    file_path.unlink()
                    removed_files.append(str(file_path))
                    removed_count += 1
            except OSError as e:
                logger.warning(f"Failed to remove cache file {file_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old model cache files")
            
        return removed_count, removed_files
    
    def run_full_cleanup(self) -> Dict[str, int]:
        """Run all cleanup tasks and return statistics."""
        logger.info("Starting scheduled cleanup...")
        
        stats = {
            "uploads_removed": 0,
            "reports_removed": 0,
            "temp_files_removed": 0,
            "cache_files_removed": 0,
            "total_removed": 0,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            # Run all cleanup tasks
            uploads_count, _ = self.cleanup_uploads()
            reports_count, _ = self.cleanup_reports()
            temp_count, _ = self.cleanup_temp_files()
            cache_count, _ = self.cleanup_model_cache()
            
            stats["uploads_removed"] = uploads_count
            stats["reports_removed"] = reports_count
            stats["temp_files_removed"] = temp_count
            stats["cache_files_removed"] = cache_count
            stats["total_removed"] = uploads_count + reports_count + temp_count + cache_count
            
            logger.info(f"Cleanup completed: {stats['total_removed']} files removed")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            stats["error"] = str(e)
        
        return stats
    
    def start_scheduled_cleanup(self, interval_hours: int = 1):
        """Start scheduled cleanup job."""
        if self.running:
            logger.warning("Cleanup service already running")
            return
            
        try:
            import schedule
            
            def job():
                try:
                    self.run_full_cleanup()
                except Exception as e:
                    logger.error(f"Scheduled cleanup job failed: {e}")
            
            # Schedule the job
            schedule.every(interval_hours).hours.do(job)
            self.running = True
            
            logger.info(f"Started scheduled cleanup service (every {interval_hours} hours)")
            
            # Run initial cleanup
            job()
            
        except ImportError:
            logger.warning("Schedule module not installed, cannot start scheduled cleanup")
            self.running = False
    
    def stop_scheduled_cleanup(self):
        """Stop scheduled cleanup job."""
        if not self.running:
            return
            
        try:
            import schedule
            schedule.clear()
            self.running = False
            logger.info("Stopped scheduled cleanup service")
        except ImportError:
            logger.warning("Schedule module not installed")
            self.running = False


# Singleton instance
_cleanup_service: Optional[CleanupService] = None

def get_cleanup_service() -> CleanupService:
    """Get or create the cleanup service singleton."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = CleanupService()
    return _cleanup_service


# API endpoint for manual cleanup trigger
async def cleanup_endpoint():
    """FastAPI endpoint for manual cleanup."""
    service = get_cleanup_service()
    stats = service.run_full_cleanup()
    return {
        "status": "success",
        "message": f"Cleanup completed: {stats['total_removed']} files removed",
        "stats": stats,
    }


def setup_cleanup_scheduler(app):
    """Set up cleanup scheduler on application startup."""
    settings = get_settings()
    
    # Only enable scheduled cleanup in production or if explicitly enabled
    if settings.is_production or os.getenv("ENABLE_SCHEDULED_CLEANUP", "false").lower() == "true":
        service = get_cleanup_service()
        
        # Determine cleanup interval (default: 1 hour)
        interval_hours = int(os.getenv("CLEANUP_INTERVAL_HOURS", "1"))
        
        # Start scheduler in background thread
        import threading
        
        def run_scheduler():
            service.start_scheduled_cleanup(interval_hours)
            if service.running:
                try:
                    import schedule
                    while service.running:
                        schedule.run_pending()
                        time.sleep(60)  # Check every minute
                except ImportError:
                    logger.warning("Schedule module not installed, scheduler not running")
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        app.state.cleanup_service = service
        logger.info(f"Cleanup scheduler started (interval: {interval_hours} hours)")