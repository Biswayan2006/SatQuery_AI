"""
SatQuery AI — Degradation Monitoring Service
Monitors system health and provides clear degradation reporting.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Deque, Dict, List, Optional, Set

from config import get_settings

logger = logging.getLogger("satquery.degradation")


class DegradationSeverity(Enum):
    """Severity levels for degradation events."""
    INFO = "info"           # Minor issue, system functional
    WARNING = "warning"     # Some features degraded
    ERROR = "error"         # Major feature unavailable
    CRITICAL = "critical"   # System barely functional


class DegradationType(Enum):
    """Types of degradation that can occur."""
    MODEL_LOAD_FAILURE = "model_load_failure"
    MODEL_INFERENCE_FAILURE = "model_inference_failure"
    GPU_MEMORY_EXHAUSTION = "gpu_memory_exhaustion"
    CONCURRENCY_QUEUE_FULL = "concurrency_queue_full"
    INFERENCE_TIMEOUT = "inference_timeout"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    NETWORK_ISSUE = "network_issue"
    AUTHENTICATION_FAILURE = "authentication_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CLEANUP_FAILURE = "cleanup_failure"
    LOGGING_FAILURE = "logging_failure"
    CONFIGURATION_ERROR = "configuration_error"


@dataclass
class DegradationEvent:
    """Represents a single degradation event."""
    timestamp: datetime
    degradation_type: DegradationType
    severity: DegradationSeverity
    component: str
    message: str
    details: Dict = field(default_factory=dict)
    recovery_time: Optional[datetime] = None
    incident_id: str = field(default_factory=lambda: f"inc_{int(time.time())}")
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "degradation_type": self.degradation_type.value,
            "severity": self.severity.value,
            "component": self.component,
            "message": self.message,
            "details": self.details,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "incident_id": self.incident_id,
            "duration_seconds": (
                (self.recovery_time - self.timestamp).total_seconds() 
                if self.recovery_time else None
            ),
            "is_resolved": self.recovery_time is not None,
        }


class DegradationMonitor:
    """Monitors system degradation and provides reporting."""
    
    def __init__(self, max_history: int = 1000, retention_days: int = 7):
        self.max_history = max_history
        self.retention_days = retention_days
        self.history: Deque[DegradationEvent] = deque(maxlen=max_history)
        self.active_incidents: Dict[str, DegradationEvent] = {}
        self.lock = threading.Lock()
        self.settings = get_settings()
        
        logger.info(f"Degradation monitor initialized (max_history={max_history}, retention_days={retention_days})")
    
    def record_degradation(
        self,
        degradation_type: DegradationType,
        severity: DegradationSeverity,
        component: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> str:
        """
        Record a degradation event.
        
        Returns:
            Incident ID for tracking
        """
        event = DegradationEvent(
            timestamp=datetime.now(),
            degradation_type=degradation_type,
            severity=severity,
            component=component,
            message=message,
            details=details or {},
        )
        
        with self.lock:
            self.history.append(event)
            self.active_incidents[event.incident_id] = event
            
            # Log the degradation
            log_method = {
                DegradationSeverity.INFO: logger.info,
                DegradationSeverity.WARNING: logger.warning,
                DegradationSeverity.ERROR: logger.error,
                DegradationSeverity.CRITICAL: logger.critical,
            }[severity]
            
            log_method(
                f"Degradation recorded: {component} - {message}",
                extra={
                    "degradation_type": degradation_type.value,
                    "severity": severity.value,
                    "component": component,
                    "incident_id": event.incident_id,
                    **(details or {}),
                },
            )
        
        return event.incident_id
    
    def record_recovery(self, incident_id: str, message: str = "Recovered"):
        """Mark a degradation event as recovered."""
        with self.lock:
            if incident_id in self.active_incidents:
                event = self.active_incidents[incident_id]
                event.recovery_time = datetime.now()
                event.details["recovery_message"] = message
                
                # Remove from active incidents
                del self.active_incidents[incident_id]
                
                logger.info(
                    f"Degradation recovered: {event.component} - {message}",
                    extra={
                        "incident_id": incident_id,
                        "duration_seconds": (event.recovery_time - event.timestamp).total_seconds(),
                        "degradation_type": event.degradation_type.value,
                    },
                )
                return True
        return False
    
    def record_model_load_failure(
        self,
        model_name: str,
        error: str,
        gpu_memory_mb: Optional[float] = None,
    ) -> str:
        """Record a model load failure."""
        details = {
            "model": model_name,
            "error": error,
            "gpu_memory_mb": gpu_memory_mb,
        }
        
        severity = (
            DegradationSeverity.CRITICAL 
            if "cuda" in error.lower() or "memory" in error.lower()
            else DegradationSeverity.ERROR
        )
        
        return self.record_degradation(
            degradation_type=DegradationType.MODEL_LOAD_FAILURE,
            severity=severity,
            component=f"model.{model_name}",
            message=f"Failed to load model '{model_name}': {error}",
            details=details,
        )
    
    def record_model_inference_failure(
        self,
        model_name: str,
        request_id: str,
        error: str,
        duration: Optional[float] = None,
    ) -> str:
        """Record a model inference failure."""
        details = {
            "model": model_name,
            "request_id": request_id,
            "error": error,
            "duration": duration,
        }
        
        return self.record_degradation(
            degradation_type=DegradationType.MODEL_INFERENCE_FAILURE,
            severity=DegradationSeverity.ERROR,
            component=f"inference.{model_name}",
            message=f"Inference failed for model '{model_name}' in request {request_id}: {error}",
            details=details,
        )
    
    def record_gpu_memory_exhaustion(
        self,
        device_id: int,
        required_mb: float,
        available_mb: float,
        model_name: str,
    ) -> str:
        """Record GPU memory exhaustion."""
        details = {
            "device_id": device_id,
            "required_memory_mb": required_mb,
            "available_memory_mb": available_mb,
            "model": model_name,
            "deficit_mb": required_mb - available_mb,
        }
        
        return self.record_degradation(
            degradation_type=DegradationType.GPU_MEMORY_EXHAUSTION,
            severity=DegradationSeverity.WARNING,
            component=f"gpu.{device_id}",
            message=f"GPU memory exhausted for model '{model_name}' (required: {required_mb}MB, available: {available_mb}MB)",
            details=details,
        )
    
    def record_concurrency_queue_full(
        self,
        model_name: str,
        queue_length: int,
        max_concurrent: int,
        wait_time_seconds: float,
    ) -> str:
        """Record concurrency queue full."""
        details = {
            "model": model_name,
            "queue_length": queue_length,
            "max_concurrent": max_concurrent,
            "wait_time_seconds": wait_time_seconds,
            "queue_pressure_percentage": (queue_length / max_concurrent) * 100,
        }
        
        severity = (
            DegradationSeverity.WARNING 
            if wait_time_seconds < 30 
            else DegradationSeverity.ERROR
        )
        
        return self.record_degradation(
            degradation_type=DegradationType.CONCURRENCY_QUEUE_FULL,
            severity=severity,
            component=f"concurrency.{model_name}",
            message=f"Concurrency queue full for model '{model_name}' ({queue_length} waiting, {wait_time_seconds:.1f}s wait)",
            details=details,
        )
    
    def record_inference_timeout(
        self,
        model_name: str,
        request_id: str,
        timeout_seconds: float,
        elapsed_seconds: float,
    ) -> str:
        """Record inference timeout."""
        details = {
            "model": model_name,
            "request_id": request_id,
            "timeout_seconds": timeout_seconds,
            "elapsed_seconds": elapsed_seconds,
            "exceeded_by_seconds": elapsed_seconds - timeout_seconds,
        }
        
        return self.record_degradation(
            degradation_type=DegradationType.INFERENCE_TIMEOUT,
            severity=DegradationSeverity.WARNING,
            component=f"timeout.{model_name}",
            message=f"Inference timeout for model '{model_name}' in request {request_id} ({elapsed_seconds:.1f}s > {timeout_seconds}s)",
            details=details,
        )
    
    def get_system_status(self) -> Dict:
        """Get overall system degradation status."""
        with self.lock:
            now = datetime.now()
            recent_cutoff = now - timedelta(hours=24)
            
            # Count incidents by severity
            recent_events = [
                e for e in self.history 
                if e.timestamp > recent_cutoff
            ]
            
            active_by_severity = {}
            recent_by_severity = {}
            
            for severity in DegradationSeverity:
                active_by_severity[severity.value] = sum(
                    1 for e in self.active_incidents.values() 
                    if e.severity == severity
                )
                recent_by_severity[severity.value] = sum(
                    1 for e in recent_events 
                    if e.severity == severity
                )
            
            # Calculate overall system status
            if active_by_severity[DegradationSeverity.CRITICAL.value] > 0:
                system_status = "critical"
            elif active_by_severity[DegradationSeverity.ERROR.value] > 0:
                system_status = "degraded"
            elif active_by_severity[DegradationSeverity.WARNING.value] > 0:
                system_status = "warning"
            else:
                system_status = "healthy"
            
            # Get most critical active incident
            most_critical = None
            severity_order = [
                DegradationSeverity.CRITICAL,
                DegradationSeverity.ERROR,
                DegradationSeverity.WARNING,
                DegradationSeverity.INFO,
            ]
            
            for severity in severity_order:
                for event in self.active_incidents.values():
                    if event.severity == severity:
                        most_critical = event
                        break
                if most_critical:
                    break
            
            return {
                "system_status": system_status,
                "active_incidents": len(self.active_incidents),
                "active_by_severity": active_by_severity,
                "recent_24h": len(recent_events),
                "recent_by_severity": recent_by_severity,
                "most_critical_incident": most_critical.to_dict() if most_critical else None,
                "degradation_score": self._calculate_degradation_score(),
                "timestamp": now.isoformat(),
            }
    
    def _calculate_degradation_score(self) -> float:
        """Calculate a degradation score (0-100, higher is worse)."""
        with self.lock:
            now = datetime.now()
            recent_cutoff = now - timedelta(hours=1)
            
            # Weight incidents by severity and recency
            score = 0.0
            severity_weights = {
                DegradationSeverity.CRITICAL: 10.0,
                DegradationSeverity.ERROR: 5.0,
                DegradationSeverity.WARNING: 2.0,
                DegradationSeverity.INFO: 0.5,
            }
            
            for event in self.active_incidents.values():
                # Active incidents get full weight
                score += severity_weights[event.severity]
            
            for event in self.history:
                if event.recovery_time and event.recovery_time > recent_cutoff:
                    # Recent recoveries get reduced weight
                    age_factor = 1.0 - ((now - event.recovery_time).total_seconds() / 3600.0)
                    score += severity_weights[event.severity] * age_factor * 0.5
            
            # Cap at 100
            return min(score, 100.0)
    
    def get_incident_history(
        self,
        limit: int = 50,
        severity: Optional[DegradationSeverity] = None,
        component: Optional[str] = None,
        include_resolved: bool = True,
    ) -> List[Dict]:
        """Get incident history with filtering."""
        with self.lock:
            result = []
            
            for event in reversed(self.history):
                if not include_resolved and event.recovery_time:
                    continue
                if severity and event.severity != severity:
                    continue
                if component and component not in event.component:
                    continue
                
                result.append(event.to_dict())
                if len(result) >= limit:
                    break
            
            return result
    
    def cleanup_old_events(self):
        """Clean up events older than retention period."""
        with self.lock:
            cutoff = datetime.now() - timedelta(days=self.retention_days)
            original_count = len(self.history)
            
            # Remove old events from history
            self.history = deque(
                (e for e in self.history if e.timestamp > cutoff),
                maxlen=self.max_history,
            )
            
            # Also clean active incidents that are very old (shouldn't happen but just in case)
            very_old_cutoff = datetime.now() - timedelta(days=self.retention_days * 2)
            active_to_remove = [
                incident_id for incident_id, event in self.active_incidents.items()
                if event.timestamp < very_old_cutoff
            ]
            for incident_id in active_to_remove:
                del self.active_incidents[incident_id]
            
            if original_count != len(self.history) or active_to_remove:
                logger.info(
                    f"Cleaned up degradation events: removed {original_count - len(self.history)} from history, "
                    f"{len(active_to_remove)} from active incidents",
                    extra={
                        "remaining_history": len(self.history),
                        "remaining_active": len(self.active_incidents),
                    },
                )


# Singleton instance
_degradation_monitor: Optional[DegradationMonitor] = None

def get_degradation_monitor() -> DegradationMonitor:
    """Get or create the degradation monitor singleton."""
    global _degradation_monitor
    if _degradation_monitor is None:
        _degradation_monitor = DegradationMonitor()
    return _degradation_monitor


def record_model_load_failure(
    model_name: str,
    error: str,
    gpu_memory_mb: Optional[float] = None,
) -> str:
    """Convenience function to record model load failure."""
    monitor = get_degradation_monitor()
    return monitor.record_model_load_failure(model_name, error, gpu_memory_mb)


def record_model_inference_failure(
    model_name: str,
    request_id: str,
    error: str,
    duration: Optional[float] = None,
) -> str:
    """Convenience function to record model inference failure."""
    monitor = get_degradation_monitor()
    return monitor.record_model_inference_failure(model_name, request_id, error, duration)


def record_gpu_memory_exhaustion(
    device_id: int,
    required_mb: float,
    available_mb: float,
    model_name: str,
) -> str:
    """Convenience function to record GPU memory exhaustion."""
    monitor = get_degradation_monitor()
    return monitor.record_gpu_memory_exhaustion(device_id, required_mb, available_mb, model_name)


def get_system_degradation_status() -> Dict:
    """Convenience function to get system degradation status."""
    monitor = get_degradation_monitor()
    return monitor.get_system_status()