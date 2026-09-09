"""
SatQuery AI — Degradation Response Schemas
"""
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DegradationIncidentResponse(BaseModel):
    """Response for a single degradation incident."""
    incident_id: str = Field(..., description="Unique incident identifier")
    timestamp: datetime = Field(..., description="When the incident started")
    degradation_type: str = Field(..., description="Type of degradation")
    severity: str = Field(..., description="Severity level: info, warning, error, critical")
    component: str = Field(..., description="Component affected")
    message: str = Field(..., description="Human-readable message")
    details: Dict = Field(..., description="Additional details about the incident")
    recovery_time: Optional[datetime] = Field(None, description="When the incident was resolved")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds (if resolved)")
    is_resolved: bool = Field(..., description="Whether the incident has been resolved")


class DegradationStatusResponse(BaseModel):
    """Response for system degradation status."""
    system_status: str = Field(..., description="Overall system status: healthy, warning, degraded, critical")
    active_incidents: int = Field(..., description="Number of active incidents")
    active_by_severity: Dict[str, int] = Field(..., description="Active incidents by severity")
    recent_24h: int = Field(..., description="Incidents in the last 24 hours")
    recent_by_severity: Dict[str, int] = Field(..., description="Recent incidents by severity")
    degradation_score: float = Field(..., description="Degradation score (0-100, higher is worse)")
    most_critical_incident: Optional[DegradationIncidentResponse] = Field(
        None, description="Most critical active incident"
    )
    timestamp: datetime = Field(..., description="When this status was generated")


class DegradationHistoryResponse(BaseModel):
    """Response for degradation incident history."""
    incidents: List[DegradationIncidentResponse] = Field(..., description="List of incidents")
    total_count: int = Field(..., description="Total incidents in history")
    limit: int = Field(..., description="Limit applied to results")
    has_more: bool = Field(..., description="Whether there are more incidents beyond the limit")


class DegradationRecoveryResponse(BaseModel):
    """Response for recovery operation."""
    success: bool = Field(..., description="Whether recovery was recorded")
    incident_id: str = Field(..., description="Incident identifier")
    recovery_time: Optional[datetime] = Field(None, description="When recovery was recorded")
    message: str = Field(..., description="Recovery message")