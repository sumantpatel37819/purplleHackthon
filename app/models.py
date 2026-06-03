"""
Pydantic models — event schema and API response models.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
import uuid


# ─── Event Types ────────────────────────────────────────────────────────────

VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
}


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int = 1


class Event(BaseModel):
    """Core event schema — matches the required output schema exactly."""
    event_id: str = Field(..., description="UUID v4 — globally unique")
    store_id: str = Field(..., description="Store identifier, e.g. STORE_BLR_002")
    camera_id: str = Field(..., description="Camera identifier, e.g. CAM_ENTRY_01")
    visitor_id: str = Field(..., description="Per-session Re-ID token, e.g. VIS_c8a2f1")
    event_type: str = Field(..., description="Event type from the catalogue")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(None, description="Zone; null for ENTRY/EXIT events")
    dwell_ms: int = Field(0, ge=0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(False, description="True if classified as store staff")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence 0-1")
    metadata: Optional[EventMetadata] = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        if v not in VALID_EVENT_TYPES:
            raise ValueError(f"event_type must be one of {VALID_EVENT_TYPES}, got '{v}'")
        return v

    @field_validator("event_id")
    @classmethod
    def validate_event_id(cls, v):
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("event_id must be a valid UUID v4")
        return v


class IngestRequest(BaseModel):
    events: List[Event] = Field(..., max_length=500, description="Batch of up to 500 events")


class IngestResponse(BaseModel):
    ingested: int
    duplicates: int
    errors: int
    error_details: List[str] = []


# ─── Metrics ────────────────────────────────────────────────────────────────

class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_sec: float
    visitor_count: int


class MetricsResponse(BaseModel):
    store_id: str
    window: str = "today"
    unique_visitors: int
    conversion_rate: float = Field(..., description="0.0 – 1.0; buyers / unique visitors")
    avg_dwell_sec: float
    total_dwell_observations: int
    queue_depth_current: int
    abandonment_rate: float = Field(..., description="BILLING_QUEUE_ABANDON events / BILLING_QUEUE_JOIN events")
    zone_breakdown: List[ZoneDwellMetric] = []
    data_confidence: str = Field("HIGH", description="HIGH / LOW if fewer than 20 sessions")


# ─── Funnel ─────────────────────────────────────────────────────────────────

class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float = Field(0.0, description="% drop-off from previous stage")


class FunnelResponse(BaseModel):
    store_id: str
    window: str = "today"
    stages: List[FunnelStage]
    note: str = ""


# ─── Heatmap ────────────────────────────────────────────────────────────────

class HeatmapZone(BaseModel):
    zone_id: str
    sku_zone: Optional[str]
    visit_count: int
    avg_dwell_sec: float
    normalized_score: float = Field(..., ge=0.0, le=100.0)


class HeatmapResponse(BaseModel):
    store_id: str
    window: str = "today"
    zones: List[HeatmapZone]
    data_confidence: str = "HIGH"


# ─── Anomalies ──────────────────────────────────────────────────────────────

class Anomaly(BaseModel):
    anomaly_id: str
    anomaly_type: str  # BILLING_QUEUE_SPIKE | CONVERSION_DROP | DEAD_ZONE | STALE_FEED
    severity: str      # INFO | WARN | CRITICAL
    description: str
    suggested_action: str
    detected_at: str
    zone_id: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None


class AnomaliesResponse(BaseModel):
    store_id: str
    active_anomalies: List[Anomaly]
    checked_at: str


# ─── Health ─────────────────────────────────────────────────────────────────

class StoreHealth(BaseModel):
    store_id: str
    status: str  # OK | STALE | NO_DATA
    last_event_timestamp: Optional[str]
    lag_seconds: Optional[float]
    stale_feed: bool = False


class HealthResponse(BaseModel):
    service: str = "store-intelligence-api"
    status: str  # OK | DEGRADED
    version: str = "1.0.0"
    stores: List[StoreHealth]
    checked_at: str


# ─── Error ──────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    trace_id: Optional[str] = None
