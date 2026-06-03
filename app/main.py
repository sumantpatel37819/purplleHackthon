"""
main.py — FastAPI entrypoint for Store Intelligence API

Features:
- 6 REST endpoints (ingest, metrics, funnel, heatmap, anomalies, health)
- Structured request logging: trace_id, store_id, endpoint, latency_ms, status_code
- Graceful degradation: DB unavailable → HTTP 503 with structured error body
- No raw stack traces in responses
- Store ID aliases: STORE_BLR_002 ↔ ST1008 both accepted
"""
import time
import uuid
import logging
import json
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

# Setup path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import init_db, get_db
from app.models import (
    IngestRequest, IngestResponse, MetricsResponse,
    FunnelResponse, HeatmapResponse, AnomaliesResponse,
    HealthResponse, ErrorResponse
)
from app.ingestion import ingest_events
from app.metrics import get_metrics
from app.funnel import get_funnel
from app.heatmap import get_heatmap
from app.anomalies import detect_anomalies
from app.health import get_health

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("store-intelligence")

# ─── Store ID Aliases ────────────────────────────────────────────────────────

STORE_ALIASES = {
    "ST1008": "STORE_BLR_002",
    "STORE_BLR_002": "STORE_BLR_002",
}

VALID_STORES = set(STORE_ALIASES.values())


def resolve_store_id(store_id: str) -> str:
    resolved = STORE_ALIASES.get(store_id.upper(), store_id)
    return resolved


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Store Intelligence API...")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down Store Intelligence API.")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Store Intelligence API",
    description="Real-time retail analytics from CCTV footage — Purplle Tech Challenge 2026",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: Structured Request Logging ──────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    trace_id = str(uuid.uuid4())[:8]
    request.state.trace_id = trace_id
    start = time.time()

    # Extract store_id from path if present
    store_id = None
    parts = request.url.path.split("/")
    if "stores" in parts:
        idx = parts.index("stores")
        if idx + 1 < len(parts):
            store_id = parts[idx + 1]

    try:
        response = await call_next(request)
        latency_ms = round((time.time() - start) * 1000, 2)

        log_data = {
            "trace_id": trace_id,
            "method": request.method,
            "endpoint": request.url.path,
            "store_id": store_id,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
        }
        logger.info(json.dumps(log_data))
        response.headers["X-Trace-Id"] = trace_id
        return response

    except OperationalError as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        logger.error(json.dumps({
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "error": "DATABASE_UNAVAILABLE",
            "latency_ms": latency_ms,
        }))
        return JSONResponse(
            status_code=503,
            content={
                "error": "SERVICE_UNAVAILABLE",
                "detail": "Database is temporarily unavailable. Please retry.",
                "trace_id": trace_id,
            },
        )
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        logger.error(json.dumps({
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "error": str(type(e).__name__),
            "latency_ms": latency_ms,
        }))
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "detail": "An unexpected error occurred.",
                "trace_id": trace_id,
            },
        )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def validate_store(store_id: str) -> str:
    """Resolve and validate a store ID. Raises 404 if unknown."""
    resolved = resolve_store_id(store_id)
    if resolved not in VALID_STORES:
        raise HTTPException(
            status_code=404,
            detail=f"Store '{store_id}' not found. Valid stores: {sorted(VALID_STORES)}"
        )
    return resolved


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/events/ingest", response_model=IngestResponse, tags=["Events"])
async def post_ingest(
    request_body: IngestRequest,
    db: Session = Depends(get_db),
):
    """
    Ingest a batch of up to 500 events from the detection pipeline.
    - **Idempotent**: Duplicate event_ids are silently ignored (not counted as errors).
    - **Partial success**: Malformed events return error details but valid ones are stored.
    """
    return ingest_events(request_body, db)


@app.get("/stores/{store_id}/metrics", response_model=MetricsResponse, tags=["Analytics"])
async def get_store_metrics(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Real-time store analytics metrics for today.
    - unique_visitors, conversion_rate, avg_dwell, queue_depth, abandonment_rate
    - Zone breakdown with per-zone dwell times
    """
    resolved = validate_store(store_id)
    return get_metrics(resolved, db, date)


@app.get("/stores/{store_id}/funnel", response_model=FunnelResponse, tags=["Analytics"])
async def get_store_funnel(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
    Sessions are deduplicated — re-entries don't double-count visitors.
    """
    resolved = validate_store(store_id)
    return get_funnel(resolved, db, date)


@app.get("/stores/{store_id}/heatmap", response_model=HeatmapResponse, tags=["Analytics"])
async def get_store_heatmap(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Zone visit frequency + avg dwell time, normalized 0-100, ready for grid heatmap rendering.
    Includes data_confidence flag if fewer than 20 sessions in window.
    """
    resolved = validate_store(store_id)
    return get_heatmap(resolved, db, date)


@app.get("/stores/{store_id}/anomalies", response_model=AnomaliesResponse, tags=["Analytics"])
async def get_store_anomalies(
    store_id: str,
    date: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Active operational anomalies: queue spike, conversion drop, dead zones, stale feeds.
    Each anomaly includes severity (INFO/WARN/CRITICAL) and suggested_action.
    """
    resolved = validate_store(store_id)
    return detect_anomalies(resolved, db, date)


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check(db: Session = Depends(get_db)):
    """
    Service health status. Includes STALE_FEED warning if any camera
    has no events for >10 minutes. Used by on-call engineers.
    """
    return get_health(db)


@app.get("/", tags=["Operations"])
async def root():
    """API root — returns service info."""
    return {
        "service": "Store Intelligence API",
        "version": "1.0.0",
        "challenge": "Purplle Tech Challenge 2026",
        "docs": "/docs",
        "health": "/health",
    }
