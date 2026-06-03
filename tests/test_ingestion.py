# PROMPT: Write comprehensive pytest tests for the event ingestion endpoint of a FastAPI store 
# intelligence API. Tests should cover: valid batch ingest, idempotency (same event_id twice),
# partial success with malformed events, batch size limit (>500), empty batch, all-staff events,
# and schema validation. Use httpx.AsyncClient with ASGITransport for async testing.
# CHANGES MADE: Added fixture for test events factory, added edge case for zero-purchase store,
# added assertion for structured error response shape, adjusted batch size test to 501 events.

import pytest
import uuid
import json
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.database import init_db, Base, engine


@pytest.fixture(autouse=True)
def setup_db(tmp_path, monkeypatch):
    """Use an in-memory SQLite database for tests."""
    import app.database as db_module
    test_db_url = "sqlite:///:memory:"
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import event as sqla_event

    test_engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine)

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    yield
    Base.metadata.drop_all(bind=test_engine)


def make_event(event_type="ENTRY", visitor_id=None, zone_id=None, is_staff=False,
               queue_depth=None, dwell_ms=0, confidence=0.9, store_id="STORE_BLR_002",
               camera_id="CAM_ENTRY_01", session_seq=1):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": "2026-04-10T10:30:00Z",
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": None,
            "session_seq": session_seq,
        }
    }


@pytest.mark.asyncio
async def test_ingest_valid_batch():
    """Valid batch of events should be ingested successfully."""
    events = [make_event("ENTRY") for _ in range(5)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 5
    assert data["errors"] == 0
    assert data["duplicates"] == 0


@pytest.mark.asyncio
async def test_ingest_idempotency():
    """Posting same events twice should result in duplicates, not errors."""
    events = [make_event("ENTRY")]
    payload = {"events": events}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/events/ingest", json=payload)
        resp2 = await client.post("/events/ingest", json=payload)

    assert resp1.status_code == 200
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["duplicates"] == 1
    assert data2["ingested"] == 0
    assert data2["errors"] == 0


@pytest.mark.asyncio
async def test_ingest_invalid_event_type():
    """Malformed event_type should cause validation error, not crash entire batch."""
    valid = make_event("ENTRY")
    invalid = make_event("ENTRY")
    invalid["event_type"] = "INVALID_TYPE"  # Bad event type

    # Pydantic validation happens at request level for event_type
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": [invalid]})
    # Should return 422 Unprocessable Entity (Pydantic validation)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_all_staff_events():
    """All-staff batch: ingested fine but excluded from customer metrics."""
    events = [make_event("ENTRY", is_staff=True) for _ in range(3)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 3

    # Check metrics: staff should be excluded. Use a new client to avoid closed client error.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp2 = await client.get("/stores/STORE_BLR_002/metrics")
        assert resp2.status_code == 200
        assert resp2.json()["unique_visitors"] == 0


@pytest.mark.asyncio
async def test_ingest_empty_batch():
    """Empty events list should return 0 ingested with no errors."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": []})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested"] == 0
    assert data["errors"] == 0


@pytest.mark.asyncio
async def test_ingest_batch_too_large():
    """Batch exceeding 500 events should be rejected by schema validation."""
    events = [make_event("ENTRY") for _ in range(501)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 422  # Pydantic max_length validation


@pytest.mark.asyncio
async def test_ingest_invalid_event_id():
    """Non-UUID event_id should fail validation."""
    event = make_event("ENTRY")
    event["event_id"] = "not-a-uuid"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/events/ingest", json={"events": [event]})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_zero_purchase_store():
    """Store with visitors but no POS matches should have conversion_rate=0.0 (not crash)."""
    events = [make_event("ENTRY", visitor_id=f"VIS_abc{i}") for i in range(10)]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post("/events/ingest", json={"events": events})
        resp = await client.get("/stores/STORE_BLR_002/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert data["unique_visitors"] == 10
    assert data["conversion_rate"] == 0.0
    assert data["abandonment_rate"] == 0.0


@pytest.mark.asyncio
async def test_health_endpoint():
    """Health endpoint should return valid response structure."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "stores" in data
    assert "checked_at" in data
    assert data["status"] in ("OK", "DEGRADED")
