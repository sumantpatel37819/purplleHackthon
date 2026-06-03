# PROMPT: Write pytest tests for metrics, funnel, heatmap, and anomaly endpoints. Include: 
# metrics with visitors and POS data, funnel drop-off calculation, heatmap normalization,
# anomaly detection for dead zones and conversion drops. Use the same DB fixture pattern.
# CHANGES MADE: Added re-entry test to verify visitor deduplication in funnel,
# adjusted anomaly test thresholds to match our configured constants.

import pytest
import uuid
from httpx import AsyncClient, ASGITransport

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app
from app.database import Base, engine, SessionLocal


@pytest.fixture(autouse=True)
def setup_db(monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import app.database as db_module

    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSession = sessionmaker(bind=test_engine)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "SessionLocal", TestSession)
    yield
    Base.metadata.drop_all(bind=test_engine)


def make_event(event_type, visitor_id=None, zone_id=None, is_staff=False,
               dwell_ms=0, confidence=0.9, camera_id="CAM_ENTRY_01",
               queue_depth=None, session_seq=1, timestamp="2026-04-10T10:30:00Z"):
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": "STORE_BLR_002",
        "camera_id": camera_id,
        "visitor_id": visitor_id or f"VIS_{uuid.uuid4().hex[:6]}",
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {"queue_depth": queue_depth, "sku_zone": zone_id, "session_seq": session_seq},
    }


async def post_events(client, events):
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 200
    return resp.json()


@pytest.mark.asyncio
async def test_metrics_with_visitors():
    """Metrics should correctly count unique non-staff visitors."""
    events = [
        make_event("ENTRY", visitor_id="VIS_001"),
        make_event("ENTRY", visitor_id="VIS_002"),
        make_event("ENTRY", visitor_id="VIS_003"),
        make_event("ENTRY", visitor_id="VIS_001", is_staff=False),  # same visitor (dedup by visitor_id)
        make_event("ENTRY", is_staff=True),  # staff - excluded
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await post_events(client, events)
        resp = await client.get("/stores/STORE_BLR_002/metrics")
    assert resp.status_code == 200
    data = resp.json()
    # 3 unique non-staff visitor_ids
    assert data["unique_visitors"] == 3
    assert data["conversion_rate"] == 0.0  # no POS data in test DB
    assert "zone_breakdown" in data


@pytest.mark.asyncio
async def test_metrics_store_alias():
    """Both ST1008 and STORE_BLR_002 should return the same metrics."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r1 = await client.get("/stores/STORE_BLR_002/metrics")
        r2 = await client.get("/stores/ST1008/metrics")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["store_id"] == r2.json()["store_id"]


@pytest.mark.asyncio
async def test_metrics_unknown_store():
    """Unknown store_id should return 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stores/STORE_XYZ_999/metrics")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_funnel_deduplication():
    """Re-entries should not double-count in funnel — ENTRY deduplicated by visitor_id."""
    events = [
        make_event("ENTRY", visitor_id="VIS_001"),
        make_event("ENTRY", visitor_id="VIS_002"),
        # Re-entry of VIS_001 — should not create a 3rd funnel entry
        make_event("REENTRY", visitor_id="VIS_001"),
        # VIS_001 and VIS_002 visit zone
        make_event("ZONE_ENTER", visitor_id="VIS_001", zone_id="SKINCARE", camera_id="CAM_FLOOR_01"),
        make_event("ZONE_ENTER", visitor_id="VIS_002", zone_id="MAKEUP", camera_id="CAM_FLOOR_01"),
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await post_events(client, events)
        resp = await client.get("/stores/STORE_BLR_002/funnel")
    assert resp.status_code == 200
    data = resp.json()
    stages = {s["stage"]: s["count"] for s in data["stages"]}
    assert stages["Entry"] == 2           # 2 unique ENTRY events (not REENTRY)
    assert stages["Zone Visit"] == 2      # Both visited zones
    assert stages["Purchase"] == 0        # No POS in test DB


@pytest.mark.asyncio
async def test_funnel_empty_store():
    """Empty store (no events) should return all zeros without crashing."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stores/STORE_BLR_002/funnel")
    assert resp.status_code == 200
    data = resp.json()
    assert all(s["count"] == 0 for s in data["stages"])


@pytest.mark.asyncio
async def test_heatmap_normalization():
    """Zone with most visits should have normalized_score=100.0."""
    events = []
    # SKINCARE: 10 visits
    for i in range(10):
        events.append(make_event("ZONE_ENTER", zone_id="SKINCARE",
                                  visitor_id=f"VIS_{i:03d}", camera_id="CAM_FLOOR_01",
                                  dwell_ms=30000))
    # MAKEUP: 5 visits
    for i in range(5):
        events.append(make_event("ZONE_ENTER", zone_id="MAKEUP",
                                  visitor_id=f"VIS_M{i:03d}", camera_id="CAM_FLOOR_01",
                                  dwell_ms=20000))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await post_events(client, events)
        resp = await client.get("/stores/STORE_BLR_002/heatmap")
    assert resp.status_code == 200
    data = resp.json()
    zones = {z["zone_id"]: z for z in data["zones"]}
    assert "SKINCARE" in zones
    assert zones["SKINCARE"]["normalized_score"] == 100.0
    assert zones["MAKEUP"]["normalized_score"] == 50.0


@pytest.mark.asyncio
async def test_anomalies_response_structure():
    """Anomalies endpoint should return structured response with correct fields."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_anomalies" in data
    assert "checked_at" in data
    assert isinstance(data["active_anomalies"], list)
    # With no events, expect DEAD_ZONE and STALE_FEED anomalies
    anomaly_types = {a["anomaly_type"] for a in data["active_anomalies"]}
    assert "STALE_FEED" in anomaly_types or "DEAD_ZONE" in anomaly_types


@pytest.mark.asyncio
async def test_anomaly_billing_queue_spike():
    """Queue depth above threshold should trigger BILLING_QUEUE_SPIKE anomaly."""
    events = [
        make_event("BILLING_QUEUE_JOIN", zone_id="BILLING", queue_depth=6,
                   visitor_id=f"VIS_{i:03d}", camera_id="CAM_BILLING_01")
        for i in range(3)
    ]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await post_events(client, events)
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
    assert resp.status_code == 200
    types = [a["anomaly_type"] for a in resp.json()["active_anomalies"]]
    assert "BILLING_QUEUE_SPIKE" in types


@pytest.mark.asyncio
async def test_anomaly_severity_fields():
    """Each anomaly must have required fields: severity, suggested_action, detected_at."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/stores/STORE_BLR_002/anomalies")
    data = resp.json()
    for anomaly in data["active_anomalies"]:
        assert anomaly["severity"] in ("INFO", "WARN", "CRITICAL")
        assert len(anomaly["suggested_action"]) > 0
        assert "detected_at" in anomaly
