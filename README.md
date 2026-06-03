# Store Intelligence API — Purplle Tech Challenge 2026

Real-time retail analytics from CCTV footage. Processes 5 camera feeds to produce
behavioural events, then serves them via a production-ready REST API.

**North Star Metric**: Offline Store Conversion Rate = Buyers ÷ Unique Visitors

---

## Quick Start (5 Commands)

```bash
# 1. Clone and enter
git clone <repo-url> && cd purplletask

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 4. (Separate terminal) Run detection pipeline on all camera clips
python pipeline/detect.py --all

# 5. Ingest events into API + verify
python pipeline/ingest_to_api.py
```

Or with Docker (recommended):

```bash
docker compose up --build
```

---

## Detection Pipeline

### Run on all cameras
```bash
python pipeline/detect.py --all
```

### Run on a single camera (faster for testing)
```bash
python pipeline/detect.py --cam 1
```

### Run with frame limit (for quick smoke test)
```bash
python pipeline/detect.py --cam 1 --max-frames 300
```

### Using PowerShell script
```powershell
.\pipeline\run.ps1          # All cameras
.\pipeline\run.ps1 -Cam 1  # Single camera
```

### Output
Events are written to `data/events/cam{N}_events.jsonl` (one file per camera).
Each line is a JSON event in the required schema.

### Ingest events into API
```bash
python pipeline/ingest_to_api.py
```

---

## API Endpoints

Base URL: `http://localhost:8080`

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/ingest` | Ingest batch (≤500 events), idempotent by event_id |
| GET | `/stores/STORE_BLR_002/metrics` | Unique visitors, conversion rate, dwell, queue depth |
| GET | `/stores/STORE_BLR_002/funnel` | Entry→ZoneVisit→BillingQueue→Purchase drop-off |
| GET | `/stores/STORE_BLR_002/heatmap` | Zone heatmap, normalized 0-100 |
| GET | `/stores/STORE_BLR_002/anomalies` | Active anomalies with severity + suggested action |
| GET | `/health` | Service status, STALE_FEED detection |
| GET | `/docs` | Interactive Swagger UI |

### Example: Get Metrics
```bash
curl http://localhost:8080/stores/STORE_BLR_002/metrics
```

```json
{
  "store_id": "STORE_BLR_002",
  "unique_visitors": 47,
  "conversion_rate": 0.2553,
  "avg_dwell_sec": 142.3,
  "queue_depth_current": 2,
  "abandonment_rate": 0.083,
  "zone_breakdown": [
    {"zone_id": "MAKEUP", "avg_dwell_sec": 180.5, "visitor_count": 31},
    {"zone_id": "SKINCARE", "avg_dwell_sec": 95.2, "visitor_count": 28}
  ]
}
```

### Example: Ingest Events
```bash
curl -X POST http://localhost:8080/events/ingest \
  -H "Content-Type: application/json" \
  -d '{"events": [{"event_id": "...", "store_id": "STORE_BLR_002", ...}]}'
```

Store ID aliases: `ST1008` and `STORE_BLR_002` both work.

---

## Live Dashboard (Bonus Part E)

```bash
# Start API first, then:
python dashboard/live_dashboard.py

# With event replay (simulates real-time from video):
python dashboard/live_dashboard.py --replay
```

Shows live-updating terminal dashboard with metrics, funnel, zone heatmap, and anomalies.

---

## Running Tests

```bash
# All tests with coverage
pytest

# Specific test file
pytest tests/test_ingestion.py -v

# Coverage report only
pytest --cov=app --cov-report=term-missing
```

Target: >70% statement coverage on `app/` and `pipeline/` modules.

---

## Project Structure

```
purplletask/
├── pipeline/           # CCTV processing pipeline
│   ├── detect.py       # Main: YOLOv8n + ByteTrack
│   ├── zone_mapper.py  # Polygon zone containment
│   ├── staff_detector.py  # Uniform color heuristic
│   ├── tracker.py      # Re-ID, re-entry, dwell tracking
│   ├── emit.py         # Event schema + JSONL writer
│   ├── ingest_to_api.py # Batch POST to API
│   └── run.ps1         # PowerShell runner
├── app/                # FastAPI REST API
│   ├── main.py         # Entrypoint + middleware
│   ├── models.py       # Pydantic schemas
│   ├── database.py     # SQLite + SQLAlchemy
│   ├── ingestion.py    # POST /events/ingest
│   ├── metrics.py      # GET /metrics
│   ├── funnel.py       # GET /funnel
│   ├── heatmap.py      # GET /heatmap
│   ├── anomalies.py    # GET /anomalies
│   └── health.py       # GET /health
├── dashboard/
│   └── live_dashboard.py  # Rich terminal dashboard
├── tests/              # pytest test suite
├── docs/
│   ├── DESIGN.md       # Architecture + AI decisions
│   └── CHOICES.md      # 3 engineering decisions
├── data/
│   ├── store_layout.json  # Zone polygon definitions
│   └── events/            # Pipeline output (JSONL)
├── resources/          # Input data (CCTV clips, POS CSV)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Store Layout

**Store**: Brigade Road, Bangalore (`STORE_BLR_002` / `ST1008`)

| Camera | ID | Zones |
|--------|----|-------|
| CAM 1 | CAM_ENTRY_01 | Entry/Exit threshold |
| CAM 2 | CAM_ENTRY_02 | Secondary Entry/Exit |
| CAM 3 | CAM_FLOOR_01 | SKINCARE, MAKEUP, HAIRCARE |
| CAM 4 | CAM_BILLING_01 | BILLING |
| CAM 5 | CAM_FLOOR_02 | FRAGRANCE, ACCESSORIES |

---

## Architecture Decisions

See [docs/DESIGN.md](docs/DESIGN.md) for full architecture and AI-assisted decisions.
See [docs/CHOICES.md](docs/CHOICES.md) for 3 key engineering trade-offs.

---

## Key Design Choices

1. **YOLOv8n + ByteTrack**: CPU-friendly detection and tracking
2. **SQLite with WAL mode**: Zero dependencies, idempotent by `event_id` PRIMARY KEY
3. **Unified event schema**: Single Pydantic model for all 8 event types
4. **POS correlation**: 5-minute time window: billing zone visit → purchase
5. **Staff detection**: HSV torso color uniformity (>55% uniform = staff)

---

## Edge Cases Handled

| Edge Case | Handling |
|-----------|---------|
| Group entry | N separate ENTRY events for N people within 2s |
| Re-entry | REENTRY event (not ENTRY) after EXIT; visitor_id preserved |
| Staff movement | `is_staff=true`; excluded from all customer metrics |
| Partial occlusion | Low-confidence events emitted with confidence field set (not suppressed) |
| Camera overlap | Cross-camera dedup within 5-second window by visitor_id |
| Billing queue | `BILLING_QUEUE_JOIN` with queue_depth; `BILLING_QUEUE_ABANDON` if leaves without purchase |
| Empty store | API returns 0 counts (not null/crash) |
| Zero purchases | `conversion_rate=0.0`, `data_confidence=LOW` |
| DB unavailable | HTTP 503 with structured JSON error, no raw stack traces |

---

## Requirements

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- ~4GB disk space (YOLOv8n model + video files)
- No GPU required (CPU inference)
- No paid API keys required

---

## Contact

Challenge submission — Purplle Tech Challenge 2026 Round 2
