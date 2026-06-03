# DESIGN.md — Store Intelligence System Architecture

## Overview

This system transforms raw CCTV footage from a Purplle retail store into real-time
business analytics. The pipeline runs offline video through a detection engine,
emits structured behavioral events, and exposes them via a REST API.

```
Raw CCTV (5 cameras)
        │
        ▼
┌───────────────────┐
│  Detection Layer  │  YOLOv8n + ByteTrack per frame
│  (pipeline/)      │  → person detection + tracking
│                   │  → staff classification (HSV uniform)
│                   │  → zone mapping (polygon containment)
│                   │  → re-entry detection (trajectory fingerprint)
└────────┬──────────┘
         │ JSONL events (5 files, one per camera)
         ▼
┌───────────────────┐
│  Ingestion Layer  │  POST /events/ingest
│  (app/ingestion)  │  → Pydantic validation
│                   │  → Idempotent upsert by event_id
│                   │  → Partial success (malformed events don't fail batch)
└────────┬──────────┘
         │ SQLite events.db
         ▼
┌───────────────────┐
│  Analytics Layer  │  GET /stores/{id}/metrics
│  (app/metrics,    │  GET /stores/{id}/funnel
│   funnel, heatmap │  GET /stores/{id}/heatmap
│   anomalies)      │  GET /stores/{id}/anomalies
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Live Dashboard   │  Rich terminal (real-time replay)
│  (dashboard/)     │  2s refresh, reads from API
└───────────────────┘
```

## System Constraints

- **No GPU required**: YOLOv8n (nano) runs on CPU. Inference speed: ~2-4 fps on CPU,
  which is fine since we sample every 3rd frame (5fps effective from 15fps video).
- **Single store deployment**: This submission targets STORE_BLR_002 (Brigade Road, Bangalore).
  The architecture supports multi-store via the `store_id` field in all events and queries.
- **Batch mode**: The detection pipeline processes video offline and ingests events into the API.
  Part E (Live Dashboard) simulates real-time by replaying events at 2s intervals.

## Component Details

### Detection Layer (`pipeline/`)

**`detect.py`** — Main orchestrator:
- Loads YOLOv8n model (COCO pretrained, `person` class = class ID 0)
- Opens each MP4 with OpenCV
- Runs `model.track()` with ByteTrack for stable track IDs across frames
- Samples every 3rd frame to balance speed vs. accuracy
- Delegates to zone_mapper, staff_detector, tracker, and emit

**`zone_mapper.py`** — Spatial reasoning:
- Stores zone polygons in normalized coordinates (0-1), resolution-independent
- Uses ray-casting algorithm for point-in-polygon tests
- Entry line detection: monitors centroid crossing a horizontal threshold line
- Direction: "top-to-bottom" = entering; "bottom-to-top" = exiting

**`staff_detector.py`** — Uniform classification:
- Extracts torso region (25%-65% of bounding box height)
- Converts to HSV, checks against known uniform color ranges
- Ranges: black (value < 60), dark navy (hue 100-130), purple (hue 130-160), white
- Returns `is_staff=True` if >55% of torso pixels match a uniform color

**`tracker.py`** — Session and Re-ID management:
- Maintains active track states (position, zone, session_seq)
- Re-entry detection: compares aspect ratio fingerprint of new detections against
  recently-exited tracks within a 90-second window
- Group entry: detects when 2+ ENTRY events occur within 2 seconds
- Zone dwell: tracks time in zone, emits ZONE_DWELL every 30s of continuous presence

**`emit.py`** — Event schema enforcement:
- Generates UUID v4 for every event_id
- Converts frame index to ISO-8601 UTC timestamp using video base time
- Validates event_type against allowed catalogue before emitting

### Intelligence API (`app/`)

**`main.py`** — FastAPI app:
- 6 REST endpoints (see API reference below)
- Request logging middleware: logs `trace_id`, `store_id`, `endpoint`, `latency_ms`, `status_code`
- Graceful degradation: `OperationalError` → 503 with structured JSON body
- Store ID aliases: `ST1008` ↔ `STORE_BLR_002` both accepted

**`database.py`** — SQLite with SQLAlchemy:
- WAL mode enabled for better concurrent read performance
- `event_id` is `PRIMARY KEY` — enforces idempotency at DB level
- POS transactions loaded from CSV on startup (idempotent)
- Indexes on `(store_id, timestamp)` and `event_type` for fast query performance

**`metrics.py`** — Real-time analytics:
- Conversion rate via POS correlation: visitor in billing zone within 5-min window before transaction
- Excludes `is_staff=true` events from all customer metrics
- Returns `data_confidence=LOW` if fewer than 20 visitor sessions

**`funnel.py`** — Conversion funnel:
- 4 stages: Entry → Zone Visit → Billing Queue → Purchase
- Deduplication by `visitor_id` — re-entries don't double-count
- `REENTRY` events excluded from Entry count

**`anomalies.py`** — Operational anomaly detection:
- `BILLING_QUEUE_SPIKE`: queue_depth > 4 (WARN), > 8 (CRITICAL)
- `CONVERSION_DROP`: rate < 30% below baseline (configurable)
- `DEAD_ZONE`: no zone visits for >30 minutes during store hours
- `STALE_FEED`: camera with no events for today

---

## AI-Assisted Decisions

### 1. ByteTrack vs. DeepSORT for People Tracking

**What AI suggested (Claude Sonnet)**: Use DeepSORT with ReID because it handles re-identification
natively and is well-documented for retail scenarios.

**What I chose**: ByteTrack (built into ultralytics `model.track()`).

**Why I disagreed**: DeepSORT requires an appearance model (typically a trained ReID network)
that needs GPU for real-time performance. ByteTrack assigns stable track IDs using only
bounding box IoU — much faster on CPU. For our retail use case (one camera, 15fps),
ByteTrack provides sufficient track continuity. Re-identification across camera cuts and
re-entry detection is handled separately by `tracker.py` using trajectory fingerprinting.

### 2. Staff Detection Approach

**What AI suggested (ChatGPT)**: Fine-tune a classifier on staff vs. customer images, or use
a VLM (GPT-4V) to classify each bounding box crop as staff/customer.

**What I chose**: HSV uniform color heuristic in `staff_detector.py`.

**Why I partially agreed**: The VLM approach would be more accurate but introduces API costs,
latency per frame, and rate limits. For batch processing of 5 CCTV clips, calling GPT-4V
on every bounding box is impractical. The HSV heuristic works because retail staff reliably
wear uniforms while customers wear varied clothing. I document the limitation: if staff wear
multi-colored uniforms, this heuristic will misclassify them.

**AI's insight I kept**: The idea of focusing on the torso region (not the whole bounding box)
was suggested by AI — it reduces noise from bags, shoes, and background.

### 3. POS Correlation Window

**What AI suggested (Gemini)**: Use customer_id or phone number from POS to link visitors
to transactions exactly.

**What I chose**: Time-window correlation — visitor in billing zone within 5 minutes before
a POS transaction = converted visitor.

**Why**: The POS data has no customer identity that maps to our visitor_ids (which are
anonymized CCTV-derived tokens). The 5-minute window is a standard assumption in retail
analytics for correlating dwell time with purchase. I document this limitation in CHOICES.md.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /events/ingest | Batch ingest (≤500 events), idempotent by event_id |
| GET | /stores/{id}/metrics | Unique visitors, conversion rate, dwell, queue depth |
| GET | /stores/{id}/funnel | Entry→ZoneVisit→BillingQueue→Purchase with drop-off % |
| GET | /stores/{id}/heatmap | Zone visit frequency + avg dwell, normalized 0-100 |
| GET | /stores/{id}/anomalies | Active anomalies with severity and suggested_action |
| GET | /health | Service status, per-store last event, STALE_FEED detection |

## Limitations and Known Edge Cases

1. **Face blur**: Videos have full-face blur applied — our staff detection relies on uniform
   color, not facial recognition. This is actually a privacy advantage.

2. **Camera overlap**: CAM_ENTRY_01 and CAM_FLOOR_01 have overlapping field of view near the
   entrance. We implement cross-camera deduplication by checking if the same visitor_id
   appears on both cameras within a 5-second window.

3. **No audio**: CCTV has no audio, so verbal interactions are invisible to the system.

4. **POS correlation accuracy**: Time-window correlation has false positives (multiple customers
   in billing zone before a single transaction) and false negatives (customer who doesn't dwell
   long in billing zone). Stated confidence metric acknowledges this.

5. **Lighting variation**: The footage includes mixed lighting (natural, fluorescent). YOLO handles
   this reasonably well, but we observe confidence drops during bright sunlight glare.
