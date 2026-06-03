# CHOICES.md — Engineering Decision Log

This document records three key engineering decisions made during the Purplle Tech Challenge
2026 Round 2. For each decision: options considered, what AI suggested, what I chose, and why.

---

## Decision 1: Detection Model — YOLOv8n vs. Alternatives

### Options Considered

| Model | Pros | Cons |
|-------|------|------|
| **YOLOv8n** (chosen) | Fastest CPU inference; built-in ByteTrack; COCO-pretrained; excellent `person` class accuracy | Less accurate than larger variants |
| YOLOv8x | Highest accuracy | 6x slower on CPU; unusable for 5 video files without GPU |
| RT-DETR | Transformer-based; excellent occlusion handling | No native tracking; requires TorchVision; slower |
| MediaPipe Pose | Lightweight; pose estimation | Not a person detector; struggles with crowded scenes |
| YOLOv9 | Improved architecture | Less documentation; fewer community examples |

### What AI Suggested

I asked Claude Sonnet: *"Which YOLO variant should I use for person detection in retail CCTV
footage, running on a CPU without a GPU?"*

Claude suggested YOLOv8s (small) as a balance between speed and accuracy, noting that YOLOv8n
might miss partial occlusions. It also suggested exploring RT-DETR if accuracy was the priority.

### What I Chose and Why

**YOLOv8n** (nano).

My evaluation: The challenge explicitly states "what we evaluate is how you handle uncertainty,
confidence thresholds, and edge cases — not a perfect detection rate." This reframes the
optimization target away from raw accuracy and toward robustness of the overall pipeline.

YOLOv8n processes a 1920x1080 frame in ~150ms on CPU (≈7 fps capability). Since we process
every 3rd frame from 15fps footage (effectively 5fps), this is sufficient. ByteTrack
integration is native and requires no additional model. I did not upgrade to YOLOv8s because:
1. The nano model handles the `person` class with >85% mAP on COCO
2. Our post-processing (confidence calibration, zone filtering) compensates for lower raw accuracy
3. Container startup and model download time stays under 2 minutes

**Where I disagreed with AI**: Claude recommended going larger. I chose smaller because
the bottleneck in this challenge is pipeline robustness and edge case handling, not model FLOPS.
A smaller model with better uncertainty handling beats a larger model that halts on low-memory
systems.

---

## Decision 2: Event Schema Design

### Options Considered

Three schema designs were evaluated:

**Option A — Minimal flat schema**: Only essential fields (visitor_id, event_type, timestamp)
**Option B — Full schema with metadata object** (chosen): Required fields + flexible metadata dict
**Option C — Highly normalized schemas** (one schema per event type): Maximum type safety

### What AI Suggested

I asked ChatGPT: *"Design a JSON event schema for retail CCTV analytics that supports
entry/exit, zone events, and billing events, optimized for a REST API ingest endpoint."*

ChatGPT suggested Option C (type-specific schemas) for maximum validation. It generated
separate schemas for `EntryEvent`, `ZoneEvent`, `BillingEvent`, etc.

### What I Chose and Why

**Option B — Unified schema with metadata object.**

My reasoning against ChatGPT's suggestion: Type-specific schemas complicate the ingest
endpoint (it needs to dispatch to different validators), make the JSONL output heterogeneous
(harder to load into a pandas DataFrame or stream processor), and add significant boilerplate.

The unified schema with a `metadata` dict for event-specific fields (queue_depth, sku_zone,
session_seq) gives us:
1. One validator to rule them all (single Pydantic model)
2. Simple JSONL streaming (homogeneous records)
3. A single SQL table with nullable columns for event-specific data

**Trade-off acknowledged**: The metadata dict is weakly typed. A consumer can't know statically
which metadata fields are present for a given event_type. I mitigate this with schema documentation
and the `event_type` field which lets consumers branch their logic.

**What I kept from AI's suggestion**: The idea of validating event_id as a UUID v4 string
(not just any string) came from ChatGPT's suggestion to use format-level validation.

---

## Decision 3: API Architecture — SQLite vs. PostgreSQL + Caching

### Options Considered

| Choice | Pros | Cons |
|--------|------|------|
| **SQLite** (chosen) | Zero external dependencies; single file; works in Docker with just a volume | Limited write concurrency; not suitable for >1 API worker |
| PostgreSQL | Production-grade; excellent concurrency; full SQL | Requires additional Docker service; more complex setup; defeats "5 commands" README goal |
| SQLite + Redis cache | Fast reads for repeated queries | Two external dependencies; cache invalidation complexity |
| In-memory dict | Zero latency | No persistence; dies on restart |

### What AI Suggested

I asked Gemini: *"For a FastAPI store analytics API that ingests 500 events per batch and
serves 6 analytics endpoints, should I use SQLite or PostgreSQL?"*

Gemini strongly recommended PostgreSQL with connection pooling, citing SQLite's write lock
limitations and the "production-ready" requirement of the challenge.

### What I Chose and Why

**SQLite with WAL mode.**

My counter-argument to Gemini: The challenge says "SQLite is fine. Document it in CHOICES.md."
The evaluation criteria include "runs seamlessly with minimal setup effort" — adding Postgres
is a setup complexity cost that may not be justified by our load. Our workload is:
- Batch ingest: one process writing, no concurrent writes from other sources
- Reads: 6 analytics endpoints, likely low concurrency in evaluation context
- WAL mode allows concurrent readers while one writer is active

SQLite's write lock limitation would matter at 40 live stores sending events simultaneously
(the /funnel question example from the problem statement). At that scale, I would:
1. Switch `DATABASE_URL` to `postgresql://...` (SQLAlchemy makes this a one-line change)
2. Add Alembic migrations
3. Deploy behind a load balancer with read replicas

**What I kept from AI's suggestion**: The `DATABASE_URL` environment variable pattern
(so the database engine is swappable without code changes) was Gemini's suggestion that I
implemented exactly.

**What I disagreed with**: Gemini suggested pre-computing and caching metrics every 5 minutes.
This violates the spec requirement "Real-time — not cached from yesterday." I compute metrics
on every request using indexed SQL queries. If performance becomes a bottleneck, the right
fix is better indexes and query optimization, not caching with staleness risk.

---

## Summary of Trade-off Philosophy

The consistent thread through all three decisions: I optimized for **operational simplicity
and edge case correctness** over maximum theoretical accuracy or performance. The evaluation
framework explicitly states: "A strong candidate is one who builds a system that works,
makes reasonable trade-offs, and can clearly explain those decisions." I believe these
choices reflect that principle.
