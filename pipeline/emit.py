"""
Event Emitter — generates structured events in the required schema.

All events conform to:
{
  "event_id": "uuid-v4",
  "store_id": "STORE_BLR_002",
  "camera_id": "CAM_ENTRY_01",
  "visitor_id": "VIS_xxxxxx",
  "event_type": "ENTRY",
  "timestamp": "2026-04-10T10:22:10Z",
  "zone_id": null,
  "dwell_ms": 0,
  "is_staff": false,
  "confidence": 0.91,
  "metadata": {
    "queue_depth": null,
    "sku_zone": null,
    "session_seq": 1
  }
}
"""
import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, IO


STORE_ID = "STORE_BLR_002"

# Canonical event types
EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY"
}


def make_visitor_id(track_id: int, camera_id: str, session_token: Optional[str] = None) -> str:
    """
    Generate a stable visitor_id from track_id + camera_id + optional session token.
    Format: VIS_xxxxxx (6-char hex suffix for readability).
    """
    seed = f"{track_id}:{camera_id}:{session_token or ''}"
    h = hashlib.md5(seed.encode()).hexdigest()[:6]
    return f"VIS_{h}"


def frame_to_timestamp(frame_idx: int, fps: float, base_datetime: datetime) -> str:
    """
    Convert frame index to ISO-8601 UTC timestamp.
    base_datetime: the real-world datetime of frame 0 (store open time + date).
    """
    offset_sec = frame_idx / fps
    ts = base_datetime + timedelta(seconds=offset_sec)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: str,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 1.0,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: int = 1,
) -> dict:
    """Build a complete event dict conforming to the required schema."""
    assert event_type in EVENT_TYPES, f"Unknown event type: {event_type}"
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": bool(is_staff),
        "confidence": round(float(confidence), 4),
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone,
            "session_seq": session_seq,
        },
    }


class EventWriter:
    """Writes events to a JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file: Optional[IO] = None
        self._count = 0

    def __enter__(self):
        self._file = open(self.output_path, "w", encoding="utf-8")
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def write(self, event: dict):
        """Write a single event as a JSON line."""
        if self._file:
            self._file.write(json.dumps(event) + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


def load_events_jsonl(path: str) -> list:
    """Load all events from a JSONL file."""
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events
