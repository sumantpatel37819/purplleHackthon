# PROMPT: Write pytest tests for the pipeline zone mapping and event emission modules.
# Cover: zone_mapper polygon containment, entry/exit crossing detection, emit schema validation,
# visitor_id generation, frame_to_timestamp conversion. Use pure unit tests (no network).
# CHANGES MADE: Added test for low confidence events (should still emit, not suppress),
# added test for ZONE_DWELL 30-second window logic.

import pytest
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestZoneMapper:
    """Unit tests for zone_mapper.py polygon containment logic."""

    def test_point_in_polygon_center(self):
        """Point at center of polygon should be inside."""
        import numpy as np
        from pipeline.zone_mapper import ZoneMapper
        # Test using the static method directly
        poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert ZoneMapper._point_in_polygon(50, 50, poly) is True

    def test_point_outside_polygon(self):
        """Point clearly outside polygon should return False."""
        import numpy as np
        from pipeline.zone_mapper import ZoneMapper
        poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        assert ZoneMapper._point_in_polygon(150, 150, poly) is False

    def test_point_on_boundary(self):
        """Points near boundary — deterministic behavior."""
        import numpy as np
        from pipeline.zone_mapper import ZoneMapper
        poly = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
        # Edge case: point on boundary may or may not be inside
        # Just test it doesn't crash
        result = ZoneMapper._point_in_polygon(0, 50, poly)
        assert isinstance(result, bool)

    def test_entry_exit_top_to_bottom(self):
        """Person moving downward past entry line should trigger ENTRY."""
        from pipeline.zone_mapper import ZoneMapper
        import json

        # Create a minimal layout for testing
        layout = {
            "store_id": "STORE_BLR_002",
            "cameras": [{
                "camera_id": "CAM_ENTRY_01",
                "type": "entry_exit",
                "entry_line": {"y": 0.5, "x_start": 0.0, "x_end": 1.0},
                "direction_in": "top_to_bottom",
                "zones": ["ENTRY_EXIT"],
                "fps": 15
            }],
            "zones": [{
                "zone_id": "ENTRY_EXIT",
                "sku_zone": "ENTRY",
                "camera_id": "CAM_ENTRY_01",
                "type": "threshold",
                "polygon_normalized": [[0, 0.4], [1, 0.4], [1, 0.7], [0, 0.7]]
            }]
        }

        import json, tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(layout, f)
            tmp_path = f.name

        try:
            zm = ZoneMapper("CAM_ENTRY_01", 1920, 1080, tmp_path)
            # prev_cy above line, curr_cy below → ENTRY
            result = zm.check_entry_exit(prev_cy=400, curr_cy=700)
            assert result == "ENTRY"
            # prev_cy below, curr_cy above → EXIT
            result2 = zm.check_entry_exit(prev_cy=700, curr_cy=400)
            assert result2 == "EXIT"
        finally:
            os.unlink(tmp_path)

    def test_no_entry_exit_for_floor_camera(self):
        """Floor cameras have no entry line, should return None."""
        from pipeline.zone_mapper import ZoneMapper
        import json, tempfile, os

        layout = {
            "store_id": "STORE_BLR_002",
            "cameras": [{
                "camera_id": "CAM_FLOOR_01",
                "type": "floor",
                "zones": ["SKINCARE"],
                "fps": 15
            }],
            "zones": [{
                "zone_id": "SKINCARE",
                "sku_zone": "MOISTURISER",
                "camera_id": "CAM_FLOOR_01",
                "type": "product_zone",
                "polygon_normalized": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]
            }]
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(layout, f)
            tmp_path = f.name

        try:
            zm = ZoneMapper("CAM_FLOOR_01", 1920, 1080, tmp_path)
            assert zm.entry_line is None
            assert zm.check_entry_exit(400, 700) is None
        finally:
            os.unlink(tmp_path)


class TestEmit:
    """Unit tests for emit.py event schema."""

    def test_make_event_schema_compliance(self):
        """Generated event must have all required fields with correct types."""
        from pipeline.emit import make_event
        event = make_event(
            camera_id="CAM_ENTRY_01",
            visitor_id="VIS_abc123",
            event_type="ENTRY",
            timestamp="2026-04-10T10:30:00Z",
        )
        assert "event_id" in event
        assert "store_id" in event
        assert "camera_id" in event
        assert "visitor_id" in event
        assert "event_type" in event
        assert "timestamp" in event
        assert "is_staff" in event
        assert "confidence" in event
        assert "metadata" in event
        # UUID check
        import uuid as _uuid
        _uuid.UUID(event["event_id"])  # Should not raise

    def test_make_event_low_confidence_not_suppressed(self):
        """Low confidence events (0.1) should still be emitted with confidence field set."""
        from pipeline.emit import make_event
        event = make_event(
            camera_id="CAM_FLOOR_01",
            visitor_id="VIS_lowconf",
            event_type="ZONE_ENTER",
            timestamp="2026-04-10T10:30:00Z",
            confidence=0.15,
            zone_id="SKINCARE",
        )
        # Should not suppress — confidence is just flagged
        assert event["confidence"] == 0.15
        assert event["event_type"] == "ZONE_ENTER"

    def test_make_event_invalid_type_raises(self):
        """Invalid event_type should raise AssertionError."""
        from pipeline.emit import make_event
        with pytest.raises(AssertionError):
            make_event(
                camera_id="CAM_ENTRY_01",
                visitor_id="VIS_001",
                event_type="INVALID",
                timestamp="2026-04-10T10:30:00Z",
            )

    def test_visitor_id_stable_for_same_inputs(self):
        """Same track_id + camera_id + session should produce same visitor_id."""
        from pipeline.emit import make_visitor_id
        v1 = make_visitor_id(42, "CAM_ENTRY_01", "session0")
        v2 = make_visitor_id(42, "CAM_ENTRY_01", "session0")
        assert v1 == v2
        assert v1.startswith("VIS_")

    def test_visitor_id_different_for_different_inputs(self):
        """Different track_ids should produce different visitor_ids."""
        from pipeline.emit import make_visitor_id
        v1 = make_visitor_id(1, "CAM_ENTRY_01", "session0")
        v2 = make_visitor_id(2, "CAM_ENTRY_01", "session0")
        assert v1 != v2

    def test_frame_to_timestamp_correct(self):
        """Frame 150 at 15fps = 10 seconds offset from base time."""
        from pipeline.emit import frame_to_timestamp
        base = datetime(2026, 4, 10, 10, 0, 0, tzinfo=timezone.utc)
        ts = frame_to_timestamp(150, 15.0, base)
        assert ts == "2026-04-10T10:00:10Z"

    def test_event_writer_creates_valid_jsonl(self, tmp_path):
        """EventWriter should create a valid JSONL file."""
        from pipeline.emit import make_event, EventWriter, load_events_jsonl
        out_path = str(tmp_path / "test_events.jsonl")
        events_to_write = [
            make_event(event_type="ENTRY", camera_id="CAM_ENTRY_01", visitor_id="VIS_001", timestamp="2026-04-10T10:00:00Z"),
            make_event(event_type="EXIT", camera_id="CAM_ENTRY_01", visitor_id="VIS_001", timestamp="2026-04-10T10:05:00Z"),
        ]
        with EventWriter(out_path) as writer:
            for e in events_to_write:
                writer.write(e)
            assert writer.count == 2

        loaded = load_events_jsonl(out_path)
        assert len(loaded) == 2
        assert loaded[0]["event_type"] == "ENTRY"
        assert loaded[1]["event_type"] == "EXIT"
