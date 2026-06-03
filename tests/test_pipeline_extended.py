# PROMPT: Write pytest unit tests for pipeline/tracker.py, pipeline/staff_detector.py,
# and pipeline/ingest_to_api.py. Cover: Tracker register/update/exit/re-entry/group-entry/
# zone-dwell/cross-camera-dedup/billing-queue; StaffDetector torso extraction and color
# uniformity; ingest_file function with mocked HTTP. No real video or GPU needed.
# CHANGES MADE: Used numpy to create synthetic BGR frames for staff detector tests;
# mocked httpx in ingest tests to avoid needing a live API; added edge cases for empty
# torso and zero-pixel crops.

import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Tracker Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTracker:
    """Unit tests for pipeline/tracker.py"""

    def _make_tracker(self, camera_id="CAM_ENTRY_01", fps=15.0):
        from pipeline.tracker import Tracker
        return Tracker(camera_id=camera_id, fps=fps)

    def test_register_track_creates_state(self):
        """Registering a new track should create a TrackState with correct fields."""
        tracker = self._make_tracker()
        state = tracker.register_track(
            track_id=1, visitor_id="VIS_001", frame=10,
            cx=500.0, cy=300.0, bbox=(400, 200, 600, 500), is_staff=False
        )
        assert state.track_id == 1
        assert state.visitor_id == "VIS_001"
        assert state.is_active is True
        assert state.is_staff is False
        assert state.current_zone is None
        assert 1 in tracker.active_tracks

    def test_register_staff_track(self):
        """Staff track should have is_staff=True."""
        tracker = self._make_tracker()
        state = tracker.register_track(
            track_id=99, visitor_id="VIS_STAFF", frame=5,
            cx=100.0, cy=200.0, bbox=(50, 100, 150, 400), is_staff=True
        )
        assert state.is_staff is True

    def test_update_track_position(self):
        """Update should change track position."""
        tracker = self._make_tracker()
        tracker.register_track(1, "VIS_001", 10, 500.0, 300.0, (400, 200, 600, 500), False)
        updated = tracker.update_track(1, frame=20, cx=550.0, cy=350.0, bbox=(450, 250, 650, 550))
        assert updated.last_cx == 550.0
        assert updated.last_cy == 350.0
        assert updated.last_seen_frame == 20

    def test_update_nonexistent_track_returns_none(self):
        """Updating a track that doesn't exist should return None."""
        tracker = self._make_tracker()
        result = tracker.update_track(999, frame=10, cx=100.0, cy=200.0, bbox=(0, 0, 100, 200))
        assert result is None

    def test_mark_exit_removes_from_active(self):
        """mark_exit should remove track from active_tracks and add to exited_tracks."""
        tracker = self._make_tracker()
        tracker.register_track(1, "VIS_001", 10, 500.0, 300.0, (400, 200, 600, 500), False)
        assert 1 in tracker.active_tracks

        tracker.mark_exit(track_id=1, frame=100)
        assert 1 not in tracker.active_tracks
        assert len(tracker.exited_tracks) == 1
        assert tracker.exited_tracks[0].is_active is False
        assert tracker.exited_tracks[0].exited_at_frame == 100

    def test_reentry_detected_within_window(self):
        """Same person re-entering within 90s window should be detected."""
        tracker = self._make_tracker(fps=15.0)
        # Register and exit a track
        tracker.register_track(1, "VIS_001", frame=0, cx=500.0, cy=300.0,
                                bbox=(400, 100, 600, 500), is_staff=False)
        tracker.mark_exit(track_id=1, frame=10)

        # Re-entry at frame 50 (well within 90s * 15fps = 1350 frames)
        match = tracker.check_reentry(cx=510.0, cy=310.0, bbox=(410, 110, 610, 510), frame=50)
        assert match is not None
        assert match.visitor_id == "VIS_001"

    def test_reentry_not_detected_outside_window(self):
        """Re-entry outside 90-second window should not be detected."""
        tracker = self._make_tracker(fps=15.0)
        tracker.register_track(1, "VIS_001", frame=0, cx=500.0, cy=300.0,
                                bbox=(400, 100, 600, 500), is_staff=False)
        tracker.mark_exit(track_id=1, frame=0)

        # 90s * 15fps = 1350 frames; try frame 2000 (way past window)
        match = tracker.check_reentry(cx=500.0, cy=300.0, bbox=(400, 100, 600, 500), frame=2000)
        assert match is None

    def test_no_reentry_when_no_exits(self):
        """check_reentry with empty exited_tracks should return None."""
        tracker = self._make_tracker()
        match = tracker.check_reentry(cx=500.0, cy=300.0, bbox=(400, 100, 600, 500), frame=50)
        assert match is None

    def test_group_entry_single_person(self):
        """Single entry should not be flagged as group entry."""
        tracker = self._make_tracker()
        is_group = tracker.detect_group_entry(frame=100)
        assert is_group is False

    def test_group_entry_two_people_close_together(self):
        """Two entries within GROUP_ENTRY_WINDOW_FRAMES should be flagged as group."""
        tracker = self._make_tracker(fps=15.0)
        tracker.detect_group_entry(frame=100)   # First person
        is_group = tracker.detect_group_entry(frame=110)  # Second person 10 frames later
        assert is_group is True

    def test_group_entry_two_people_far_apart(self):
        """Two entries far apart in time should NOT be grouped."""
        tracker = self._make_tracker(fps=15.0)
        tracker.detect_group_entry(frame=0)
        # GROUP_ENTRY_WINDOW_FRAMES = 30; frame 200 is way past window
        is_group = tracker.detect_group_entry(frame=200)
        assert is_group is False

    def test_zone_dwell_not_triggered_before_30s(self):
        """ZONE_DWELL should not trigger before 30 seconds of dwell."""
        tracker = self._make_tracker(fps=15.0)
        state = tracker.register_track(1, "VIS_001", 0, 500.0, 300.0, (400, 100, 600, 500), False)
        state.zone_entry_frame = 0  # entered zone at frame 0

        # Check at frame 200 = ~13s (< 30s)
        should_emit = tracker.check_zone_dwell(state, frame=200)
        assert should_emit is False

    def test_zone_dwell_triggered_after_30s(self):
        """ZONE_DWELL should trigger after 30+ seconds of continuous dwell."""
        tracker = self._make_tracker(fps=15.0)
        state = tracker.register_track(1, "VIS_001", 0, 500.0, 300.0, (400, 100, 600, 500), False)
        state.zone_entry_frame = 0  # entered zone at frame 0

        # 30s * 15fps = 450 frames
        should_emit = tracker.check_zone_dwell(state, frame=500)
        assert should_emit is True

    def test_zone_dwell_no_zone_entry_frame(self):
        """check_zone_dwell with no zone_entry_frame should return False."""
        tracker = self._make_tracker()
        state = tracker.register_track(1, "VIS_001", 0, 500.0, 300.0, (400, 100, 600, 500), False)
        # zone_entry_frame is None by default
        assert tracker.check_zone_dwell(state, frame=1000) is False

    def test_get_dwell_ms_correct(self):
        """Dwell time calculation should be correct."""
        tracker = self._make_tracker(fps=15.0)
        state = tracker.register_track(1, "VIS_001", 0, 500.0, 300.0, (400, 100, 600, 500), False)
        state.zone_entry_frame = 0
        # 150 frames / 15fps = 10 seconds = 10000ms
        dwell = tracker.get_dwell_ms(state, frame=150)
        assert dwell == 10000

    def test_get_dwell_ms_no_zone(self):
        """get_dwell_ms with no zone_entry_frame returns 0."""
        tracker = self._make_tracker()
        state = tracker.register_track(1, "VIS_001", 0, 500.0, 300.0, (400, 100, 600, 500), False)
        assert tracker.get_dwell_ms(state, frame=500) == 0

    def test_cross_cam_dedup_same_visitor_other_camera(self):
        """Same visitor_id seen on overlapping camera within window → duplicate."""
        tracker = self._make_tracker(camera_id="CAM_ENTRY_01", fps=15.0)

        # First seen on CAM_FLOOR_01
        is_dup1 = tracker.is_duplicate_cross_cam(
            visitor_id="VIS_001", camera_id="CAM_FLOOR_01",
            frame=100, cx=500.0, cy=300.0, timestamp="2026-04-10T10:00:00Z",
            overlap_cameras=["CAM_FLOOR_01"]
        )
        assert is_dup1 is False  # First time, not a dup

        # Same visitor on CAM_ENTRY_01 close to first sighting → duplicate
        is_dup2 = tracker.is_duplicate_cross_cam(
            visitor_id="VIS_001", camera_id="CAM_ENTRY_01",
            frame=110, cx=505.0, cy=305.0, timestamp="2026-04-10T10:00:01Z",
            overlap_cameras=["CAM_FLOOR_01"]
        )
        assert is_dup2 is True

    def test_cross_cam_dedup_different_visitor(self):
        """Different visitor_ids should NOT be flagged as duplicate."""
        tracker = self._make_tracker(fps=15.0)
        tracker.is_duplicate_cross_cam(
            "VIS_001", "CAM_FLOOR_01", 100, 500.0, 300.0, "2026-04-10T10:00:00Z",
            overlap_cameras=["CAM_FLOOR_01"]
        )
        is_dup = tracker.is_duplicate_cross_cam(
            "VIS_002", "CAM_ENTRY_01", 110, 500.0, 300.0, "2026-04-10T10:00:01Z",
            overlap_cameras=["CAM_FLOOR_01"]
        )
        assert is_dup is False

    def test_billing_queue_count_zero(self):
        """No active tracks in billing zone → count = 0."""
        tracker = self._make_tracker()
        assert tracker.count_billing_queue() == 0

    def test_billing_queue_count_with_customers(self):
        """Active tracks in BILLING zone (non-staff) should be counted."""
        tracker = self._make_tracker()
        s1 = tracker.register_track(1, "VIS_001", 0, 100.0, 200.0, (50, 100, 150, 400), False)
        s2 = tracker.register_track(2, "VIS_002", 0, 200.0, 200.0, (150, 100, 250, 400), False)
        s3 = tracker.register_track(3, "VIS_STAFF", 0, 300.0, 200.0, (250, 100, 350, 400), True)

        s1.current_zone = "BILLING"
        s2.current_zone = "BILLING"
        s3.current_zone = "BILLING"  # staff — should NOT count

        assert tracker.count_billing_queue() == 2  # only non-staff

    def test_billing_queue_excludes_other_zones(self):
        """Visitors in non-billing zones should not be counted."""
        tracker = self._make_tracker()
        s1 = tracker.register_track(1, "VIS_001", 0, 100.0, 200.0, (50, 100, 150, 400), False)
        s1.current_zone = "SKINCARE"
        assert tracker.count_billing_queue() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Staff Detector Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestStaffDetector:
    """Unit tests for pipeline/staff_detector.py"""

    def _black_frame(self, h=200, w=100):
        """Create a black BGR frame (simulates black uniform)."""
        return np.zeros((h, w, 3), dtype=np.uint8)

    def _white_frame(self, h=200, w=100):
        """Create a white BGR frame (simulates white uniform)."""
        return np.full((h, w, 3), 255, dtype=np.uint8)

    def _random_colorful_frame(self, h=200, w=100):
        """Create a random colorful frame (simulates varied customer clothing)."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_extract_torso_region_correct_bounds(self):
        """Torso region should be 25%-65% of bounding box height."""
        from pipeline.staff_detector import extract_torso_region
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 500)   # height = 400px
        torso = extract_torso_region(frame, bbox)
        # Expected: y1 = 100 + 0.25*400 = 200, y2 = 100 + 0.65*400 = 360
        # torso height = 360 - 200 = 160
        assert torso.shape[0] == 160
        assert torso.shape[1] == 100  # x2 - x1

    def test_extract_torso_clamps_to_frame_bounds(self):
        """Torso extraction should not go outside frame boundaries."""
        from pipeline.staff_detector import extract_torso_region
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox near edge
        bbox = (0, 0, 100, 500)  # y2 way beyond frame
        torso = extract_torso_region(frame, bbox)
        # Should not raise and height should be clamped to frame
        assert torso.shape[0] <= frame.shape[0]
        assert torso.shape[1] <= frame.shape[1]

    def test_empty_torso_not_uniform(self):
        """Empty/tiny torso crop should return not uniform."""
        from pipeline.staff_detector import compute_color_uniformity
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        is_uniform, score = compute_color_uniformity(empty)
        assert is_uniform is False
        assert score == 0.0

    def test_tiny_torso_not_uniform(self):
        """Too-small torso (< 5px) should return not uniform."""
        from pipeline.staff_detector import compute_color_uniformity
        tiny = np.zeros((3, 3, 3), dtype=np.uint8)
        is_uniform, score = compute_color_uniformity(tiny)
        assert is_uniform is False

    def test_black_torso_is_staff(self):
        """Mostly black torso should be classified as staff (black uniform)."""
        from pipeline.staff_detector import compute_color_uniformity
        black_crop = np.zeros((80, 50, 3), dtype=np.uint8)  # pure black
        is_uniform, score = compute_color_uniformity(black_crop)
        assert bool(is_uniform) is True
        assert score > 0.55

    def test_white_torso_is_staff(self):
        """Mostly white torso should be classified as staff (white uniform)."""
        from pipeline.staff_detector import compute_color_uniformity
        white_crop = np.full((80, 50, 3), 255, dtype=np.uint8)  # pure white
        is_uniform, score = compute_color_uniformity(white_crop)
        assert bool(is_uniform) is True
        assert score > 0.55

    def test_is_staff_main_function_black_uniform(self):
        """is_staff() on a black-uniformed person should return True."""
        from pipeline.staff_detector import is_staff
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)  # all black frame
        bbox = (500, 100, 700, 900)
        result, conf = is_staff(frame, bbox, yolo_conf=0.9)
        assert bool(result) is True
        assert 0.0 <= conf <= 1.0

    def test_is_staff_main_function_confidence_range(self):
        """Confidence returned by is_staff() should always be in [0, 1]."""
        from pipeline.staff_detector import is_staff
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bbox = (100, 100, 300, 500)
        _, conf = is_staff(frame, bbox, yolo_conf=0.5)
        assert 0.0 <= conf <= 1.0

    def test_color_uniformity_score_between_0_and_1(self):
        """Uniformity score should always be in [0, 1]."""
        from pipeline.staff_detector import compute_color_uniformity
        for _ in range(5):
            rng = np.random.default_rng()
            crop = rng.integers(0, 255, (60, 40, 3), dtype=np.uint8)
            _, score = compute_color_uniformity(crop)
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Ingest to API Tests (mocked HTTP)
# ─────────────────────────────────────────────────────────────────────────────

class TestIngestToApi:
    """Unit tests for pipeline/ingest_to_api.py — HTTP is mocked."""

    def _make_jsonl_file(self, tmp_path, events):
        import json
        p = tmp_path / "cam1_events.jsonl"
        with open(p, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        return p

    def _sample_event(self, i=0):
        import uuid
        return {
            "event_id": str(uuid.uuid4()),
            "store_id": "STORE_BLR_002",
            "camera_id": "CAM_ENTRY_01",
            "visitor_id": f"VIS_{i:03d}",
            "event_type": "ENTRY",
            "timestamp": "2026-04-10T10:00:00Z",
            "zone_id": None,
            "dwell_ms": 0,
            "is_staff": False,
            "confidence": 0.9,
            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": 1}
        }

    def test_ingest_file_success(self, tmp_path):
        """ingest_file should POST events and return ingested count."""
        from pipeline.ingest_to_api import ingest_file

        events = [self._sample_event(i) for i in range(5)]
        jsonl = self._make_jsonl_file(tmp_path, events)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ingested": 5, "duplicates": 0, "errors": 0}

        with patch("pipeline.ingest_to_api.httpx.post", return_value=mock_resp) as mock_post:
            result = ingest_file("http://localhost:8000", jsonl)

        assert result["ingested"] == 5
        assert result["errors"] == 0
        mock_post.assert_called_once()

    def test_ingest_file_empty_file(self, tmp_path):
        """ingest_file on an empty JSONL should return 0 ingested, no HTTP call."""
        from pipeline.ingest_to_api import ingest_file

        jsonl = self._make_jsonl_file(tmp_path, [])

        with patch("pipeline.ingest_to_api.httpx.post") as mock_post:
            result = ingest_file("http://localhost:8000", jsonl)

        assert result["ingested"] == 0
        mock_post.assert_not_called()

    def test_ingest_file_api_error_status(self, tmp_path):
        """Non-200 API response should increment error count."""
        from pipeline.ingest_to_api import ingest_file

        events = [self._sample_event(i) for i in range(3)]
        jsonl = self._make_jsonl_file(tmp_path, events)

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("pipeline.ingest_to_api.httpx.post", return_value=mock_resp):
            result = ingest_file("http://localhost:8000", jsonl)

        assert result["errors"] == 3  # all events errored

    def test_ingest_file_batches_large_input(self, tmp_path):
        """Large input (>BATCH_SIZE) should be sent in multiple POST calls."""
        from pipeline.ingest_to_api import ingest_file, BATCH_SIZE

        events = [self._sample_event(i) for i in range(BATCH_SIZE + 50)]
        jsonl = self._make_jsonl_file(tmp_path, events)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"ingested": BATCH_SIZE, "duplicates": 0, "errors": 0}

        with patch("pipeline.ingest_to_api.httpx.post", return_value=mock_resp) as mock_post:
            result = ingest_file("http://localhost:8000", jsonl)

        # Should have been called twice (200 events + 50 events)
        assert mock_post.call_count == 2

    def test_ingest_file_connect_error_exits(self, tmp_path):
        """Connection error should raise SystemExit."""
        import httpx as _httpx
        from pipeline.ingest_to_api import ingest_file

        events = [self._sample_event(0)]
        jsonl = self._make_jsonl_file(tmp_path, events)

        with patch("pipeline.ingest_to_api.httpx.post", side_effect=_httpx.ConnectError("refused")):
            with pytest.raises(SystemExit):
                ingest_file("http://localhost:9999", jsonl)
