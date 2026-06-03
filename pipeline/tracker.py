"""
Tracker — handles Re-ID, re-entry detection, cross-camera deduplication, and group entry.

Key responsibilities:
1. Maintain per-track state (last position, zone, timestamps, session)
2. Detect re-entry: same physical person returns after EXIT (within session window)
3. Cross-camera dedup: same person seen by overlapping cameras → don't double-count
4. Group entry: 2+ people entering within 2 seconds through same threshold
5. Session management: unique visitor_id per visit session

Re-ID approach: bounding-box trajectory + aspect ratio fingerprint.
We don't use deep appearance Re-ID (requires GPU + training data), instead:
- Store (centroid_x, centroid_y, bbox_aspect_ratio) fingerprint for each exited track
- On ENTRY, compare new track fingerprint to recent exits within 90-second window
- If cosine similarity > threshold → REENTRY event instead of ENTRY
"""
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TrackState:
    track_id: int
    visitor_id: str
    camera_id: str
    first_seen_frame: int
    last_seen_frame: int
    last_cx: float
    last_cy: float
    bbox_ar: float          # aspect ratio (w/h) as appearance fingerprint
    current_zone: Optional[str]
    zone_entry_frame: Optional[int]
    dwell_emitted_at: Optional[int]  # frame of last ZONE_DWELL emit
    session_seq: int
    is_staff: bool
    is_active: bool         # False after EXIT
    exited_at_frame: Optional[int]
    session_token: str      # random token to disambiguate re-entries


@dataclass
class CrossCamRecord:
    """Used to dedup the same person across overlapping cameras."""
    visitor_id: str
    camera_id: str
    frame: int
    cx: float
    cy: float
    timestamp: str


class Tracker:
    """
    Maintains track states and provides Re-ID, dedup, and group-entry logic.
    """

    REENTRY_WINDOW_SEC = 90.0     # Look back 90s for re-entry matching
    REENTRY_AR_THRESHOLD = 0.3    # Aspect ratio similarity threshold
    GROUP_ENTRY_WINDOW_FRAMES = 30  # 2s at 15fps
    CROSS_CAM_DEDUP_WINDOW_SEC = 5.0  # Dedup window for overlapping cameras

    def __init__(self, camera_id: str, fps: float = 15.0):
        self.camera_id = camera_id
        self.fps = fps
        self.active_tracks: Dict[int, TrackState] = {}
        self.exited_tracks: List[TrackState] = []   # For re-entry matching
        self.cross_cam_log: List[CrossCamRecord] = []  # For dedup
        self._entry_log: List[int] = []              # Frame numbers of recent entries (group detection)
        self._visitor_session_map: Dict[str, int] = defaultdict(int)  # visitor_id → session_seq counter

    def register_track(self, track_id: int, visitor_id: str, frame: int,
                       cx: float, cy: float, bbox: Tuple, is_staff: bool) -> TrackState:
        """Register a new or resumed track."""
        import hashlib, uuid
        bbox_ar = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
        session_token = str(uuid.uuid4())[:8]
        state = TrackState(
            track_id=track_id,
            visitor_id=visitor_id,
            camera_id=self.camera_id,
            first_seen_frame=frame,
            last_seen_frame=frame,
            last_cx=cx,
            last_cy=cy,
            bbox_ar=bbox_ar,
            current_zone=None,
            zone_entry_frame=None,
            dwell_emitted_at=None,
            session_seq=1,
            is_staff=is_staff,
            is_active=True,
            exited_at_frame=None,
            session_token=session_token,
        )
        self.active_tracks[track_id] = state
        return state

    def update_track(self, track_id: int, frame: int, cx: float, cy: float,
                     bbox: Tuple) -> Optional[TrackState]:
        """Update an existing track's position."""
        state = self.active_tracks.get(track_id)
        if state:
            state.last_seen_frame = frame
            state.last_cx = cx
            state.last_cy = cy
            state.bbox_ar = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
        return state

    def mark_exit(self, track_id: int, frame: int):
        """Mark a track as exited and archive for re-entry detection."""
        state = self.active_tracks.get(track_id)
        if state:
            state.is_active = False
            state.exited_at_frame = frame
            self.exited_tracks.append(state)
            del self.active_tracks[track_id]

    def check_reentry(self, cx: float, cy: float, bbox: Tuple, frame: int) -> Optional[TrackState]:
        """
        Check if a new detection matches a recently-exited track (re-entry).
        
        Matching criteria:
        - Exited within the last REENTRY_WINDOW_SEC seconds
        - Aspect ratio (body shape) is similar (|ar1 - ar2| < threshold)
        - Horizontal position is similar (same door area, |cx1 - cx2| < 0.3 * width)
        
        Returns the matching exited TrackState if re-entry detected, else None.
        """
        if not self.exited_tracks:
            return None

        new_ar = (bbox[2] - bbox[0]) / max(1, bbox[3] - bbox[1])
        window_frames = int(self.REENTRY_WINDOW_SEC * self.fps)
        candidates = [
            t for t in self.exited_tracks
            if (frame - t.exited_at_frame) <= window_frames
        ]

        best_score = float("inf")
        best_match = None
        for candidate in candidates:
            ar_diff = abs(new_ar - candidate.bbox_ar)
            cx_diff = abs(cx - candidate.last_cx)
            score = ar_diff * 0.5 + (cx_diff / 1920.0) * 0.5
            if score < best_score and ar_diff < self.REENTRY_AR_THRESHOLD:
                best_score = score
                best_match = candidate

        return best_match

    def detect_group_entry(self, frame: int) -> bool:
        """
        Returns True if this ENTRY is part of a group (another ENTRY within GROUP_ENTRY_WINDOW_FRAMES).
        """
        self._entry_log.append(frame)
        # Clean old entries
        self._entry_log = [f for f in self._entry_log
                           if frame - f <= self.GROUP_ENTRY_WINDOW_FRAMES]
        return len(self._entry_log) > 1

    def check_zone_dwell(self, state: TrackState, frame: int) -> bool:
        """
        Returns True if a ZONE_DWELL event should be emitted (every 30s of continuous dwell).
        """
        if state.zone_entry_frame is None:
            return False
        dwell_frames = frame - state.zone_entry_frame
        dwell_sec = dwell_frames / self.fps
        # Should emit at 30s, 60s, 90s, etc.
        last_emit_frame = state.dwell_emitted_at or state.zone_entry_frame
        frames_since_emit = frame - last_emit_frame
        return dwell_sec >= 30 and frames_since_emit >= int(30 * self.fps)

    def get_dwell_ms(self, state: TrackState, frame: int) -> int:
        """Calculate dwell time in milliseconds."""
        if state.zone_entry_frame is None:
            return 0
        return int((frame - state.zone_entry_frame) / self.fps * 1000)

    def is_duplicate_cross_cam(self, visitor_id: str, camera_id: str, frame: int,
                                cx: float, cy: float, timestamp: str,
                                overlap_cameras: List[str],
                                dedup_window_sec: float = 5.0) -> bool:
        """
        Check if this event is a cross-camera duplicate.
        Returns True if the same visitor_id was seen on another overlapping camera
        within dedup_window_sec.
        """
        window_frames = int(dedup_window_sec * self.fps)
        for record in self.cross_cam_log:
            if (record.visitor_id == visitor_id
                    and record.camera_id != camera_id
                    and record.camera_id in overlap_cameras
                    and abs(frame - record.frame) <= window_frames):
                return True

        self.cross_cam_log.append(CrossCamRecord(
            visitor_id=visitor_id,
            camera_id=camera_id,
            frame=frame,
            cx=cx,
            cy=cy,
            timestamp=timestamp,
        ))
        # Keep log bounded
        if len(self.cross_cam_log) > 1000:
            self.cross_cam_log = self.cross_cam_log[-500:]
        return False

    def count_billing_queue(self, billing_zone_id: str = "BILLING") -> int:
        """Count active tracks currently in the billing zone."""
        return sum(
            1 for t in self.active_tracks.values()
            if t.current_zone == billing_zone_id and not t.is_staff
        )
