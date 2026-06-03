"""
detect.py — Main CCTV Detection Pipeline

Processes each CCTV video clip using YOLOv8n + ByteTrack to:
1. Detect people in each frame
2. Track them across frames (ByteTrack assigns stable track_ids)
3. Determine entry/exit events from boundary line crossings
4. Map positions to store zones (polygon containment)
5. Detect staff via uniform color heuristics
6. Handle re-entry, group entry, zone dwell, billing queue
7. Emit structured events to JSONL files

Usage:
    python pipeline/detect.py --cam 1
    python pipeline/detect.py --all
    python pipeline/detect.py --cam 1 --max-frames 500  # for testing

AI-Assisted Decisions:
- YOLOv8n selected over RT-DETR after testing: 3x faster on CPU, acceptable accuracy
- ByteTrack chosen over DeepSORT: no appearance model needed (works well on CPU)
- Frame sampling at every 3rd frame (5fps effective) to balance speed/accuracy
"""

import argparse
import json
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EVENTS_DIR = DATA_DIR / "events"
FOOTAGE_DIR = BASE_DIR / "resources" / "CCTV Footage-20260529T160731Z-3-00144614ea" / "CCTV Footage"
LAYOUT_PATH = DATA_DIR / "store_layout.json"

# Camera configuration
CAMERA_MAP = {
    1: ("CAM_ENTRY_01", "CAM 1.mp4"),
    2: ("CAM_ENTRY_02", "CAM 2.mp4"),
    3: ("CAM_FLOOR_01", "CAM 3.mp4"),
    4: ("CAM_BILLING_01", "CAM 4.mp4"),
    5: ("CAM_FLOOR_02", "CAM 5.mp4"),
}

# Video base timestamp: store opens at 10:00 AM on 2026-04-10
VIDEO_BASE_TIME = datetime(2026, 4, 10, 10, 0, 0, tzinfo=timezone.utc)

# Cross-camera overlap pairs (for dedup)
OVERLAP_CAMERAS = {
    "CAM_ENTRY_01": ["CAM_FLOOR_01"],
    "CAM_FLOOR_01": ["CAM_ENTRY_01"],
}

# Processing parameters
FRAME_SKIP = 3          # Process every 3rd frame (5fps effective from 15fps)
CONF_THRESHOLD = 0.25   # YOLO confidence threshold (low to capture all detections)
IOU_THRESHOLD = 0.45    # ByteTrack IoU threshold
PERSON_CLASS_ID = 0     # COCO class 0 = person


def load_layout() -> dict:
    with open(LAYOUT_PATH) as f:
        return json.load(f)


def get_camera_config(camera_id: str, layout: dict) -> Optional[dict]:
    for cam in layout["cameras"]:
        if cam["camera_id"] == camera_id:
            return cam
    return None


def frame_to_ts(frame_idx: int, fps: float) -> str:
    offset = frame_idx / fps
    ts = VIDEO_BASE_TIME + timedelta(seconds=offset)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_visitor_id(track_id: int, camera_id: str, session: int = 0) -> str:
    import hashlib
    seed = f"{track_id}:{camera_id}:{session}"
    h = hashlib.md5(seed.encode()).hexdigest()[:6]
    return f"VIS_{h}"


def process_camera(cam_num: int, max_frames: Optional[int] = None, verbose: bool = True):
    """
    Main processing function for a single camera.
    Returns number of events emitted.
    """
    from zone_mapper import ZoneMapper
    from staff_detector import is_staff as detect_staff
    from tracker import Tracker
    from emit import make_event, EventWriter

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        return 0

    camera_id, video_file = CAMERA_MAP[cam_num]
    video_path = FOOTAGE_DIR / video_file
    output_path = EVENTS_DIR / f"cam{cam_num}_events.jsonl"

    if not video_path.exists():
        print(f"[WARN] Video not found: {video_path}")
        return 0

    print(f"\n{'='*60}")
    print(f"Processing Camera {cam_num}: {camera_id}")
    print(f"Video: {video_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")

    # Load YOLO model (downloads automatically on first run)
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # nano = fastest, CPU-friendly

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps:.1f}, Frames: {total_frames}, Resolution: {width}x{height}")

    # Initialize zone mapper and tracker
    layout = load_layout()
    zone_mapper = ZoneMapper(camera_id, width, height, str(LAYOUT_PATH))
    tracker = Tracker(camera_id, fps)
    cam_config = get_camera_config(camera_id, layout)
    is_entry_cam = cam_config and cam_config.get("type") in ("entry_exit",)
    is_billing_cam = cam_config and cam_config.get("type") == "billing"

    # Track state: track_id → TrackState
    track_states: Dict[int, dict] = {}
    visitor_sessions: Dict[int, int] = {}  # track_id → session_count (for re-entries)

    # Entry log for group detection
    recent_entries = []  # list of (frame, visitor_id)

    frame_idx = 0
    events_count = 0
    start_time = time.time()

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)

    with EventWriter(str(output_path)) as writer:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            # Skip frames for efficiency
            if frame_idx % FRAME_SKIP != 0:
                continue

            # YOLO inference with ByteTrack
            results = model.track(
                frame,
                persist=True,
                classes=[PERSON_CLASS_ID],
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if not results or results[0].boxes is None:
                continue

            boxes = results[0].boxes
            if boxes.id is None:
                continue

            # Get track IDs, bboxes, confidences
            track_ids = boxes.id.cpu().numpy().astype(int)
            bboxes = boxes.xyxy.cpu().numpy().astype(int)  # x1,y1,x2,y2
            confs = boxes.conf.cpu().numpy()

            current_track_ids = set(track_ids.tolist())
            timestamp = frame_to_ts(frame_idx, fps)

            # Process each detected person
            for i, track_id in enumerate(track_ids):
                bbox = tuple(bboxes[i])
                conf = float(confs[i])
                x1, y1, x2, y2 = bbox
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Staff detection (sample every 10th frame for performance)
                if frame_idx % 30 == 0 or track_id not in track_states:
                    staff_flag, staff_conf = detect_staff(frame, bbox, conf)
                else:
                    existing = track_states.get(track_id, {})
                    staff_flag = existing.get("is_staff", False)

                # Get visitor_id
                session_count = visitor_sessions.get(track_id, 0)
                visitor_id = make_visitor_id(track_id, camera_id, session_count)

                # --- New track: potential ENTRY ---
                if track_id not in track_states:
                    is_reentry = False
                    reentry_match = None

                    # Check re-entry
                    if is_entry_cam:
                        bbox_ar = (x2 - x1) / max(1, y2 - y1)
                        for exited in [s for s in track_states.values() if not s.get("is_active", True)]:
                            prev_ar = exited.get("bbox_ar", 1.0)
                            prev_frame = exited.get("exit_frame", 0)
                            if (abs(bbox_ar - prev_ar) < 0.3
                                    and (frame_idx - prev_frame) < int(90 * fps)):
                                is_reentry = True
                                reentry_match = exited
                                # Restore visitor_id from matched session
                                visitor_id = exited["visitor_id"]
                                break

                    # Initialize track state
                    track_states[track_id] = {
                        "visitor_id": visitor_id,
                        "is_staff": staff_flag,
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "prev_cy": cy,
                        "cx": cx, "cy": cy,
                        "bbox_ar": (x2 - x1) / max(1, y2 - y1),
                        "current_zone": None,
                        "zone_entry_frame": None,
                        "dwell_last_emit_frame": None,
                        "session_seq": 1,
                        "is_active": True,
                        "exit_frame": None,
                    }

                    # Emit ENTRY or REENTRY for entry cameras
                    if is_entry_cam:
                        event_type = "REENTRY" if is_reentry else "ENTRY"
                        # Check for group entry
                        recent_entries = [(f, v) for f, v in recent_entries
                                         if frame_idx - f < int(2 * fps)]
                        is_group = len(recent_entries) >= 1
                        recent_entries.append((frame_idx, visitor_id))

                        event = make_event(
                            camera_id=camera_id,
                            visitor_id=visitor_id,
                            event_type=event_type,
                            timestamp=timestamp,
                            zone_id=None,
                            dwell_ms=0,
                            is_staff=staff_flag,
                            confidence=conf,
                            session_seq=1,
                        )
                        writer.write(event)
                        events_count += 1
                        if verbose:
                            print(f"  [{timestamp}] {event_type}: {visitor_id} "
                                  f"{'(GROUP)' if is_group else ''} "
                                  f"{'(STAFF)' if staff_flag else ''}")
                else:
                    # Update existing track
                    state = track_states[track_id]
                    prev_cy = state["cy"]
                    state["prev_cy"] = prev_cy
                    state["cx"] = cx
                    state["cy"] = cy
                    state["last_frame"] = frame_idx
                    state["visitor_id"] = visitor_id
                    state["is_staff"] = staff_flag

                    # --- ENTRY/EXIT detection (direction crossing) for entry cameras ---
                    if is_entry_cam and zone_mapper.entry_line:
                        crossing = zone_mapper.check_entry_exit(prev_cy, cy)
                        if crossing == "EXIT":
                            event = make_event(
                                camera_id=camera_id,
                                visitor_id=visitor_id,
                                event_type="EXIT",
                                timestamp=timestamp,
                                is_staff=staff_flag,
                                confidence=conf,
                                session_seq=state["session_seq"],
                            )
                            writer.write(event)
                            events_count += 1
                            state["is_active"] = False
                            state["exit_frame"] = frame_idx
                            if verbose:
                                print(f"  [{timestamp}] EXIT: {visitor_id}")

                    # --- Zone detection (for floor/billing cameras) ---
                    if not is_entry_cam:
                        new_zone = zone_mapper.get_zone(cx, cy)
                        prev_zone = state.get("current_zone")

                        if new_zone != prev_zone:
                            # Zone exit
                            if prev_zone is not None:
                                dwell_ms = int((frame_idx - state["zone_entry_frame"]) / fps * 1000) \
                                    if state["zone_entry_frame"] else 0
                                event = make_event(
                                    camera_id=camera_id,
                                    visitor_id=visitor_id,
                                    event_type="ZONE_EXIT",
                                    timestamp=timestamp,
                                    zone_id=prev_zone,
                                    dwell_ms=dwell_ms,
                                    is_staff=staff_flag,
                                    confidence=conf,
                                    sku_zone=zone_mapper.get_zone_sku(prev_zone),
                                    session_seq=state["session_seq"],
                                )
                                writer.write(event)
                                events_count += 1
                                state["session_seq"] += 1

                            # Zone enter
                            if new_zone is not None:
                                # Check billing queue depth
                                queue_depth = None
                                if is_billing_cam and new_zone == "BILLING":
                                    queue_depth = sum(
                                        1 for s in track_states.values()
                                        if s.get("current_zone") == "BILLING"
                                        and not s.get("is_staff", False)
                                        and s.get("is_active", True)
                                    )

                                event_type = ("BILLING_QUEUE_JOIN"
                                              if is_billing_cam and new_zone == "BILLING" and queue_depth and queue_depth > 0
                                              else "ZONE_ENTER")
                                event = make_event(
                                    camera_id=camera_id,
                                    visitor_id=visitor_id,
                                    event_type=event_type,
                                    timestamp=timestamp,
                                    zone_id=new_zone,
                                    dwell_ms=0,
                                    is_staff=staff_flag,
                                    confidence=conf,
                                    queue_depth=queue_depth,
                                    sku_zone=zone_mapper.get_zone_sku(new_zone),
                                    session_seq=state["session_seq"],
                                )
                                writer.write(event)
                                events_count += 1
                                state["session_seq"] += 1
                                state["zone_entry_frame"] = frame_idx
                                state["dwell_last_emit_frame"] = frame_idx
                                if verbose and not staff_flag:
                                    print(f"  [{timestamp}] ZONE_ENTER: {visitor_id} → {new_zone}")

                            state["current_zone"] = new_zone

                        # --- ZONE_DWELL (every 30s of continuous dwell) ---
                        elif new_zone is not None and state.get("zone_entry_frame"):
                            dwell_frames = frame_idx - state["zone_entry_frame"]
                            last_emit = state.get("dwell_last_emit_frame") or state["zone_entry_frame"]
                            frames_since_emit = frame_idx - last_emit
                            if dwell_frames >= int(30 * fps) and frames_since_emit >= int(30 * fps):
                                dwell_ms = int(dwell_frames / fps * 1000)
                                event = make_event(
                                    camera_id=camera_id,
                                    visitor_id=visitor_id,
                                    event_type="ZONE_DWELL",
                                    timestamp=timestamp,
                                    zone_id=new_zone,
                                    dwell_ms=dwell_ms,
                                    is_staff=staff_flag,
                                    confidence=conf,
                                    sku_zone=zone_mapper.get_zone_sku(new_zone),
                                    session_seq=state["session_seq"],
                                )
                                writer.write(event)
                                events_count += 1
                                state["dwell_last_emit_frame"] = frame_idx

            # --- Handle disappeared tracks (exited without crossing line) ---
            disappeared = set(track_states.keys()) - current_track_ids
            for track_id in disappeared:
                state = track_states.get(track_id)
                if state and state.get("is_active"):
                    # Check if they were in a zone → emit ZONE_EXIT
                    if state.get("current_zone") and not is_entry_cam:
                        dwell_ms = int((frame_idx - (state.get("zone_entry_frame") or frame_idx)) / fps * 1000)
                        event = make_event(
                            camera_id=camera_id,
                            visitor_id=state["visitor_id"],
                            event_type="ZONE_EXIT",
                            timestamp=timestamp,
                            zone_id=state["current_zone"],
                            dwell_ms=dwell_ms,
                            is_staff=state["is_staff"],
                            confidence=0.5,
                            session_seq=state["session_seq"],
                        )
                        writer.write(event)
                        events_count += 1

                    state["is_active"] = False
                    state["exit_frame"] = frame_idx

            # Progress reporting
            if verbose and frame_idx % 300 == 0:
                elapsed = time.time() - start_time
                progress = frame_idx / max(total_frames, 1) * 100
                print(f"  Progress: {progress:.1f}% | Frame: {frame_idx}/{total_frames} | "
                      f"Events: {events_count} | Time: {elapsed:.0f}s")

    cap.release()
    elapsed = time.time() - start_time
    print(f"\n✓ Camera {cam_num} done: {events_count} events in {elapsed:.1f}s")
    print(f"  Output: {output_path}")
    return events_count


def main():
    parser = argparse.ArgumentParser(description="CCTV Detection Pipeline — Purplle Challenge")
    parser.add_argument("--cam", type=int, choices=[1, 2, 3, 4, 5],
                        help="Camera number to process (1-5)")
    parser.add_argument("--all", action="store_true", help="Process all cameras")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process (for testing)")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if not args.cam and not args.all:
        parser.error("Specify --cam N or --all")

    cameras = list(CAMERA_MAP.keys()) if args.all else [args.cam]
    total_events = 0

    print(f"\nPurplle Store Intelligence - Detection Pipeline")
    print(f"Store: STORE_BLR_002 (Brigade Road, Bangalore)")
    print(f"Video base time: {VIDEO_BASE_TIME.isoformat()}")
    print(f"Processing {len(cameras)} camera(s): {cameras}")

    for cam_num in cameras:
        events = process_camera(cam_num, max_frames=args.max_frames, verbose=args.verbose)
        total_events += events

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE: {total_events} total events")
    print(f"Events stored in: {EVENTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
