"""
Zone Mapper — maps bounding box centroids to store zones using polygon containment.
Uses normalized coordinates (0-1) for resolution independence.
"""
import json
import numpy as np
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent.parent / "data"


def load_store_layout(layout_path: Optional[str] = None) -> dict:
    """Load store layout JSON."""
    path = Path(layout_path) if layout_path else DATA_DIR / "store_layout.json"
    with open(path) as f:
        return json.load(f)


class ZoneMapper:
    """
    Maps pixel coordinates to zone IDs using polygon containment (ray-casting algorithm).
    Coordinates are normalized 0–1 relative to frame width/height.
    """

    def __init__(self, camera_id: str, frame_width: int, frame_height: int,
                 layout_path: Optional[str] = None):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.layout = load_store_layout(layout_path)

        # Build zone list for this camera
        self.zones = [
            z for z in self.layout["zones"]
            if z["camera_id"] == camera_id
        ]
        # Precompute polygons in pixel space
        self._polygons = []
        for zone in self.zones:
            poly_norm = np.array(zone["polygon_normalized"])
            poly_px = poly_norm.copy()
            poly_px[:, 0] *= frame_width
            poly_px[:, 1] *= frame_height
            self._polygons.append(poly_px)

        # Entry line for entry/exit cameras
        self.entry_line = None
        for cam in self.layout["cameras"]:
            if cam["camera_id"] == camera_id and "entry_line" in cam:
                line = cam["entry_line"]
                self.entry_line = {
                    "y_px": int(line["y"] * frame_height),
                    "x_start_px": int(line["x_start"] * frame_width),
                    "x_end_px": int(line["x_end"] * frame_width),
                    "direction_in": cam.get("direction_in", "top_to_bottom")
                }

    def get_zone(self, cx: float, cy: float) -> Optional[str]:
        """
        Get zone ID for a centroid point (pixel coordinates).
        Returns None if not in any zone.
        """
        for i, poly in enumerate(self._polygons):
            if self._point_in_polygon(cx, cy, poly):
                return self.zones[i]["zone_id"]
        return None

    def get_zone_sku(self, zone_id: str) -> Optional[str]:
        """Get SKU zone label for a zone_id."""
        for z in self.zones:
            if z["zone_id"] == zone_id:
                return z.get("sku_zone")
        return None

    def check_entry_exit(self, prev_cy: float, curr_cy: float) -> Optional[str]:
        """
        Check if a person crossed the entry/exit line.
        Returns 'ENTRY', 'EXIT', or None.
        """
        if self.entry_line is None:
            return None
        line_y = self.entry_line["y_px"]
        direction_in = self.entry_line["direction_in"]

        if direction_in == "top_to_bottom":
            if prev_cy < line_y and curr_cy >= line_y:
                return "ENTRY"
            if prev_cy >= line_y and curr_cy < line_y:
                return "EXIT"
        else:  # bottom_to_top
            if prev_cy > line_y and curr_cy <= line_y:
                return "ENTRY"
            if prev_cy <= line_y and curr_cy > line_y:
                return "EXIT"
        return None

    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        px, py = x, y
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
