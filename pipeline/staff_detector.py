"""
Staff Detector — classifies whether a detected person is store staff.

Approach:
1. HSV color analysis of torso region: staff typically wear uniform-colored clothing
2. Dominant color heuristic: if top 60%+ of torso pixels share a narrow HSV hue range
   → likely uniform → is_staff = True
3. Fallback: YOLO class confidence hints (if model fine-tuned)

This is a heuristic — not a trained classifier. The key insight is that retail staff
in beauty stores typically wear branded uniforms (often black, navy, or branded colors).
We look for HIGH color saturation + HIGH uniformity in the torso crop.
"""
import cv2
import numpy as np
from typing import Tuple


# Define known staff uniform color ranges in HSV
# Purplle / beauty retail typically: black, dark navy, or brand-purple uniforms
STAFF_COLOR_RANGES = [
    # Black/dark: low value
    {"name": "black", "h": (0, 180), "s": (0, 80), "v": (0, 60)},
    # Dark navy
    {"name": "navy", "h": (100, 130), "s": (80, 255), "v": (30, 100)},
    # Purple/violet (Purplle brand color)
    {"name": "purple", "h": (130, 160), "s": (80, 255), "v": (50, 200)},
    # White (some beauty brands use white uniforms)
    {"name": "white", "h": (0, 180), "s": (0, 30), "v": (200, 255)},
]


def extract_torso_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract the torso region from a bounding box (middle 40% height, full width).
    bbox: (x1, y1, x2, y2) in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    # Torso: from 25% to 65% of bounding box height
    torso_y1 = y1 + int(h * 0.25)
    torso_y2 = y1 + int(h * 0.65)
    # Ensure within frame bounds
    torso_y1 = max(0, torso_y1)
    torso_y2 = min(frame.shape[0], torso_y2)
    x1 = max(0, x1)
    x2 = min(frame.shape[1], x2)
    return frame[torso_y1:torso_y2, x1:x2]


def compute_color_uniformity(torso_crop: np.ndarray) -> Tuple[bool, float]:
    """
    Check if the torso region is dominated by a single uniform color.
    Returns (is_uniform, confidence_score 0-1).
    
    Strategy: Convert to HSV, compute histogram of Hue channel,
    check if top bin accounts for >50% of pixels.
    """
    if torso_crop.size == 0 or torso_crop.shape[0] < 5 or torso_crop.shape[1] < 5:
        return False, 0.0

    hsv = cv2.cvtColor(torso_crop, cv2.COLOR_BGR2HSV)
    
    # Check against each known staff color range
    total_pixels = torso_crop.shape[0] * torso_crop.shape[1]
    best_match = 0.0
    
    for color_def in STAFF_COLOR_RANGES:
        lower = np.array([color_def["h"][0], color_def["s"][0], color_def["v"][0]])
        upper = np.array([color_def["h"][1], color_def["s"][1], color_def["v"][1]])
        mask = cv2.inRange(hsv, lower, upper)
        match_ratio = np.count_nonzero(mask) / total_pixels
        best_match = max(best_match, match_ratio)
    
    # If >55% of torso matches a uniform color → likely staff
    is_uniform = best_match > 0.55
    return is_uniform, best_match


def is_staff(frame: np.ndarray, bbox: Tuple[int, int, int, int],
             yolo_conf: float = 1.0) -> Tuple[bool, float]:
    """
    Main staff classification function.
    
    Args:
        frame: Full BGR video frame
        bbox: (x1, y1, x2, y2) bounding box of detected person
        yolo_conf: YOLO detection confidence (for weighting)
    
    Returns:
        (is_staff: bool, confidence: float 0-1)
    
    Decision: heuristic color uniformity. Staff wear uniforms;
    customers wear varied clothing → lower color uniformity.
    """
    torso = extract_torso_region(frame, bbox)
    uniform, uniform_score = compute_color_uniformity(torso)
    
    # Confidence is combination of uniformity score and YOLO conf
    confidence = uniform_score * 0.8 + (1 - yolo_conf) * 0.2 if uniform else 0.1
    confidence = float(np.clip(confidence, 0.0, 1.0))
    
    return uniform, confidence
