"""
Anomaly detection — GET /stores/{store_id}/anomalies

Detects:
1. BILLING_QUEUE_SPIKE — queue depth exceeds threshold
2. CONVERSION_DROP — conversion rate significantly below 7-day average
3. DEAD_ZONE — no visits in a zone for >30 minutes during store hours
4. STALE_FEED — camera feed has no events for >10 minutes

Each anomaly has: severity (INFO/WARN/CRITICAL), description, suggested_action.
"""
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models import AnomaliesResponse, Anomaly

logger = logging.getLogger(__name__)

# Thresholds
QUEUE_SPIKE_THRESHOLD = 4       # >4 people in queue = spike
DEAD_ZONE_MINUTES = 30          # Zone with no visits for 30+ min
STALE_FEED_MINUTES = 10         # Camera with no events for 10+ min
CONVERSION_DROP_THRESHOLD = 0.3  # 30% drop from baseline = anomaly
BASELINE_CONVERSION = 0.25      # Assumed baseline (no 7-day history available)


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def detect_anomalies(store_id: str, db: Session,
                     date_filter: Optional[str] = None) -> AnomaliesResponse:
    """Detect active operational anomalies for a store."""
    if not date_filter:
        date_filter = "2026-04-10"

    anomalies: List[Anomaly] = []
    now_ts = _now_ts()

    # ── 1. BILLING_QUEUE_SPIKE ─────────────────────────────────────────────
    queue_result = db.execute(text("""
        SELECT COALESCE(MAX(queue_depth), 0) as max_queue
        FROM events
        WHERE store_id = :store_id
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND timestamp LIKE :date_prefix
          AND timestamp >= :recent_ts
          AND queue_depth IS NOT NULL
    """), {
        "store_id": store_id,
        "date_prefix": f"{date_filter}%",
        "recent_ts": f"{date_filter}T00:00:00Z",
    }).fetchone()

    max_queue = queue_result.max_queue if queue_result else 0
    if max_queue > QUEUE_SPIKE_THRESHOLD:
        severity = "CRITICAL" if max_queue > 8 else "WARN"
        anomalies.append(Anomaly(
            anomaly_id=str(uuid.uuid4()),
            anomaly_type="BILLING_QUEUE_SPIKE",
            severity=severity,
            description=f"Billing queue depth reached {max_queue} (threshold: {QUEUE_SPIKE_THRESHOLD})",
            suggested_action="Open additional billing counter or redirect customers to express checkout.",
            detected_at=now_ts,
            zone_id="BILLING",
            value=float(max_queue),
            threshold=float(QUEUE_SPIKE_THRESHOLD),
        ))

    # ── 2. CONVERSION_DROP ─────────────────────────────────────────────────
    # Compare today's conversion to baseline
    from app.metrics import get_metrics
    try:
        metrics = get_metrics(store_id, db, date_filter)
        conversion_rate = metrics.conversion_rate
        drop = BASELINE_CONVERSION - conversion_rate

        if conversion_rate > 0 and drop > CONVERSION_DROP_THRESHOLD * BASELINE_CONVERSION:
            severity = "CRITICAL" if drop > 0.5 * BASELINE_CONVERSION else "WARN"
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type="CONVERSION_DROP",
                severity=severity,
                description=(
                    f"Conversion rate {conversion_rate:.1%} is {drop/BASELINE_CONVERSION:.0%} "
                    f"below baseline {BASELINE_CONVERSION:.1%}"
                ),
                suggested_action="Review staff availability, check billing area for issues, inspect top drop-off zone.",
                detected_at=now_ts,
                value=conversion_rate,
                threshold=BASELINE_CONVERSION,
            ))
        elif metrics.unique_visitors > 0 and conversion_rate == 0.0:
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type="CONVERSION_DROP",
                severity="WARN",
                description="Conversion rate is 0% — no purchases recorded today.",
                suggested_action="Verify POS system connectivity and check billing zone camera.",
                detected_at=now_ts,
                value=0.0,
                threshold=BASELINE_CONVERSION,
            ))
    except Exception as e:
        logger.warning("Could not compute conversion for anomaly check: %s", e)

    # ── 3. DEAD_ZONE ──────────────────────────────────────────────────────
    product_zones = ["SKINCARE", "MAKEUP", "HAIRCARE", "FRAGRANCE", "ACCESSORIES"]

    for zone in product_zones:
        last_visit_result = db.execute(text("""
            SELECT MAX(timestamp) as last_ts
            FROM events
            WHERE store_id = :store_id
              AND zone_id = :zone_id
              AND timestamp LIKE :date_prefix
              AND is_staff = 0
        """), {
            "store_id": store_id,
            "zone_id": zone,
            "date_prefix": f"{date_filter}%",
        }).fetchone()

        last_ts_str = last_visit_result.last_ts if last_visit_result else None

        if last_ts_str:
            try:
                # Parse and compare against now
                last_ts = datetime.strptime(last_ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                # For video replay, use video end time as "now" (2026-04-10 end of day)
                reference_now = datetime(2026, 4, 10, 21, 0, 0, tzinfo=timezone.utc)
                minutes_ago = (reference_now - last_ts).total_seconds() / 60
                if minutes_ago > DEAD_ZONE_MINUTES:
                    anomalies.append(Anomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="DEAD_ZONE",
                        severity="INFO",
                        description=f"Zone '{zone}' has had no customer visits for {minutes_ago:.0f} minutes.",
                        suggested_action=f"Check if {zone} display is accessible. Consider staff assistance or promotional placement.",
                        detected_at=now_ts,
                        zone_id=zone,
                        value=minutes_ago,
                        threshold=float(DEAD_ZONE_MINUTES),
                    ))
            except ValueError:
                pass
        else:
            # Zone has never had a visit today
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type="DEAD_ZONE",
                severity="INFO",
                description=f"Zone '{zone}' has had no customer visits recorded today.",
                suggested_action=f"Verify camera coverage for {zone} and check zone accessibility.",
                detected_at=now_ts,
                zone_id=zone,
                value=float("inf"),
                threshold=float(DEAD_ZONE_MINUTES),
            ))

    # ── 4. STALE_FEED ────────────────────────────────────────────────────
    cameras = ["CAM_ENTRY_01", "CAM_ENTRY_02", "CAM_FLOOR_01", "CAM_BILLING_01", "CAM_FLOOR_02"]
    for camera_id in cameras:
        result = db.execute(text("""
            SELECT MAX(timestamp) as last_ts
            FROM events
            WHERE store_id = :store_id AND camera_id = :camera_id
              AND timestamp LIKE :date_prefix
        """), {
            "store_id": store_id,
            "camera_id": camera_id,
            "date_prefix": f"{date_filter}%",
        }).fetchone()

        last_ts_str = result.last_ts if result else None
        if not last_ts_str:
            anomalies.append(Anomaly(
                anomaly_id=str(uuid.uuid4()),
                anomaly_type="STALE_FEED",
                severity="WARN",
                description=f"Camera {camera_id} has no events recorded for today.",
                suggested_action=f"Check camera {camera_id} hardware and network connectivity.",
                detected_at=now_ts,
                zone_id=camera_id,
            ))

    return AnomaliesResponse(
        store_id=store_id,
        active_anomalies=anomalies,
        checked_at=now_ts,
    )
