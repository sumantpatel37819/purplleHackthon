"""
Metrics computation — GET /stores/{store_id}/metrics

Returns:
- unique_visitors: count of unique visitor_ids with ENTRY events (non-staff)
- conversion_rate: buyers (in billing zone before POS txn) / unique visitors
- avg_dwell_sec: average dwell time across all ZONE_DWELL events
- queue_depth_current: current billing queue depth (from latest BILLING_QUEUE_JOIN)
- abandonment_rate: BILLING_QUEUE_ABANDON / BILLING_QUEUE_JOIN
- zone_breakdown: per-zone visitor count and avg dwell
"""
from datetime import datetime, timezone, date
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models import MetricsResponse, ZoneDwellMetric

logger = logging.getLogger(__name__)

# POS correlation window: visitor in billing zone within 5 minutes before transaction
POS_CORRELATION_WINDOW_SEC = 300  # 5 minutes


def get_metrics(store_id: str, db: Session, date_filter: Optional[str] = None) -> MetricsResponse:
    """Compute real-time metrics for a store."""

    # Use today's date or provided filter (format: YYYY-MM-DD)
    if not date_filter:
        date_filter = "2026-04-10"  # Our video date (Brigade Road POS data)
    # Timestamp prefix for events (stored as YYYY-MM-DDTHH:MM:SSZ)
    ts_prefix = f"{date_filter}%"

    # 1. Unique visitors (ENTRY events, non-staff, today)
    unique_visitors_result = db.execute(text("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id = :store_id
          AND event_type IN ('ENTRY', 'REENTRY')
          AND is_staff = 0
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchone()
    unique_visitors = unique_visitors_result.cnt if unique_visitors_result else 0

    # 2. Conversion rate via POS correlation
    # Count visitors who were in BILLING zone within 5 min before any POS transaction
    converted_result = db.execute(text("""
        SELECT COUNT(DISTINCT e.visitor_id) as cnt
        FROM events e
        JOIN pos_transactions p ON p.store_id = :store_id
        WHERE e.store_id = :store_id
          AND e.event_type IN ('ZONE_ENTER', 'BILLING_QUEUE_JOIN')
          AND e.zone_id = 'BILLING'
          AND e.is_staff = 0
          AND e.timestamp LIKE :date_prefix
          AND p.order_date = :date_filter
          AND (
            CAST(strftime('%s', REPLACE(p.timestamp, 'Z', '')) AS INTEGER) -
            CAST(strftime('%s', REPLACE(e.timestamp, 'Z', '')) AS INTEGER)
          ) BETWEEN 0 AND :window_sec
    """), {
        "store_id": store_id,
        "date_prefix": ts_prefix,
        "date_filter": date_filter,
        "window_sec": POS_CORRELATION_WINDOW_SEC,
    }).fetchone()
    converted = converted_result.cnt if converted_result else 0
    conversion_rate = round(converted / unique_visitors, 4) if unique_visitors > 0 else 0.0

    # 3. Average dwell time across ZONE_DWELL events
    dwell_result = db.execute(text("""
        SELECT AVG(dwell_ms) as avg_dwell, COUNT(*) as cnt
        FROM events
        WHERE store_id = :store_id
          AND event_type IN ('ZONE_DWELL', 'ZONE_EXIT')
          AND is_staff = 0
          AND dwell_ms > 0
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchone()
    avg_dwell_ms = dwell_result.avg_dwell or 0
    avg_dwell_sec = round(avg_dwell_ms / 1000.0, 2)
    total_dwell_obs = dwell_result.cnt or 0

    # 4. Current queue depth (latest BILLING_QUEUE_JOIN count)
    queue_result = db.execute(text("""
        SELECT COALESCE(MAX(queue_depth), 0) as qd
        FROM events
        WHERE store_id = :store_id
          AND event_type = 'BILLING_QUEUE_JOIN'
          AND timestamp LIKE :date_prefix
          AND queue_depth IS NOT NULL
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchone()
    queue_depth_current = queue_result.qd if queue_result else 0

    # 5. Abandonment rate
    join_result = db.execute(text("""
        SELECT COUNT(*) as cnt FROM events
        WHERE store_id = :store_id AND event_type = 'BILLING_QUEUE_JOIN'
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchone()
    abandon_result = db.execute(text("""
        SELECT COUNT(*) as cnt FROM events
        WHERE store_id = :store_id AND event_type = 'BILLING_QUEUE_ABANDON'
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchone()

    joins = join_result.cnt or 0
    abandons = abandon_result.cnt or 0
    abandonment_rate = round(abandons / joins, 4) if joins > 0 else 0.0

    # 6. Zone breakdown
    zone_rows = db.execute(text("""
        SELECT zone_id,
               COUNT(DISTINCT visitor_id) as visitor_count,
               AVG(dwell_ms) as avg_dwell
        FROM events
        WHERE store_id = :store_id
          AND zone_id IS NOT NULL
          AND is_staff = 0
          AND event_type IN ('ZONE_DWELL', 'ZONE_EXIT')
          AND timestamp LIKE :date_prefix
        GROUP BY zone_id
        ORDER BY visitor_count DESC
    """), {"store_id": store_id, "date_prefix": ts_prefix}).fetchall()

    zone_breakdown = [
        ZoneDwellMetric(
            zone_id=row.zone_id,
            avg_dwell_sec=round((row.avg_dwell or 0) / 1000.0, 2),
            visitor_count=row.visitor_count,
        )
        for row in zone_rows
    ]

    data_confidence = "LOW" if unique_visitors < 20 else "HIGH"

    return MetricsResponse(
        store_id=store_id,
        unique_visitors=unique_visitors,
        conversion_rate=conversion_rate,
        avg_dwell_sec=avg_dwell_sec,
        total_dwell_observations=total_dwell_obs,
        queue_depth_current=queue_depth_current,
        abandonment_rate=abandonment_rate,
        zone_breakdown=zone_breakdown,
        data_confidence=data_confidence,
    )
