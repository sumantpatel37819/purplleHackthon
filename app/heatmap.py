"""
Heatmap — GET /stores/{store_id}/heatmap

Zone visit frequency + avg dwell time, normalized 0-100, ready for grid heatmap rendering.
"""
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models import HeatmapResponse, HeatmapZone

logger = logging.getLogger(__name__)

MIN_SESSIONS_FOR_HIGH_CONFIDENCE = 20


def get_heatmap(store_id: str, db: Session, date_filter: Optional[str] = None) -> HeatmapResponse:
    """Compute zone heatmap with normalized scores."""
    if not date_filter:
        date_filter = "2026-04-10"

    rows = db.execute(text("""
        SELECT zone_id,
               sku_zone,
               COUNT(DISTINCT visitor_id) as visit_count,
               AVG(CASE WHEN dwell_ms > 0 THEN dwell_ms ELSE NULL END) as avg_dwell_ms
        FROM events
        WHERE store_id = :store_id
          AND zone_id IS NOT NULL
          AND zone_id NOT IN ('ENTRY_EXIT', 'ENTRY_EXIT_2')
          AND is_staff = 0
          AND timestamp LIKE :date_prefix
          AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL', 'ZONE_EXIT', 'BILLING_QUEUE_JOIN')
        GROUP BY zone_id, sku_zone
        ORDER BY visit_count DESC
    """), {"store_id": store_id, "date_prefix": f"{date_filter}%"}).fetchall()

    if not rows:
        return HeatmapResponse(store_id=store_id, zones=[], data_confidence="LOW")

    # Normalize visit_count to 0-100
    max_visits = max(r.visit_count for r in rows) or 1
    total_sessions = sum(r.visit_count for r in rows)

    zones = []
    for row in rows:
        normalized = round((row.visit_count / max_visits) * 100, 1)
        zones.append(HeatmapZone(
            zone_id=row.zone_id,
            sku_zone=row.sku_zone,
            visit_count=row.visit_count,
            avg_dwell_sec=round((row.avg_dwell_ms or 0) / 1000.0, 2),
            normalized_score=normalized,
        ))

    data_confidence = "HIGH" if total_sessions >= MIN_SESSIONS_FOR_HIGH_CONFIDENCE else "LOW"

    return HeatmapResponse(
        store_id=store_id,
        zones=zones,
        data_confidence=data_confidence,
    )
