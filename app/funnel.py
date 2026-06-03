"""
Funnel computation — GET /stores/{store_id}/funnel

Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase
Session is the unit — re-entries must not double-count a visitor.
"""
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models import FunnelResponse, FunnelStage

logger = logging.getLogger(__name__)


def get_funnel(store_id: str, db: Session, date_filter: Optional[str] = None) -> FunnelResponse:
    """
    Compute conversion funnel stages. 
    Unit of analysis: unique visitor_id per session (ENTRY events, no double-counting for re-entries).
    """
    if not date_filter:
        date_filter = "2026-04-10"

    # Stage 1: Total unique visitors (ENTRY only, not REENTRY — don't double count)
    entry_result = db.execute(text("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id = :store_id
          AND event_type = 'ENTRY'
          AND is_staff = 0
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": f"{date_filter}%"}).fetchone()
    entries = entry_result.cnt or 0

    # Stage 2: Visitors who visited at least one product zone (ZONE_ENTER on non-billing zone)
    zone_visit_result = db.execute(text("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id = :store_id
          AND event_type IN ('ZONE_ENTER', 'ZONE_DWELL')
          AND zone_id NOT IN ('BILLING', 'ENTRY_EXIT', 'ENTRY_EXIT_2')
          AND zone_id IS NOT NULL
          AND is_staff = 0
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": f"{date_filter}%"}).fetchone()
    zone_visitors = zone_visit_result.cnt or 0

    # Stage 3: Visitors who reached billing queue
    billing_result = db.execute(text("""
        SELECT COUNT(DISTINCT visitor_id) as cnt
        FROM events
        WHERE store_id = :store_id
          AND event_type IN ('BILLING_QUEUE_JOIN', 'ZONE_ENTER')
          AND zone_id = 'BILLING'
          AND is_staff = 0
          AND timestamp LIKE :date_prefix
    """), {"store_id": store_id, "date_prefix": f"{date_filter}%"}).fetchone()
    billing_visitors = billing_result.cnt or 0

    # Stage 4: Purchasers (correlated with POS)
    purchase_result = db.execute(text("""
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
          ) BETWEEN 0 AND 300
    """), {
        "store_id": store_id,
        "date_prefix": f"{date_filter}%",
        "date_filter": date_filter,
    }).fetchone()
    purchasers = purchase_result.cnt or 0

    def drop_off(current: int, previous: int) -> float:
        if previous == 0:
            return 0.0
        return round((1 - current / previous) * 100, 1)

    stages = [
        FunnelStage(stage="Entry", count=entries, drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit", count=zone_visitors,
                    drop_off_pct=drop_off(zone_visitors, entries)),
        FunnelStage(stage="Billing Queue", count=billing_visitors,
                    drop_off_pct=drop_off(billing_visitors, zone_visitors)),
        FunnelStage(stage="Purchase", count=purchasers,
                    drop_off_pct=drop_off(purchasers, billing_visitors)),
    ]

    note = ""
    if entries == 0:
        note = "No visitor data for this date. Ensure pipeline has been run."
    elif entries < 20:
        note = f"Low sample size ({entries} visitors). Results may not be statistically significant."

    return FunnelResponse(
        store_id=store_id,
        stages=stages,
        note=note,
    )
