"""
Health endpoint — GET /health

Reports:
- Overall service status (OK / DEGRADED)
- Per-store: last event timestamp, lag_seconds, STALE_FEED warning if >10 min lag
"""
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging

from app.models import HealthResponse, StoreHealth

logger = logging.getLogger(__name__)

STALE_FEED_THRESHOLD_MINUTES = 10
KNOWN_STORES = ["STORE_BLR_002"]


def get_health(db: Session) -> HealthResponse:
    """Check service health and per-store feed freshness."""
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Reference time for the video scenario (store close)
    reference_now = datetime(2026, 4, 10, 21, 0, 0, tzinfo=timezone.utc)
    stale_threshold = timedelta(minutes=STALE_FEED_THRESHOLD_MINUTES)

    store_healths = []
    overall_ok = True

    for store_id in KNOWN_STORES:
        try:
            result = db.execute(text("""
                SELECT MAX(timestamp) as last_ts
                FROM events
                WHERE store_id = :store_id
            """), {"store_id": store_id}).fetchone()

            last_ts_str = result.last_ts if result else None

            if not last_ts_str:
                store_healths.append(StoreHealth(
                    store_id=store_id,
                    status="NO_DATA",
                    last_event_timestamp=None,
                    lag_seconds=None,
                    stale_feed=True,
                ))
                overall_ok = False
            else:
                last_ts = datetime.strptime(last_ts_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                lag = (reference_now - last_ts).total_seconds()
                stale = lag > stale_threshold.total_seconds()

                store_healths.append(StoreHealth(
                    store_id=store_id,
                    status="STALE" if stale else "OK",
                    last_event_timestamp=last_ts_str,
                    lag_seconds=round(lag, 1),
                    stale_feed=stale,
                ))
                if stale:
                    overall_ok = False

        except Exception as e:
            logger.error("Health check failed for store %s: %s", store_id, e)
            store_healths.append(StoreHealth(
                store_id=store_id,
                status="NO_DATA",
                last_event_timestamp=None,
                lag_seconds=None,
                stale_feed=True,
            ))
            overall_ok = False

    return HealthResponse(
        status="OK" if overall_ok else "DEGRADED",
        stores=store_healths,
        checked_at=now_ts,
    )
