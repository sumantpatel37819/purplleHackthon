"""
Ingestion endpoint — POST /events/ingest

Key requirements:
- Idempotent by event_id (safe to call twice with same payload)
- Partial success: malformed events return errors but valid ones are ingested
- Batch up to 500 events per request
- Structured error response
"""
import uuid
from datetime import datetime, timezone
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException
import logging

from app.models import Event, IngestRequest, IngestResponse
from app.database import Event as EventDB

logger = logging.getLogger(__name__)


def ingest_events(request: IngestRequest, db: Session) -> IngestResponse:
    """
    Ingest a batch of events. Idempotent by event_id.
    Returns counts of ingested, duplicates, and errors.
    """
    ingested = 0
    duplicates = 0
    errors = 0
    error_details = []

    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for event in request.events:
        try:
            db_event = EventDB(
                event_id=event.event_id,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                queue_depth=event.metadata.queue_depth if event.metadata else None,
                sku_zone=event.metadata.sku_zone if event.metadata else None,
                session_seq=event.metadata.session_seq if event.metadata else 1,
                ingested_at=now_ts,
            )
            db.add(db_event)
            db.flush()
            ingested += 1

        except IntegrityError:
            # Duplicate event_id — idempotent, not an error
            db.rollback()
            duplicates += 1

        except Exception as e:
            db.rollback()
            errors += 1
            error_details.append(f"event_id={event.event_id}: {str(e)[:100]}")
            logger.warning("Ingest error for event %s: %s", event.event_id, e)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error("Batch commit failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Database commit failed: {e}")

    logger.info("Ingest: ingested=%d duplicates=%d errors=%d", ingested, duplicates, errors)
    return IngestResponse(
        ingested=ingested,
        duplicates=duplicates,
        errors=errors,
        error_details=error_details[:10],  # cap at 10 error messages
    )
