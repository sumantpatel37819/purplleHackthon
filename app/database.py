"""
Database setup — SQLite via SQLAlchemy.

Schema design decisions (documented in CHOICES.md):
- SQLite chosen for zero-dependency deployment; swappable via DATABASE_URL env var
- event_id is PRIMARY KEY to enforce idempotency at database level
- Indexes on (store_id, timestamp) and (visitor_id) for fast analytics queries
- WAL mode enabled for better concurrent read performance
"""
import os
from pathlib import Path
from contextlib import contextmanager
from sqlalchemy import (
    create_engine, text, Column, String, Integer, Float,
    Boolean, DateTime, JSON, Index, event as sqla_event
)
from sqlalchemy.orm import declarative_base, Session, sessionmaker
from sqlalchemy.exc import OperationalError
import logging

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "events.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")

Base = declarative_base()


class Event(Base):
    """Stores raw events from the detection pipeline."""
    __tablename__ = "events"

    event_id = Column(String, primary_key=True)
    store_id = Column(String, nullable=False, index=True)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False, index=True)
    timestamp = Column(String, nullable=False)
    zone_id = Column(String, nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)
    queue_depth = Column(Integer, nullable=True)
    sku_zone = Column(String, nullable=True)
    session_seq = Column(Integer, default=1)
    ingested_at = Column(String, nullable=True)  # Server-side ingestion timestamp

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_type", "store_id", "event_type"),
    )


class POSTransaction(Base):
    """POS transaction data loaded from CSV."""
    __tablename__ = "pos_transactions"

    transaction_id = Column(String, primary_key=True)
    store_id = Column(String, nullable=False, index=True)
    order_date = Column(String, nullable=False)
    order_time = Column(String, nullable=False)
    timestamp = Column(String, nullable=True)  # Combined ISO timestamp
    total_amount = Column(Float, nullable=True)
    customer_name = Column(String, nullable=True)
    customer_number = Column(String, nullable=True)
    product_name = Column(String, nullable=True)
    brand_name = Column(String, nullable=True)
    sub_category = Column(String, nullable=True)
    qty = Column(Integer, nullable=True)


# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

# Enable WAL mode for better concurrent reads
@sqla_event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create all tables and load POS data."""
    DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    _load_pos_data()
    logger.info("Database initialized: %s", DATABASE_URL)


def _load_pos_data():
    """Load POS transaction CSV into database (idempotent)."""
    import pandas as pd
    csv_path = BASE_DIR / "resources" / "Brigade_Bangalore_10_April_26 (1)bc6219c.csv"
    if not csv_path.exists():
        logger.warning("POS CSV not found: %s", csv_path)
        return

    with SessionLocal() as session:
        existing = session.execute(text("SELECT COUNT(*) FROM pos_transactions")).scalar()
        if existing > 0:
            return  # Already loaded

    df = pd.read_csv(csv_path)
    store_id = "STORE_BLR_002"  # Map from ST1008

    with SessionLocal() as session:
        for idx, row in df.iterrows():
            try:
                # Build ISO timestamp from date + time
                # CSV date format: "10-04-2026" (DD-MM-YYYY)
                date_raw = str(row.get("order_date", "10-04-2026")).strip()
                time_str = str(row.get("order_time", "12:00:00")).strip()

                # Convert DD-MM-YYYY → YYYY-MM-DD
                if "-" in date_raw and len(date_raw) == 10:
                    parts = date_raw.split("-")
                    if len(parts[0]) == 2:  # DD-MM-YYYY
                        date_iso = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    else:  # YYYY-MM-DD already
                        date_iso = date_raw
                else:
                    date_iso = "2026-04-10"

                ts = f"{date_iso}T{time_str}Z"

                # Use order_id + row_index as unique transaction_id
                # (invoice_number is shared across multi-item orders)
                order_id = str(row.get("order_id", idx))
                txn_id = f"{order_id}_{idx}"

                txn = POSTransaction(
                    transaction_id=txn_id,
                    store_id=store_id,
                    order_date=date_iso,
                    order_time=time_str,
                    timestamp=ts,
                    total_amount=float(row.get("total_amount", 0) or 0),
                    customer_name=str(row.get("customer_name", "")).strip(),
                    customer_number=str(row.get("customer_number", "")),
                    product_name=str(row.get("product_name", ""))[:500],
                    brand_name=str(row.get("brand_name", "")),
                    sub_category=str(row.get("sub_category", "")),
                    qty=int(row.get("qty", 1) or 1),
                )
                session.add(txn)
            except Exception as e:
                logger.warning("Error loading POS row %s: %s", idx, e)
        session.commit()

    logger.info("POS transactions loaded from CSV")


@contextmanager
def get_db_session():
    """Context manager for database sessions with error handling."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except OperationalError as e:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    except OperationalError:
        raise
    finally:
        db.close()
