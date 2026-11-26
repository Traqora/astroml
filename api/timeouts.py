from __future__ import annotations
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
logger = logging.getLogger(__name__)
def set_sync_db_timeout(session: Session, timeout_seconds: int) -> None:
    """Issue SET LOCAL statement_timeout for PostgreSQL sessions only.
    Uses session.bind to access the engine (SQLAlchemy 2.x compatible;
    session.get_bind() was removed in 2.0). Skipped on SQLite and any
    non-PostgreSQL dialect. Call once at the start of a sync route handler.
    """
    engine = session.bind
    if engine is None:
        return
    if engine.dialect.name == "postgresql":
        session.execute(
            text("SET LOCAL statement_timeout = :ms"),
            {"ms": timeout_seconds * 1000},
        )
