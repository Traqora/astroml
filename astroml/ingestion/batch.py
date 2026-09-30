from astroml.utils.exceptions import AstroMLError
"""Chunked UPSERT batching for ingestion writes.

Provides batch accumulation and chunked persistence of ORM models
to reduce database round-trips during high-throughput ingestion.

Key components:
- BatchBuffer: Accumulates models and flushes in configurable chunks
- chunked_merge: Context manager for chunked merge + commit cycles
- batch_upsert: One-shot chunked upsert for a sequence of models
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

from sqlalchemy.orm import Session

from astroml.ingestion.metrics import (
    BATCH_BUFFER_SIZE,
    BATCH_FLUSH_DURATION,
    BATCH_FLUSH_TOTAL,
)

logger = logging.getLogger("astroml.ingestion.batch")

T = TypeVar("T")


class BatchBuffer:
    """Accumulates ORM models and flushes them in chunked batches.

    Models are buffered in memory until either the chunk_size is reached
    or an explicit flush is called. Each flush issues a single commit
    for all buffered models, reducing database round-trips.

    Args:
        session: SQLAlchemy session to use for persistence.
        chunk_size: Maximum number of models per flush batch.
        flush_on_exit: Whether to auto-flush remaining models on close.

    Example:
        buffer = BatchBuffer(session, chunk_size=50)
        for record in records:
            buffer.add(parse_model(record))
        buffer.flush()
    """

    def __init__(
        self,
        session: Session,
        chunk_size: int = 100,
        flush_on_exit: bool = True,
    ) -> None:
        self._session = session
        self._chunk_size = chunk_size
        self._flush_on_exit = flush_on_exit
        self._buffer: list[T] = []
        self._total_flushed: int = 0
        self._flush_count: int = 0

    def add(self, model: T) -> None:
        """Add a model to the buffer, flushing if chunk_size is reached."""
        self._buffer.append(model)
        BATCH_BUFFER_SIZE.set(len(self._buffer))
        if len(self._buffer) >= self._chunk_size:
            self._flush()

    def add_many(self, models: Iterable[T]) -> None:
        """Add multiple models, flushing as each chunk fills."""
        for model in models:
            self.add(model)

    def flush(self) -> int:
        """Flush all remaining models in the buffer.

        Returns:
            Number of models flushed.
        """
        count = len(self._buffer)
        if count > 0:
            self._flush()
        return count

    def _flush(self) -> None:
        """Internal flush: upsert all buffered models and commit."""
        if not self._buffer:
            return

        start = time.time()
        try:
            for model in self._buffer:
                self._merge_one(model)
            self._session.commit()
            duration = time.time() - start
            flushed = len(self._buffer)
            self._total_flushed += flushed
            self._flush_count += 1
            BATCH_FLUSH_TOTAL.labels(status="success").inc()
            BATCH_FLUSH_DURATION.observe(duration)
            logger.debug(
                "Batch flush: %d models in %.3fs (total: %d, flush #%d)",
                flushed,
                duration,
                self._total_flushed,
                self._flush_count,
            )
        except AstroMLError:
            self._session.rollback()
            BATCH_FLUSH_TOTAL.labels(status="error").inc()
            logger.exception("Batch flush failed after %d models", len(self._buffer))
            raise
        finally:
            self._buffer.clear()
            BATCH_BUFFER_SIZE.set(0)

    def _merge_one(self, model: object) -> None:
        """Route one buffered model through the upsert that fits its key.

        Most models carry a real primary key and ``merge()`` resolves them
        correctly.  ``normalized_transactions`` does not: its primary key is a
        surrogate ``id`` that a freshly normalized row does not have, so
        ``merge()`` inserts on every replay and duplicates activity (#728).
        Those rows are keyed on their natural key instead.

        Imported here rather than at module scope because
        :mod:`astroml.db.repositories` imports this module.
        """
        from astroml.db.models import NormalizedTransaction
        from astroml.db.repositories import NormalizedTransactionRepository

        if isinstance(model, NormalizedTransaction):
            NormalizedTransactionRepository(self._session).upsert(model)
            return
        self._session.merge(model)

    def close(self) -> None:
        """Close the buffer, flushing remaining models if configured."""
        if self._flush_on_exit:
            self.flush()

    @property
    def pending(self) -> int:
        """Number of models currently buffered."""
        return len(self._buffer)

    @property
    def total_flushed(self) -> int:
        """Total number of models flushed so far."""
        return self._total_flushed

    @property
    def flush_count(self) -> int:
        """Number of flush operations performed."""
        return self._flush_count

    def __enter__(self) -> BatchBuffer:
        return self

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self._session.rollback()
            except AstroMLError:
                logger.exception("Rollback on exit failed")


@contextmanager
def chunked_merge(
    session: Session,
    chunk_size: int = 100,
) -> Iterator[BatchBuffer]:
    """Context manager for chunked merge + commit cycles.

    Yields a :class:`BatchBuffer` that accumulates models and flushes
    them in chunks. Any remaining models are flushed on exit.

    Args:
        session: SQLAlchemy session.
        chunk_size: Maximum models per commit.

    Yields:
        BatchBuffer instance.

    Example:
        with chunked_merge(session, chunk_size=50) as buf:
            for record in records:
                buf.add(parse_model(record))
    """
    buf = BatchBuffer(session, chunk_size=chunk_size)
    try:
        yield buf
    finally:
        buf.close()


def batch_upsert(
    session: Session,
    models: Sequence[T],
    chunk_size: int = 100,
) -> int:
    """Upsert a sequence of models in chunked batches.

    Convenience function for one-shot chunked upserts. Merges all models
    into the session and commits in chunks of ``chunk_size``.

    Args:
        session: SQLAlchemy session.
        models: Sequence of ORM models to upsert.
        chunk_size: Maximum models per commit.

    Returns:
        Total number of models upserted.

    Example:
        count = batch_upsert(session, parsed_effects, chunk_size=50)
    """
    total = 0
    with chunked_merge(session, chunk_size=chunk_size) as buf:
        buf.add_many(models)
        total = buf.total_flushed + buf.pending
    return total
