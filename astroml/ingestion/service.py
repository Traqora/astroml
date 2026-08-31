"""Ingestion service for processing Stellar network ledgers.

This module provides the core ingestion service for processing Stellar ledger data
with idempotency guarantees and state management.

Key components:
- IngestionService: Main service for ledger ingestion (implements Ingestor ABC)
- IngestionResult: Summary of ingestion results
- LedgerOutcome: Per-ledger processing outcome

Dependencies:
- StateStore: Persistent state management
- observability.metrics: Job tracking metrics
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from astroml.core.abstracts import IngestionResult as BaseIngestionResult
from astroml.core.abstracts import Ingestor
from astroml.utils.validators import validate_positive_int, validate_range

from .state import StateStore

logger = logging.getLogger("astroml.ingestion.service")


@dataclass
class IngestionResult(BaseIngestionResult):
    """Summary of ingestion results.

    Attributes:
        attempted: List of ledger IDs that were attempted
        processed: List of ledger IDs that were successfully processed
        skipped: List of ledger IDs that were skipped (already processed)
        start_time: When ingestion started
        end_time: When ingestion completed
        errors: List of errors encountered
    """

    attempted: List[int]
    processed: List[int]
    skipped: List[int]
    start_time: datetime
    end_time: datetime
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass(frozen=True)
class LedgerOutcome:
    """Per-ledger result yielded by :meth:`IngestionService.ingest_stream` — issue #547.

    One of these is produced per ledger as it's processed, instead of
    accumulating into range-sized lists the way :class:`IngestionResult` does.
    """

    ledger_id: int
    status: Literal["processed", "skipped"]


class IngestionService(Ingestor):
    """Service for ingesting Stellar ledger data with idempotency guarantees.

    Implements the Ingestor abstract base class for dependency injection
    and implementation swapping (issue #573).
    """

    def __init__(self, state_store: Optional[StateStore] = None) -> None:
        """Initialize the ingestion service.

        Args:
            state_store: Optional state store for tracking processed ledgers.
                        Defaults to a new StateStore instance.
        """
        self.state = state_store or StateStore()

    def ingest(
        self,
        start_ledger: int | None = None,
        end_ledger: int | None = None,
        fetch_fn: Callable[[int], object] | None = None,
        process_fn: Callable[[int, object], None] | None = None,
        batch_size: int = 100,
    ) -> IngestionResult:
        """Ingest ledgers incrementally and idempotently.

        - start_ledger: starting ledger id (inclusive). If None, resume from last_processed_ledger+1 or 0.
        - end_ledger: ending ledger id (inclusive). If None, will process only the start_ledger if provided,
                      or nothing if no bounds are provided.
        - fetch_fn: function to fetch data for a ledger id; defaults to identity payload
        - process_fn: function to handle processing; defaults to no-op
        - batch_size: progress-logging granularity, forwarded to :meth:`ingest_stream`
          (see its docstring — issue #547). Does not change this method's return value.

        The function will skip any ledger already recorded as processed. State is updated per-ledger,
        ensuring safe retries.

        For large ranges (e.g. a 1M-ledger backfill) prefer :meth:`ingest_stream`: this
        method accumulates every attempted/processed/skipped ledger id into the returned
        :class:`IngestionResult`, which is O(N) memory in the size of the range. It's kept
        as-is (rather than changed to return a smaller summary) to avoid breaking existing
        callers that rely on the full id lists.

        Returns:
            IngestionResult with timestamps and error tracking (issue #573)
        """
        start_time = datetime.utcnow()
        attempted: list[int] = []
        processed: list[int] = []
        skipped: list[int] = []
        errors: list[str] = []

        try:
            for ledger_id, outcome in self.ingest_stream(
                start_ledger=start_ledger,
                end_ledger=end_ledger,
                fetch_fn=fetch_fn,
                process_fn=process_fn,
                batch_size=batch_size,
            ):
                attempted.append(ledger_id)
                if outcome.status == "processed":
                    processed.append(ledger_id)
                else:
                    skipped.append(ledger_id)
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Ingestion error: {e}")

        end_time = datetime.utcnow()

        return IngestionResult(
            attempted=attempted,
            processed=processed,
            skipped=skipped,
            start_time=start_time,
            end_time=end_time,
            errors=errors,
        )

    @validate_positive_int("batch_size")
    @validate_range("batch_size", start=1)
    def ingest_stream(
        self,
        start_ledger: int | None = None,
        end_ledger: int | None = None,
        fetch_fn: Callable[[int], object] | None = None,
        process_fn: Callable[[int, object], None] | None = None,
        batch_size: int = 100,
    ) -> Iterator[tuple[int, LedgerOutcome]]:
        """Stream ledger ingestion results one ledger at a time — issue #547.

        Unlike :meth:`ingest`, this never accumulates the requested range into a
        list: it's a generator that yields ``(ledger_id, LedgerOutcome)`` as each
        ledger is fetched/processed, so a caller driving a large backfill (e.g.
        1M ledgers) holds only the current ledger's payload in memory, not the
        whole range's worth of ids/results.

        Backpressure: fetch → process → persist happens synchronously per
        ledger before the next ``fetch_fn`` call, so a slow ``process_fn``
        naturally throttles how fast ``fetch_fn`` runs — there's no read-ahead
        buffer to overflow.

        ``batch_size`` (default 100, must be >= 1) controls two things, both
        bounded rather than growing with the range size:

        - How often progress is logged.
        - How often processed-ledger state is flushed to the
          :class:`~astroml.ingestion.state.StateStore`. The naive per-ledger
          ``StateStore.mark_processed`` call reloads *and* re-serialises the
          entire processed-ledger set from disk on every single call — fine
          for a handful of ledgers, but quadratic (and, empirically,
          minutes-slow past a few thousand ledgers) over a large backfill.
          Flushing once per batch instead turns that into a small constant
          factor of ``1/batch_size`` while keeping the flush cadence bounded
          and configurable.

        Trade-off: on a crash mid-batch, up to ``batch_size - 1`` already
        processed ledgers may not yet be durably recorded and will be
        reprocessed on restart — the existing idempotency contract
        (``process_fn`` should tolerate being called again for the same
        ledger) already covers this. The final partial batch is always
        flushed before the generator returns or is closed early (e.g. the
        caller stops iterating partway through), via a ``finally`` block.
        """

        state = self.state.load()
        processed_set = state.processed_ledgers

        if start_ledger is None and end_ledger is None:
            # default behavior: attempt only the next ledger after last processed
            if state.last_processed_ledger is None:
                return
            start_ledger = state.last_processed_ledger + 1
            end_ledger = start_ledger

        if start_ledger is None and state.last_processed_ledger is not None:
            start_ledger = state.last_processed_ledger + 1

        if end_ledger is None and start_ledger is not None:
            end_ledger = start_ledger

        if start_ledger is None or end_ledger is None:
            return

        if end_ledger < start_ledger:
            raise ValueError("end_ledger must be >= start_ledger")

        fetch = fetch_fn or (lambda ledger_id: {"ledger": ledger_id})
        process = process_fn or (lambda ledger_id, payload: None)

        from astroml.observability.metrics import track_active_job

        pending_flush = 0
        try:
            # Active ingestion jobs gauge (issue #567). Entering the context
            # here keeps the gauge balanced even if the caller abandons the
            # generator partway through — GeneratorExit unwinds this `with`.
            with track_active_job("ingestion"):
                for offset, ledger_id in enumerate(range(start_ledger, end_ledger + 1), start=1):
                    if ledger_id in processed_set:
                        yield ledger_id, LedgerOutcome(ledger_id=ledger_id, status="skipped")
                    else:
                        payload = fetch(ledger_id)
                        process(ledger_id, payload)
                        processed_set.add(ledger_id)
                        state.last_processed_ledger = (
                            ledger_id
                            if state.last_processed_ledger is None
                            else max(state.last_processed_ledger, ledger_id)
                        )
                        pending_flush += 1
                        if pending_flush >= batch_size:
                            self.state.save(state)
                            pending_flush = 0
                        yield ledger_id, LedgerOutcome(ledger_id=ledger_id, status="processed")

                    if offset % batch_size == 0:
                        logger.info(
                            "ingest_stream progress: %d/%d ledgers (up to %d)",
                            offset,
                            end_ledger - start_ledger + 1,
                            ledger_id,
                        )
        finally:
            if pending_flush:
                self.state.save(state)

    def ingest_incremental(
        self,
        latest_ledger_fn: Callable[[], int | None],
        fetch_fn: Callable[[int], object] | None = None,
        process_fn: Callable[[int, object], None] | None = None,
        batch_size: int = 100,
        max_ledgers: int | None = None,
        start_from: int | None = None,
    ) -> IngestionResult:
        """Ingest only ledgers newer than the last one already ingested (issue #729).

        A repeated run of :meth:`ingest` over a fixed range still walks every
        ledger in that range — each already-processed id is looked up and yielded
        as ``skipped`` — so the cost of "catch me up" grows with history rather
        than with how much is actually new. This mode instead asks the state
        store where it got to, asks the network where the head is, and touches
        only what lies between.

        Args:
            latest_ledger_fn: Returns the newest ledger available upstream. May
                return ``None`` when the head is unknown (e.g. the source is
                unreachable), in which case no work is attempted.
            fetch_fn: Fetches a ledger's payload; defaults to an identity payload.
            process_fn: Handles a fetched ledger; defaults to a no-op.
            batch_size: Progress-logging and state-flush granularity, forwarded
                to :meth:`ingest_stream`.
            max_ledgers: Optional cap on how many ledgers a single run will
                process. Lets a scheduled run bound its own duration when it has
                fallen a long way behind; the remainder is picked up next run.
            start_from: Where to begin on a **cold** state store (no ledger has
                ever been processed). Defaults to the current head, i.e. "start
                watching from now" rather than backfilling all of history, which
                is what an incremental mode is for. Ignored once state exists.

        Returns:
            An :class:`IngestionResult`. When nothing new is available the result
            is empty and ``fetch_fn`` is never called — the point of the mode is
            that an up-to-date run does no work.

        Idempotency and restart-safety are unchanged: this delegates to
        :meth:`ingest`, so per-ledger state, the processed-set skip check and the
        batched flush all behave exactly as they do for an explicit range.
        """
        state = self.state.load()
        head = latest_ledger_fn()

        if head is None:
            logger.info("ingest_incremental: upstream head unknown, nothing to do")
            return self._empty_result()

        if state.last_processed_ledger is None:
            start = head if start_from is None else start_from
        else:
            start = state.last_processed_ledger + 1

        if start > head:
            logger.info(
                "ingest_incremental: already up to date at ledger %d",
                state.last_processed_ledger if state.last_processed_ledger is not None else head,
            )
            return self._empty_result()

        end = head
        if max_ledgers is not None:
            if max_ledgers < 1:
                raise ValueError("max_ledgers must be >= 1")
            end = min(head, start + max_ledgers - 1)

        logger.info(
            "ingest_incremental: fetching ledgers %d..%d (head=%d)",
            start,
            end,
            head,
        )

        return self.ingest(
            start_ledger=start,
            end_ledger=end,
            fetch_fn=fetch_fn,
            process_fn=process_fn,
            batch_size=batch_size,
        )

    @staticmethod
    def _empty_result() -> IngestionResult:
        """An IngestionResult representing a run that had nothing to do."""
        now = datetime.utcnow()
        return IngestionResult(
            attempted=[],
            processed=[],
            skipped=[],
            start_time=now,
            end_time=now,
            errors=[],
        )

    def ingest_backfill_chunked(
        self,
        start_ledger: int,
        end_ledger: int,
        chunk_size: int = 10_000,
        fetch_fn: Callable[[int], object] | None = None,
        process_fn: Callable[[int, object], None] | None = None,
        batch_size: int = 100,
    ) -> Generator[dict[str, Any], None, None]:
        """Memory-efficient backfill for very large ledger ranges — issue #766.

        For multi-million-ledger ranges :meth:`ingest` keeps every ledger id
        in three Python lists (attempted/processed/skipped) for the duration
        of the call. At 100M ledgers that alone exceeds 1 GiB of resident
        memory before any payload work begins.

        This method partitions the range into ``chunk_size`` sub-ranges and
        processes each independently via :meth:`ingest_stream`, discarding
        accumulated ids and explicitly running the garbage collector between
        chunks. Peak RSS is therefore proportional to ``chunk_size`` instead
        of the full range length.

        Yields one summary ``dict`` per chunk:

        .. code-block:: python

            {
                "chunk_start": int,
                "chunk_end": int,
                "processed": int,
                "skipped": int,
                "errors": int,
            }

        Args:
            start_ledger: First ledger to process (inclusive).
            end_ledger: Last ledger to process (inclusive).
            chunk_size: Number of ledgers per memory-bounded batch. Default 10 000.
            fetch_fn: Forwarded to :meth:`ingest_stream`.
            process_fn: Forwarded to :meth:`ingest_stream`.
            batch_size: State-flush cadence inside each chunk, forwarded to
                :meth:`ingest_stream`.
        """
        if end_ledger < start_ledger:
            raise ValueError("end_ledger must be >= start_ledger")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        current = start_ledger
        while current <= end_ledger:
            chunk_end = min(current + chunk_size - 1, end_ledger)
            n_processed = 0
            n_skipped = 0
            n_errors = 0

            try:
                for _ledger_id, outcome in self.ingest_stream(
                    start_ledger=current,
                    end_ledger=chunk_end,
                    fetch_fn=fetch_fn,
                    process_fn=process_fn,
                    batch_size=batch_size,
                ):
                    if outcome.status == "processed":
                        n_processed += 1
                    else:
                        n_skipped += 1
            except Exception as exc:
                logger.error(
                    "Backfill chunk %d-%d failed: %s",
                    current,
                    chunk_end,
                    exc,
                )
                n_errors += 1

            yield {
                "chunk_start": current,
                "chunk_end": chunk_end,
                "processed": n_processed,
                "skipped": n_skipped,
                "errors": n_errors,
            }

            # Release per-chunk temporaries and compact the heap before the
            # next chunk's fetch allocations begin.  gc.collect() is a no-op
            # when the GC would have run anyway, so the overhead is negligible.
            gc.collect()
            current = chunk_end + 1

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the ingestor (issue #573).

        Returns:
            Dictionary with status information including last processed ledger,
            processed ledger count, and state store status.
        """
        state = self.state.load()
        return {
            "last_processed_ledger": state.last_processed_ledger,
            "processed_ledger_count": len(state.processed_ledgers),
            "state_store_path": (
                str(self.state.state_file) if hasattr(self.state, "state_file") else "memory"
            ),
        }
