from astroml.utils.exceptions import AstroMLError
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

Intended use (issue #969):
- Backfilling or incrementally ingesting a range of Stellar ledgers via
  ``ingest``/``ingest_stream``/``ingest_backfill_chunked``, and catching up
  to the network head via ``ingest_incremental``.
- ``fetch_fn``/``process_fn`` are supplied by the caller; this module owns
  ordering, idempotency (skip-if-already-processed), state persistence, and
  batching — not how a ledger is fetched or what "processing" it means.

Limitations:
- No built-in retry/backoff for a failing ``fetch_fn``/``process_fn``: an
  exception is logged and aborts the current ``ingest``/``ingest_stream``
  call (see ``ingest_stream``). Callers that need resilience to transient
  fetch/process failures must implement retries in their own callbacks.
- Idempotency relies on ``process_fn`` itself tolerating being re-invoked
  for the same ledger id — this module only prevents *re-attempting* a
  ledger already recorded as processed; it does not undo partial side
  effects from an attempt that failed midway.
- Not safe for concurrent ``ingest*`` calls sharing the same ``StateStore``.

Test coverage: ``tests/test_ingestion_service_streaming.py``,
``tests/test_backfill_chunked.py``, ``tests/ingestion/test_incremental_ingestion.py``.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Literal, Optional

from astroml.core.abstracts import IngestionResult as BaseIngestionResult
from astroml.core.abstracts import Ingestor
from astroml.utils.logging import CorrelationId, get_correlation_id
from astroml.utils.validators import validate_positive_int, validate_range

from .batch_metrics import BatchMetricsRecorder
from .state import StateStore

logger = logging.getLogger("astroml.ingestion.service")


class BackfillCheckpointManager:
    """Manages checkpoint persistence for chunked backfill operations.

    Stores the last successfully completed chunk position to enable
    resuming from that point on restart. The checkpoint file is written
    atomically to avoid corruption on crash.
    """

    def __init__(self, checkpoint_path: str | None = None):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_path: Path to the checkpoint file. If None, uses
                default path in .astroml_state directory.
        """
        if checkpoint_path is None:
            state_dir = os.path.join(os.getcwd(), ".astroml_state")
            os.makedirs(state_dir, exist_ok=True)
            checkpoint_path = os.path.join(state_dir, "backfill_checkpoint.json")
        self.checkpoint_path = checkpoint_path

    def save(self, last_ledger: int) -> None:
        """Save checkpoint position.

        Args:
            last_ledger: The last ledger successfully processed.
        """
        checkpoint_data = {
            "last_ledger": last_ledger,
            "updated_at": datetime.utcnow().isoformat(),
        }
        tmp_path = f"{self.checkpoint_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2)
        os.replace(tmp_path, self.checkpoint_path)

    def load(self) -> int | None:
        """Load the last checkpoint position.

        Returns:
            The last ledger processed, or None if no checkpoint exists.
        """
        if not os.path.exists(self.checkpoint_path):
            return None
        try:
            with open(self.checkpoint_path, encoding="utf-8") as f:
                data = json.load(f)
            return data.get("last_ledger")
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def clear(self) -> None:
        """Remove the checkpoint file."""
        if os.path.exists(self.checkpoint_path):
            os.remove(self.checkpoint_path)


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

    def __post_init__(self) -> Any:
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

    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        notifier: Optional[Callable[[str], Any]] = None,
    ) -> None:
        """Initialize the ingestion service.

        Args:
            state_store: Optional state store for tracking processed ledgers.
                        Defaults to a new StateStore instance.
            notifier: Optional ``(message) -> Any`` callback invoked when
                :meth:`ingest` fails, e.g.
                ``SlackIntegration(config).send_webhook`` (issue #986).
        """
        self.state = state_store or StateStore()
        self.notifier = notifier

    def ingest(
        self,
        start_ledger: int | None = None,
        end_ledger: int | None = None,
        fetch_fn: Callable[[int], object] | None = None,
        process_fn: Callable[[int, object], None] | None = None,
        batch_size: int = 100,
    ) -> IngestionResult:
        """Ingest ledgers incrementally and idempotently.

        The function will skip any ledger already recorded as processed. State is updated per-ledger,
        ensuring safe retries.

        For large ranges (e.g. a 1M-ledger backfill) prefer :meth:`ingest_stream`: this
        method accumulates every attempted/processed/skipped ledger id into the returned
        :class:`IngestionResult`, which is O(N) memory in the size of the range. It's kept
        as-is (rather than changed to return a smaller summary) to avoid breaking existing
        callers that rely on the full id lists.

        Returns:
            IngestionResult with timestamps and error tracking (issue #573)

        Args:
            start_ledger: Starting ledger id (inclusive). If None, resume from
                last_processed_ledger+1 or 0.
            end_ledger: Ending ledger id (inclusive). If None, only
                ``start_ledger`` is processed when provided, otherwise nothing.
            fetch_fn: Function to fetch data for a ledger id; defaults to identity payload.
            process_fn: Function to handle processing; defaults to no-op.
            batch_size: Progress-logging granularity, forwarded to
                :meth:`ingest_stream` (issue #547). Does not change the return value.
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
        except AstroMLError as e:
            errors.append(str(e))
            logger.error(f"Ingestion error: {e}")
            self._notify_failure(e, attempted, processed)

        end_time = datetime.utcnow()

        return IngestionResult(
            attempted=attempted,
            processed=processed,
            skipped=skipped,
            start_time=start_time,
            end_time=end_time,
            errors=errors,
        )

    def _notify_failure(self, error: Exception, attempted: list[int], processed: list[int]) -> None:
        """Send an ingestion-failure alert via ``self.notifier`` (issue #986).

        Notifier errors are logged and swallowed so alerting can never mask
        the original ingestion failure.
        """
        if self.notifier is None:
            return
        last = attempted[-1] if attempted else None
        message = (
            f":rotating_light: AstroML ingestion failed after ledger {last}: {error} "
            f"({len(processed)} processed before failure)"
        )
        try:
            self.notifier(message)
        except Exception:
            logger.warning("Ingestion failure notifier raised", exc_info=True)

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

        Tracing/correlation (issues #944, #950): every log line emitted while
        this generator runs — including from ``fetch_fn``/``process_fn`` if
        they log through the standard logging module — carries a
        ``request_id`` field (see :mod:`astroml.utils.logging`), so a single
        ingestion run can be correlated end-to-end in structured logs. If the
        caller already established a correlation id (e.g. an HTTP handler or
        an outer ``ingest_backfill_chunked`` chunk loop), it's inherited
        as-is; otherwise a fresh one is generated for this run and scoped to
        the lifetime of the generator via :class:`~astroml.utils.logging.CorrelationId`.

        Args:
            start_ledger: First ledger to process (inclusive). If None, resumes
                from ``last_processed_ledger + 1``; yields nothing on a cold
                state store.
            end_ledger: Last ledger to process (inclusive). If None, only
                ``start_ledger`` is processed.
            fetch_fn: Fetches a ledger's payload; defaults to an identity payload.
            process_fn: Handles a fetched ledger; defaults to a no-op.
            batch_size: Progress-logging and state-flush granularity; must be
                ``>= 1``.

        Yields:
            ``(ledger_id, LedgerOutcome)`` per ledger, with a status of
            ``"processed"`` or ``"skipped"``. An exception from ``fetch_fn`` or
            ``process_fn`` is logged and re-raised, aborting the generator.
        """
        inherited_correlation_id = get_correlation_id()
        with CorrelationId(inherited_correlation_id):
            yield from self._ingest_stream_impl(
                start_ledger=start_ledger,
                end_ledger=end_ledger,
                fetch_fn=fetch_fn,
                process_fn=process_fn,
                batch_size=batch_size,
            )

    def _ingest_stream_impl(
        self,
        start_ledger: int | None,
        end_ledger: int | None,
        fetch_fn: Callable[[int], object] | None,
        process_fn: Callable[[int, object], None] | None,
        batch_size: int,
    ) -> Iterator[tuple[int, LedgerOutcome]]:
        """Body of :meth:`ingest_stream`, run inside its correlation-id scope."""

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

        logger.info(
            "ingest_stream starting: ledgers %d..%d (request_id=%s)",
            start_ledger,
            end_ledger,
            get_correlation_id(),
        )

        pending_flush = 0
        batch_metrics = BatchMetricsRecorder()
        batch_metrics.start()
        try:
            # Active ingestion jobs gauge (issue #567). Entering the context
            # here keeps the gauge balanced even if the caller abandons the
            # generator partway through — GeneratorExit unwinds this `with`.
            with track_active_job("ingestion"):
                for offset, ledger_id in enumerate(range(start_ledger, end_ledger + 1), start=1):
                    if ledger_id in processed_set:
                        outcome = LedgerOutcome(ledger_id=ledger_id, status="skipped")
                    else:
                        try:
                            payload = fetch(ledger_id)
                            process(ledger_id, payload)
                        except Exception as exc:
                            batch_metrics.observe(
                                LedgerOutcome(ledger_id=ledger_id, status="error")
                            )
                            batch_metrics.finish()
                            logger.error("Ingestion error for ledger %d: %s", ledger_id, exc)
                            raise
                        # Also stamps ``state.last_processed_at`` — the heartbeat
                        # the ingestion staleness probe reads.
                        state.record_processed(ledger_id)
                        pending_flush += 1
                        if pending_flush >= batch_size:
                            self.state.save(state)
                            pending_flush = 0
                        outcome = LedgerOutcome(ledger_id=ledger_id, status="processed")

                    batch_metrics.observe(outcome)
                    yield ledger_id, outcome

                    if offset % batch_size == 0:
                        batch_metrics.finish()
                        batch_metrics.start()
                        logger.info(
                            "ingest_stream progress: %d/%d ledgers (up to %d)",
                            offset,
                            end_ledger - start_ledger + 1,
                            ledger_id,
                        )
        finally:
            if pending_flush:
                self.state.save(state)
            batch_metrics.finish()

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
        resume_from_checkpoint: bool = False,
        checkpoint_path: str | None = None,
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

        Checkpoint support: When ``resume_from_checkpoint=True``, the method
        saves the last successfully completed chunk position to a checkpoint
        file. On restart, it resumes from that position rather than from
        ``start_ledger``. This enables resuming large backfills after crashes
        or interruptions without reprocessing already-completed chunks.

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
            start_ledger: First ledger to process (inclusive). If
                ``resume_from_checkpoint=True`` and a checkpoint exists,
                this is overridden by the checkpoint position.
            end_ledger: Last ledger to process (inclusive).
            chunk_size: Number of ledgers per memory-bounded batch. Default 10 000.
            fetch_fn: Forwarded to :meth:`ingest_stream`.
            process_fn: Forwarded to :meth:`ingest_stream`.
            batch_size: State-flush cadence inside each chunk, forwarded to
                :meth:`ingest_stream`.
            resume_from_checkpoint: If True, resume from the last checkpoint
                instead of starting from ``start_ledger``. Defaults to False.
            checkpoint_path: Optional path to the checkpoint file. If None,
                uses ``.astroml_state/backfill_checkpoint.json``.

        Yields one summary ``dict`` per chunk, where ``errors`` counts the
        chunks that failed. A failed chunk is also reported through
        ``self.notifier`` (e.g. ``SlackIntegration(config).send_webhook``),
        matching :meth:`ingest` — see issue #993.

        Correlation (issue #957): each chunk delegates to :meth:`ingest_stream`,
        which mints a fresh correlation id whenever none is already set on entry.
        Without a shared scope around the whole chunk loop, that means every
        chunk of one backfill run got a *different* request_id in the logs,
        defeating the point of #944/#950's tracing for exactly the run most in
        need of it: a multi-million-ledger backfill spanning many chunks. This
        method now establishes one correlation id (inherited from the caller if
        already set, otherwise freshly generated) before the loop starts, so
        every chunk's logs carry the same ``request_id``.
        """
        if end_ledger < start_ledger:
            raise ValueError("end_ledger must be >= start_ledger")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")


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
                # A chunked backfill swallows the per-chunk exception and keeps
                # going, so without this the operator gets no alert at all: a
                # backfill where every chunk fails looks exactly like a
                # successful one from the notifier's point of view (issue #993).
                self._notify_failure(exc, [], [])

            yield {
                "chunk_start": current,
                "chunk_end": chunk_end,
                "processed": n_processed,
                "skipped": n_skipped,
                "errors": n_errors,
            }

            # Save checkpoint after successful chunk completion
            if checkpoint_mgr is not None and n_errors == 0:
                checkpoint_mgr.save(chunk_end)
                logger.debug("Checkpoint saved at ledger %d", chunk_end)

            # Release per-chunk temporaries and compact the heap before the
            # next chunk's fetch allocations begin.  gc.collect() is a no-op
            # when the GC would have run anyway, so the overhead is negligible.
            gc.collect()
            current = chunk_end + 1

        # Clear checkpoint on successful completion
        if checkpoint_mgr is not None:
            checkpoint_mgr.clear()
            logger.info("Backfill completed, checkpoint cleared")

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
