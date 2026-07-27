from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, List, Literal, Optional

from .state import StateStore

logger = logging.getLogger("astroml.ingestion.service")


@dataclass
class IngestionResult:
    attempted: List[int]
    processed: List[int]
    skipped: List[int]


@dataclass(frozen=True)
class LedgerOutcome:
    """Per-ledger result yielded by :meth:`IngestionService.ingest_stream` — issue #547.

    One of these is produced per ledger as it's processed, instead of
    accumulating into range-sized lists the way :class:`IngestionResult` does.
    """

    ledger_id: int
    status: Literal["processed", "skipped"]


class IngestionService:
    def __init__(self, state_store: Optional[StateStore] = None) -> None:
        self.state = state_store or StateStore()

    def ingest(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
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
        """
        attempted: List[int] = []
        processed: List[int] = []
        skipped: List[int] = []

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

        return IngestionResult(attempted=attempted, processed=processed, skipped=skipped)

    def ingest_stream(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
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
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

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
                for offset, ledger_id in enumerate(
                    range(start_ledger, end_ledger + 1), start=1
                ):
                    if ledger_id in processed_set:
                        yield ledger_id, LedgerOutcome(
                            ledger_id=ledger_id, status="skipped"
                        )
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
                        yield ledger_id, LedgerOutcome(
                            ledger_id=ledger_id, status="processed"
                        )

                    if offset % batch_size == 0:
                        logger.info(
                            "ingest_stream progress: %d/%d ledgers (up to %d)",
                            offset, end_ledger - start_ledger + 1, ledger_id,
                        )
        finally:
            if pending_flush:
                self.state.save(state)
