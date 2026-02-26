from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

from .state import StateStore


@dataclass
class IngestionResult:
    attempted: List[int]
    processed: List[int]
    skipped: List[int]


class IngestionService:
    def __init__(self, state_store: Optional[StateStore] = None) -> None:
        self.state = state_store or StateStore()

    def ingest(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
    ) -> IngestionResult:
        """Ingest ledgers incrementally and idempotently.

        - start_ledger: starting ledger id (inclusive). If None, resume from last_processed_ledger+1 or 0.
        - end_ledger: ending ledger id (inclusive). If None, will process only the start_ledger if provided,
                      or nothing if no bounds are provided.
        - fetch_fn: function to fetch data for a ledger id; defaults to identity payload
        - process_fn: function to handle processing; defaults to no-op

        The function will skip any ledger already recorded as processed. State is updated per-ledger,
        ensuring safe retries.
        """
        state = self.state.load()
        processed_set = set(state.processed_ledgers)

        if start_ledger is None and end_ledger is None:
            # default behavior: attempt only the next ledger after last processed
            if state.last_processed_ledger is None:
                return IngestionResult(attempted=[], processed=[], skipped=[])
            start_ledger = state.last_processed_ledger + 1
            end_ledger = start_ledger

        if start_ledger is None and state.last_processed_ledger is not None:
            start_ledger = state.last_processed_ledger + 1

        if end_ledger is None and start_ledger is not None:
            end_ledger = start_ledger

        if start_ledger is None or end_ledger is None:
            return IngestionResult(attempted=[], processed=[], skipped=[])

        if end_ledger < start_ledger:
            raise ValueError("end_ledger must be >= start_ledger")

        fetch = fetch_fn or (lambda ledger_id: {"ledger": ledger_id})
        process = process_fn or (lambda ledger_id, payload: None)

        attempted: List[int] = []
        processed: List[int] = []
        skipped: List[int] = []

        for ledger_id in range(start_ledger, end_ledger + 1):
            attempted.append(ledger_id)
            if ledger_id in processed_set:
                skipped.append(ledger_id)
                continue

            payload = fetch(ledger_id)
            process(ledger_id, payload)
            self.state.mark_processed(ledger_id)
            processed_set.add(ledger_id)
            processed.append(ledger_id)

        return IngestionResult(attempted=attempted, processed=processed, skipped=skipped)
