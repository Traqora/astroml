"""Per-batch ingestion progress/throughput metrics recorder (Issue #727)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from astroml.ingestion.metrics import (
    INGESTION_BATCH_DURATION_SECONDS,
    INGESTION_BATCH_LEDGERS,
    INGESTION_BATCH_THROUGHPUT,
)


@dataclass
class BatchCounters:
    """Mutable counters for a single batch window."""

    processed: int = 0
    skipped: int = 0
    errors: int = 0

    def observe(self, outcome: Any) -> None:
        """Record one ledger outcome.

        Args:
            outcome: An object with a ``status`` attribute of either
                ``"processed"``, ``"skipped"``, or ``"error"``.
        """
        status = getattr(outcome, "status", "unknown")
        if status == "processed":
            self.processed += 1
        elif status == "skipped":
            self.skipped += 1
        elif status == "error":
            self.errors += 1


class BatchMetricsRecorder:
    """Records per-batch ingestion metrics.

    Call :meth:`start` at the beginning of a batch, :meth:`observe` for each
    ledger outcome, and :meth:`finish` when the batch ends to publish Prometheus
    metrics.
    """

    def __init__(self) -> None:
        self._counters = BatchCounters()
        self._start_time: float = 0.0

    def start(self) -> None:
        """Reset counters and start the batch timer."""
        self._counters = BatchCounters()
        self._start_time = time.perf_counter()

    def observe(self, outcome: Any) -> None:
        """Record one ledger outcome in the current batch."""
        self._counters.observe(outcome)

    def finish(self) -> None:
        """Publish metrics for the current batch."""
        elapsed = time.perf_counter() - self._start_time
        INGESTION_BATCH_DURATION_SECONDS.observe(elapsed)
        INGESTION_BATCH_LEDGERS.labels(status="processed").inc(self._counters.processed)
        INGESTION_BATCH_LEDGERS.labels(status="skipped").inc(self._counters.skipped)
        INGESTION_BATCH_LEDGERS.labels(status="error").inc(self._counters.errors)
        total = self._counters.processed + self._counters.skipped + self._counters.errors
        INGESTION_BATCH_THROUGHPUT.set(total / elapsed if elapsed > 0 else 0.0)
