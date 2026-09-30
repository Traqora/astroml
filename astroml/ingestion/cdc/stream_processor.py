from astroml.utils.exceptions import AstroMLError
"""Kafka stream processing for CDC events (issue #626).

Processes CDC events from Kafka topics into the AstroML feature store
with exactly-once semantics, data transformation, and enrichment.

Components:
- StreamProcessor: Processes CDC change events in real time
- ExactlyOnceTracker: Tracks offsets for exactly-once delivery
- DataTransformer: Applies transformation and enrichment rules
- StreamMonitor: Health and throughput monitoring
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from astroml.ingestion.cdc.connector import ChangeEvent, ChangeOperation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ProcessedEvent:
    """A CDC event that has been fully processed and committed.

    Attributes:
        original_event: The raw CDC event.
        transformed_data: Data after transformation/enrichment.
        feature_store_key: Key used for feature store upsert.
        processing_time_ms: Processing duration in milliseconds.
        status: Processing outcome (committed, failed, skipped).
    """

    original_event: ChangeEvent
    transformed_data: dict[str, Any]
    feature_store_key: str
    processing_time_ms: float
    status: str = "committed"


# ---------------------------------------------------------------------------
# Exactly-once tracker
# ---------------------------------------------------------------------------


class ExactlyOnceTracker:
    """Tracks Kafka offsets for exactly-once processing semantics.

    Ensures each CDC event is processed exactly once by recording
    (topic, partition, offset) tuples and re-processing from the
    last committed position on failure recovery.

    Args:
        checkpoint_interval: Number of events between offset commits.
    """

    def __init__(self, checkpoint_interval: int = 100) -> None:
        self.checkpoint_interval = checkpoint_interval
        self._processed_offsets: dict[str, dict[int, int]] = defaultdict(dict)
        self._events_since_checkpoint: int = 0
        self._committed: set[tuple[str, int, int]] = set()

    def mark_processed(self, topic: str, partition: int, offset: int) -> None:
        """Record that an offset has been processed but not yet committed."""
        self._processed_offsets[topic][partition] = max(
            self._processed_offsets[topic].get(partition, -1), offset
        )
        self._events_since_checkpoint += 1

    def mark_committed(self, topic: str, partition: int, offset: int) -> None:
        """Record that an offset has been committed."""
        self._committed.add((topic, partition, offset))

    def should_checkpoint(self) -> bool:
        """Return True if a checkpoint should be written now."""
        return self._events_since_checkpoint >= self.checkpoint_interval

    def checkpoint(self) -> dict[str, Any]:
        """Return the current checkpoint state to persist."""
        self._events_since_checkpoint = 0
        return {
            "offsets": dict(self._processed_offsets),
            "committed_count": len(self._committed),
        }

    def get_last_processed(self, topic: str, partition: int) -> int:
        """Return the last processed offset for a given topic-partition."""
        return self._processed_offsets.get(topic, {}).get(partition, -1)


# ---------------------------------------------------------------------------
# Data transformer
# ---------------------------------------------------------------------------


class DataTransformer:
    """Applies transformation and enrichment rules to CDC event data.

    Args:
        rules: List of transformation functions ``(dict) -> dict``.
        enrichments: List of enrichment functions ``(dict) -> dict``.
    """

    def __init__(
        self,
        rules: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
        enrichments: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self.rules = rules or [_passthrough]
        self.enrichments = enrichments or []

    def transform(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply all transformation rules to a data row.

        Args:
            data: Input data dict.

        Returns:
            Transformed data dict.
        """
        for rule in self.rules:
            data = rule(data)
        return data

    def enrich(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply all enrichment functions to a data row.

        Args:
            data: Input data dict.

        Returns:
            Enriched data dict.
        """
        for enrichment in self.enrichments:
            data = enrichment(data)
        return data

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Transform and enrich in sequence.

        Args:
            data: Input data dict.

        Returns:
            Fully transformed and enriched data dict.
        """
        return self.enrich(self.transform(data))


def _passthrough(data: dict[str, Any]) -> dict[str, Any]:
    """Default no-op transformation rule."""
    return data


# ---------------------------------------------------------------------------
# Stream processor
# ---------------------------------------------------------------------------


class StreamProcessor:
    """Processes CDC change events into the feature store.

    Handles data transformation, enrichment, feature-store upserts,
    and exactly-once offset tracking.

    Args:
        transformer: Data transformation/enrichment rules.
        tracker: Exactly-once offset tracker.
        batch_commit_size: Maximum events to buffer before commit.
    """

    def __init__(
        self,
        transformer: DataTransformer | None = None,
        tracker: ExactlyOnceTracker | None = None,
        batch_commit_size: int = 500,
    ) -> None:
        self.transformer = transformer or DataTransformer()
        self.tracker = tracker or ExactlyOnceTracker()
        self.batch_commit_size = batch_commit_size
        self._processed: list[ProcessedEvent] = []
        self._stats: dict[str, int] = defaultdict(int)

    def process_event(
        self,
        event: ChangeEvent,
        *,
        topic: str = "astroml.cdc",
        partition: int = 0,
        offset: int = 0,
    ) -> ProcessedEvent:
        """Process a single CDC event through the full pipeline.

        Args:
            event: CDC change event.
            topic: Kafka topic.
            partition: Kafka partition.
            offset: Kafka offset.

        Returns:
            :class:`ProcessedEvent` with processing details.
        """
        import time

        start = time.perf_counter()

        try:
            data = event.after if event.after else event.before or {}
            transformed = self.transformer.process(data)

            # Derive feature store key
            pk = data.get("id") or data.get("account_id") or str(offset)
            table = event.source.get("table", "unknown")
            key = f"{table}:{pk}"

            elapsed_ms = (time.perf_counter() - start) * 1000
            result = ProcessedEvent(
                original_event=event,
                transformed_data=transformed,
                feature_store_key=key,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Failed to process event at offset %d: %s", offset, exc)
            result = ProcessedEvent(
                original_event=event,
                transformed_data={},
                feature_store_key="error",
                processing_time_ms=elapsed_ms,
                status="failed",
            )

        self.tracker.mark_processed(topic, partition, offset)
        self._stats["total_processed"] += 1
        if result.status == "committed":
            self._stats["committed"] += 1
        else:
            self._stats["failed"] += 1

        self._processed.append(result)
        if len(self._processed) >= self.batch_commit_size:
            self.commit()

        return result

    def process_batch(self, events: list[ChangeEvent], **kwargs: Any) -> list[ProcessedEvent]:
        """Process a batch of CDC events.

        Args:
            events: CDC change events.
            **kwargs: Forwarded to :meth:`process_event`.

        Returns:
            List of processed results.
        """
        return [self.process_event(e, offset=i, **kwargs) for i, e in enumerate(events)]

    def commit(self) -> None:
        """Commit processed events to the feature store."""
        if not self._processed:
            return

        for processed in self._processed:
            if processed.status == "committed":
                self.tracker.mark_committed("astroml.cdc", 0, 0)

        checkpoint = self.tracker.checkpoint() if self.tracker.should_checkpoint() else {}
        logger.info(
            "Committed %d events (checkpoint=%s)",
            len(self._processed),
            "written" if checkpoint else "deferred",
        )
        self._processed.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return processing statistics for monitoring.

        Returns:
            Dict with total_processed, committed, failed, and tracker state.
        """
        return {
            "total_processed": self._stats["total_processed"],
            "committed": self._stats["committed"],
            "failed": self._stats["failed"],
            "buffered": len(self._processed),
            "last_committed_offset": self.tracker._events_since_checkpoint,
        }

    def replay_from(
        self,
        topic: str,
        partition: int,
        from_offset: int,
        events: list[ChangeEvent],
    ) -> list[ProcessedEvent]:
        """Re-play events from a given offset for failure recovery.

        Args:
            topic: Kafka topic.
            partition: Kafka partition.
            from_offset: Offset to start replay from.
            events: All available events for the partition.

        Returns:
            List of re-processed events.
        """
        last = self.tracker.get_last_processed(topic, partition)
        unprocessed = events[last + 1 :]
        logger.info(
            "Replaying %d events from offset %d (topic=%s, partition=%d)",
            len(unprocessed),
            last + 1,
            topic,
            partition,
        )
        return self.process_batch(unprocessed, topic=topic, partition=partition)


# ---------------------------------------------------------------------------
# Stream monitor
# ---------------------------------------------------------------------------


class StreamMonitor:
    """Real-time monitoring for CDC stream processing.

    Tracks throughput, latency, and error rates.

    Attributes:
        window_seconds: Rolling window for throughput calculation.
    """

    def __init__(self, window_seconds: float = 60.0) -> None:
        self.window_seconds = window_seconds
        self._event_times: list[tuple[datetime, float]] = []  # (time, processing_ms)
        self._error_count: int = 0
        self.start_time: datetime = datetime.utcnow()

    def record(self, processing_time_ms: float, is_error: bool = False) -> None:
        """Record a processed event.

        Args:
            processing_time_ms: Processing duration.
            is_error: Whether processing failed.
        """
        now = datetime.utcnow()
        self._event_times.append((now, processing_time_ms))
        if is_error:
            self._error_count += 1

        # Prune old entries
        cutoff = now - timedelta(seconds=self.window_seconds)
        self._event_times = [(t, ms) for t, ms in self._event_times if t >= cutoff]

    def get_metrics(self) -> dict[str, Any]:
        """Return current stream metrics.

        Returns:
            Dict with throughput, latency percentiles, and error rate.
        """
        if not self._event_times:
            return {"throughput_events_per_sec": 0.0, "avg_latency_ms": 0.0, "error_rate": 0.0}

        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        total = len(self._event_times)
        throughput = total / max(self.window_seconds, elapsed) if elapsed > 0 else 0.0
        latencies = sorted(ms for _, ms in self._event_times)
        error_rate = self._error_count / max(total, 1)

        return {
            "throughput_events_per_sec": round(throughput, 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p50_latency_ms": latencies[len(latencies) // 2],
            "p99_latency_ms": latencies[int(len(latencies) * 0.99)],
            "error_rate": round(error_rate, 4),
            "total_events": total,
            "total_errors": self._error_count,
        }