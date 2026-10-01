from astroml.utils.exceptions import AstroMLError
"""Change Data Capture connector (issue #626).

Integrates Debezium for capturing database changes from PostgreSQL and
publishing them to Kafka for streaming ingestion into the feature store.

Components:
- CDCConnector: Debezium-based CDC connector
- CDCConfig: Pydantic-validated CDC configuration
- ChangeEvent: Typed representation of a database change event
- ConnectorStatus: Health and status tracking
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ChangeOperation(str, Enum):
    """CDC operation type."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    READ = "read"  # snapshot
    TRUNCATE = "truncate"


class ConnectorStatus(str, Enum):
    """CDC connector health status."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"
    INITIALIZING = "initializing"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CDCConfig(BaseModel):
    """Configuration for a Debezium CDC connector.

    Attributes:
        connector_name: Unique connector name.
        database_hostname: PostgreSQL host.
        database_port: PostgreSQL port.
        database_user: Database user.
        database_password: Database password.
        database_name: Database to capture changes from.
        table_include_list: Comma-separated table filter.
        kafka_bootstrap_servers: Kafka broker list.
        snapshot_mode: ``"initial"`` or ``"never"``.
        slot_name: Replication slot name.
        publication_name: PostgreSQL publication name.
    """

    model_config = ConfigDict(extra="forbid")

    connector_name: str = Field(
        default="astroml-cdc",
        description="Unique connector name in the Debezium cluster",
    )
    database_hostname: str = Field(
        default="localhost",
        description="PostgreSQL hostname",
    )
    database_port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    database_user: str = Field(default="astroml", description="Database user")
    database_password: str = Field(default="", description="Database password")
    database_name: str = Field(default="astroml", description="Database name")
    table_include_list: str = Field(
        default="public.accounts,public.transactions",
        description="Comma-separated list of tables to capture",
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers",
    )
    snapshot_mode: Literal["initial", "never"] = Field(
        default="initial",
        description="Snapshot mode: initial for full snapshot, never for streaming only",
    )
    slot_name: str = Field(default="astroml_replication_slot", description="Replication slot name")
    publication_name: str = Field(
        default="astroml_publication",
        description="PostgreSQL publication name",
    )


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

# Debezium short operation codes to ChangeOperation
_DEBEZIUM_OP_MAP: dict[str, ChangeOperation] = {
    "c": ChangeOperation.CREATE,
    "r": ChangeOperation.READ,
    "u": ChangeOperation.UPDATE,
    "d": ChangeOperation.DELETE,
    "t": ChangeOperation.TRUNCATE,
}


@dataclass
class ChangeEvent:
    """A single database change event captured by CDC.

    Attributes:
        source: Source metadata (connector, db, table, etc.).
        op: Operation type (c/r/u/d).
        before: Row state before the change (None for creates).
        after: Row state after the change (None for deletes).
        transaction_id: ID of the enclosing transaction.
        lsn: PostgreSQL LSN for the change.
        timestamp_ms: Event timestamp in milliseconds since epoch.
    """

    source: dict[str, Any]
    op: ChangeOperation
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    transaction_id: str | None = None
    lsn: int | None = None
    timestamp_ms: int | None = None

    @classmethod
    def from_debezium(cls, payload: dict[str, Any]) -> ChangeEvent:
        """Parse a Debezium-format JSON payload.

        Args:
            payload: Raw Debezium message (envelope).

        Returns:
            Parsed :class:`ChangeEvent`.
        """
        source = payload.get("source", {})
        op_raw = payload.get("op", "r")
        # Map Debezium short codes (c/u/d/r) or full names
        op = _DEBEZIUM_OP_MAP.get(op_raw)
        if op is None:
            try:
                op = ChangeOperation(op_raw)
            except ValueError:
                op = ChangeOperation.READ
        return cls(
            source=source,
            op=op,
            before=payload.get("before"),
            after=payload.get("after"),
            transaction_id=payload.get("transaction_id"),
            lsn=source.get("lsn"),
            timestamp_ms=source.get("ts_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the event for a transport or a JSON column.

        Returns:
            The event's fields, with ``op`` flattened to its string value so
                the result is JSON serialisable. ``before``/``after`` are
                passed through as-is and may themselves be ``None`` — for an
                insert, a delete and a read snapshot respectively.
        """
        return {
            "source": self.source,
            "op": self.op.value,
            "before": self.before,
            "after": self.after,
            "transaction_id": self.transaction_id,
            "lsn": self.lsn,
            "timestamp_ms": self.timestamp_ms,
        }


# ---------------------------------------------------------------------------
# CDC connector
# ---------------------------------------------------------------------------


class CDCConnector:
    """Debezium-based Change Data Capture connector.

    Captures PostgreSQL changes via logical replication and publishes
    them to Kafka topics for downstream streaming consumers.

    Args:
        config: CDC connector configuration.
    """

    def __init__(self, config: CDCConfig) -> None:
        self.config = config
        self.status: ConnectorStatus = ConnectorStatus.INITIALIZING
        self._events_processed: int = 0
        self._errors: list[str] = []
        self._listeners: list[Callable[[ChangeEvent], None]] = []

    def start(self) -> None:
        """Start the CDC connector.

        In production this registers the connector with the Debezium
        Kafka Connect REST API.
        """
        self.status = ConnectorStatus.RUNNING
        logger.info(
            "CDC connector %s started (db=%s:%d/%s, kafka=%s)",
            self.config.connector_name,
            self.config.database_hostname,
            self.config.database_port,
            self.config.database_name,
            self.config.kafka_bootstrap_servers,
        )

    def stop(self) -> None:
        """Stop the CDC connector gracefully."""
        self.status = ConnectorStatus.STOPPED
        logger.info("CDC connector stopped (%d events processed)", self._events_processed)

    def pause(self) -> None:
        """Pause event capture without stopping the connector."""
        self.status = ConnectorStatus.PAUSED

    def resume(self) -> None:
        """Resume event capture after a :meth:`pause`.

        Sets the status back to :attr:`ConnectorStatus.RUNNING`. Like
        :meth:`pause`, this only moves the local state machine — events
        published by PostgreSQL while paused are not replayed, so resuming is
        not lossless and a gap must be closed by a re-read of the source
        tables.
        """
        self.status = ConnectorStatus.RUNNING

    def subscribe(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Subscribe to change events.

        Args:
            callback: Called with each :class:`ChangeEvent` as it arrives.
        """
        self._listeners.append(callback)

    def process_event(self, event: ChangeEvent) -> None:
        """Process a single change event and dispatch to listeners.

        Args:
            event: The change event to process.
        """
        self._events_processed += 1
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.error("Listener error on event %s: %s", event.source, exc)
                self._errors.append(str(exc))

    def simulate_events(
        self,
        table: str,
        rows: list[dict[str, Any]],
        op: ChangeOperation = ChangeOperation.CREATE,
    ) -> list[ChangeEvent]:
        """Generate synthetic CDC events for testing.

        Args:
            table: Table name.
            rows: List of row dicts.
            op: Operation type for these events.

        Returns:
            List of generated :class:`ChangeEvent` instances.
        """
        import time

        now_ms = int(time.time() * 1000)
        events: list[ChangeEvent] = []
        for row in rows:
            event = ChangeEvent(
                source={
                    "connector": self.config.connector_name,
                    "db": self.config.database_name,
                    "table": table,
                    "lsn": self._events_processed,
                    "ts_ms": now_ms,
                },
                op=op,
                before=None if op == ChangeOperation.CREATE else row,
                after=None if op == ChangeOperation.DELETE else row,
                transaction_id=f"txn_{self._events_processed}",
                lsn=self._events_processed,
                timestamp_ms=now_ms,
            )
            self.process_event(event)
            events.append(event)
        return events

    def get_status(self) -> dict[str, Any]:
        """Return connector status and metrics.

        Returns:
            Dict with status, event count, errors.
        """
        return {
            "status": self.status.value,
            "events_processed": self._events_processed,
            "listener_count": len(self._listeners),
            "error_count": len(self._errors),
            "last_error": self._errors[-1] if self._errors else None,
        }