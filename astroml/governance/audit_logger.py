from astroml.utils.exceptions import AstroMLError
"""Audit logging for model operations.

Issue #637 Step 1: Implements comprehensive audit logging for all model operations,
including training, deployment, inference, and lifecycle management.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable model events."""

    # Training events
    TRAINING_STARTED = auto()
    TRAINING_COMPLETED = auto()
    TRAINING_FAILED = auto()

    # Deployment events
    DEPLOYMENT_REQUESTED = auto()
    DEPLOYMENT_APPROVED = auto()
    DEPLOYMENT_REJECTED = auto()
    DEPLOYMENT_STARTED = auto()
    DEPLOYMENT_COMPLETED = auto()
    DEPLOYMENT_ROLLED_BACK = auto()

    # Inference events
    INFERENCE_REQUEST = auto()
    INFERENCE_COMPLETED = auto()
    INFERENCE_ERROR = auto()

    # Model lifecycle
    MODEL_REGISTERED = auto()
    MODEL_VERSIONED = auto()
    MODEL_RETIRED = auto()
    MODEL_ARCHIVED = auto()

    # Governance
    APPROVAL_REQUESTED = auto()
    APPROVAL_GRANTED = auto()
    APPROVAL_DENIED = auto()
    RISK_ASSESSMENT_COMPLETED = auto()
    COMPLIANCE_CHECK_COMPLETED = auto()

    # System
    CONFIGURATION_CHANGED = auto()
    ACCESS_GRANTED = auto()
    ACCESS_REVOKED = auto()
    SECURITY_INCIDENT = auto()


@dataclass(frozen=True)
class AuditEvent:
    """An immutable audit event record.

    Attributes:
        event_id: Unique event identifier.
        event_type: Type of the event.
        timestamp: UTC timestamp.
        actor: Who triggered the event (user ID, service name, etc.).
        model_id: Model identifier (name + version).
        details: Arbitrary event-specific metadata.
        outcome: "success", "failure", or "pending".
        trace_id: Optional distributed tracing ID for correlation.
    """

    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_type: AuditEventType = AuditEventType.MODEL_REGISTERED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    actor: str = "system"
    model_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    trace_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.name,
            "timestamp": self.timestamp.isoformat(),
            "actor": self.actor,
            "model_id": self.model_id,
            "details": self.details,
            "outcome": self.outcome,
            "trace_id": self.trace_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditEvent:
        """Deserialize from dictionary."""
        return cls(
            event_id=data.get("event_id", uuid.uuid4().hex),
            event_type=AuditEventType[data["event_type"]],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if "timestamp" in data
                else datetime.now(timezone.utc)
            ),
            actor=data.get("actor", "system"),
            model_id=data.get("model_id", ""),
            details=data.get("details", {}),
            outcome=data.get("outcome", "success"),
            trace_id=data.get("trace_id"),
        )


class AuditStore(ABC):
    """Abstract audit event store."""

    @abstractmethod
    def write(self, event: AuditEvent) -> None:
        """Persist an audit event."""
        ...

    @abstractmethod
    def query(
        self,
        model_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with filters."""
        ...

    @abstractmethod
    def get_event(self, event_id: str) -> AuditEvent | None:
        """Retrieve a specific audit event."""
        ...


class InMemoryAuditStore(AuditStore):
    """In-memory audit event store for development/testing."""

    def __init__(self, max_events: int = 100_000) -> None:
        self._events: list[AuditEvent] = []
        self._index: dict[str, AuditEvent] = {}
        self._max_events = max_events
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._index[event.event_id] = event
            # Evict oldest if over limit
            while len(self._events) > self._max_events:
                removed = self._events.pop(0)
                self._index.pop(removed.event_id, None)

    def query(
        self,
        model_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        results: list[AuditEvent] = []
        with self._lock:
            for event in reversed(self._events):
                if len(results) >= limit:
                    break
                if model_id and event.model_id != model_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if actor and event.actor != actor:
                    continue
                results.append(event)
        return results

    def get_event(self, event_id: str) -> AuditEvent | None:
        return self._index.get(event_id)


class FileAuditStore(AuditStore):
    """File-based audit event store.

    Writes events as newline-delimited JSON (NDJSON) for append-only durability.
    """

    def __init__(self, log_dir: str | Path = "audit_logs", max_file_size_mb: int = 100) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self._current_file: Path | None = None
        self._lock = threading.Lock()

    def _get_current_file(self) -> Path:
        if self._current_file is None:
            self._rotate()
        assert self._current_file is not None
        return self._current_file

    def _rotate(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_dir / f"audit_{timestamp}.ndjson"

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            fpath = self._get_current_file()
            if fpath.exists() and fpath.stat().st_size > self.max_file_size:
                self._rotate()
                fpath = self._get_current_file()

            with open(fpath, "a") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")

    def query(
        self,
        model_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        results: list[AuditEvent] = []
        # Read from most recent file backwards
        files = sorted(self.log_dir.glob("audit_*.ndjson"), reverse=True)
        for fpath in files:
            if len(results) >= limit:
                break
            try:
                for line in reversed(list(open(fpath))):
                    if len(results) >= limit:
                        break
                    try:
                        evt = AuditEvent.from_dict(json.loads(line))
                        if model_id and evt.model_id != model_id:
                            continue
                        if event_type and evt.event_type != event_type:
                            continue
                        if start_time and evt.timestamp < start_time:
                            continue
                        if end_time and evt.timestamp > end_time:
                            continue
                        if actor and evt.actor != actor:
                            continue
                        results.append(evt)
                    except (json.JSONDecodeError, KeyError):
                        pass
            except OSError as e:
                logger.warning(f"Error reading audit file {fpath}: {e}")
        return results

    def get_event(self, event_id: str) -> AuditEvent | None:
        for fpath in sorted(self.log_dir.glob("audit_*.ndjson"), reverse=True):
            try:
                for line in open(fpath):
                    try:
                        data = json.loads(line)
                        if data.get("event_id") == event_id:
                            return AuditEvent.from_dict(data)
                    except (json.JSONDecodeError, KeyError):
                        pass
            except OSError:
                pass
        return None


class ModelAuditLogger:
    """Primary audit logging service for model governance.

    Provides a structured logging interface for all model operations
    with support for multiple storage backends.

    Example:
        audit = ModelAuditLogger(store=FileAuditStore())

        @audit.log_event(AuditEventType.DEPLOYMENT_STARTED)
        def deploy_model(model_id: str) -> bool:
            ...  # deployment logic
            return True
    """

    def __init__(self, store: AuditStore | None = None) -> None:
        """Initialize audit logger.

        Args:
            store: Audit event store. Uses InMemoryAuditStore if None.
        """
        self.store = store or InMemoryAuditStore()

    def log(
        self,
        event_type: AuditEventType,
        actor: str = "system",
        model_id: str = "",
        details: dict[str, Any] | None = None,
        outcome: str = "success",
        trace_id: str | None = None,
    ) -> AuditEvent:
        """Log an audit event.

        Args:
            event_type: Type of event.
            actor: Entity that triggered the event.
            model_id: Model identifier.
            details: Event-specific metadata.
            outcome: "success", "failure", or "pending".
            trace_id: Optional distributed tracing ID.

        Returns:
            The created AuditEvent.
        """
        event = AuditEvent(
            event_type=event_type,
            actor=actor,
            model_id=model_id,
            details=details or {},
            outcome=outcome,
            trace_id=trace_id,
        )

        self.store.write(event)
        logger.debug(f"Audit: {event_type.name} | {actor} | {model_id} | {outcome}")

        return event

    def log_event(
        self,
        event_type: AuditEventType,
        capture_args: bool = True,
        capture_result: bool = False,
    ):
        """Decorator to automatically log audit events around function calls.

        Example:
            @audit.log_event(AuditEventType.TRAINING_COMPLETED)
            def train_model(model_id: str, data_path: str) -> dict:
                return {"epochs": 10, "loss": 0.05}
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start = time.monotonic()
                details: dict[str, Any] = {}

                if capture_args:
                    details["args"] = str(args)[:500] if args else ""
                    details["kwargs"] = {k: str(v)[:200] for k, v in kwargs.items()}

                try:
                    result = func(*args, **kwargs)
                    elapsed = time.monotonic() - start
                    details["duration_ms"] = round(elapsed * 1000, 2)

                    if capture_result:
                        details["result"] = str(result)[:500]

                    self.log(
                        event_type=event_type,
                        actor=kwargs.get("actor", "system"),
                        model_id=kwargs.get("model_id", ""),
                        details=details,
                        outcome="success",
                    )
                    return result
                except AstroMLError as e:
                    elapsed = time.monotonic() - start
                    details["duration_ms"] = round(elapsed * 1000, 2)
                    details["error"] = str(e)
                    details["error_type"] = type(e).__name__

                    self.log(
                        event_type=event_type,
                        actor=kwargs.get("actor", "system"),
                        model_id=kwargs.get("model_id", ""),
                        details=details,
                        outcome="failure",
                    )
                    raise

            return wrapper

        return decorator

    def query(
        self,
        model_id: str | None = None,
        event_type: AuditEventType | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        actor: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query audit events with filters.

        Args:
            model_id: Filter by model.
            event_type: Filter by event type.
            start_time: Filter events after this time.
            end_time: Filter events before this time.
            actor: Filter by actor.
            limit: Maximum events to return.

        Returns:
            List of matching AuditEvent objects, most recent first.
        """
        return self.store.query(
            model_id=model_id,
            event_type=event_type,
            start_time=start_time,
            end_time=end_time,
            actor=actor,
            limit=limit,
        )

    def get_model_audit_trail(self, model_id: str) -> list[AuditEvent]:
        """Get the complete audit trail for a specific model.

        Args:
            model_id: Model identifier.

        Returns:
            All audit events for this model, chronological order.
        """
        events = self.store.query(model_id=model_id, limit=10_000)
        return sorted(events, key=lambda e: e.timestamp)

    def export(self, filepath: str | Path, model_id: str | None = None) -> int:
        """Export audit events to a JSON file.

        Args:
            filepath: Output file path.
            model_id: Optional model filter.

        Returns:
            Number of events exported.
        """
        events = self.query(model_id=model_id, limit=1_000_000)
        with open(filepath, "w") as f:
            json.dump([e.to_dict() for e in events], f, indent=2, default=str)
        return len(events)


__all__ = [
    "AuditEventType",
    "AuditEvent",
    "AuditStore",
    "InMemoryAuditStore",
    "FileAuditStore",
    "ModelAuditLogger",
]
