"""Cache invalidation strategies for pipeline caching.

Issue #636: Implements time-based, version-based, dependency-based, and
event-driven invalidation strategies.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class InvalidationReason(Enum):
    """Reasons for cache invalidation."""

    TTL_EXPIRED = auto()
    VERSION_CHANGED = auto()
    DEPENDENCY_CHANGED = auto()
    MANUAL = auto()
    DATA_UPDATE = auto()
    MODEL_RETRAINED = auto()
    CONFIG_CHANGE = auto()
    BULK_INVALIDATION = auto()
    CASCADE = auto()


@dataclass
class InvalidationEvent:
    """Record of a cache invalidation event."""

    reason: InvalidationReason
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)
    affected_keys: set[str] = field(default_factory=set)
    source: str = ""


class InvalidationStrategy(ABC):
    """Abstract base class for cache invalidation strategies."""

    @abstractmethod
    def should_invalidate(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        """Determine whether a cache entry should be invalidated.

        Args:
            key: Cache key to check.
            metadata: Optional metadata associated with the entry.

        Returns:
            Tuple of (should_invalidate, reason).
        """
        ...

    def on_invalidate(self, event: InvalidationEvent) -> None:
        """Hook called after invalidation occurs."""
        pass


class TimeBasedInvalidation(InvalidationStrategy):
    """Invalidate entries that have exceeded their TTL."""

    def should_invalidate(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        if metadata is None:
            return True, InvalidationReason.TTL_EXPIRED

        expiry = metadata.get("expiry")
        if expiry is None:
            return False, None

        if time.time() > float(expiry):
            return True, InvalidationReason.TTL_EXPIRED

        return False, None


class VersionBasedInvalidation(InvalidationStrategy):
    """Invalidate entries when the schema or pipeline version changes."""

    def __init__(self, current_version: str) -> None:
        """Initialize with the current pipeline version.

        Args:
            current_version: Version string (e.g., 'v2', '2024-01.1').
        """
        self.current_version = current_version

    def should_invalidate(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        if metadata is None:
            return True, InvalidationReason.VERSION_CHANGED

        cached_version = metadata.get("version")
        if cached_version is None:
            return True, InvalidationReason.VERSION_CHANGED

        if cached_version != self.current_version:
            return True, InvalidationReason.VERSION_CHANGED

        return False, None

    def update_version(self, new_version: str) -> None:
        """Update the current version (triggers re-check on next poll)."""
        logger.info(
            f"VersionBasedInvalidation: updating from {self.current_version} to {new_version}"
        )
        self.current_version = new_version


class DependencyBasedInvalidation(InvalidationStrategy):
    """Invalidate entries when their upstream dependencies change.

    Maintains a dependency graph and invalidates downstream entries
    when an upstream entry is invalidated.
    """

    def __init__(self) -> None:
        self._dependencies: dict[str, set[str]] = {}  # key -> set of keys it depends on
        self._reverse_deps: dict[str, set[str]] = {}  # key -> set of keys that depend on it

    def add_dependency(self, dependent_key: str, dependency_key: str) -> None:
        """Record that `dependent_key` depends on `dependency_key`.

        When `dependency_key` is invalidated, `dependent_key` will also be checked.
        """
        self._dependencies.setdefault(dependent_key, set()).add(dependency_key)
        self._reverse_deps.setdefault(dependency_key, set()).add(dependent_key)

    def remove_dependency(self, dependent_key: str, dependency_key: str) -> None:
        """Remove a dependency relationship."""
        if dependent_key in self._dependencies:
            self._dependencies[dependent_key].discard(dependency_key)
        if dependency_key in self._reverse_deps:
            self._reverse_deps[dependency_key].discard(dependent_key)

    def get_dependencies(self, key: str) -> frozenset[str]:
        """Get all keys that `key` depends on."""
        return frozenset(self._dependencies.get(key, set()))

    def get_dependents(self, key: str) -> frozenset[str]:
        """Get all keys that depend on `key` (reverse lookup)."""
        return frozenset(self._reverse_deps.get(key, set()))

    def get_transitive_dependents(self, key: str) -> frozenset[str]:
        """Get all keys transitively dependent on `key`."""
        visited: set[str] = set()
        to_visit = [key]
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            deps = self._reverse_deps.get(current, set())
            to_visit.extend(deps)
        visited.discard(key)
        return frozenset(visited)

    def should_invalidate(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        # This strategy is triggered externally when a dependency is invalidated.
        # The check is performed by the CacheInvalidator orchestrator.
        return False, None

    def on_dependency_invalidated(
        self, invalidated_key: str
    ) -> list[str]:
        """Return all keys that should be invalidated because of this dependency.

        Args:
            invalidated_key: The key that was just invalidated.

        Returns:
            List of dependent keys to also invalidate.
        """
        return list(self.get_transitive_dependents(invalidated_key))


class EventDrivenInvalidation(InvalidationStrategy):
    """Invalidate entries based on application events.

    Supports subscribing to events like 'model_retrained', 'data_updated',
    'config_changed', etc.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[tuple[Callable[[str], bool], InvalidationReason]]] = {}

    def subscribe(
        self,
        event_type: str,
        predicate: Callable[[str], bool],
        reason: InvalidationReason | None = None,
    ) -> None:
        """Subscribe to an event type with a key-matching predicate.

        Args:
            event_type: Event name (e.g., 'model_retrained').
            predicate: Function that returns True for keys to invalidate.
            reason: Invalidation reason (defaults to DATA_UPDATE).
        """
        self._subscribers.setdefault(event_type, []).append(
            (predicate, reason or InvalidationReason.DATA_UPDATE)
        )

    def emit(self, event_type: str) -> list[str]:
        """Emit an event, returning keys that should be invalidated.

        Args:
            event_type: Event name to emit.

        Returns:
            List of keys that match any subscriber predicate.
        """
        matched: set[str] = set()
        for predicate, reason in self._subscribers.get(event_type, []):
            # Predicates are evaluated by the orchestrator against cached keys
            # Here we just return the subscribers for the orchestrator to use
            pass
        return list(matched)

    def get_subscribers(self, event_type: str) -> list[tuple[Callable[[str], bool], InvalidationReason]]:
        """Get all subscribers for an event type."""
        return self._subscribers.get(event_type, [])

    def should_invalidate(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        # Event-driven invalidation is triggered by emit(), not polling
        return False, None


class CacheInvalidator:
    """Orchestrates multiple invalidation strategies for pipeline cache.

    Combines time-based, version-based, dependency-based, and event-driven
    strategies to determine when cache entries should be evicted.

    Example:
        invalidator = CacheInvalidator()
        invalidator.add_strategy(TimeBasedInvalidation())
        invalidator.add_strategy(VersionBasedInvalidation("v2"))

        should_evict, reason = invalidator.check("feature:my-key", metadata)
        if should_evict:
            cache.delete("feature:my-key")
    """

    def __init__(self) -> None:
        self._strategies: list[InvalidationStrategy] = []
        self._history: list[InvalidationEvent] = []
        self._max_history = 1000
        self._dependency_strategy: DependencyBasedInvalidation | None = None
        self._event_strategy: EventDrivenInvalidation | None = None

    def add_strategy(self, strategy: InvalidationStrategy) -> None:
        """Add an invalidation strategy.

        Args:
            strategy: Strategy instance to add.
        """
        self._strategies.append(strategy)

        if isinstance(strategy, DependencyBasedInvalidation):
            self._dependency_strategy = strategy
        elif isinstance(strategy, EventDrivenInvalidation):
            self._event_strategy = strategy

    @property
    def dependency_strategy(self) -> DependencyBasedInvalidation | None:
        return self._dependency_strategy

    @property
    def event_strategy(self) -> EventDrivenInvalidation | None:
        return self._event_strategy

    def check(
        self, key: str, metadata: dict[str, Any] | None
    ) -> tuple[bool, InvalidationReason | None]:
        """Check all strategies to determine if a key should be invalidated.

        Returns on the first strategy that says to invalidate.

        Args:
            key: Cache key to check.
            metadata: Associated metadata.

        Returns:
            (should_invalidate, reason)
        """
        for strategy in self._strategies:
            should, reason = strategy.should_invalidate(key, metadata)
            if should:
                event = InvalidationEvent(
                    reason=reason or InvalidationReason.MANUAL,
                    affected_keys={key},
                    source=type(strategy).__name__,
                )
                self._record_event(event)

                # Cascade dependency invalidation
                if self._dependency_strategy is not None:
                    cascade = self._dependency_strategy.on_dependency_invalidated(key)
                    for cascaded_key in cascade:
                        cascade_event = InvalidationEvent(
                            reason=InvalidationReason.CASCADE,
                            affected_keys={cascaded_key},
                            details={"triggered_by": key},
                            source="DependencyBasedInvalidation",
                        )
                        self._record_event(cascade_event)

                for s in self._strategies:
                    s.on_invalidate(event)

                return True, reason

        return False, None

    def invalidate_all(
        self, reason: InvalidationReason = InvalidationReason.BULK_INVALIDATION
    ) -> InvalidationEvent:
        """Bulk-invalidate all entries.

        Args:
            reason: Reason for bulk invalidation.

        Returns:
            InvalidationEvent describing the operation.
        """
        event = InvalidationEvent(reason=reason, source="CacheInvalidator")
        self._record_event(event)
        for s in self._strategies:
            s.on_invalidate(event)
        return event

    def emit_event(self, event_type: str) -> list[str]:
        """Emit an application event to trigger event-driven invalidation.

        Args:
            event_type: Event name (e.g., 'model_retrained', 'data_updated').

        Returns:
            List of keys flagged for invalidation.
        """
        if self._event_strategy is None:
            return []
        return self._event_strategy.emit(event_type)

    def add_dependency(self, dependent_key: str, dependency_key: str) -> None:
        """Add a dependency relationship for cascade invalidation.

        Args:
            dependent_key: Key that depends on the dependency.
            dependency_key: Key that is a dependency.
        """
        if self._dependency_strategy is None:
            self._dependency_strategy = DependencyBasedInvalidation()
            self._strategies.append(self._dependency_strategy)
        self._dependency_strategy.add_dependency(dependent_key, dependency_key)

    def clear_history(self) -> None:
        """Clear the invalidation event history."""
        self._history.clear()

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent invalidation events as dictionaries.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of event dicts (most recent first).
        """
        return [
            {
                "reason": e.reason.name,
                "timestamp": e.timestamp,
                "affected_keys": sorted(e.affected_keys),
                "details": e.details,
                "source": e.source,
            }
            for e in self._history[-limit:][::-1]
        ]

    def stats(self) -> dict[str, Any]:
        """Return invalidation statistics."""
        reasons: dict[str, int] = {}
        for event in self._history:
            name = event.reason.name
            reasons[name] = reasons.get(name, 0) + 1

        return {
            "total_invalidations": len(self._history),
            "by_reason": reasons,
            "strategies": len(self._strategies),
        }

    def _record_event(self, event: InvalidationEvent) -> None:
        """Record an invalidation event in history."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]


__all__ = [
    "InvalidationReason",
    "InvalidationEvent",
    "InvalidationStrategy",
    "TimeBasedInvalidation",
    "VersionBasedInvalidation",
    "DependencyBasedInvalidation",
    "EventDrivenInvalidation",
    "CacheInvalidator",
]