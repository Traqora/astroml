"""Cost tracking for ML operations (compute, API, storage).

Resolves part of #647.

The tracker is deliberately provider-agnostic and dependency-free: callers
record :class:`CostRecord` entries (directly, or via the ``record_compute`` /
``record_api`` / ``record_storage`` helpers) and the tracker maintains rolling
totals, per-dimension breakdowns and a simple linear forecast.

Typical usage::

    tracker = CostTracker()
    with tracker.track_compute("train-gnn", ResourceType.GPU_A100, project="fraud"):
        train_model()
    print(tracker.total_cost_usd())
    print(tracker.forecast(horizon_days=30).projected_cost_usd)

Prices default to a small built-in table of public on-demand rates; override
them with :meth:`CostTracker.set_price` or by passing ``prices=`` to the
constructor.  All money values are USD.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "CostCategory",
    "CostForecast",
    "CostRecord",
    "CostTracker",
    "ResourceType",
    "utcnow",
]


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class CostCategory(str, Enum):
    """High-level bucket a cost record belongs to."""

    COMPUTE = "compute"
    API = "api"
    STORAGE = "storage"
    NETWORK = "network"
    OTHER = "other"


class ResourceType(str, Enum):
    """Compute resource types with a known hourly price."""

    CPU = "cpu"
    GPU_T4 = "gpu_t4"
    GPU_V100 = "gpu_v100"
    GPU_A100 = "gpu_a100"
    GPU_H100 = "gpu_h100"
    MEMORY_GB = "memory_gb"


#: Default on-demand USD/hour prices.  Indicative only — override per deployment.
DEFAULT_COMPUTE_PRICES_USD_PER_HOUR: dict[ResourceType, float] = {
    ResourceType.CPU: 0.0425,
    ResourceType.GPU_T4: 0.526,
    ResourceType.GPU_V100: 2.48,
    ResourceType.GPU_A100: 3.06,
    ResourceType.GPU_H100: 9.98,
    ResourceType.MEMORY_GB: 0.0057,
}

#: Default USD per 1,000 API calls for well-known external services.
DEFAULT_API_PRICES_USD_PER_1K_CALLS: dict[str, float] = {
    "horizon": 0.0,
    "stellar-rpc": 0.0,
    "coingecko": 0.5,
    "openai": 2.0,
    "anthropic": 3.0,
}

#: Default USD per GB-month of storage.
DEFAULT_STORAGE_PRICE_USD_PER_GB_MONTH: float = 0.023


@dataclass(frozen=True)
class CostRecord:
    """A single billable event.

    Attributes
    ----------
    category:
        Which :class:`CostCategory` the spend belongs to.
    resource:
        Free-form resource identifier (``"gpu_a100"``, ``"openai"``, ``"s3"``).
    quantity:
        Amount consumed, in ``unit`` units.
    unit:
        Unit ``quantity`` is expressed in (``"hours"``, ``"calls"``, ``"gb_month"``).
    cost_usd:
        Resolved cost in USD.
    project / team / model:
        Allocation dimensions; ``None`` means "unallocated".
    timestamp:
        When the spend occurred (timezone-aware UTC).
    metadata:
        Arbitrary extra context, retained verbatim in exports.
    """

    category: CostCategory
    resource: str
    quantity: float
    unit: str
    cost_usd: float
    project: str | None = None
    team: str | None = None
    model: str | None = None
    timestamp: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the record."""
        return {
            "category": self.category.value,
            "resource": self.resource,
            "quantity": self.quantity,
            "unit": self.unit,
            "cost_usd": self.cost_usd,
            "project": self.project,
            "team": self.team,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CostForecast:
    """Projection of future spend produced by :meth:`CostTracker.forecast`."""

    horizon_days: int
    daily_rate_usd: float
    projected_cost_usd: float
    observed_days: float
    basis_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the forecast."""
        return {
            "horizon_days": self.horizon_days,
            "daily_rate_usd": self.daily_rate_usd,
            "projected_cost_usd": self.projected_cost_usd,
            "observed_days": self.observed_days,
            "basis_cost_usd": self.basis_cost_usd,
        }


class CostTracker:
    """Thread-safe in-memory ledger of ML operating costs.

    Parameters
    ----------
    compute_prices:
        Override for the built-in USD/hour compute price table.
    api_prices:
        Override for the built-in USD per 1,000 calls API price table.
    storage_price_usd_per_gb_month:
        Override for the built-in storage price.
    max_records:
        Ring-buffer bound.  Once exceeded, the oldest records are dropped while
        their contribution is retained in the running totals.
    """

    def __init__(
        self,
        *,
        compute_prices: dict[ResourceType, float] | None = None,
        api_prices: dict[str, float] | None = None,
        storage_price_usd_per_gb_month: float | None = None,
        max_records: int = 100_000,
    ) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self._compute_prices: dict[ResourceType, float] = dict(
            compute_prices or DEFAULT_COMPUTE_PRICES_USD_PER_HOUR
        )
        self._api_prices: dict[str, float] = dict(api_prices or DEFAULT_API_PRICES_USD_PER_1K_CALLS)
        self._storage_price = (
            DEFAULT_STORAGE_PRICE_USD_PER_GB_MONTH
            if storage_price_usd_per_gb_month is None
            else storage_price_usd_per_gb_month
        )
        self._max_records = max_records
        self._records: list[CostRecord] = []
        self._dropped_cost_usd = 0.0
        self._lock = threading.RLock()

    # ── Pricing ──────────────────────────────────────────────────────────────

    def set_price(self, resource: ResourceType | str, price: float) -> None:
        """Set the unit price for a compute resource or an API service.

        ``ResourceType`` keys set the USD/hour compute price; string keys set
        the USD per 1,000 calls API price.
        """
        if price < 0:
            raise ValueError("price must be non-negative")
        with self._lock:
            if isinstance(resource, ResourceType):
                self._compute_prices[resource] = price
            else:
                self._api_prices[resource] = price

    def compute_price(self, resource: ResourceType) -> float:
        """Return the USD/hour price for ``resource``."""
        with self._lock:
            return self._compute_prices.get(resource, 0.0)

    def api_price(self, service: str) -> float:
        """Return the USD per 1,000 calls price for ``service``."""
        with self._lock:
            return self._api_prices.get(service, 0.0)

    # ── Recording ────────────────────────────────────────────────────────────

    def record(self, record: CostRecord) -> CostRecord:
        """Append a pre-built :class:`CostRecord` to the ledger."""
        with self._lock:
            self._records.append(record)
            if len(self._records) > self._max_records:
                evicted = self._records.pop(0)
                self._dropped_cost_usd += evicted.cost_usd
        return record

    def record_compute(
        self,
        resource: ResourceType,
        hours: float,
        *,
        count: float = 1.0,
        project: str | None = None,
        team: str | None = None,
        model: str | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record compute spend for ``count`` units of ``resource`` over ``hours``.

        ``cost_usd`` overrides the price-table calculation when the provider
        already reports an authoritative amount.
        """
        if hours < 0 or count < 0:
            raise ValueError("hours and count must be non-negative")
        quantity = hours * count
        resolved = self.compute_price(resource) * quantity if cost_usd is None else cost_usd
        return self.record(
            CostRecord(
                category=CostCategory.COMPUTE,
                resource=resource.value,
                quantity=quantity,
                unit="hours",
                cost_usd=resolved,
                project=project,
                team=team,
                model=model,
                metadata=metadata or {},
            )
        )

    def record_api(
        self,
        service: str,
        calls: int,
        *,
        project: str | None = None,
        team: str | None = None,
        model: str | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record spend for ``calls`` requests against an external ``service``."""
        if calls < 0:
            raise ValueError("calls must be non-negative")
        resolved = self.api_price(service) * (calls / 1000.0) if cost_usd is None else cost_usd
        return self.record(
            CostRecord(
                category=CostCategory.API,
                resource=service,
                quantity=float(calls),
                unit="calls",
                cost_usd=resolved,
                project=project,
                team=team,
                model=model,
                metadata=metadata or {},
            )
        )

    def record_storage(
        self,
        gb: float,
        *,
        days: float = 30.0,
        resource: str = "object-store",
        project: str | None = None,
        team: str | None = None,
        cost_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CostRecord:
        """Record storage spend for ``gb`` gigabytes held for ``days`` days."""
        if gb < 0 or days < 0:
            raise ValueError("gb and days must be non-negative")
        gb_months = gb * (days / 30.0)
        resolved = self._storage_price * gb_months if cost_usd is None else cost_usd
        return self.record(
            CostRecord(
                category=CostCategory.STORAGE,
                resource=resource,
                quantity=gb_months,
                unit="gb_month",
                cost_usd=resolved,
                project=project,
                team=team,
                metadata=metadata or {},
            )
        )

    @contextmanager
    def track_compute(
        self,
        label: str,
        resource: ResourceType,
        *,
        count: float = 1.0,
        project: str | None = None,
        team: str | None = None,
        model: str | None = None,
    ) -> Iterator[None]:
        """Context manager that records wall-clock compute time for a block.

        The record is written even when the wrapped block raises, so failed
        training runs are still billed.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            hours = max(time.monotonic() - started, 0.0) / 3600.0
            self.record_compute(
                resource,
                hours,
                count=count,
                project=project,
                team=team,
                model=model,
                metadata={"label": label},
            )

    # ── Querying ─────────────────────────────────────────────────────────────

    def records(
        self,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        category: CostCategory | None = None,
        project: str | None = None,
        team: str | None = None,
    ) -> list[CostRecord]:
        """Return records matching the supplied filters, oldest first."""
        with self._lock:
            snapshot = list(self._records)
        return [
            rec
            for rec in snapshot
            if (since is None or rec.timestamp >= since)
            and (until is None or rec.timestamp <= until)
            and (category is None or rec.category is category)
            and (project is None or rec.project == project)
            and (team is None or rec.team == team)
        ]

    def total_cost_usd(self, **filters: Any) -> float:
        """Return total spend, optionally filtered as per :meth:`records`.

        Records evicted by the ring buffer are included in the unfiltered total
        so long-running processes do not appear to lose spend.
        """
        matched = self.records(**filters)
        total = sum(rec.cost_usd for rec in matched)
        if not filters:
            total += self._dropped_cost_usd
        return total

    def breakdown(self, dimension: str, **filters: Any) -> dict[str, float]:
        """Return spend grouped by ``dimension``.

        ``dimension`` is one of ``category``, ``resource``, ``project``,
        ``team``, ``model`` or ``day``.
        """
        if dimension not in _DIMENSIONS:
            raise ValueError(
                f"unknown dimension {dimension!r}; expected one of {list(_DIMENSIONS)}"
            )
        buckets: defaultdict[str, float] = defaultdict(float)
        for rec in self.records(**filters):
            buckets[_dimension_key(rec, dimension)] += rec.cost_usd
        return dict(sorted(buckets.items(), key=lambda kv: kv[1], reverse=True))

    def allocation(self, **filters: Any) -> dict[str, dict[str, float]]:
        """Return spend allocated by project and by team in one call."""
        return {
            "by_project": self.breakdown("project", **filters),
            "by_team": self.breakdown("team", **filters),
        }

    def daily_costs(self, **filters: Any) -> dict[str, float]:
        """Return spend keyed by ISO date (``YYYY-MM-DD``), oldest first."""
        buckets: defaultdict[str, float] = defaultdict(float)
        for rec in self.records(**filters):
            buckets[rec.timestamp.date().isoformat()] += rec.cost_usd
        return dict(sorted(buckets.items()))

    def forecast(self, *, horizon_days: int = 30, lookback_days: int = 7) -> CostForecast:
        """Project spend over ``horizon_days`` from the last ``lookback_days``.

        Uses a flat daily-rate extrapolation, which is stable for the bursty,
        low-volume ledgers typical of ML workloads.  With no observations the
        forecast is zero rather than an error.
        """
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")

        window_start = utcnow() - timedelta(days=lookback_days)
        window = self.records(since=window_start)
        if not window:
            return CostForecast(horizon_days, 0.0, 0.0, 0.0, 0.0)

        basis = sum(rec.cost_usd for rec in window)
        earliest = min(rec.timestamp for rec in window)
        span_days = max((utcnow() - earliest).total_seconds() / 86_400.0, 1e-9)
        observed = min(span_days, float(lookback_days))
        # Sub-day windows would otherwise extrapolate a huge daily rate.
        daily_rate = basis / max(observed, 1.0)
        return CostForecast(
            horizon_days=horizon_days,
            daily_rate_usd=daily_rate,
            projected_cost_usd=daily_rate * horizon_days,
            observed_days=observed,
            basis_cost_usd=basis,
        )

    def summary(self, **filters: Any) -> dict[str, Any]:
        """Return a compact overview suitable for an API response or a report."""
        matched = self.records(**filters)
        return {
            "record_count": len(matched),
            "total_cost_usd": self.total_cost_usd(**filters),
            "by_category": self.breakdown("category", **filters),
            "by_resource": self.breakdown("resource", **filters),
            "by_project": self.breakdown("project", **filters),
            "by_team": self.breakdown("team", **filters),
            "daily": self.daily_costs(**filters),
        }

    # ── Persistence ──────────────────────────────────────────────────────────

    def export_json(self, path: str | Path) -> Path:
        """Write the full ledger to ``path`` as JSON and return the path."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = [rec.to_dict() for rec in self._records]
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return destination

    def extend(self, records: Iterable[CostRecord]) -> None:
        """Bulk-append records, e.g. when restoring a ledger from disk."""
        for rec in records:
            self.record(rec)

    def reset(self) -> None:
        """Drop every record and reset running totals."""
        with self._lock:
            self._records.clear()
            self._dropped_cost_usd = 0.0

    def __len__(self) -> int:
        """Return the number of retained records."""
        with self._lock:
            return len(self._records)


_DIMENSIONS: Sequence[str] = ("category", "resource", "project", "team", "model", "day")


def _dimension_key(record: CostRecord, dimension: str) -> str:
    """Return the grouping key of ``record`` along ``dimension``."""
    if dimension == "category":
        return record.category.value
    if dimension == "resource":
        return record.resource
    if dimension == "day":
        return record.timestamp.date().isoformat()
    # Only the six names in _DIMENSIONS reach here; breakdown() validates first.
    return getattr(record, dimension) or "unallocated"


#: Process-wide tracker, convenient for instrumentation that has no DI hook.
default_cost_tracker = CostTracker()
