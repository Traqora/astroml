from astroml.utils.exceptions import AstroMLError
"""Budget definition, enforcement and alerting for ML spend.

Resolves part of #647.

A :class:`Budget` scopes a spend limit to a period (daily / weekly / monthly)
and, optionally, to a project or team.  :class:`BudgetManager` evaluates
budgets against a :class:`~astroml.tracking.cost_tracker.CostTracker`, raises
threshold alerts exactly once per threshold per period, and can block further
spend once a hard limit is exhausted.

Example::

    manager = BudgetManager(tracker)
    manager.add_budget(Budget("fraud-monthly", 500.0, BudgetPeriod.MONTHLY, project="fraud"))
    manager.on_alert(lambda alert: logger.warning(alert.message))
    manager.evaluate()
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from astroml.tracking.cost_tracker import CostTracker, utcnow

logger = logging.getLogger(__name__)

__all__ = [
    "AlertSeverity",
    "Budget",
    "BudgetAlert",
    "BudgetExceededError",
    "BudgetManager",
    "BudgetPeriod",
    "BudgetStatus",
]

#: Fractions of a budget at which an alert fires, ascending.
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.5, 0.8, 1.0)


class BudgetPeriod(str, Enum):
    """Reset cadence of a budget."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class AlertSeverity(str, Enum):
    """Severity attached to a :class:`BudgetAlert`."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BudgetExceededError(RuntimeError):
    """Raised by :meth:`BudgetManager.check_spend` when a hard limit is hit."""


@dataclass(frozen=True)
class Budget:
    """A spend limit scoped to a period and, optionally, an owner.

    Attributes
    ----------
    name:
        Unique identifier of the budget.
    limit_usd:
        Maximum spend allowed within one period.
    period:
        How often the budget resets.
    project / team:
        Restrict the budget to spend allocated to this project/team.  ``None``
        matches every record.
    thresholds:
        Fractions of ``limit_usd`` at which alerts fire.
    hard_limit:
        When true, :meth:`BudgetManager.check_spend` raises once exhausted.
    """

    name: str
    limit_usd: float
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    project: str | None = None
    team: str | None = None
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    hard_limit: bool = False

    def __post_init__(self) -> None:
        """Validate the budget definition."""
        if not self.name:
            raise ValueError("budget name must not be empty")
        if self.limit_usd <= 0:
            raise ValueError("limit_usd must be positive")
        if not self.thresholds:
            raise ValueError("at least one threshold is required")
        if any(t <= 0 for t in self.thresholds):
            raise ValueError("thresholds must be positive fractions")

    def period_start(self, now: datetime | None = None) -> datetime:
        """Return the UTC start of the period containing ``now``."""
        moment = (now or utcnow()).astimezone(timezone.utc)
        midnight = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.period is BudgetPeriod.DAILY:
            return midnight
        if self.period is BudgetPeriod.WEEKLY:
            return midnight - timedelta(days=midnight.weekday())
        if self.period is BudgetPeriod.MONTHLY:
            return midnight.replace(day=1)
        quarter_first_month = 3 * ((midnight.month - 1) // 3) + 1
        return midnight.replace(month=quarter_first_month, day=1)

    def period_key(self, now: datetime | None = None) -> str:
        """Return a stable identifier for the current period."""
        return f"{self.name}:{self.period.value}:{self.period_start(now).isoformat()}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the budget."""
        return {
            "name": self.name,
            "limit_usd": self.limit_usd,
            "period": self.period.value,
            "project": self.project,
            "team": self.team,
            "thresholds": list(self.thresholds),
            "hard_limit": self.hard_limit,
        }


@dataclass(frozen=True)
class BudgetStatus:
    """Point-in-time evaluation of a single budget."""

    budget: Budget
    spent_usd: float
    limit_usd: float
    period_start: datetime
    utilization: float
    remaining_usd: float
    exceeded: bool
    projected_period_cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the status."""
        return {
            "budget": self.budget.to_dict(),
            "spent_usd": self.spent_usd,
            "limit_usd": self.limit_usd,
            "period_start": self.period_start.isoformat(),
            "utilization": self.utilization,
            "remaining_usd": self.remaining_usd,
            "exceeded": self.exceeded,
            "projected_period_cost_usd": self.projected_period_cost_usd,
        }


@dataclass(frozen=True)
class BudgetAlert:
    """Emitted the first time a budget crosses one of its thresholds."""

    budget_name: str
    threshold: float
    severity: AlertSeverity
    spent_usd: float
    limit_usd: float
    utilization: float
    message: str
    timestamp: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the alert."""
        return {
            "budget_name": self.budget_name,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "spent_usd": self.spent_usd,
            "limit_usd": self.limit_usd,
            "utilization": self.utilization,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class BudgetManager:
    """Evaluates budgets against a cost tracker and dispatches alerts."""

    def __init__(self, tracker: CostTracker, *, max_alert_history: int = 1_000) -> None:
        if max_alert_history <= 0:
            raise ValueError("max_alert_history must be positive")
        self._tracker = tracker
        self._budgets: dict[str, Budget] = {}
        self._fired: set[tuple[str, float]] = set()
        self._alerts: list[BudgetAlert] = []
        self._subscribers: list[Callable[[BudgetAlert], None]] = []
        self._max_alert_history = max_alert_history
        self._lock = threading.RLock()

    # ── Budget registry ──────────────────────────────────────────────────────

    def add_budget(self, budget: Budget) -> Budget:
        """Register ``budget``, replacing any existing budget with the same name."""
        with self._lock:
            self._budgets[budget.name] = budget
        return budget

    def remove_budget(self, name: str) -> bool:
        """Remove the named budget; return whether it existed."""
        with self._lock:
            return self._budgets.pop(name, None) is not None

    def get_budget(self, name: str) -> Budget | None:
        """Return the named budget, or ``None``."""
        with self._lock:
            return self._budgets.get(name)

    def budgets(self) -> list[Budget]:
        """Return every registered budget."""
        with self._lock:
            return list(self._budgets.values())

    # ── Alert subscribers ────────────────────────────────────────────────────

    def on_alert(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register ``callback`` to receive every future alert."""
        with self._lock:
            self._subscribers.append(callback)

    def alerts(self) -> list[BudgetAlert]:
        """Return the retained alert history, oldest first."""
        with self._lock:
            return list(self._alerts)

    # ── Evaluation ───────────────────────────────────────────────────────────

    def status(self, name: str, *, now: datetime | None = None) -> BudgetStatus:
        """Return the current :class:`BudgetStatus` for the named budget."""
        budget = self.get_budget(name)
        if budget is None:
            raise KeyError(f"unknown budget {name!r}")
        return self._status_for(budget, now=now)

    def statuses(self, *, now: datetime | None = None) -> list[BudgetStatus]:
        """Return the status of every registered budget."""
        return [self._status_for(budget, now=now) for budget in self.budgets()]

    def evaluate(self, *, now: datetime | None = None) -> list[BudgetAlert]:
        """Evaluate all budgets and dispatch newly crossed threshold alerts.

        Each (period, threshold) pair fires at most once, so calling this on a
        schedule will not spam subscribers.
        """
        fired: list[BudgetAlert] = []
        for status in self.statuses(now=now):
            fired.extend(self._fire_thresholds(status, now=now))
        return fired

    def check_spend(
        self,
        additional_usd: float,
        *,
        project: str | None = None,
        team: str | None = None,
        now: datetime | None = None,
    ) -> list[BudgetStatus]:
        """Validate that ``additional_usd`` of spend is affordable.

        Returns the statuses of budgets covering the spend.  Raises
        :class:`BudgetExceededError` if any matching hard-limit budget would be
        pushed over its limit.
        """
        if additional_usd < 0:
            raise ValueError("additional_usd must be non-negative")
        matching: list[BudgetStatus] = []
        for budget in self.budgets():
            if budget.project is not None and budget.project != project:
                continue
            if budget.team is not None and budget.team != team:
                continue
            status = self._status_for(budget, now=now)
            matching.append(status)
            if budget.hard_limit and status.spent_usd + additional_usd > budget.limit_usd:
                raise BudgetExceededError(
                    f"budget {budget.name!r} would be exceeded: "
                    f"{status.spent_usd + additional_usd:.2f} USD > "
                    f"{budget.limit_usd:.2f} USD limit"
                )
        return matching

    def report(self, *, now: datetime | None = None) -> dict[str, Any]:
        """Return a dashboard-friendly summary of every budget."""
        statuses = self.statuses(now=now)
        return {
            "generated_at": (now or utcnow()).isoformat(),
            "budget_count": len(statuses),
            "exceeded_count": sum(1 for s in statuses if s.exceeded),
            "total_limit_usd": sum(s.limit_usd for s in statuses),
            "total_spent_usd": sum(s.spent_usd for s in statuses),
            "budgets": [s.to_dict() for s in statuses],
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _status_for(self, budget: Budget, *, now: datetime | None = None) -> BudgetStatus:
        """Compute the status of ``budget`` for the period containing ``now``."""
        start = budget.period_start(now)
        spent = self._tracker.total_cost_usd(since=start, project=budget.project, team=budget.team)
        elapsed = max(((now or utcnow()) - start).total_seconds(), 0.0)
        period_seconds = max((_period_end(budget, start) - start).total_seconds(), 1.0)
        fraction_elapsed = min(max(elapsed / period_seconds, 1e-6), 1.0)
        return BudgetStatus(
            budget=budget,
            spent_usd=spent,
            limit_usd=budget.limit_usd,
            period_start=start,
            utilization=spent / budget.limit_usd,
            remaining_usd=budget.limit_usd - spent,
            exceeded=spent > budget.limit_usd,
            projected_period_cost_usd=spent / fraction_elapsed,
        )

    def _fire_thresholds(
        self, status: BudgetStatus, *, now: datetime | None = None
    ) -> list[BudgetAlert]:
        """Emit alerts for thresholds newly crossed by ``status``."""
        budget = status.budget
        period_key = budget.period_key(now)
        emitted: list[BudgetAlert] = []
        for threshold in sorted(budget.thresholds):
            if status.utilization < threshold:
                continue
            key = (period_key, threshold)
            with self._lock:
                if key in self._fired:
                    continue
                self._fired.add(key)
            alert = BudgetAlert(
                budget_name=budget.name,
                threshold=threshold,
                severity=_severity_for(threshold),
                spent_usd=status.spent_usd,
                limit_usd=status.limit_usd,
                utilization=status.utilization,
                message=(
                    f"Budget {budget.name!r} reached {status.utilization:.0%} of its "
                    f"{budget.limit_usd:.2f} USD {budget.period.value} limit "
                    f"({status.spent_usd:.2f} USD spent)"
                ),
            )
            self._dispatch(alert)
            emitted.append(alert)
        return emitted

    def _dispatch(self, alert: BudgetAlert) -> None:
        """Record ``alert`` and notify subscribers, isolating callback failures."""
        with self._lock:
            self._alerts.append(alert)
            if len(self._alerts) > self._max_alert_history:
                self._alerts.pop(0)
            subscribers = list(self._subscribers)
        for callback in subscribers:
            try:
                callback(alert)
            except AstroMLError:  # pragma: no cover - defensive
                logger.exception("budget alert subscriber failed")


def _period_end(budget: Budget, start: datetime) -> datetime:
    """Return the exclusive end of the period beginning at ``start``."""
    if budget.period is BudgetPeriod.DAILY:
        return start + timedelta(days=1)
    if budget.period is BudgetPeriod.WEEKLY:
        return start + timedelta(days=7)
    months = 1 if budget.period is BudgetPeriod.MONTHLY else 3
    month = start.month - 1 + months
    return start.replace(year=start.year + month // 12, month=month % 12 + 1, day=1)


def _severity_for(threshold: float) -> AlertSeverity:
    """Map a threshold fraction onto an alert severity."""
    if threshold >= 1.0:
        return AlertSeverity.CRITICAL
    if threshold >= 0.8:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


#: Thresholds are documented here so callers can reuse them when defining budgets.
SUPPORTED_THRESHOLDS: Sequence[float] = DEFAULT_THRESHOLDS
