"""Transport-agnostic health check primitives (Issue #569).

The FastAPI routers in ``api/routers/healthz.py`` are thin wrappers around
the types defined here, so the health semantics (status vocabulary,
aggregation rules, remediation text, readiness gating) are unit-testable
without an HTTP client or live dependencies.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Iterable, Mapping


class HealthStatus(str, Enum):
    """Status vocabulary shared by every health check.

    ``OK`` — the component is fully functional.
    ``DEGRADED`` — the component works but is close to a limit; traffic is
    still served.
    ``FAIL`` — the component is unusable; readiness must not pass.
    """

    OK = "ok"
    DEGRADED = "degraded"
    FAIL = "fail"

    @property
    def is_serving(self) -> bool:
        """True when a component in this state can still serve traffic."""
        return self is not HealthStatus.FAIL


#: Severity ordering used when aggregating multiple checks.
_SEVERITY: Final[dict[HealthStatus, int]] = {
    HealthStatus.OK: 0,
    HealthStatus.DEGRADED: 1,
    HealthStatus.FAIL: 2,
}

#: HTTP status code returned for each health status.
HTTP_STATUS_FOR: Final[dict[HealthStatus, int]] = {
    HealthStatus.OK: 200,
    HealthStatus.DEGRADED: 200,
    HealthStatus.FAIL: 503,
}


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single component health check.

    Attributes:
        name: Component name, e.g. ``"db"``.
        status: Health status of the component.
        details: Structured, component-specific measurements.
        remediation: Operator-facing next step. Empty when status is ``OK``.
        duration_ms: Wall-clock duration of the check.
    """

    name: str
    status: HealthStatus
    details: Mapping[str, Any] = field(default_factory=dict)
    remediation: str = ""
    duration_ms: float = 0.0

    @property
    def http_status(self) -> int:
        """HTTP status code this result should be served with."""
        return HTTP_STATUS_FOR[self.status]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the JSON body shape used by the ``/healthz`` API."""
        return {
            "status": self.status.value,
            "component": self.name,
            "details": dict(self.details),
            "remediation": self.remediation,
            "duration_ms": round(self.duration_ms, 2),
        }


def aggregate_status(results: Iterable[CheckResult]) -> HealthStatus:
    """Reduce several check results to the worst status observed.

    Args:
        results: Individual component results.

    Returns:
        ``OK`` for an empty iterable, otherwise the most severe status.
    """
    worst = HealthStatus.OK
    for result in results:
        if _SEVERITY[result.status] > _SEVERITY[worst]:
            worst = result.status
    return worst


def aggregate_to_dict(
    results: Iterable[CheckResult],
    *,
    remediation_prefix: str = "",
) -> dict[str, Any]:
    """Build the aggregate JSON body for a multi-component probe.

    Args:
        results: Individual component results.
        remediation_prefix: Optional sentence prepended to the combined
            remediation text.

    Returns:
        A mapping with ``status``, per-component ``details`` and a combined
        ``remediation`` string.
    """
    materialised = list(results)
    status = aggregate_status(materialised)
    remediations = [r.remediation for r in materialised if r.remediation]
    remediation = " ".join(filter(None, [remediation_prefix, *remediations]))

    return {
        "status": status.value,
        "details": {r.name: r.to_dict() for r in materialised},
        "remediation": remediation,
    }


#: Free-space ratio below which the disk check reports ``DEGRADED``.
DISK_DEGRADED_RATIO: Final[float] = 0.20

#: Free-space ratio below which the disk check reports ``FAIL``.
DISK_FAIL_RATIO: Final[float] = 0.10


def check_disk(
    path: str = ".",
    *,
    degraded_ratio: float = DISK_DEGRADED_RATIO,
    fail_ratio: float = DISK_FAIL_RATIO,
) -> CheckResult:
    """Check free disk space on the filesystem backing ``path``.

    Args:
        path: Any path on the filesystem to inspect.
        degraded_ratio: Free-space ratio below which status is ``DEGRADED``.
        fail_ratio: Free-space ratio below which status is ``FAIL``.

    Returns:
        A :class:`CheckResult` named ``"disk"``.
    """
    started = time.perf_counter()
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return CheckResult(
            name="disk",
            status=HealthStatus.FAIL,
            details={"path": path, "error": str(exc)},
            remediation=(
                f"Cannot stat {path}. Verify the volume is mounted and the "
                "process user has read access."
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    free_ratio = usage.free / usage.total if usage.total else 0.0
    if free_ratio < fail_ratio:
        status = HealthStatus.FAIL
        remediation = (
            f"Only {free_ratio:.1%} of {path} is free. Free space immediately "
            "— prune model artifacts, rotate logs, or expand the volume; "
            "writes will start failing."
        )
    elif free_ratio < degraded_ratio:
        status = HealthStatus.DEGRADED
        remediation = (
            f"{free_ratio:.1%} of {path} is free, below the "
            f"{degraded_ratio:.0%} warning threshold. Schedule cleanup of "
            "artifacts and logs before it becomes critical."
        )
    else:
        status = HealthStatus.OK
        remediation = ""

    return CheckResult(
        name="disk",
        status=status,
        details={
            "path": path,
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "free_ratio": round(free_ratio, 4),
            "degraded_ratio": degraded_ratio,
            "fail_ratio": fail_ratio,
        },
        remediation=remediation,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


class ReadinessState:
    """Tracks whether the process has finished initialising.

    Kubernetes startup probes poll :meth:`snapshot` until ``started`` is
    true; readiness probes additionally require dependency checks to pass.
    Readiness can be flipped off manually (``set_ready(False)``) to drain a
    pod before shutdown.
    """

    def __init__(self) -> None:
        self._started_at: float | None = None
        self._ready: bool = False
        self._detail: str = "Process has not completed startup."

    def mark_started(self, detail: str = "Startup complete.") -> None:
        """Record that application startup finished."""
        self._started_at = time.time()
        self._ready = True
        self._detail = detail

    def set_ready(self, ready: bool, detail: str = "") -> None:
        """Manually gate readiness, e.g. to drain traffic before shutdown."""
        self._ready = ready
        self._detail = detail or (
            "Ready." if ready else "Readiness disabled by operator."
        )

    def reset(self) -> None:
        """Return to the pre-startup state (used by tests)."""
        self._started_at = None
        self._ready = False
        self._detail = "Process has not completed startup."

    @property
    def started(self) -> bool:
        """True once :meth:`mark_started` has been called."""
        return self._started_at is not None

    @property
    def ready(self) -> bool:
        """True when startup finished and readiness has not been revoked."""
        return self.started and self._ready

    @property
    def uptime_seconds(self) -> float:
        """Seconds since startup completed; ``0.0`` before that."""
        if self._started_at is None:
            return 0.0
        return time.time() - self._started_at

    def snapshot(self) -> CheckResult:
        """Return the startup gate as a :class:`CheckResult`."""
        if not self.started:
            return CheckResult(
                name="startup",
                status=HealthStatus.FAIL,
                details={"started": False, "ready": False},
                remediation=(
                    "The application is still initialising. If this persists "
                    "past the startup probe budget, inspect the container "
                    "logs for a stalled lifespan handler."
                ),
            )
        if not self._ready:
            return CheckResult(
                name="startup",
                status=HealthStatus.FAIL,
                details={
                    "started": True,
                    "ready": False,
                    "uptime_seconds": round(self.uptime_seconds, 2),
                },
                remediation=(
                    f"{self._detail} Traffic is being drained; this is "
                    "expected during a graceful shutdown."
                ),
            )
        return CheckResult(
            name="startup",
            status=HealthStatus.OK,
            details={
                "started": True,
                "ready": True,
                "uptime_seconds": round(self.uptime_seconds, 2),
            },
        )


#: Process-wide readiness gate used by the API lifespan and probes.
readiness_state: Final[ReadinessState] = ReadinessState()
