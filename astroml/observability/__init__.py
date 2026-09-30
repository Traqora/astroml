"""Operational observability primitives: health checks and Prometheus metrics.

Three independent surfaces live here:

* :mod:`astroml.observability.health` — transport-agnostic health check
  results used by the ``/healthz/*`` endpoints (Issue #569).
* :mod:`astroml.observability.metrics` — the Prometheus metric registry and
  the decorators used to instrument critical paths (Issue #567).
* :mod:`astroml.observability.ingestion` — the ingestion heartbeat check that
  turns ``last_processed_at`` in the ingestion state store into a
  :class:`~astroml.observability.health.CheckResult` and the data-freshness
  gauges, backing ``GET /healthz/ingestion`` and the stale-data alerts.

The ingestion module is intentionally *not* re-exported here: importing
``astroml.observability`` is on the hot path for every health check, and the
heartbeat check pulls in the ingestion state store. Import it explicitly
(``from astroml.observability.ingestion import check_ingestion_heartbeat``)
where it is used.
"""

from astroml.observability.health import (
    CheckResult,
    HealthStatus,
    ReadinessState,
    aggregate_status,
    check_disk,
    readiness_state,
)
from astroml.observability.metrics import (
    ACTIVE_JOBS,
    DB_POOL_SIZE,
    DB_POOL_UTILIZATION,
    FEATURE_COMPUTE_TIME,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_TOTAL,
    MODEL_INFERENCE_LATENCY,
    observe_http_request,
    render_latest,
    track_active_job,
    track_feature_compute,
    track_inference,
    track_time,
    update_db_pool_metrics,
)

__all__ = [
    "ACTIVE_JOBS",
    "CheckResult",
    "DB_POOL_SIZE",
    "DB_POOL_UTILIZATION",
    "FEATURE_COMPUTE_TIME",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION",
    "HealthStatus",
    "MODEL_INFERENCE_LATENCY",
    "ReadinessState",
    "aggregate_status",
    "check_disk",
    "observe_http_request",
    "readiness_state",
    "render_latest",
    "track_active_job",
    "track_feature_compute",
    "track_inference",
    "track_time",
    "update_db_pool_metrics",
]
