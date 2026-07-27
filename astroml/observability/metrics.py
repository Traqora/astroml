"""Prometheus metrics for critical AstroML paths (Issue #567).

Metric reference
----------------

==============================  =========  ==================================
Name                            Type       Labels
==============================  =========  ==================================
http_request_duration_seconds   Histogram  method, endpoint, status
http_requests_total             Counter    method, endpoint, status
db_pool_size                    Gauge      pool, state
db_pool_utilization_ratio       Gauge      pool
feature_compute_time            Histogram  feature
model_inference_latency         Histogram  model
active_jobs                     Gauge      job_type
==============================  =========  ==================================

Instrumentation helpers
-----------------------

``track_time`` wraps sync *or* async callables and records into any
histogram. ``track_feature_compute`` / ``track_inference`` are the
pre-bound variants for the feature and model paths, and ``track_active_job``
is a context manager that keeps the ``active_jobs`` gauge balanced even when
the job raises.
"""

from __future__ import annotations

import functools
import inspect
import time
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Final, Iterator, TypeVar, cast

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.metrics import MetricWrapperBase
from prometheus_client.registry import CollectorRegistry

F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

#: Latency buckets tuned for sub-second API work with a long tail.
LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)


def _get_or_create(
    metric_cls: type[MetricWrapperBase],
    name: str,
    documentation: str,
    labelnames: tuple[str, ...] = (),
    registry: CollectorRegistry = REGISTRY,
    **kwargs: Any,
) -> Any:
    """Return an existing collector by name, or register a new one.

    Module reloads (common under pytest) would otherwise raise
    ``Duplicated timeseries in CollectorRegistry``.

    Args:
        metric_cls: ``Counter``, ``Gauge`` or ``Histogram``.
        name: Metric name.
        documentation: HELP text.
        labelnames: Label names for the metric.
        registry: Registry to look in. Defaults to the global registry.
        **kwargs: Extra keyword arguments for the metric constructor.

    Returns:
        The registered collector.
    """
    existing = registry._names_to_collectors.get(name)  # noqa: SLF001
    if existing is not None:
        return existing
    return metric_cls(
        name,
        documentation,
        labelnames,
        registry=registry,
        **kwargs,
    )


HTTP_REQUEST_DURATION: Final[Histogram] = _get_or_create(
    Histogram,
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ("method", "endpoint", "status"),
    buckets=LATENCY_BUCKETS,
)

HTTP_REQUESTS_TOTAL: Final[Counter] = _get_or_create(
    Counter,
    "http_requests_total",
    "Total HTTP requests by endpoint and response status.",
    ("method", "endpoint", "status"),
)

DB_POOL_SIZE: Final[Gauge] = _get_or_create(
    Gauge,
    "db_pool_size",
    (
        "Database connection pool connections by state "
        "(configured, in_use, idle, overflow, capacity)."
    ),
    ("pool", "state"),
)

DB_POOL_UTILIZATION: Final[Gauge] = _get_or_create(
    Gauge,
    "db_pool_utilization_ratio",
    "Database connection pool utilization as a ratio of total capacity.",
    ("pool",),
)

FEATURE_COMPUTE_TIME: Final[Histogram] = _get_or_create(
    Histogram,
    "feature_compute_time",
    "Feature computation time in seconds.",
    ("feature",),
    buckets=LATENCY_BUCKETS,
)

MODEL_INFERENCE_LATENCY: Final[Histogram] = _get_or_create(
    Histogram,
    "model_inference_latency",
    "Model inference latency in seconds.",
    ("model",),
    buckets=LATENCY_BUCKETS,
)

ACTIVE_JOBS: Final[Gauge] = _get_or_create(
    Gauge,
    "active_jobs",
    "Number of jobs currently running, by job type (e.g. ingestion).",
    ("job_type",),
)


def observe_http_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration_seconds: float,
) -> None:
    """Record one HTTP request in both the latency histogram and counter.

    Args:
        method: HTTP method, e.g. ``"GET"``.
        endpoint: Route template (not the raw path) to keep cardinality low.
        status_code: Response status code.
        duration_seconds: Request duration in seconds.
    """
    labels = (method.upper(), endpoint, str(status_code))
    HTTP_REQUEST_DURATION.labels(*labels).observe(duration_seconds)
    HTTP_REQUESTS_TOTAL.labels(*labels).inc()


def update_db_pool_metrics(stats: Any, pool_name: str = "default") -> None:
    """Publish connection pool statistics to the pool gauges.

    Args:
        stats: An :class:`astroml.db.pool_health.PoolStats` instance (any
            object exposing the same attributes works).
        pool_name: Label distinguishing multiple pools.
    """
    DB_POOL_SIZE.labels(pool_name, "configured").set(stats.pool_size)
    DB_POOL_SIZE.labels(pool_name, "in_use").set(stats.checked_out)
    DB_POOL_SIZE.labels(pool_name, "idle").set(stats.checked_in)
    DB_POOL_SIZE.labels(pool_name, "overflow").set(stats.overflow)
    DB_POOL_SIZE.labels(pool_name, "capacity").set(stats.capacity)
    DB_POOL_UTILIZATION.labels(pool_name).set(stats.utilization)


def track_time(histogram: Histogram, *label_values: str) -> Callable[[F], F]:
    """Decorate a sync or async callable, timing it into ``histogram``.

    Args:
        histogram: Target histogram.
        *label_values: Label values, in the histogram's label order.

    Returns:
        A decorator preserving the wrapped function's signature.

    Example:
        >>> from astroml.observability.metrics import FEATURE_COMPUTE_TIME
        >>> @track_time(FEATURE_COMPUTE_TIME, "degree_centrality")
        ... def compute() -> int:
        ...     return 1
        >>> compute()
        1
    """
    target = histogram.labels(*label_values) if label_values else histogram

    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return await cast(Callable[..., Awaitable[Any]], func)(
                        *args, **kwargs
                    )
                finally:
                    target.observe(time.perf_counter() - started)

            return cast(F, async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                target.observe(time.perf_counter() - started)

        return cast(F, sync_wrapper)

    return decorator


def track_feature_compute(feature: str) -> Callable[[F], F]:
    """Time a feature computation into ``feature_compute_time``.

    Args:
        feature: Feature name used as the metric label.
    """
    return track_time(FEATURE_COMPUTE_TIME, feature)


def track_inference(model: str) -> Callable[[F], F]:
    """Time a model prediction into ``model_inference_latency``.

    Args:
        model: Model name used as the metric label.
    """
    return track_time(MODEL_INFERENCE_LATENCY, model)


@contextmanager
def track_active_job(job_type: str = "ingestion") -> Iterator[None]:
    """Increment ``active_jobs`` for the duration of a job.

    The gauge is decremented in a ``finally`` block, so a raising job does
    not leak a permanently elevated count.

    Args:
        job_type: Label value, e.g. ``"ingestion"`` or ``"training"``.
    """
    gauge = ACTIVE_JOBS.labels(job_type)
    gauge.inc()
    try:
        yield
    finally:
        gauge.dec()


def render_latest(registry: CollectorRegistry = REGISTRY) -> tuple[bytes, str]:
    """Render the Prometheus exposition payload.

    Args:
        registry: Registry to scrape. Defaults to the global registry.

    Returns:
        A ``(body, content_type)`` tuple ready to be returned from an
        HTTP handler.
    """
    return generate_latest(registry), CONTENT_TYPE_LATEST
