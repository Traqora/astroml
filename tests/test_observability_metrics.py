"""Tests for Prometheus metric coverage and instrumentation (Issue #567)."""

from __future__ import annotations

import asyncio

import pytest
from prometheus_client import REGISTRY

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
from astroml.db.pool_health import PoolStats


def _sample(name: str, **labels: str) -> float:
    value = REGISTRY.get_sample_value(name, labels)
    return 0.0 if value is None else value


class TestMetricRegistration:
    @pytest.mark.parametrize(
        "name",
        [
            "http_request_duration_seconds",
            "http_requests_total",
            "db_pool_size",
            "db_pool_utilization_ratio",
            "feature_compute_time",
            "model_inference_latency",
            "active_jobs",
        ],
    )
    def test_metric_is_registered(self, name: str) -> None:
        assert name in REGISTRY._names_to_collectors

    def test_labels_are_declared(self) -> None:
        assert HTTP_REQUEST_DURATION._labelnames == ("method", "endpoint", "status")
        assert HTTP_REQUESTS_TOTAL._labelnames == ("method", "endpoint", "status")
        assert DB_POOL_SIZE._labelnames == ("pool", "state")
        assert DB_POOL_UTILIZATION._labelnames == ("pool",)
        assert FEATURE_COMPUTE_TIME._labelnames == ("feature",)
        assert MODEL_INFERENCE_LATENCY._labelnames == ("model",)
        assert ACTIVE_JOBS._labelnames == ("job_type",)


class TestHttpMetrics:
    def test_request_is_counted_and_timed(self) -> None:
        labels = {"method": "GET", "endpoint": "/healthz", "status": "200"}
        before = _sample("http_requests_total", **labels)

        observe_http_request("get", "/healthz", 200, 0.05)

        assert _sample("http_requests_total", **labels) == before + 1
        assert _sample("http_request_duration_seconds_count", **labels) >= 1
        assert _sample("http_request_duration_seconds_sum", **labels) >= 0.05

    def test_status_codes_are_separate_series(self) -> None:
        observe_http_request("GET", "/healthz/db", 503, 0.01)

        assert (
            _sample(
                "http_requests_total",
                method="GET",
                endpoint="/healthz/db",
                status="503",
            )
            >= 1
        )


class TestDbPoolMetrics:
    def test_pool_stats_are_published(self) -> None:
        stats = PoolStats(
            pool_size=10,
            checked_in=3,
            checked_out=8,
            overflow=1,
            max_overflow=10,
            implementation="QueuePool",
        )

        update_db_pool_metrics(stats, pool_name="test-pool")

        assert _sample("db_pool_size", pool="test-pool", state="configured") == 10
        assert _sample("db_pool_size", pool="test-pool", state="in_use") == 8
        assert _sample("db_pool_size", pool="test-pool", state="idle") == 3
        assert _sample("db_pool_size", pool="test-pool", state="overflow") == 1
        assert _sample("db_pool_size", pool="test-pool", state="capacity") == 20
        assert _sample("db_pool_utilization_ratio", pool="test-pool") == pytest.approx(
            0.4
        )


class TestTrackTime:
    def test_sync_function_is_timed(self) -> None:
        @track_time(FEATURE_COMPUTE_TIME, "sync_feature")
        def compute() -> int:
            return 42

        assert compute() == 42
        assert _sample("feature_compute_time_count", feature="sync_feature") == 1

    def test_async_function_is_timed(self) -> None:
        @track_time(MODEL_INFERENCE_LATENCY, "async_model")
        async def infer() -> str:
            await asyncio.sleep(0)
            return "done"

        assert asyncio.run(infer()) == "done"
        assert _sample("model_inference_latency_count", model="async_model") == 1

    def test_exception_is_still_timed_and_propagates(self) -> None:
        @track_time(FEATURE_COMPUTE_TIME, "boom")
        def compute() -> None:
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            compute()

        assert _sample("feature_compute_time_count", feature="boom") == 1

    def test_async_exception_is_still_timed(self) -> None:
        @track_time(MODEL_INFERENCE_LATENCY, "async_boom")
        async def infer() -> None:
            raise RuntimeError("bad")

        with pytest.raises(RuntimeError, match="bad"):
            asyncio.run(infer())

        assert _sample("model_inference_latency_count", model="async_boom") == 1

    def test_metadata_is_preserved(self) -> None:
        @track_time(FEATURE_COMPUTE_TIME, "documented")
        def compute() -> None:
            """Docstring."""

        assert compute.__name__ == "compute"
        assert compute.__doc__ == "Docstring."

    def test_unlabelled_histogram_is_supported(self) -> None:
        from prometheus_client import CollectorRegistry, Histogram

        registry = CollectorRegistry()
        histogram = Histogram("bare_seconds", "doc", registry=registry)

        @track_time(histogram)
        def compute() -> int:
            return 1

        compute()

        assert registry.get_sample_value("bare_seconds_count") == 1


class TestPreBoundDecorators:
    def test_track_feature_compute(self) -> None:
        @track_feature_compute("degree_centrality")
        def compute() -> int:
            return 7

        assert compute() == 7
        assert _sample("feature_compute_time_count", feature="degree_centrality") == 1

    def test_track_inference(self) -> None:
        @track_inference("gnn")
        def predict() -> int:
            return 1

        predict()
        assert _sample("model_inference_latency_count", model="gnn") == 1


class TestTrackActiveJob:
    def test_gauge_rises_then_falls(self) -> None:
        assert _sample("active_jobs", job_type="ingestion") == 0

        with track_active_job("ingestion"):
            assert _sample("active_jobs", job_type="ingestion") == 1

        assert _sample("active_jobs", job_type="ingestion") == 0

    def test_gauge_is_balanced_on_exception(self) -> None:
        with pytest.raises(ValueError):
            with track_active_job("training"):
                raise ValueError("job failed")

        assert _sample("active_jobs", job_type="training") == 0

    def test_nested_jobs_accumulate(self) -> None:
        with track_active_job("backfill"):
            with track_active_job("backfill"):
                assert _sample("active_jobs", job_type="backfill") == 2
        assert _sample("active_jobs", job_type="backfill") == 0


class TestRenderLatest:
    def test_exposition_contains_every_metric(self) -> None:
        observe_http_request("GET", "/metrics", 200, 0.001)
        body, content_type = render_latest()
        text = body.decode()

        assert "text/plain" in content_type
        for name in (
            "http_request_duration_seconds",
            "http_requests_total",
            "db_pool_size",
            "feature_compute_time",
            "model_inference_latency",
            "active_jobs",
        ):
            assert name in text

    def test_custom_registry_is_scraped(self) -> None:
        from prometheus_client import CollectorRegistry, Counter

        registry = CollectorRegistry()
        Counter("isolated_total", "doc", registry=registry).inc()

        body, _ = render_latest(registry)

        assert "isolated_total" in body.decode()
