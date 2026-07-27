"""HTTP-level tests for the /healthz probes and /metrics/db-pool (#569, #550).

The router is mounted on a bare FastAPI app so these tests exercise the probe
contract without importing the full application graph (auth middleware,
GraphQL schema, scheduler).
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterator

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from astroml.db.pool_health import PoolStats  # noqa: E402
from astroml.observability.health import (  # noqa: E402
    CheckResult,
    HealthStatus,
    readiness_state,
)


@pytest.fixture(scope="module")
def healthz_module() -> Any:
    """Load ``api/routers/healthz.py`` in isolation.

    Importing it as ``api.routers.healthz`` would execute
    ``api/routers/__init__.py``, which eagerly imports every router in the
    service (auth, GraphQL, LLM providers, …) along with dependencies the
    CPU CI image does not install. The probe module only needs FastAPI,
    SQLAlchemy and ``api.database``, so it is loaded directly from its path.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "api" / "routers" / "healthz.py"
    spec = importlib.util.spec_from_file_location("astroml_healthz_probe", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        pytest.skip(f"Cannot load health router from {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"API dependencies unavailable: {exc}")

    return module


@pytest.fixture()
def client(healthz_module: Any) -> Iterator[TestClient]:
    app = FastAPI()
    app.include_router(healthz_module.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _ready_process() -> Iterator[None]:
    readiness_state.mark_started()
    yield
    readiness_state.reset()


@pytest.fixture()
def healthy_pool(healthz_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        healthz_module,
        "_pool_check",
        lambda: CheckResult(
            name="db_pool",
            status=HealthStatus.OK,
            details=PoolStats(10, 8, 2, 0, 10, "QueuePool").to_dict(),
        ),
    )


def _ok_db() -> Any:
    async def _check() -> CheckResult:
        return CheckResult(
            name="db", status=HealthStatus.OK, details={"latency_ms": 1.0}
        )

    return _check


def _failing(name: str, remediation: str = "Fix it.") -> Any:
    async def _check() -> CheckResult:
        return CheckResult(
            name=name,
            status=HealthStatus.FAIL,
            details={"error": "boom"},
            remediation=remediation,
        )

    return _check


def _degraded(name: str) -> Any:
    async def _check() -> CheckResult:
        return CheckResult(
            name=name, status=HealthStatus.DEGRADED, remediation="Watch it."
        )

    return _check


@pytest.fixture()
def all_healthy(healthz_module: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(healthz_module, "check_database", _ok_db())
    monkeypatch.setattr(
        healthz_module,
        "check_cache",
        lambda: _resolved(CheckResult("cache", HealthStatus.OK)),
    )
    monkeypatch.setattr(
        healthz_module,
        "check_disk_space",
        lambda: _resolved(CheckResult("disk", HealthStatus.OK)),
    )


async def _resolved(result: CheckResult) -> CheckResult:
    return result


class _FakeSession:
    """Minimal async session that answers ``SELECT 1``."""

    async def execute(self, _statement: Any) -> None:
        return None

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _FakeSessionFactory() -> _FakeSession:  # noqa: N802 - mimics a factory call
    return _FakeSession()


class TestLiveness:
    def test_live_is_always_ok(self, client: TestClient) -> None:
        response = client.get("/healthz/live")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["component"] == "live"
        assert "uptime_seconds" in body["details"]

    def test_live_ignores_dependency_failures(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(healthz_module, "check_database", _failing("db"))

        assert client.get("/healthz/live").status_code == 200


class TestStartupProbe:
    def test_startup_ok_after_mark_started(self, client: TestClient) -> None:
        response = client.get("/healthz/startup")

        assert response.status_code == 200
        assert response.json()["details"]["started"] is True

    def test_startup_fails_before_initialisation(self, client: TestClient) -> None:
        readiness_state.reset()

        response = client.get("/healthz/startup")

        assert response.status_code == 503
        assert response.json()["status"] == "fail"
        assert "initialising" in response.json()["remediation"]


class TestReadiness:
    def test_ready_when_dependencies_pass(
        self, client: TestClient, all_healthy: None
    ) -> None:
        response = client.get("/healthz/ready")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["probe"] == "ready"
        assert set(body["details"]) == {"startup", "db", "cache", "disk"}

    def test_not_ready_before_startup(self, client: TestClient) -> None:
        readiness_state.reset()

        response = client.get("/healthz/ready")

        assert response.status_code == 503
        assert set(response.json()["details"]) == {"startup"}

    def test_database_failure_blocks_readiness(
        self,
        client: TestClient,
        all_healthy: None,
        healthz_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            healthz_module, "check_database", _failing("db", "Check DATABASE_URL.")
        )

        response = client.get("/healthz/ready")

        assert response.status_code == 503
        assert response.json()["status"] == "fail"
        assert "Check DATABASE_URL." in response.json()["remediation"]

    def test_cache_degradation_still_serves_traffic(
        self,
        client: TestClient,
        all_healthy: None,
        healthz_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(healthz_module, "check_cache", _degraded("cache"))

        response = client.get("/healthz/ready")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_draining_pod_is_not_ready(
        self, client: TestClient, all_healthy: None
    ) -> None:
        readiness_state.set_ready(False, "Application is shutting down.")

        response = client.get("/healthz/ready")

        assert response.status_code == 503
        assert "shutting down" in response.json()["remediation"]


class TestAggregateHealthz:
    def test_aggregate_reports_every_component(
        self, client: TestClient, all_healthy: None
    ) -> None:
        response = client.get("/healthz")

        assert response.status_code == 200
        body = response.json()
        assert body["probe"] == "healthz"
        assert set(body["details"]) == {"startup", "db", "cache", "disk"}

    def test_aggregate_surfaces_worst_status(
        self,
        client: TestClient,
        all_healthy: None,
        healthz_module: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(healthz_module, "check_disk_space", _failing("disk"))

        response = client.get("/healthz")

        assert response.status_code == 503
        assert response.json()["status"] == "fail"


class TestComponentProbes:
    def test_db_probe(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(healthz_module, "check_database", _ok_db())

        response = client.get("/healthz/db")

        assert response.status_code == 200
        assert response.json()["component"] == "db"

    def test_db_probe_failure_returns_503(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            healthz_module, "check_database", _failing("db", "Verify DATABASE_URL.")
        )

        response = client.get("/healthz/db")

        assert response.status_code == 503
        assert "Verify DATABASE_URL." in response.json()["remediation"]

    def test_cache_probe(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            healthz_module,
            "check_cache",
            lambda: _resolved(
                CheckResult("cache", HealthStatus.OK, {"latency_ms": 0.5})
            ),
        )

        response = client.get("/healthz/cache")

        assert response.status_code == 200
        assert response.json()["details"]["latency_ms"] == 0.5

    def test_cache_outage_is_degraded_by_default(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(healthz_module, "check_cache", _degraded("cache"))

        response = client.get("/healthz/cache")

        assert response.status_code == 200
        assert response.json()["status"] == "degraded"

    def test_disk_probe(self, client: TestClient) -> None:
        response = client.get("/healthz/disk")

        assert response.status_code in (200, 503)
        assert response.json()["component"] == "disk"
        assert "free_bytes" in response.json()["details"]


class TestProbeResilience:
    def test_timeout_is_reported_as_failure(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _hang() -> CheckResult:
            await asyncio.sleep(10)
            raise AssertionError("unreachable")

        monkeypatch.setattr(healthz_module, "CHECK_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(healthz_module, "check_database", _hang)

        response = client.get("/healthz/db")

        assert response.status_code == 503
        assert "timed out" in response.json()["details"]["error"]

    def test_unexpected_exception_is_reported_as_failure(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom() -> CheckResult:
            raise RuntimeError("driver exploded")

        monkeypatch.setattr(healthz_module, "check_database", _boom)

        response = client.get("/healthz/db")

        assert response.status_code == 503
        body = response.json()
        assert body["details"]["error_type"] == "RuntimeError"
        assert "driver exploded" in body["remediation"]


class TestDbPoolMetricsEndpoint:
    def test_pool_snapshot_is_served(
        self, client: TestClient, healthy_pool: None
    ) -> None:
        response = client.get("/metrics/db-pool")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["pool"]["pool_size"] == 10
        assert body["pool"]["alert_threshold_percent"] == 80.0

    def test_saturated_pool_is_degraded(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from astroml.db.pool_health import evaluate_pool_health

        stats = PoolStats(10, 2, 17, 7, 10, "QueuePool")
        monkeypatch.setattr(
            healthz_module, "_pool_check", lambda: evaluate_pool_health(stats)
        )

        response = client.get("/metrics/db-pool")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["pool"]["saturated"] is True
        assert "DB_POOL_MAX_SIZE" in body["remediation"]

    def test_exhausted_pool_returns_503(
        self, client: TestClient, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from astroml.db.pool_health import evaluate_pool_health

        stats = PoolStats(10, 0, 20, 10, 10, "QueuePool")
        monkeypatch.setattr(
            healthz_module, "_pool_check", lambda: evaluate_pool_health(stats)
        )

        response = client.get("/metrics/db-pool")

        assert response.status_code == 503
        assert response.json()["pool"]["exhausted"] is True


class TestDatabaseCheckImplementation:
    def test_connection_error_is_reported(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _factory() -> Any:
            raise OSError("connection refused")

        monkeypatch.setattr(healthz_module, "get_async_session_factory", _factory)

        result = asyncio.run(healthz_module.check_database())

        assert result.status is HealthStatus.FAIL
        assert result.details["error_type"] == "OSError"
        assert "DATABASE_URL" in result.remediation

    def test_cache_ping_failure_degrades(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_url: str) -> float:
            raise ConnectionError("no route to host")

        monkeypatch.setattr(healthz_module, "_ping_redis", _boom)
        monkeypatch.setattr(healthz_module, "CACHE_REQUIRED", False)

        result = asyncio.run(healthz_module.check_cache())

        assert result.status is HealthStatus.DEGRADED
        assert "REDIS_URL" in result.remediation

    def test_cache_ping_success(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(healthz_module, "_ping_redis", lambda _url: 1.5)

        result = asyncio.run(healthz_module.check_cache())

        assert result.status is HealthStatus.OK
        assert result.details["latency_ms"] == 1.5

    def test_cache_required_turns_outage_into_failure(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_url: str) -> float:
            raise ConnectionError("refused")

        monkeypatch.setattr(healthz_module, "_ping_redis", _boom)
        monkeypatch.setattr(healthz_module, "CACHE_REQUIRED", True)

        result = asyncio.run(healthz_module.check_cache())

        assert result.status is HealthStatus.FAIL
        assert result.details["required"] is True

    def test_successful_query_folds_in_pool_status(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            healthz_module, "get_async_session_factory", lambda: _FakeSessionFactory
        )
        monkeypatch.setattr(
            healthz_module,
            "_pool_check",
            lambda: CheckResult(
                name="db_pool",
                status=HealthStatus.DEGRADED,
                details=PoolStats(10, 2, 17, 7, 10, "QueuePool").to_dict(),
                remediation="Raise DB_POOL_MAX_SIZE.",
            ),
        )

        result = asyncio.run(healthz_module.check_database())

        assert result.status is HealthStatus.DEGRADED
        assert result.details["query"] == "SELECT 1"
        assert result.details["pool"]["saturated"] is True
        assert result.remediation == "Raise DB_POOL_MAX_SIZE."

    def test_pool_check_publishes_prometheus_gauges(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from prometheus_client import REGISTRY

        class _Pool:
            def size(self) -> int:
                return 4

            def checkedin(self) -> int:
                return 3

            def checkedout(self) -> int:
                return 1

            def overflow(self) -> int:
                return -4

        class _Engine:
            pool = _Pool()

        monkeypatch.setattr(healthz_module, "get_async_engine", lambda: _Engine())

        result = healthz_module._pool_check()

        assert result.status is HealthStatus.OK
        assert (
            REGISTRY.get_sample_value(
                "db_pool_size", {"pool": "default", "state": "in_use"}
            )
            == 1
        )

    def test_ping_redis_uses_client_and_closes_it(
        self, healthz_module: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        closed: list[bool] = []

        class _Client:
            def ping(self) -> bool:
                return True

            def close(self) -> None:
                closed.append(True)

        fake_redis = types.ModuleType("redis")
        fake_redis.Redis = types.SimpleNamespace(  # type: ignore[attr-defined]
            from_url=lambda url, **kwargs: _Client()
        )
        monkeypatch.setitem(sys.modules, "redis", fake_redis)

        latency_ms = healthz_module._ping_redis("redis://localhost:6379/0")

        assert latency_ms >= 0.0
        assert closed == [True]
