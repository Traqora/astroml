"""Docker Compose restart policies and healthcheck wiring (issue #775).

Validates that every long-running compose service declares an appropriate
restart policy, a readiness healthcheck, and graceful shutdown settings.
One-off batch services (training, soroban-build/test, e2e pytest runner) are
exempt from healthchecks but must still declare restart: "no".
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

COMPOSE_FILES = (
    REPO_ROOT / "docker-compose.yml",
    REPO_ROOT / "docker-compose.e2e.yml",
)

# Batch / one-shot services — restart: "no", healthcheck optional.
ONE_OFF_SERVICES = frozenset(
    {
        "training-gpu",
        "training-cpu",
        "soroban-build",
        "soroban-test",
        "pytest-e2e",
    }
)

# Services without healthchecks (explicit allow-list for one-off jobs only).
HEALTHCHECK_EXEMPT = ONE_OFF_SERVICES

# HTTP services that must probe the readiness endpoint, not legacy /health.
READINESS_ENDPOINTS = {
    "api": "/healthz/ready",
    "api-e2e": "/healthz/ready",
}

# Long-running services that must allow in-flight work to finish on stop.
GRACEFUL_SHUTDOWN_REQUIRED = frozenset(
    {
        "postgres",
        "redis",
        "api",
        "celery-worker",
        "flower",
        "feature-store",
        "ingestion",
        "streaming",
        "dev",
        "production",
        "prometheus",
        "grafana",
        "soroban-dev",
        "postgres-e2e",
        "redis-e2e",
        "api-e2e",
    }
)


def _load_compose(path: pathlib.Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    assert isinstance(loaded, dict), f"{path.name} must be a mapping"
    return loaded


def _service_names(compose: dict) -> list[str]:
    services = compose.get("services", {})
    assert isinstance(services, dict), "services must be a mapping"
    return sorted(services)


@pytest.mark.parametrize(
    "compose_path",
    COMPOSE_FILES,
    ids=[p.name for p in COMPOSE_FILES],
)
class TestComposeFileParses:
    def test_is_valid_yaml(self, compose_path: pathlib.Path):
        _load_compose(compose_path)

    def test_declares_services(self, compose_path: pathlib.Path):
        compose = _load_compose(compose_path)
        assert _service_names(compose), f"{compose_path.name} has no services"


class TestDockerComposePolicies:
    @pytest.fixture(scope="class")
    def main_compose(self) -> dict:
        return _load_compose(REPO_ROOT / "docker-compose.yml")

    def test_every_service_has_restart_policy(self, main_compose: dict):
        missing: list[str] = []
        for name, service in main_compose["services"].items():
            if "restart" not in service:
                missing.append(name)
        assert not missing, f"services missing restart policy: {missing}"

    def test_long_running_services_have_healthchecks(self, main_compose: dict):
        missing: list[str] = []
        for name, service in main_compose["services"].items():
            if name in HEALTHCHECK_EXEMPT:
                continue
            if "healthcheck" not in service:
                missing.append(name)
        assert not missing, f"services missing healthcheck: {missing}"

    def test_one_off_services_do_not_auto_restart(self, main_compose: dict):
        offenders: list[str] = []
        for name in ONE_OFF_SERVICES:
            service = main_compose["services"].get(name)
            if service is None:
                continue
            if service.get("restart") != "no":
                offenders.append(f"{name}={service.get('restart')!r}")
        assert not offenders, f"one-off services must use restart: 'no': {offenders}"

    def test_http_services_use_readiness_endpoint(self, main_compose: dict):
        offenders: list[str] = []
        for name, endpoint in READINESS_ENDPOINTS.items():
            service = main_compose["services"].get(name)
            if service is None:
                continue
            healthcheck = service.get("healthcheck", {})
            test_cmd = healthcheck.get("test", [])
            probe = " ".join(str(part) for part in test_cmd)
            if endpoint not in probe:
                offenders.append(f"{name} probes {probe!r}, expected {endpoint}")
            if "/health" in probe and endpoint not in probe:
                offenders.append(f"{name} still uses legacy /health probe")
        assert not offenders, offenders

    def test_graceful_shutdown_on_long_running_services(self, main_compose: dict):
        missing: list[str] = []
        for name in GRACEFUL_SHUTDOWN_REQUIRED:
            service = main_compose["services"].get(name)
            if service is None:
                continue
            if "stop_grace_period" not in service:
                missing.append(name)
            if service.get("stop_signal", "SIGTERM") != "SIGTERM":
                missing.append(f"{name}:stop_signal")
        assert not missing, f"services missing graceful shutdown: {missing}"

    def test_flower_waits_for_healthy_celery_worker(self, main_compose: dict):
        flower = main_compose["services"]["flower"]
        depends = flower.get("depends_on", {})
        assert isinstance(depends, dict), "flower depends_on must use condition form"
        celery_dep = depends.get("celery-worker", {})
        assert celery_dep.get("condition") == "service_healthy"

    def test_grafana_waits_for_healthy_prometheus(self, main_compose: dict):
        grafana = main_compose["services"]["grafana"]
        depends = grafana.get("depends_on", {})
        assert isinstance(depends, dict), "grafana depends_on must use condition form"
        prom_dep = depends.get("prometheus", {})
        assert prom_dep.get("condition") == "service_healthy"


class TestE2eComposePolicies:
    @pytest.fixture(scope="class")
    def e2e_compose(self) -> dict:
        return _load_compose(REPO_ROOT / "docker-compose.e2e.yml")

    def test_e2e_services_have_restart_and_healthchecks(self, e2e_compose: dict):
        missing_restart: list[str] = []
        missing_health: list[str] = []
        for name, service in e2e_compose["services"].items():
            if name == "pytest-e2e":
                if service.get("restart", "no") != "no":
                    missing_restart.append(name)
                continue
            if "restart" not in service:
                missing_restart.append(name)
            if "healthcheck" not in service:
                missing_health.append(name)
        assert not missing_restart, missing_restart
        assert not missing_health, missing_health

    def test_e2e_api_uses_readiness_endpoint(self, e2e_compose: dict):
        healthcheck = e2e_compose["services"]["api-e2e"]["healthcheck"]
        probe = " ".join(str(part) for part in healthcheck.get("test", []))
        assert "/healthz/ready" in probe
