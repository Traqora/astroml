"""Tests for explicit external-service test markers."""

from pathlib import Path

import pytest


def test_external_marker_is_registered() -> None:
    """The pytest configuration declares the supported external markers."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"external:' in text
    assert '"external_service:' in text


def test_live_service_tests_are_marked() -> None:
    """Known live-service test modules/classes explicitly opt in to external."""
    root = Path(__file__).resolve().parents[1]
    checks = {
        "tests/contract/test_stellar_horizon.py": "pytestmark = pytest.mark.external",
        "tests/e2e/test_deployment_smoke.py": "pytestmark = pytest.mark.external",
        "tests/contract/test_mlflow.py": "@pytest.mark.external",
        "tests/infrastructure/test_alembic_migrations.py": "@pytest.mark.external",
        "tests/performance/test_benchmarks.py": "@pytest.mark.external",
    }

    for relative_path, marker in checks.items():
        assert marker in (root / relative_path).read_text(encoding="utf-8")


def test_external_marker_skip_decision(monkeypatch) -> None:
    """The opt-in switch can be enabled through configuration or environment."""
    from conftest import external_tests_enabled

    class Config:
        def getoption(self, name: str) -> bool:
            assert name == "--run-external"
            return False

    monkeypatch.delenv("ASTROML_RUN_EXTERNAL_TESTS", raising=False)
    assert external_tests_enabled(Config()) is False

    monkeypatch.setenv("ASTROML_RUN_EXTERNAL_TESTS", "1")
    assert external_tests_enabled(Config()) is True
