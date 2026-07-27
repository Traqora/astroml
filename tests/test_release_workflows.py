"""Regression tests for release automation artifacts."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _read(relpath: str) -> str:
    path = ROOT / relpath
    assert path.exists(), f"Expected {relpath} to exist"
    return path.read_text(encoding="utf-8")


def test_ci_workflow_exists_and_runs_checks() -> None:
    text = _read(".github/workflows/ci.yml")
    assert "black --check" in text
    assert "ruff check" in text
    assert "pytest tests/" in text


def test_docker_workflow_exists_and_builds_multiarch() -> None:
    text = _read(".github/workflows/docker.yml")
    assert "docker/build-push-action" in text
    assert "linux/amd64,linux/arm64" in text
    assert "ghcr.io" in text


def test_validate_build_script_exists() -> None:
    text = _read("scripts/validate_build.sh")
    assert "python -m build" in text
    assert "twine check dist/*" in text


def test_makefile_has_validate_build_target() -> None:
    text = _read("Makefile")
    assert "validate-build" in text
