"""Regression tests for CI matrix configuration (GitHub Issue #156).

Verifies that:
- The `gpu` pytest marker is registered in pyproject.toml so pytest
  never emits PytestUnknownMarkWarning when running GPU-gated tests.
- The `e2e` marker is also registered (used by #163).
- The CI workflow file contains a CPU-only run that excludes GPU tests,
  preventing accidental CUDA imports from breaking standard CI.
"""
from __future__ import annotations

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).parent.parent


def _pyproject_markers() -> list[str]:
    """Extract registered marker names from pyproject.toml."""
    pyproject = _ROOT / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not found")
    text = pyproject.read_text()
    markers: list[str] = []
    in_markers = False
    for line in text.splitlines():
        stripped = line.strip()
        if "markers" in stripped and "[" in stripped:
            in_markers = True
            continue
        if in_markers:
            if stripped.startswith("]"):
                break
            if stripped.startswith('"') or stripped.startswith("'"):
                name = stripped.strip('"\'').split(":")[0].strip()
                markers.append(name)
    return markers


def test_gpu_marker_registered() -> None:
    """The `gpu` marker must be declared in pyproject.toml (#156)."""
    markers = _pyproject_markers()
    assert "gpu" in markers, (
        "pytest marker 'gpu' is not registered in pyproject.toml — "
        "add it under [tool.pytest.ini_options] markers to silence PytestUnknownMarkWarning"
    )


def test_e2e_marker_registered() -> None:
    """The `e2e` marker must be declared in pyproject.toml (#163)."""
    markers = _pyproject_markers()
    assert "e2e" in markers, (
        "pytest marker 'e2e' is not registered in pyproject.toml"
    )


def test_ci_workflow_excludes_gpu_on_cpu_runs() -> None:
    """pytest.yml must run CPU jobs with `-m 'not gpu'` (#156)."""
    workflow = _ROOT / ".github" / "workflows" / "pytest.yml"
    if not workflow.exists():
        pytest.skip(".github/workflows/pytest.yml not found")
    text = workflow.read_text()
    assert "not gpu" in text, (
        "CI CPU job must pass `-m 'not gpu'` to pytest so GPU-gated tests "
        "are not attempted on CPU-only runners"
    )


def test_ci_workflow_has_gpu_flavor() -> None:
    """pytest.yml must define a gpu flavor in the matrix (#156)."""
    workflow = _ROOT / ".github" / "workflows" / "pytest.yml"
    if not workflow.exists():
        pytest.skip(".github/workflows/pytest.yml not found")
    text = workflow.read_text()
    assert "gpu" in text, (
        "CI matrix must include a gpu flavor entry"
    )


def test_ci_gpu_job_is_optional() -> None:
    """GPU CI job must be marked continue-on-error so CPU CI still passes (#156)."""
    workflow = _ROOT / ".github" / "workflows" / "pytest.yml"
    if not workflow.exists():
        pytest.skip(".github/workflows/pytest.yml not found")
    text = workflow.read_text()
    assert "continue-on-error" in text, (
        "GPU CI job must set continue-on-error: true so the matrix passes "
        "on GitHub-hosted (CPU-only) runners"
    )
