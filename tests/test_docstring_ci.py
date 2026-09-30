"""Tests for the repository-wide docstring validation wiring."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    """Return a UTF-8 repository file as text."""
    return (ROOT / path).read_text(encoding="utf-8")


def test_interrogate_config_enforces_documentation_baseline() -> None:
    """Ensure the checked-in configuration keeps the current coverage floor."""
    pyproject = _read("pyproject.toml")

    assert "[tool.interrogate]" in pyproject
    assert "fail-under = 68.0" in pyproject
    assert "ignore-module = true" in pyproject


def test_docstring_check_is_available_in_every_developer_workflow() -> None:
    """Ensure local, pre-commit, and CI workflows all run Interrogate."""
    assert "interrogate>=1.7.0,<2.0" in _read("requirements-dev.txt")
    assert "make lint-docs" in _read("CONTRIBUTING.md")
    assert "interrogate astroml/ api/ tests/" in _read("Makefile")
    assert "id: docstring-coverage" in _read(".pre-commit-config.yaml")
    assert "interrogate astroml api tests" in _read(".github/workflows/ci.yml")
