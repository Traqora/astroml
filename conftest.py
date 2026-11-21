"""Pytest configuration for the AstroML test suite.

Ensures the repository root is importable so ``import astroml`` resolves when
pytest is invoked as a console script (``pytest -v``) from the project root,
without requiring ``pip install -e .`` first.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    """Add the opt-in switch for tests that require external services."""
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="run tests marked with 'external' (skipped by default)",
    )


def external_tests_enabled(config) -> bool:
    """Return whether external-service tests should run for this invocation."""
    return bool(
        config.getoption("--run-external")
        or os.environ.get("ASTROML_RUN_EXTERNAL_TESTS") == "1"
    )


def pytest_collection_modifyitems(config, items) -> None:
    """Skip external-service tests unless they are explicitly enabled."""
    if external_tests_enabled(config):
        return

    skip_external = pytest.mark.skip(
        reason="external-service test; pass --run-external or set ASTROML_RUN_EXTERNAL_TESTS=1"
    )
    for item in items:
        if (
            item.get_closest_marker("external") is not None
            or item.get_closest_marker("external_service") is not None
        ):
            item.add_marker(skip_external)
