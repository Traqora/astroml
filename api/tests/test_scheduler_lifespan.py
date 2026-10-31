"""Regression tests for the batch scheduler startup/shutdown logging (issue #844).

``api.app``'s ``lifespan`` context manager used to swallow a failure to
start or stop the batch scoring scheduler with a bare ``except Exception:
pass`` and no logging at all. The API would finish startup looking
perfectly healthy with no fraud-scoring scheduler actually running, and
nothing in the logs said why. These tests assert that a failure in either
direction is now logged at ``ERROR`` instead of vanishing silently.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_scheduler_start_failure_is_logged(caplog):
    """A ``start_scheduler`` exception during startup is logged, not swallowed."""
    from api.app import app, lifespan

    with (
        patch(
            "astroml.api.scheduler.start_scheduler",
            side_effect=RuntimeError("boom: scheduler config invalid"),
        ),
        patch("astroml.api.scheduler.build_score_fn", return_value=lambda account_id: 0.0),
        patch.dict("os.environ", {"DISABLE_SCHEDULER": "", "DISABLE_WS_POLLER": "true"}),
        caplog.at_level(logging.ERROR, logger="api.app"),
    ):
        async with lifespan(app):
            pass

    assert any(
        "Failed to start batch scoring scheduler" in record.message for record in caplog.records
    ), f"expected a logged scheduler-start failure, got: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_scheduler_stop_failure_is_logged(caplog):
    """A ``stop_scheduler`` exception during shutdown is logged, not swallowed."""
    from api.app import app, lifespan

    async def _raise_on_stop():
        raise RuntimeError("boom: scheduler task already gone")

    with (
        patch.dict("os.environ", {"DISABLE_SCHEDULER": "true", "DISABLE_WS_POLLER": "true"}),
        patch("astroml.api.scheduler.stop_scheduler", side_effect=_raise_on_stop),
        caplog.at_level(logging.ERROR, logger="api.app"),
    ):
        async with lifespan(app):
            pass

    assert any(
        "Failed to stop batch scoring scheduler" in record.message for record in caplog.records
    ), f"expected a logged scheduler-stop failure, got: {[r.message for r in caplog.records]}"


@pytest.mark.asyncio
async def test_scheduler_start_success_logs_nothing_at_error(caplog):
    """The happy path (scheduler disabled) must not spuriously log an error."""
    from api.app import app, lifespan

    with (
        patch.dict("os.environ", {"DISABLE_SCHEDULER": "true", "DISABLE_WS_POLLER": "true"}),
        caplog.at_level(logging.ERROR, logger="api.app"),
    ):
        async with lifespan(app):
            pass

    assert not any(
        "Failed to start batch scoring scheduler" in record.message
        or "Failed to stop batch scoring scheduler" in record.message
        for record in caplog.records
    )
