"""JSON logging regression tests for astroml.features.graph.snapshot (#943).

Verifies that the log records emitted by ``snapshot.py`` — the joblib
fallback warning and the parallel-build info line — serialize cleanly
through :class:`~astroml.utils.logging.StructuredJsonFormatter`, the
formatter used in production (``ASTROML_LOG_FORMAT=json``). A prior gap
left this module's logging entirely untested even though
``astroml/utils/logging.py`` has its own formatter-level test coverage.
"""

from __future__ import annotations

import builtins
import json
import logging
from datetime import datetime, timezone

import pytest

from astroml.features.graph.snapshot import parallel_build_snapshots
from astroml.utils.logging import StructuredJsonFormatter

_LOGGER_NAME = "astroml.features.graph.snapshot"


def _format_records_as_json(records: list[logging.LogRecord]) -> list[dict]:
    """Run each captured record through the production JSON formatter."""
    formatter = StructuredJsonFormatter()
    return [json.loads(formatter.format(record)) for record in records]


class _FakeResult:
    """Minimal stand-in for a SQLAlchemy Result — no real DB required."""

    def __init__(self, rows=(), scalar_value=None):
        self._rows = rows
        self._scalar = scalar_value

    def yield_per(self, _size):
        return iter(self._rows)

    def scalar(self):
        return self._scalar


class _FakeSession:
    def __init__(self, result: _FakeResult):
        self._result = result

    def execute(self, _query):
        return self._result

    def close(self):
        pass


def test_joblib_missing_warning_emits_valid_json(monkeypatch, caplog):
    """The joblib-not-installed fallback warning must be valid JSON in prod format."""
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError("No module named 'joblib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    # No rows in range -> iter_db_snapshots (the sequential fallback) yields nothing,
    # so we don't need to fake a scalar min-timestamp lookup beyond an empty window.
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t_now = t0
    fake_session = _FakeSession(_FakeResult(rows=[]))
    monkeypatch.setattr("astroml.db.session.get_session", lambda: fake_session)

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        result = parallel_build_snapshots(t0=t0, t_now=t_now)

    # Falls back to the sequential path and still returns a (possibly empty) list.
    assert result == []

    warning_records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert warning_records, "expected a warning to be logged when joblib is missing"

    payloads = _format_records_as_json(warning_records)
    assert any("joblib not installed" in p["message"] for p in payloads)
    for payload in payloads:
        assert payload["level"] == "WARNING"
        assert payload["logger"] == _LOGGER_NAME
        assert "timestamp" in payload


def test_json_formatter_handles_percent_style_args_from_snapshot_logger(caplog):
    """The %-style logger.info calls in snapshot.py must render fully in JSON output.

    Regression guard: StructuredJsonFormatter calls record.getMessage(), which
    performs %-interpolation; a naive formatter that used record.msg directly
    would leak the raw "%d"/"%s" placeholders into the JSON payload instead of
    the interpolated values.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
        logger.info(
            "Parallel snapshot build: %d windows, n_jobs=%d, batch_size=%s",
            3,
            -1,
            "all",
        )

    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(records) == 1

    payload = _format_records_as_json(records)[0]
    assert payload["message"] == "Parallel snapshot build: 3 windows, n_jobs=-1, batch_size=all"
    assert "%d" not in payload["message"]
    assert "%s" not in payload["message"]

    # Round-trips through json.dumps/json.loads without error or data loss.
    reserialized = json.dumps(payload, default=str)
    assert json.loads(reserialized) == payload


@pytest.mark.parametrize("level_name", ["INFO", "WARNING"])
def test_snapshot_log_records_are_always_json_serializable(caplog, level_name):
    """Any record from the snapshot logger must survive the JSON formatter.

    Guards against future log calls in this module passing non-JSON-safe
    objects (e.g. datetimes, sets) as extra fields, which StructuredJsonFormatter
    handles via `repr()` fallback rather than raising.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    level = getattr(logging, level_name)
    with caplog.at_level(level, logger=_LOGGER_NAME):
        logger.log(level, "synthetic %s event", "test")

    records = [r for r in caplog.records if r.name == _LOGGER_NAME]
    assert len(records) == 1

    # Must not raise — this is the core "prod JSON logging" contract.
    payload = _format_records_as_json(records)[0]
    assert payload["level"] == level_name
    assert payload["message"] == "synthetic test event"
