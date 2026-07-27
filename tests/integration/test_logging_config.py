"""Tests for `astroml.utils.logging.configure_logging` (issue #195).

The function was added in `0b31e91` without unit-test coverage. These
tests pin its core contracts so a regression doesn't silently disable
structured logging across services.
"""
from __future__ import annotations

import json
import logging
from io import StringIO
from typing import Iterator

import pytest

from astroml.utils import logging as astroml_logging
from astroml.utils.logging import configure_logging


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    """Force-reconfigure between tests so handlers don't pile up and
    the `_CONFIGURED` guard doesn't short-circuit."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    astroml_logging._CONFIGURED = False
    yield
    # Restore prior root logger configuration.
    root.handlers.clear()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    astroml_logging._CONFIGURED = False


def _capture_log(format_: str, level: str = "INFO") -> str:
    """Configure logging into an in-memory StringIO and emit one record.

    Returns the captured handler output as a string.
    """
    buf = StringIO()
    configure_logging(level=level, format=format_, force=True)

    # Replace the root logger's stream handler stream so we can read
    # the bytes back without touching stderr in the test environment.
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = buf

    logging.getLogger("astroml.test").info("hello world", extra={"job": "ingest"})
    for handler in root.handlers:
        handler.flush()
    return buf.getvalue()


def test_text_format_renders_human_readable_line():
    output = _capture_log("text")
    assert "hello world" in output
    assert "astroml.test" in output
    assert "INFO" in output


def test_json_format_emits_one_object_per_line():
    output = _capture_log("json")
    # One trailing newline — strip it before parsing.
    line = output.strip()
    assert line, "expected at least one log line"
    payload = json.loads(line)
    assert payload["message"] == "hello world"
    assert payload["logger"] == "astroml.test"
    assert payload["level"] == "INFO"
    # Structured extra= fields make it through.
    assert payload["job"] == "ingest"


def test_level_filter_drops_lower_severity():
    buf = StringIO()
    configure_logging(level="WARNING", format="text", force=True)
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = buf
    log = logging.getLogger("astroml.level")
    log.info("info-line-should-be-dropped")
    log.warning("warning-line-should-render")
    for handler in root.handlers:
        handler.flush()
    output = buf.getvalue()
    assert "warning-line-should-render" in output
    assert "info-line-should-be-dropped" not in output


def test_reconfigure_is_idempotent_unless_forced():
    """A second call without `force=True` should not duplicate handlers."""
    configure_logging(level="INFO", format="text")
    handler_count_first = len(logging.getLogger().handlers)
    configure_logging(level="DEBUG", format="text")  # no force
    handler_count_second = len(logging.getLogger().handlers)
    assert handler_count_first == handler_count_second


def test_env_var_overrides_pick_up_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ASTROML_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("ASTROML_LOG_FORMAT", "json")

    buf = StringIO()
    configure_logging(force=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG

    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.stream = buf
    logging.getLogger("astroml.env").debug("env-driven")
    for handler in root.handlers:
        handler.flush()
    payload = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["level"] == "DEBUG"
    assert payload["message"] == "env-driven"


def test_unknown_format_falls_back_to_text():
    output = _capture_log("yaml")  # not supported
    # Falls back to text format — line is human-readable, not JSON.
    assert "hello world" in output
    with pytest.raises(json.JSONDecodeError):
        json.loads(output.strip())
