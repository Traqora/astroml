"""Tests for structured logging (issue #568)."""
from __future__ import annotations

import json
import logging
import re
from io import StringIO

import pytest

from astroml.utils.logging import (
    StructuredJsonFormatter,
    clear_correlation_id,
    configure_logging,
    correlation_id,
    get_correlation_id,
    set_correlation_id,
    set_module_log_level,
    get_module_log_level,
)


class TestStructuredJsonFormatter:
    def setup_method(self):
        self.formatter = StructuredJsonFormatter()
        self.logger = logging.getLogger("test_logger")
        self.logger.setLevel(logging.DEBUG)

    def _capture_log(self, record: logging.LogRecord) -> dict:
        output = self.formatter.format(record)
        return json.loads(output)

    def test_has_timestamp(self):
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "test message", (), None)
        result = self._capture_log(record)
        assert "timestamp" in result
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result["timestamp"])

    def test_has_level(self):
        record = self.logger.makeRecord("test_logger", logging.WARNING, "test.py", 10, "warning message", (), None)
        result = self._capture_log(record)
        assert result["level"] == "WARNING"

    def test_has_logger_name(self):
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "msg", (), None)
        result = self._capture_log(record)
        assert result["logger"] == "test_logger"

    def test_has_message(self):
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "hello world", (), None)
        result = self._capture_log(record)
        assert result["message"] == "hello world"

    def test_contains_request_id_when_set(self):
        set_correlation_id("req-123")
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "msg", (), None)
        result = self._capture_log(record)
        assert result["request_id"] == "req-123"
        clear_correlation_id()

    def test_no_request_id_when_not_set(self):
        clear_correlation_id()
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "msg", (), None)
        result = self._capture_log(record)
        assert "request_id" not in result

    def test_includes_contextual_fields(self):
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "msg", (), None)
        record.feature_name = "fraud_detection"
        record.ledger_id = 12345
        result = self._capture_log(record)
        assert result["feature_name"] == "fraud_detection"
        assert result["ledger_id"] == 12345

    def test_output_is_valid_json(self):
        record = self.logger.makeRecord("test_logger", logging.INFO, "test.py", 10, "json test", (), None)
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "json test"

    def test_exception_info_included(self):
        import sys
        try:
            raise ValueError("test error")
        except ValueError:
            exc_info = sys.exc_info()
            record = self.logger.makeRecord("test_logger", logging.ERROR, "test.py", 10, "error occurred", (), exc_info=exc_info)
            result = self._capture_log(record)
            assert "exception" in result
            assert "test error" in result["exception"]


class TestCorrelationId:
    def test_set_and_get(self):
        cid = set_correlation_id("abc-123")
        assert cid == "abc-123"
        assert get_correlation_id() == "abc-123"
        clear_correlation_id()

    def test_auto_generates_uuid(self):
        cid = set_correlation_id()
        assert cid is not None
        assert len(cid) > 0
        clear_correlation_id()

    def test_clear(self):
        set_correlation_id("test-id")
        clear_correlation_id()
        assert get_correlation_id() is None

    def test_context_manager(self):
        with correlation_id("ctx-id") as cid:
            assert cid == "ctx-id"
            assert get_correlation_id() == "ctx-id"
        assert get_correlation_id() is None

    def test_nested_context(self):
        set_correlation_id("outer")
        with correlation_id("inner") as cid:
            assert cid == "inner"
            assert get_correlation_id() == "inner"
        assert get_correlation_id() == "outer"
        clear_correlation_id()


class TestConfigureLogging:
    def test_configure_json_format(self):
        configure_logging(level="DEBUG", format="json", force=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert any(isinstance(h.formatter, StructuredJsonFormatter) for h in root.handlers)

    def test_configure_text_format(self):
        configure_logging(level="INFO", format="text", force=True)
        root = logging.getLogger()
        assert isinstance(root.handlers[0].formatter, logging.Formatter)


class TestModuleLogLevel:
    def test_set_and_get(self):
        set_module_log_level("astroml.test", "DEBUG")
        assert get_module_log_level("astroml.test") == "DEBUG"

    def test_logger_level_is_set(self):
        set_module_log_level("astroml.test_module", "WARNING")
        logger = logging.getLogger("astroml.test_module")
        assert logger.level == logging.WARNING
