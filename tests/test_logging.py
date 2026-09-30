"""Logging tests for #506."""

from __future__ import annotations

import logging

import pytest

from astroml.features import feature_store
from astroml.utils.logging import StructuredJsonFormatter, configure_logging


@pytest.fixture(autouse=True)
def configure():
    configure_logging(level="DEBUG", force=True)


def test_module_logger_emits_info(caplog: pytest.LogCaptureFixture):
    logger = logging.getLogger("astroml.features.feature_store")
    with caplog.at_level(logging.INFO, logger="astroml.features.feature_store"):
        logger.info("feature store init")
    assert "feature store init" in caplog.text


def test_json_formatter_produces_valid_json():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    out = formatter.format(record)
    assert '"message": "hello world"' in out
    assert '"level": "INFO"' in out
