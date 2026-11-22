"""Secrets must not reach logs or exception messages (issue #1245)."""

from __future__ import annotations

import json
import logging
import sys

from astroml.utils.logging import (
    SecretRedactionFilter,
    StructuredJsonFormatter,
    sanitize_log_value,
)

SECRET_MARKERS = (
    "eyJhbGciOiJIUzI1NiJ9.abcdefghij.klmnopqrst",
    "ak_abcdef1234567890",
    "hunter2",
    "supersecretvalue",
)


def make_record(message: str, args=(), exc_info=None) -> logging.LogRecord:
    return logging.LogRecord("tests.secrets", logging.ERROR, __file__, 1, message, args, exc_info)


def make_exc_info() -> tuple:
    try:
        raise RuntimeError(
            "connect failed postgresql://svc:hunter2@db.internal:5432/app "
            "token=ak_abcdef1234567890"
        )
    except RuntimeError:
        return sys.exc_info()


def test_redacts_bearer_token_and_jwt() -> None:
    value = sanitize_log_value("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.abcdefghij.klmnopqrst")
    assert "eyJhbGciOiJIUzI1NiJ9" not in value
    assert "[REDACTED]" in value
    assert value.startswith("Authorization:")


def test_redacts_api_key_prefix() -> None:
    assert "ak_abcdef1234567890" not in sanitize_log_value("key=ak_abcdef1234567890 user=bob")


def test_redacts_secret_assignments() -> None:
    for snippet in (
        "SECRET_KEY=supersecretvalue",
        "JWT_SECRET_KEY=supersecretvalue",
        "password=supersecretvalue",
        "api_key: supersecretvalue",
    ):
        redacted = sanitize_log_value(snippet)
        assert "supersecretvalue" not in redacted, snippet
        assert "[REDACTED]" in redacted, snippet


def test_redacts_credentials_in_urls() -> None:
    redacted = sanitize_log_value("postgresql://svc:hunter2@db.internal:5432/app")
    assert "hunter2" not in redacted
    assert "postgresql://svc:[REDACTED]@db.internal:5432/app" == redacted


def test_non_secret_text_keeps_existing_behavior() -> None:
    assert sanitize_log_value("ingested 42 ledgers") == "ingested 42 ledgers"
    assert sanitize_log_value("first\nsecond\rthird") == "firstsecondthird"


def test_long_values_are_still_truncated() -> None:
    redacted = sanitize_log_value("a" * 2000, max_length=50)
    assert len(redacted) == 53
    assert redacted.endswith("...")


def test_filter_sanitizes_formatted_message_and_traceback() -> None:
    record = make_record(
        "scoring failed for %s with token %s",
        ("account-1", "ak_abcdef1234567890"),
        exc_info=make_exc_info(),
    )
    assert SecretRedactionFilter().filter(record) is True

    assert "ak_abcdef1234567890" not in record.getMessage()
    assert "account-1" in record.getMessage()
    for marker in SECRET_MARKERS:
        assert marker not in (record.exc_text or ""), marker
    assert record.exc_info is not None


def test_json_formatter_uses_the_redacted_exception_text() -> None:
    record = make_record(
        "Scoring failed: %s", ("token=ak_abcdef1234567890",), exc_info=make_exc_info()
    )
    SecretRedactionFilter().filter(record)
    payload = json.loads(StructuredJsonFormatter().format(record))

    assert "[REDACTED]" in payload["exception"]
    for marker in SECRET_MARKERS:
        assert marker not in payload["exception"], marker
        assert marker not in payload["message"], marker


def test_text_formatter_uses_the_redacted_exception_text() -> None:
    record = make_record("failed token=ak_abcdef1234567890", exc_info=make_exc_info())
    SecretRedactionFilter().filter(record)
    output = logging.Formatter("%(message)s").format(record)

    for marker in SECRET_MARKERS:
        assert marker not in output, marker
    assert "RuntimeError" in output


def test_filter_never_breaks_on_odd_records() -> None:
    record = make_record("%s", ({"nested": "value"},))
    assert SecretRedactionFilter().filter(record) is True
    assert "nested" in record.getMessage()
