"""Centralized structured logging configuration (issues #195, #334, #568, #960).

Standardized log fields:
- timestamp (ISO 8601)
- level
- logger
- message
- request_id (from context)
- feature_name / ledger_id / etc. (contextual fields)

All fields pass through :func:`mask_pii` before being serialized, so common
PII (emails, phone numbers, SSNs, API keys/tokens) is redacted regardless of
which module produced the log record.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from typing import Any, Final

_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "json"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s - %(message)s"

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)
_module_log_levels: dict[str, str] = {}
_CONFIGURED = False

#: Regex patterns for the PII we redact from log output before it is
#: serialized. Applied in order; each substitution runs on the already
#: partially-redacted text so patterns should not overlap.
_PII_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    ("[EMAIL]", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("[SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "[PHONE]",
        re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    ),
    (
        "[CREDIT_CARD]",
        re.compile(r"\b(?:\d[ -]?){13,16}\d\b"),
    ),
]

#: ``key=value`` / ``key: value`` pairs whose value looks like a credential.
_CREDENTIAL_ASSIGNMENT: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b(api[_-]?key|apikey|token|secret|password|passwd|pwd)\b(\s*[:=]\s*)\S+"
)


def mask_pii(text: str) -> str:
    """Redact common PII (emails, phone numbers, SSNs, credentials) from ``text``.

    Args:
        text: Raw string that may contain PII.

    Returns:
        ``text`` with recognized PII patterns replaced by ``[TYPE]``
        placeholders. Non-string or empty input is returned unchanged.
    """
    if not text:
        return text
    masked = _CREDENTIAL_ASSIGNMENT.sub(r"\1\2[REDACTED]", text)
    for placeholder, pattern in _PII_PATTERNS:
        masked = pattern.sub(placeholder, masked)
    return masked


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_pii(record.getMessage()),
        }
        request_id = _correlation_id.get()
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = mask_pii(self.formatException(record.exc_info))
        for key, value in record.__dict__.items():
            if key in payload:
                continue
            if key in {
                "args",
                "asctime",
                "created",
                "exc_info",
                "exc_text",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "msg",
                "name",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "thread",
                "threadName",
                "taskName",
            }:
                continue
            if isinstance(value, str):
                value = mask_pii(value)
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = mask_pii(repr(value))
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | None = None, format: str | None = None, force: bool = False
) -> None:
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    resolved_level = (level or os.environ.get("ASTROML_LOG_LEVEL") or _DEFAULT_LEVEL).upper()
    resolved_format = (format or os.environ.get("ASTROML_LOG_FORMAT") or _DEFAULT_FORMAT).lower()
    handler = logging.StreamHandler(stream=sys.stderr)
    if resolved_format == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved_level)
    _CONFIGURED = True


def set_correlation_id(correlation_id: str | None = None) -> str:
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def clear_correlation_id() -> None:
    _correlation_id.set(None)


def set_module_log_level(module_name: str, level: str) -> None:
    _module_log_levels[module_name] = level.upper()
    logging.getLogger(module_name).setLevel(level.upper())


def get_module_log_level(module_name: str) -> str | None:
    return _module_log_levels.get(module_name)


def configure_module_levels_from_env() -> None:
    env_config = os.environ.get("ASTROML_MODULE_LOG_LEVELS", "")
    if not env_config:
        return
    for config in env_config.split(","):
        config = config.strip()
        if ":" not in config:
            continue
        module, level = config.split(":", 1)
        module = module.strip()
        level = level.strip().upper()
        if module and level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            set_module_log_level(module, level)


class CorrelationId:
    def __init__(self, correlation_id: str | None = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> str:
        self.token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if self.token is not None:
            _correlation_id.reset(self.token)


def sanitize_log_value(value: str, max_length: int = 1000) -> str:
    s = str(value).replace("\r", "").replace("\n", "")
    if len(s) > max_length:
        s = s[:max_length] + "..."
    return s


__all__ = [
    "configure_logging",
    "set_correlation_id",
    "get_correlation_id",
    "clear_correlation_id",
    "set_module_log_level",
    "get_module_log_level",
    "configure_module_levels_from_env",
    "CorrelationId",
    "correlation_id",
    "sanitize_log_value",
    "mask_pii",
]

# Backward-compatible alias
correlation_id = CorrelationId
