"""Centralized structured logging configuration (issues #195, #334, #568).

Standardized log fields:
- timestamp (ISO 8601)
- level
- logger
- message
- request_id (from context)
- feature_name / ledger_id / etc. (contextual fields)
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import uuid
from typing import Any

_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "json"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s - %(message)s"
_EXCEPTION_FORMATTER = logging.Formatter()

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)
_module_log_levels: dict[str, str] = {}
_CONFIGURED = False


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = _correlation_id.get()
        if request_id:
            payload["request_id"] = request_id
        exception_text = getattr(record, "exc_text", None)
        if exception_text:
            payload["exception"] = exception_text
        elif record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
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
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
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
    handler.addFilter(SecretRedactionFilter())
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
    def __init__(self, correlation_id: str | None = None) -> None:
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> str:
        self.token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if self.token is not None:
            _correlation_id.reset(self.token)


#: Patterns for values that must never reach a log sink (issue #1245).
#: Compiled once so the hot logging path only pays for a substitution when a
#: marker is actually present.
_SECRET_MARKERS = (
    "bearer",
    "ak_",
    "eyj",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "api_key",
    "apikey",
    "private_key",
    "private-key",
    "authorization",
    "@",
)

_SECRET_PATTERNS = (
    (re.compile(r"(?i)(://[^:/\s]+:)[^@\s]+(@)"), r"\1[REDACTED]\2"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{6,}"), "Bearer [REDACTED]"),
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"),
        "[REDACTED]",
    ),
    (re.compile(r"\bak_[A-Za-z0-9_-]{6,}"), "[REDACTED]"),
    (
        re.compile(
            r"(?i)\b(password|passwd|pwd|secret(?:[_-]?key)?|"
            r"jwt[_-]?secret(?:[_-]?key)?|token|api[_-]?key|authorization|"
            r"credential|private[_-]?key)(\s*[=:]\s*)([^\s,;]+)"
        ),
        r"\1\2[REDACTED]",
    ),
)


def sanitize_log_value(value: str, max_length: int = 1000) -> str:
    """Strip control characters, bound length and redact credentials.

    The signature is unchanged from the original helper so every existing
    caller gains redaction without modification. Non-secret values pass through
    untouched apart from newline removal and truncation.
    """
    s = str(value).replace("\r", "").replace("\n", "")
    lowered = s.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        for pattern, replacement in _SECRET_PATTERNS:
            s = pattern.sub(replacement, s)
    if len(s) > max_length:
        s = s[:max_length] + "..."
    return s


class SecretRedactionFilter(logging.Filter):
    """Sanitize every record before a handler formats it.

    Attached once in :func:`configure_logging`, so both the JSON and text
    formatters and all ``exc_info=True`` call sites are covered without
    touching dozens of individual log statements.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = sanitize_log_value(record.getMessage())
            record.args = ()
            if record.exc_info:
                formatted = _EXCEPTION_FORMATTER.formatException(record.exc_info)
                record.exc_text = sanitize_log_value(formatted)
            elif record.exc_text:
                record.exc_text = sanitize_log_value(record.exc_text)
        except Exception:  # noqa: BLE001 - logging must never break the request
            return True
        return True


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
    "SecretRedactionFilter",
]

# Backward-compatible alias
correlation_id = CorrelationId
