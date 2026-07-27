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
import sys
import uuid
from typing import Any, Optional


_DEFAULT_LEVEL = "INFO"
_DEFAULT_FORMAT = "json"
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s - %(message)s"

_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("correlation_id", default=None)
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
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in payload:
                continue
            if key in {"args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno", "lineno", "module", "msecs", "msg", "name", "pathname", "process", "processName", "relativeCreated", "stack_info", "thread", "threadName", "taskName"}:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, default=str)


def configure_logging(level: Optional[str] = None, format: Optional[str] = None, force: bool = False) -> None:
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


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id.set(correlation_id)
    return correlation_id


def get_correlation_id() -> Optional[str]:
    return _correlation_id.get()


def clear_correlation_id() -> None:
    _correlation_id.set(None)


def set_module_log_level(module_name: str, level: str) -> None:
    _module_log_levels[module_name] = level.upper()
    logging.getLogger(module_name).setLevel(level.upper())


def get_module_log_level(module_name: str) -> Optional[str]:
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


class correlation_id:
    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.token = None

    def __enter__(self) -> str:
        self.token = _correlation_id.set(self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.token is not None:
            _correlation_id.reset(self.token)


def sanitize_log_value(value: str, max_length: int = 1000) -> str:
    s = str(value).replace("\r", "").replace("\n", "")
    if len(s) > max_length:
        s = s[:max_length] + "..."
    return s


__all__ = [
    "configure_logging", "set_correlation_id", "get_correlation_id",
    "clear_correlation_id", "set_module_log_level", "get_module_log_level",
    "configure_module_levels_from_env", "correlation_id", "sanitize_log_value",
]
