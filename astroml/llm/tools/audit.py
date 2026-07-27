"""Execution audit log for tool invocations."""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ToolAuditLog:
    """Records all tool invocations with metadata."""

    def __init__(self):
        self._entries: list[dict[str, Any]] = []

    def record(
        self,
        tool_name: str,
        params: dict[str, Any],
        result: Any,
        user_id: str | None = None,
        duration: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record a tool invocation."""
        entry = {
            "tool_name": tool_name,
            "params": params,
            "user_id": user_id,
            "duration": duration,
            "error": error,
            "timestamp": time.time(),
        }
        if error is None:
            entry["result"] = result
        else:
            entry["result"] = None
        self._entries.append(entry)
        logger.info(
            "Tool call: %s by %s (duration=%.2fs, error=%s)",
            tool_name, user_id, duration, error or "none",
        )

    def get_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._entries[-limit:]

    def clear(self) -> None:
        self._entries.clear()
