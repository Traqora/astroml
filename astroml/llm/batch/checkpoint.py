"""Progress checkpointing for backfill jobs."""

import json
import logging
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages job progress checkpointing via in-memory dict."""

    def __init__(self, job_id: str, initial_checkpoint: Optional[dict] = None):
        self._job_id = job_id
        self._state: dict[str, Any] = initial_checkpoint or {}
        self._processed = 0
        self._failed = 0
        self._started_at = datetime.utcnow()

    def save(self, position: str | int) -> None:
        """Save a checkpoint position."""
        self._state["last_position"] = str(position)
        self._state["updated_at"] = datetime.utcnow().isoformat()

    def load(self) -> str | int | None:
        """Load the last checkpoint position."""
        pos = self._state.get("last_position")
        if pos is None:
            return None
        try:
            return int(pos)
        except (ValueError, TypeError):
            return str(pos)

    def record_success(self, count: int = 1) -> None:
        self._processed += count

    def record_failure(self, count: int = 1) -> None:
        self._failed += count

    def get_progress(self) -> dict[str, Any]:
        return {
            "processed": self._processed,
            "failed": self._failed,
            "last_position": self._state.get("last_position"),
            "elapsed_seconds": (datetime.utcnow() - self._started_at).total_seconds(),
        }

    def to_json(self) -> str:
        self._state["_processed"] = self._processed
        self._state["_failed"] = self._failed
        return json.dumps(self._state)

    @classmethod
    def from_json(cls, job_id: str, raw: str) -> "CheckpointManager":
        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            state = {}
        mgr = cls(job_id, state)
        mgr._processed = state.get("_processed", 0)
        mgr._failed = state.get("_failed", 0)
        return mgr
