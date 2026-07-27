"""Backfill job scheduler — enqueue and manage jobs."""

import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BackfillScheduler:
    """Manages backfill job lifecycle: create, pause, resume, list."""

    def __init__(self):
        self._jobs: dict[str, dict[str, Any]] = {}

    def create_job(
        self,
        job_type: str,
        total_items: int,
        config: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Create a new backfill job record."""
        job_id = uuid.uuid4().hex[:16]
        job = {
            "id": job_id,
            "job_type": job_type,
            "status": "pending",
            "total_items": total_items,
            "processed_items": 0,
            "failed_items": 0,
            "checkpoint": "{}",
            "config": json.dumps(config or {}),
            "cost_spent": 0.0,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error_message": None,
        }
        self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        return list(self._jobs.values())

    def update_job(self, job_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.update(updates)
        return job

    def pause_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return self.update_job(job_id, {"status": "paused"})

    def resume_job(self, job_id: str) -> Optional[dict[str, Any]]:
        return self.update_job(job_id, {"status": "running"})


_scheduler: BackfillScheduler | None = None


def get_scheduler() -> BackfillScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = BackfillScheduler()
    return _scheduler
