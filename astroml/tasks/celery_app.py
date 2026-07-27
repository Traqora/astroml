"""Celery application instance for AstroML — issue #296.

Broker and result backend are driven by environment variables so the same
image can be used in Docker Compose (redis service) and in local dev
(localhost:6379).
"""
from __future__ import annotations

import os

from celery import Celery

app = Celery(
    "astroml",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
