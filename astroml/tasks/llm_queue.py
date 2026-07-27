"""Queue management for LLM backfill jobs using Celery."""

from typing import Any

from astroml.tasks.celery_app import app


def enqueue_backfill(job_id: str) -> None:
    """Enqueue a full backfill job to the Celery worker."""
    app.send_task(
        "astroml.tasks.llm_backfill.run_backfill",
        args=[job_id],
        queue="llm_backfill",
    )


def enqueue_batch(
    job_id: str,
    items: list[dict[str, Any]],
    job_type: str,
) -> None:
    """Enqueue a single batch of items for processing."""
    app.send_task(
        "astroml.tasks.llm_backfill.process_batch",
        args=[job_id, items, job_type],
        queue="llm_backfill",
    )
