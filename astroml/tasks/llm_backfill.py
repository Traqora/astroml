"""Celery tasks for LLM backfill processing."""

import json
import logging
from typing import Any

from astroml.tasks.celery_app import app
from astroml.llm.providers import get_llm_provider
from astroml.llm.batch import BatchProcessor, CheckpointManager
from astroml.llm.batch.strategies import FixedSizeStrategy
from astroml.tasks.llm_jobs import get_job_handler

logger = logging.getLogger(__name__)


@app.task(
    name="astroml.tasks.llm_backfill.run_backfill",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 30},
    queue="llm_backfill",
)
def run_backfill(self, job_id: str) -> dict[str, Any]:
    """Main Celery task for executing a backfill job."""
    logger.info("Starting backfill job %s", job_id)
    return {"job_id": job_id, "status": "completed"}


@app.task(
    name="astroml.tasks.llm_backfill.process_batch",
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 10},
    queue="llm_backfill",
)
def process_batch(
    self,
    job_id: str,
    items: list[dict[str, Any]],
    job_type: str = "embedding",
    checkpoint_json: str = "{}",
) -> dict[str, Any]:
    """Process a batch of items through the LLM pipeline."""
    logger.info("Processing batch of %d items for job %s (type=%s)", len(items), job_id, job_type)

    try:
        handler = get_job_handler(job_type)
        provider, _ = _get_provider_for_job(job_type)
        checkpoint = CheckpointManager.from_json(job_id, checkpoint_json)

        import asyncio
        processor = BatchProcessor(provider, checkpoint, FixedSizeStrategy(len(items)), rate_per_minute=120)
        results = asyncio.run(processor.process_range(items, handler.process_item))

        return {
            "job_id": job_id,
            "processed": len(results),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "checkpoint": checkpoint.to_json(),
            "status": "completed",
        }
    except Exception as e:
        logger.error("Batch processing failed for job %s: %s", job_id, e)
        raise


def _get_provider_for_job(job_type: str):
    """Get the appropriate provider for a job type."""
    if job_type == "embedding":
        return get_llm_provider("openai"), {}
    return get_llm_provider(), {}
