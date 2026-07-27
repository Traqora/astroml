"""Core batch processing engine for LLM backfill jobs."""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable

from astroml.llm.providers.base import LLMProvider
from .strategies import BatchingStrategy, FixedSizeStrategy
from .checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processes items in batches with rate limiting and checkpointing."""

    def __init__(
        self,
        provider: LLMProvider,
        checkpoint_mgr: CheckpointManager,
        strategy: BatchingStrategy | None = None,
        rate_per_minute: int = 60,
    ):
        self._provider = provider
        self._checkpoint = checkpoint_mgr
        self._strategy = strategy or FixedSizeStrategy(100)
        self._min_interval = 60.0 / rate_per_minute
        self._last_call = 0.0

    async def process_range(
        self,
        items: list[Any],
        process_fn: Callable[[Any, LLMProvider], Awaitable[Any]],
    ) -> list[dict[str, Any]]:
        """Process a list of items in batches with rate limiting."""
        results: list[dict[str, Any]] = []
        batch_size = self._strategy.get_batch_size()

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_result = await self._process_batch(batch, process_fn)
            results.extend(batch_result)

        return results

    async def _process_batch(
        self,
        batch: list[Any],
        process_fn: Callable[[Any, LLMProvider], Awaitable[Any]],
    ) -> list[dict[str, Any]]:
        """Process a single batch with rate limiting."""
        await self._rate_limit()

        tasks = [process_fn(item, self._provider) for item in batch]
        batch_results = []

        try:
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
            for item, outcome in zip(batch, outcomes):
                if isinstance(outcome, Exception):
                    logger.warning("Item %s failed: %s", item, outcome)
                    self._checkpoint.record_failure()
                    batch_results.append({"item": item, "status": "failed", "error": str(outcome)})
                else:
                    self._checkpoint.record_success()
                    batch_results.append({"item": item, "status": "completed", "result": outcome})
            self._strategy.on_success()
        except Exception as e:
            logger.error("Batch failed: %s", e)
            self._strategy.on_failure(e)
            raise

        return batch_results

    async def _rate_limit(self) -> None:
        """Ensure minimum interval between API calls."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_call = time.time()
