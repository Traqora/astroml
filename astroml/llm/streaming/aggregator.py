"""Aggregates multiple streaming sources into a unified output stream."""
from __future__ import annotations

import asyncio
import logging
from typing import List, AsyncIterator, Dict, Any

from astroml.llm.streaming.buffer import StreamBuffer

logger = logging.getLogger(__name__)


class StreamAggregator:
    """Combines/aggregates tokens from multiple streaming LLM outputs into one."""

    def __init__(self, buffer_max_size: int = 200):
        self.buffer = StreamBuffer[Dict[str, Any]](max_size=buffer_max_size)
        self._tasks: List[asyncio.Task] = []
        self._active_sources = 0

    async def add_source(self, source_id: str, stream: AsyncIterator[str]) -> None:
        """Add an asynchronous streaming source to be aggregated."""
        self._active_sources += 1
        task = asyncio.create_task(self._consume_source(source_id, stream))
        self._tasks.append(task)

    async def _consume_source(self, source_id: str, stream: AsyncIterator[str]) -> None:
        """Helper task to consume a single source and push token chunks to the shared buffer."""
        try:
            async for chunk in stream:
                if self.buffer.is_aborted:
                    break
                await self.buffer.push({
                    "source_id": source_id,
                    "token": chunk,
                    "finished": False
                })
        except Exception as e:
            logger.error("Error consuming source %s in aggregator: %s", source_id, e)
        finally:
            self._active_sources -= 1
            if self._active_sources == 0:
                # Last source finished, push sentinel done token
                await self.buffer.push({
                    "source_id": source_id,
                    "token": None,
                    "finished": True
                })

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        """Yield aggregated items from all sources as they arrive."""
        try:
            while not (self.buffer.is_aborted or (self._active_sources == 0 and self.buffer.size == 0)):
                item = await self.buffer.get()
                yield item
                self.buffer.task_done()
                if item.get("finished", False) and self._active_sources == 0:
                    break
        finally:
            self.abort()

    def abort(self) -> None:
        """Abort aggregation and cancel all running tasks."""
        self.buffer.abort()
        for t in self._tasks:
            if not t.done():
                t.cancel()
