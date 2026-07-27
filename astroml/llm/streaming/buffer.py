"""Buffering and backpressure handling for slow consumers."""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, TypeVar, Generic

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StreamBuffer(Generic[T]):
    """
    Asynchronous buffer queue with backpressure.
    If the queue grows too large, pushing is blocked or slower to handle backpressure.
    """

    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)
        self._max_size = max_size
        self._aborted = False

    async def push(self, item: T) -> bool:
        """Push an item to the buffer. Respects queue capacity limit (backpressure)."""
        if self._aborted:
            return False
            
        try:
            # If queue is full, this will wait asynchronously, applying backpressure
            await self._queue.put(item)
            return True
        except Exception as e:
            logger.error("Failed to push item to stream buffer: %s", e)
            return False

    async def get(self) -> T:
        """Retrieve the next item from the buffer."""
        return await self._queue.get()

    def task_done(self) -> None:
        """Mark task as done in the underlying queue."""
        self._queue.task_done()

    def abort(self) -> None:
        """Abort/cancel the current stream."""
        self._aborted = True
        # Clear existing items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    @property
    def size(self) -> int:
        """Return current size of the buffer."""
        return self._queue.qsize()

    @property
    def is_full(self) -> bool:
        """Check if the queue is full, indicating memory pressure / slow consumer."""
        return self._queue.full()

    @property
    def is_aborted(self) -> bool:
        """Check if the stream has been aborted."""
        return self._aborted
