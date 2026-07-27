"""Batching strategies for backfill processing."""

from abc import ABC, abstractmethod
from typing import Any


class BatchingStrategy(ABC):
    """Abstract base for batching strategies."""

    @abstractmethod
    def get_batch_size(self) -> int:
        """Return the current batch size."""
        pass

    @abstractmethod
    def on_success(self) -> None:
        """Called after a successful batch."""
        pass

    @abstractmethod
    def on_failure(self, error: Exception) -> None:
        """Called after a failed batch."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset strategy to initial state."""
        pass


class FixedSizeStrategy(BatchingStrategy):
    """Always use the same batch size."""

    def __init__(self, size: int = 100):
        self._size = size

    def get_batch_size(self) -> int:
        return self._size

    def on_success(self) -> None:
        pass

    def on_failure(self, error: Exception) -> None:
        pass

    def reset(self) -> None:
        pass


class AdaptiveStrategy(BatchingStrategy):
    """Start at max size, shrink on errors, grow on success."""

    def __init__(self, min_size: int = 10, max_size: int = 100, shrink_factor: float = 0.5, grow_factor: float = 1.1):
        self._min = min_size
        self._max = max_size
        self._shrink = shrink_factor
        self._grow = grow_factor
        self._current = max_size

    def get_batch_size(self) -> int:
        return self._current

    def on_success(self) -> None:
        self._current = min(int(self._current * self._grow), self._max)

    def on_failure(self, error: Exception) -> None:
        self._current = max(int(self._current * self._shrink), self._min)

    def reset(self) -> None:
        self._current = self._max
