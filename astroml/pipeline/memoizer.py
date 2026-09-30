"""Pipeline step memoization for AstroML.

Issue #636: Implements intelligent memoization decorators that skip redundant
pipeline steps when inputs haven't changed, using content-addressable keys.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from astroml.pipeline.cache import ContentAddressableCache, hash_inputs
from astroml.pipeline.cache_backend import BackendStats, CacheBackend, create_backend
from astroml.pipeline.cache_invalidator import CacheInvalidator, InvalidationReason

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class MemoStats:
    """Memoization statistics for a pipeline step."""

    function_name: str = ""
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    compute_time_ms: float = 0.0
    cache_time_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def avg_compute_time_ms(self) -> float:
        return self.compute_time_ms / self.calls if self.calls > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "function_name": self.function_name,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.hit_rate,
            "avg_compute_time_ms": round(self.avg_compute_time_ms, 2),
        }


class PipelineMemoizer:
    """Memoization engine for ML pipeline steps.

    Wraps individual pipeline steps and caches their outputs based on
    content-addressable input hashes. Uses configurable cache backends
    and invalidation strategies.

    Example:
        memoizer = PipelineMemoizer(backend="local", cache_dir="/tmp/astro_cache")

        @memoizer.memoize(namespace="features", tags=["critical"])
        def expensive_feature_computation(account_ids: list[str]) -> dict:
            # ... expensive computation ...
            return results
    """

    def __init__(
        self,
        backend: str = "local",
        namespace: str = "pipeline",
        version: str = "v1",
        **backend_kwargs: Any,
    ) -> None:
        """Initialize the pipeline memoizer.

        Args:
            backend: Backend type ('local', 'redis', 's3').
            namespace: Default namespace for content-addressable keys.
            version: Pipeline version for version-based invalidation.
            **backend_kwargs: Backend-specific configuration.
        """
        self._backend: CacheBackend = create_backend(backend, **backend_kwargs)
        self._cache = ContentAddressableCache(namespace=namespace, version=version)
        self._invalidator = CacheInvalidator()
        self._stats: dict[str, MemoStats] = {}
        self._enabled = True

    @property
    def backend(self) -> CacheBackend:
        return self._backend

    @property
    def invalidator(self) -> CacheInvalidator:
        return self._invalidator

    def enable(self) -> None:
        """Enable memoization (default)."""
        self._enabled = True

    def disable(self) -> None:
        """Disable memoization — all calls will compute fresh."""
        self._enabled = False

    def memoize(
        self,
        namespace: str | None = None,
        tags: Sequence[str] | None = None,
        ttl_seconds: int | None = None,
        key_fn: Callable[..., str] | None = None,
    ) -> Callable[[F], F]:
        """Decorator to memoize a pipeline step function.

        The function's arguments are hashed to create a content-addressable
        cache key. On cache hit, the cached result is returned without
        recomputation. On miss, the function is called and the result cached.

        Args:
            namespace: Cache namespace (defaults to the function name).
            tags: Tags for grouping and invalidation.
            ttl_seconds: Time-to-live in seconds.
            key_fn: Optional custom key generation function.

        Returns:
            Decorated function with memoization.
        """
        def decorator(func: F) -> F:
            ns = namespace or func.__name__
            stat = MemoStats(function_name=func.__name__)
            self._stats[func.__name__] = stat

            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                stat.calls += 1

                if not self._enabled:
                    return func(*args, **kwargs)

                # Generate cache key
                if key_fn:
                    cache_key = key_fn(*args, **kwargs)
                else:
                    cache_key = f"{ns}:{func.__name__}:{hash_inputs(*args, **kwargs)}"

                # Check backend
                t_start = time.monotonic()

                # Check with invalidator
                existing = self._backend.get(cache_key)

                if existing is not None:
                    # Check if should be invalidated
                    metadata = self._backend.get(f"{cache_key}:metadata")
                    should_evict, _ = self._invalidator.check(cache_key, metadata)
                    if should_evict:
                        self._backend.delete(cache_key)
                        self._backend.delete(f"{cache_key}:metadata")
                    else:
                        stat.cache_hits += 1
                        stat.cache_time_ms += (time.monotonic() - t_start) * 1000
                        return existing

                # Cache miss - compute
                stat.cache_misses += 1
                t_compute = time.monotonic()
                result = func(*args, **kwargs)
                stat.compute_time_ms += (time.monotonic() - t_compute) * 1000

                # Store in backend
                self._backend.put(cache_key, result, ttl_seconds=ttl_seconds)
                # Store metadata for invalidation checks
                metadata = {
                    "created_at": time.time(),
                    "namespace": ns,
                    "function": func.__name__,
                    "tags": list(tags or []),
                }
                self._backend.put(f"{cache_key}:metadata", metadata, ttl_seconds=ttl_seconds)

                return result

            # Attach stats accessor
            setattr(wrapper, "memo_stats", lambda: self._stats.get(func.__name__, stat))
            return cast(F, wrapper)

        return decorator

    def invalidate(
        self,
        namespace: str | None = None,
        tags: Sequence[str] | None = None,
        reason: InvalidationReason = InvalidationReason.MANUAL,
    ) -> int:
        """Invalidate cached pipeline results.

        Args:
            namespace: Optional namespace filter.
            tags: Optional tag filter.
            reason: Reason for invalidation.

        Returns:
            Number of entries invalidated.
        """
        count = 0
        # In production, the backend would support querying by namespace/tags.
        # For now, bulk invalidate with the invalidator.
        self._invalidator.invalidate_all(reason)
        return count

    def invalidate_by_event(self, event_type: str) -> list[str]:
        """Invalidate entries based on an application event.

        Args:
            event_type: Event name (e.g., 'model_retrained').

        Returns:
            List of invalidated keys.
        """
        return self._invalidator.emit_event(event_type)

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get memoization statistics for all wrapped functions."""
        return {name: s.to_dict() for name, s in self._stats.items()}

    def get_overall_stats(self) -> dict[str, Any]:
        """Get aggregate memoization statistics."""
        total_calls = sum(s.calls for s in self._stats.values())
        total_hits = sum(s.cache_hits for s in self._stats.values())
        total_misses = sum(s.cache_misses for s in self._stats.values())
        total_compute = sum(s.compute_time_ms for s in self._stats.values())

        return {
            "total_calls": total_calls,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            "total_compute_time_ms": round(total_compute, 2),
            "enabled": self._enabled,
            "functions_cached": len(self._stats),
            "invalidation_history": self._invalidator.get_history(limit=10),
        }

    def close(self) -> None:
        """Close the backend and release resources."""
        self._backend.close()


# Convenience: global pipeline memoizer instance
_memoizer: PipelineMemoizer | None = None


def get_pipeline_memoizer(
    backend: str = "local",
    **backend_kwargs: Any,
) -> PipelineMemoizer:
    """Get or create the global pipeline memoizer instance.

    Args:
        backend: Backend type.
        **backend_kwargs: Backend configuration.

    Returns:
        Global PipelineMemoizer singleton.
    """
    global _memoizer
    if _memoizer is None:
        _memoizer = PipelineMemoizer(backend=backend, **backend_kwargs)
    return _memoizer


def reset_memoizer() -> None:
    """Reset the global memoizer instance."""
    global _memoizer
    if _memoizer is not None:
        _memoizer.close()
    _memoizer = None


__all__ = [
    "MemoStats",
    "PipelineMemoizer",
    "get_pipeline_memoizer",
    "reset_memoizer",
]