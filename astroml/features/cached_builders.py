"""Caching layer for expensive feature computations.

Issue #743: caches feature computations keyed by ``(feature, window,
data_version, definition_hash)`` and invalidates on recompute, reusing the
cache utilities already provided by :mod:`astroml.cache` (the
:class:`~astroml.cache.redis_cache.RedisCache` backends and their metrics).

Cache / recompute semantics:
    - Cache keys include the feature name, the window string, the caller's
      data version, and a hash of the builder definition (#742). Two calls
      with the same key hit the cache; any change to window, data version, or
      definition produces a new key and therefore a recomputation.
    - ``CachedFeatureBuilder.compute`` recomputes on a miss, stores the result
      with the configured TTL, and returns it. ``invalidate`` removes entries
      explicitly; ``recompute`` is invalidate + compute.
    - Stale entries (e.g. after a definition change) are simply never read
      again; they expire via TTL or LRU eviction. ``invalidate_all`` clears
      everything for a hard reset.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from astroml.features.feature_cache import CacheConfig, FeatureCache
from astroml.features.yaml_builders import FeatureBuilderSpec

logger = logging.getLogger(__name__)

__all__ = [
    "CachedFeatureBuilder",
    "FeatureCacheResult",
    "cache_key_for_spec",
]


def cache_key_for_spec(
    spec: FeatureBuilderSpec,
    data_version: str,
    entity_id: str,
    **context: Any,
) -> str:
    """Build a deterministic cache key for a builder invocation.

    The key packs everything that can change the output: feature name,
    window, data version, definition hash (#742), entity, and any extra
    call context (e.g. an as-of timestamp).
    """
    parts = [
        "fb",
        spec.name,
        spec.window or "all",
        str(data_version),
        spec.definition_hash(),
        str(entity_id),
    ]
    if context:
        rendered = ",".join(f"{k}={context[k]}" for k in sorted(context))
        parts.append(rendered)
    return ":".join(parts)


@dataclass(frozen=True)
class FeatureCacheResult:
    """Outcome of a cached feature build.

    Attributes:
        value: The computed (or cached) feature value.
        cache_hit: True when the value came from cache.
        cache_key: The exact key used for lookup/store.
        spec: The builder spec that produced the value.
    """

    value: float
    cache_hit: bool
    cache_key: str
    spec: FeatureBuilderSpec


@dataclass
class CachedFeatureBuilder:
    """Executes feature builders with a transparent cache in front (#743).

    Args:
        cache: A :class:`~astroml.features.feature_cache.FeatureCache`
            (memory, disk, or Redis) used for storage.
        data_version: Version tag of the underlying data. Bump it when the
            source data changes to force recomputation everywhere.
        ttl_seconds: Default TTL for stored entries; individual calls may
            override it.
    """

    cache: FeatureCache
    data_version: str = "v0"
    ttl_seconds: int | None = None
    primitives: Mapping[str, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]] = field(
        default_factory=dict
    )
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def compute(
        self,
        spec: FeatureBuilderSpec,
        rows: list[dict[str, Any]],
        entity_id: str = "",
        ttl_seconds: int | None = None,
        **context: Any,
    ) -> FeatureCacheResult:
        """Compute (or fetch) a feature value for one entity.

        Args:
            spec: The builder spec (from YAML, #742).
            rows: Input rows already filtered to the entity.
            entity_id: Entity identifier for the cache key.
            ttl_seconds: Optional TTL override for the stored entry.
            **context: Extra values that participate in the cache key.

        Returns:
            A :class:`FeatureCacheResult` telling you whether the value was a
            cache hit.
        """
        key = cache_key_for_spec(spec, self.data_version, entity_id, **context)
        cached = self.cache.get(key)
        if cached is not None:
            logger.debug("feature cache hit: %s", key)
            return FeatureCacheResult(value=float(cached), cache_hit=True, cache_key=key, spec=spec)

        value = self._evaluate(spec, rows)
        self.cache.put(
            key,
            value,
            ttl_seconds=self.ttl_seconds if ttl_seconds is None else ttl_seconds,
        )
        logger.debug("feature cache miss, stored: %s", key)
        return FeatureCacheResult(value=value, cache_hit=False, cache_key=key, spec=spec)

    def compute_batch(
        self,
        spec: FeatureBuilderSpec,
        rows_by_entity: dict[str, list[dict[str, Any]]],
        ttl_seconds: int | None = None,
        **context: Any,
    ) -> dict[str, FeatureCacheResult]:
        """Compute a feature for many entities at once.

        Args:
            spec: The builder spec.
            rows_by_entity: Mapping of entity id to that entity's rows.
            ttl_seconds: Optional TTL override.
            **context: Extra values that participate in the cache key.

        Returns:
            Mapping of entity id to its :class:`FeatureCacheResult`.
        """
        return {
            entity_id: self.compute(
                spec, rows, entity_id=entity_id, ttl_seconds=ttl_seconds, **context
            )
            for entity_id, rows in rows_by_entity.items()
        }

    def invalidate(self, spec: FeatureBuilderSpec, entity_id: str = "", **context: Any) -> bool:
        """Remove the cached value for one builder/entity pair.

        Returns:
            True when an entry was removed.
        """
        key = cache_key_for_spec(spec, self.data_version, entity_id, **context)
        return self.cache.remove(key)

    def recompute(
        self,
        spec: FeatureBuilderSpec,
        rows: list[dict[str, Any]],
        entity_id: str = "",
        **context: Any,
    ) -> FeatureCacheResult:
        """Invalidate then compute, guaranteeing a fresh value."""
        self.invalidate(spec, entity_id=entity_id, **context)
        return self.compute(spec, rows, entity_id=entity_id, **context)

    def invalidate_all(self) -> None:
        """Clear every cached feature entry (hard reset)."""
        with self._lock:
            self.cache.clear()
        logger.info("feature builder cache cleared")

    def _evaluate(self, spec: FeatureBuilderSpec, rows: list[dict[str, Any]]) -> float:
        """Run the primitive named by the spec against the input rows."""
        primitive = self.primitives.get(spec.primitive)
        if primitive is None:
            raise ValueError(
                f"no primitive {spec.primitive!r} registered for builder {spec.name!r}"
            )
        return float(primitive(rows, spec))


def build_cached_feature_builder(
    config: CacheConfig | None = None,
    data_version: str = "v0",
    ttl_seconds: int | None = None,
    primitives: (
        Mapping[str, Callable[[list[dict[str, Any]], FeatureBuilderSpec], float]] | None
    ) = None,
) -> CachedFeatureBuilder:
    """Convenience factory wiring a default in-memory FeatureCache."""
    return CachedFeatureBuilder(
        cache=FeatureCache(config or CacheConfig()),
        data_version=data_version,
        ttl_seconds=ttl_seconds,
        primitives=primitives or {},
    )
