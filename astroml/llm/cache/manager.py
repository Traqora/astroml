"""Cache orchestration with multi-level storage."""
import logging
import time
from typing import Any, Dict, Optional

from .exact import ExactMatchCache
from .semantic import SemanticCache
from .store import CacheStore, RedisStore, SQLiteStore, DiskStore
from .metrics import CacheMetrics

logger = logging.getLogger(__name__)


class CacheManager:
    """Multi-level cache manager with hot/warm/cold tiers."""

    def __init__(
        self,
        enable_redis: bool = True,
        enable_sqlite: bool = True,
        enable_disk: bool = False,
        semantic_similarity_threshold: float = 0.95,
    ):
        """Initialize cache manager.

        Args:
            enable_redis: Use Redis for hot cache
            enable_sqlite: Use SQLite for warm cache
            enable_disk: Use disk for cold storage
            semantic_similarity_threshold: Threshold for semantic cache hits
        """
        self.metrics = CacheMetrics()

        # Initialize storage tiers
        self.stores = {}

        if enable_redis:
            try:
                self.stores["redis"] = RedisStore()
                logger.info("Redis hot cache enabled")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}")

        if enable_sqlite:
            try:
                self.stores["sqlite"] = SQLiteStore()
                logger.info("SQLite warm cache enabled")
            except Exception as e:
                logger.warning(f"SQLite unavailable: {e}")

        if enable_disk:
            self.stores["disk"] = DiskStore()
            logger.info("Disk cold storage enabled")

        if not self.stores:
            logger.warning("No cache stores available, caching disabled")

        # Initialize cache layers
        self.exact_caches = {}
        self.semantic_caches = {}

        for name, store in self.stores.items():
            self.exact_caches[name] = ExactMatchCache(store)
            # Semantic cache needs embedding provider (set later)
            self.semantic_caches[name] = SemanticCache(
                store,
                embedding_provider=None,
                similarity_threshold=semantic_similarity_threshold,
            )

    def get(
        self,
        prompt: str,
        use_semantic: bool = True,
        **kwargs: Any,
    ) -> Optional[str]:
        """Retrieve cached response from any tier.

        Checks in order: Redis exact → Redis semantic → SQLite exact → SQLite semantic → Disk

        Args:
            prompt: Input prompt
            use_semantic: Whether to check semantic cache
            **kwargs: Additional parameters for cache key

        Returns:
            Cached response or None
        """
        start_time = time.time()

        # Try exact match in each tier (fastest)
        for tier_name in ["redis", "sqlite", "disk"]:
            if tier_name not in self.exact_caches:
                continue

            result = self.exact_caches[tier_name].get(prompt, **kwargs)
            if result:
                latency_ms = (time.time() - start_time) * 1000
                self.metrics.record_hit(tier_name, "exact", latency_ms)
                logger.info(f"Cache hit ({tier_name}/exact) in {latency_ms:.1f}ms")

                # Promote to faster tier if applicable
                self._promote_to_hot_tier(prompt, result, **kwargs)

                return result

        # Try semantic match if enabled (slower)
        if use_semantic:
            for tier_name in ["redis", "sqlite"]:
                if tier_name not in self.semantic_caches:
                    continue

                result = self.semantic_caches[tier_name].get(prompt, **kwargs)
                if result:
                    latency_ms = (time.time() - start_time) * 1000
                    self.metrics.record_hit(tier_name, "semantic", latency_ms)
                    logger.info(f"Cache hit ({tier_name}/semantic) in {latency_ms:.1f}ms")

                    # Promote to hot tier and add as exact match
                    self._promote_to_hot_tier(prompt, result, **kwargs)

                    return result

        # Cache miss
        latency_ms = (time.time() - start_time) * 1000
        self.metrics.record_miss(latency_ms)
        logger.debug(f"Cache miss after {latency_ms:.1f}ms")

        return None

    def set(
        self,
        prompt: str,
        response: str,
        store_semantic: bool = True,
        **kwargs: Any,
    ) -> None:
        """Store response in all enabled tiers.

        Args:
            prompt: Input prompt
            response: LLM response
            store_semantic: Whether to store in semantic cache
            **kwargs: Additional parameters
        """
        # Store in exact match cache for each tier with appropriate TTL
        if "redis" in self.exact_caches:
            self.exact_caches["redis"].set(prompt, response, ttl=3600, **kwargs)  # 1 hour

        if "sqlite" in self.exact_caches:
            self.exact_caches["sqlite"].set(prompt, response, ttl=86400, **kwargs)  # 1 day

        if "disk" in self.exact_caches:
            self.exact_caches["disk"].set(prompt, response, ttl=604800, **kwargs)  # 1 week

        # Store in semantic cache
        if store_semantic:
            if "redis" in self.semantic_caches:
                self.semantic_caches["redis"].set(prompt, response, ttl=3600, **kwargs)

            if "sqlite" in self.semantic_caches:
                self.semantic_caches["sqlite"].set(prompt, response, ttl=86400, **kwargs)

        self.metrics.record_store()

    def invalidate(self, prompt: str, **kwargs: Any) -> int:
        """Invalidate cached entry across all tiers.

        Args:
            prompt: Prompt to invalidate
            **kwargs: Additional parameters

        Returns:
            Number of entries deleted
        """
        deleted = 0

        for tier_name, cache in self.exact_caches.items():
            if cache.delete(prompt, **kwargs):
                deleted += 1
                logger.info(f"Invalidated cache in {tier_name}")

        self.metrics.record_invalidation()
        return deleted

    def _promote_to_hot_tier(self, prompt: str, response: str, **kwargs: Any) -> None:
        """Promote frequently accessed items to Redis hot tier."""
        if "redis" in self.exact_caches:
            self.exact_caches["redis"].set(prompt, response, ttl=3600, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hit rates, latencies, cost savings
        """
        return self.metrics.get_stats()

    def clear_all(self) -> None:
        """Clear all cache tiers (use with caution)."""
        for tier_name in self.stores:
            logger.warning(f"Clearing {tier_name} cache")
            # Note: Actual clear implementation depends on store type
            # For safety, we don't implement automatic clear here
        self.metrics.reset()
