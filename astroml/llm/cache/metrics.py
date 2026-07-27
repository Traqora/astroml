"""Cache hit rate and cost savings tracking."""
import time
import logging
from collections import defaultdict
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CacheMetrics:
    """Tracks cache performance metrics and cost savings."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.hits_by_tier = defaultdict(int)  # tier_name -> count
        self.hits_by_type = defaultdict(int)  # "exact" or "semantic" -> count
        self.misses = 0
        self.stores = 0
        self.invalidations = 0

        self.hit_latencies = []  # List of latencies for hits (ms)
        self.miss_latencies = []  # List of latencies for misses (ms)

        self.cost_savings = 0.0  # Estimated USD saved
        self.start_time = time.time()

    def record_hit(self, tier: str, cache_type: str, latency_ms: float) -> None:
        """Record cache hit.

        Args:
            tier: Cache tier (redis/sqlite/disk)
            cache_type: Type of hit (exact/semantic)
            latency_ms: Lookup latency in milliseconds
        """
        self.hits_by_tier[tier] += 1
        self.hits_by_type[cache_type] += 1
        self.hit_latencies.append(latency_ms)

    def record_miss(self, latency_ms: float) -> None:
        """Record cache miss.

        Args:
            latency_ms: Lookup latency in milliseconds
        """
        self.misses += 1
        self.miss_latencies.append(latency_ms)

    def record_store(self) -> None:
        """Record cache store operation."""
        self.stores += 1

    def record_invalidation(self) -> None:
        """Record cache invalidation."""
        self.invalidations += 1

    def record_cost_savings(self, amount_usd: float) -> None:
        """Record cost savings from cache hit.

        Args:
            amount_usd: Amount saved in USD
        """
        self.cost_savings += amount_usd

    def get_hit_rate(self) -> float:
        """Calculate overall cache hit rate.

        Returns:
            Hit rate as percentage (0-100)
        """
        total_requests = sum(self.hits_by_tier.values()) + self.misses
        if total_requests == 0:
            return 0.0

        total_hits = sum(self.hits_by_tier.values())
        return (total_hits / total_requests) * 100

    def get_exact_hit_rate(self) -> float:
        """Calculate exact match hit rate.

        Returns:
            Exact hit rate as percentage
        """
        total_hits = sum(self.hits_by_tier.values())
        if total_hits == 0:
            return 0.0

        exact_hits = self.hits_by_type["exact"]
        return (exact_hits / (exact_hits + self.misses)) * 100

    def get_semantic_contribution(self) -> float:
        """Calculate percentage of hits from semantic cache.

        Returns:
            Semantic hit contribution as percentage
        """
        total_hits = sum(self.hits_by_tier.values())
        if total_hits == 0:
            return 0.0

        semantic_hits = self.hits_by_type["semantic"]
        return (semantic_hits / total_hits) * 100

    def get_avg_hit_latency(self) -> float:
        """Calculate average cache hit latency.

        Returns:
            Average latency in milliseconds
        """
        if not self.hit_latencies:
            return 0.0
        return sum(self.hit_latencies) / len(self.hit_latencies)

    def get_avg_miss_latency(self) -> float:
        """Calculate average cache miss latency.

        Returns:
            Average latency in milliseconds
        """
        if not self.miss_latencies:
            return 0.0
        return sum(self.miss_latencies) / len(self.miss_latencies)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.

        Returns:
            Dictionary with all metrics
        """
        total_requests = sum(self.hits_by_tier.values()) + self.misses
        uptime_hours = (time.time() - self.start_time) / 3600

        return {
            "total_requests": total_requests,
            "hits": sum(self.hits_by_tier.values()),
            "misses": self.misses,
            "hit_rate_pct": self.get_hit_rate(),
            "exact_hit_rate_pct": self.get_exact_hit_rate(),
            "semantic_contribution_pct": self.get_semantic_contribution(),
            "hits_by_tier": dict(self.hits_by_tier),
            "hits_by_type": dict(self.hits_by_type),
            "avg_hit_latency_ms": self.get_avg_hit_latency(),
            "avg_miss_latency_ms": self.get_avg_miss_latency(),
            "cost_savings_usd": self.cost_savings,
            "stores": self.stores,
            "invalidations": self.invalidations,
            "uptime_hours": uptime_hours,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.hits_by_tier.clear()
        self.hits_by_type.clear()
        self.misses = 0
        self.stores = 0
        self.invalidations = 0
        self.hit_latencies.clear()
        self.miss_latencies.clear()
        self.cost_savings = 0.0
        self.start_time = time.time()
        logger.info("Cache metrics reset")
