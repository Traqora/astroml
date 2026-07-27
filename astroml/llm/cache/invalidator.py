"""Cache invalidation logic."""
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class CacheInvalidator:
    """Manages cache invalidation with pattern matching and expiration."""

    def __init__(self, cache_manager: "CacheManager"):
        """Initialize invalidator.

        Args:
            cache_manager: CacheManager instance to invalidate
        """
        self.cache_manager = cache_manager
        self._invalidation_queue = []

    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all cache entries matching pattern.

        Args:
            pattern: Pattern to match (supports wildcards)

        Returns:
            Number of entries invalidated
        """
        count = 0

        for tier_name, store in self.cache_manager.stores.items():
            # Scan for matching keys
            entries = store.scan_prefix(pattern.replace("*", ""))

            for key, _ in entries:
                if store.delete(key):
                    count += 1
                    logger.info(f"Invalidated {key} from {tier_name}")

        return count

    def invalidate_by_age(self, max_age_seconds: int) -> int:
        """Invalidate entries older than specified age.

        Args:
            max_age_seconds: Maximum age in seconds

        Returns:
            Number of entries invalidated
        """
        count = 0
        cutoff_time = time.time() - max_age_seconds

        for tier_name, store in self.cache_manager.stores.items():
            if hasattr(store, "cleanup_expired"):
                # SQLite has built-in cleanup
                store.cleanup_expired()
                logger.info(f"Cleaned up expired entries in {tier_name}")
                count += 1

        return count

    def schedule_invalidation(self, prompt: str, delay_seconds: int = 60) -> None:
        """Schedule delayed cache invalidation.

        Args:
            prompt: Prompt to invalidate
            delay_seconds: Delay before invalidation
        """
        invalidation_time = time.time() + delay_seconds
        self._invalidation_queue.append((invalidation_time, prompt))
        logger.info(f"Scheduled invalidation for {prompt[:50]}... in {delay_seconds}s")

    def process_queue(self) -> int:
        """Process pending invalidations.

        Returns:
            Number of entries invalidated
        """
        now = time.time()
        count = 0

        # Process due invalidations
        remaining = []
        for invalidation_time, prompt in self._invalidation_queue:
            if invalidation_time <= now:
                self.cache_manager.invalidate(prompt)
                count += 1
            else:
                remaining.append((invalidation_time, prompt))

        self._invalidation_queue = remaining
        return count

    def invalidate_all(self) -> None:
        """Invalidate entire cache (use with caution)."""
        logger.warning("Invalidating all cache entries")
        self.cache_manager.clear_all()
