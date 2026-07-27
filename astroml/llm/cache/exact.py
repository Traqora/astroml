"""Exact match cache using hash-based lookups."""
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ExactMatchCache:
    """Fast exact-match cache using SHA256 hashing."""

    def __init__(self, store: "CacheStore"):
        """Initialize exact match cache.

        Args:
            store: Backend storage implementation
        """
        self.store = store

    def _hash_prompt(self, prompt: str, **kwargs) -> str:
        """Create deterministic hash from prompt and parameters."""
        # Include kwargs in hash for parameter-sensitive caching
        key_parts = [prompt]
        for k in sorted(kwargs.keys()):
            key_parts.append(f"{k}={kwargs[k]}")

        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """Retrieve cached response if exact match exists.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Cached response or None
        """
        cache_key = f"exact:{self._hash_prompt(prompt, **kwargs)}"
        return self.store.get(cache_key)

    def set(self, prompt: str, response: str, ttl: int = 3600, **kwargs) -> None:
        """Store response in cache.

        Args:
            prompt: Input prompt
            response: LLM response
            ttl: Time to live in seconds
            **kwargs: Additional parameters used in hash
        """
        cache_key = f"exact:{self._hash_prompt(prompt, **kwargs)}"
        self.store.set(cache_key, response, ttl=ttl)
        logger.debug(f"Cached exact match: {cache_key[:16]}...")

    def delete(self, prompt: str, **kwargs) -> bool:
        """Delete cached entry.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters

        Returns:
            True if entry was deleted
        """
        cache_key = f"exact:{self._hash_prompt(prompt, **kwargs)}"
        return self.store.delete(cache_key)
