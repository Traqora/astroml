from typing import Any, Callable, Dict, List, Optional, Union

from astroml.utils.exceptions import AstroMLError

"""Semantic caching for LLM responses using Redis."""

import hashlib
import os
import time

try:
    import redis
except ImportError:
    redis = None


class SemanticCache:
    """Redis-backed cache for LLM responses with TTL expiration."""

    def __init__(self, ttl: int = 3600) -> None:
        self.ttl = ttl

        if redis is None:
            self.redis_client = None
            return

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        except AstroMLError:
            self.redis_client = None

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for exact matching."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> str | None:
        """Retrieve a cached response if one exists."""
        if self.redis_client is None:
            return None

        cache_key = f"llm_cache:{self._hash_prompt(prompt)}"
        try:
            value = self.redis_client.get(cache_key)
            if value is None:
                return None
            expires_at = self.redis_client.ttl(cache_key)
            if expires_at is not None and expires_at <= 0:
                self.redis_client.delete(cache_key)
                return None
            return value
        except AstroMLError:
            return None

    def set(self, prompt: str, response: str) -> None:
        """Cache a response with a TTL."""
        if self.redis_client is None:
            return

        cache_key = f"llm_cache:{self._hash_prompt(prompt)}"
        try:
            self.redis_client.setex(cache_key, self.ttl, response)
        except AstroMLError:
            pass
