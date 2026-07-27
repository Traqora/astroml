"""Semantic caching for LLM responses using Redis."""
import hashlib
import json
import os
from typing import Optional

try:
    import redis
except ImportError:
    redis = None

class SemanticCache:
    """Redis-backed cache for LLM responses with TTL expiration."""
    
    def __init__(self, ttl: int = 3600):
        self.ttl = ttl
        
        if redis is None:
            self.redis_client = None
            return
            
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
        except Exception:
            self.redis_client = None

    def _hash_prompt(self, prompt: str) -> str:
        """Create a hash of the prompt for exact matching."""
        # For true semantic caching, we would use an embedding model
        # and store vectors in a vector DB (like Redisearch or Pinecone).
        # As a placeholder, we'll use a simple SHA256 hash.
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

    def get(self, prompt: str) -> Optional[str]:
        """Retrieve a cached response if one exists."""
        if self.redis_client is None:
            return None
            
        cache_key = f"llm_cache:{self._hash_prompt(prompt)}"
        try:
            return self.redis_client.get(cache_key)
        except Exception:
            return None

    def set(self, prompt: str, response: str) -> None:
        """Cache a response with a TTL."""
        if self.redis_client is None:
            return
            
        cache_key = f"llm_cache:{self._hash_prompt(prompt)}"
        try:
            self.redis_client.setex(cache_key, self.ttl, response)
        except Exception:
            pass
