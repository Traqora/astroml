"""LLM semantic cache metrics endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from astroml.cache.redis_cache import RedisCache
from astroml.llm.llm_cached_client import get_semantic_cache_metrics

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


@router.get("/cache/semantic/metrics", response_model=Dict[str, Any])
def semantic_cache_metrics():
    """Return semantic cache hit/miss and avg lookup latency."""
    return get_semantic_cache_metrics(redis_cache=RedisCache())

