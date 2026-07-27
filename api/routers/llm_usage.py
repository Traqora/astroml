"""LLM usage and cost monitoring endpoints.

These endpoints expose:
- recent LLM call events (all calls logged)
- rolling cost summaries

Prometheus metrics are emitted by ``LLMUsageTracker``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from astroml.tracking.llm_usage_tracker import default_llm_usage_tracker

router = APIRouter(prefix="/api/v1/llm", tags=["llm"])


@router.get("/usage/recent", response_model=List[Dict[str, Any]])
def recent_llm_usage(limit: int = Query(100, ge=1, le=1000)):
    """Return the most recent recorded LLM calls."""
    return default_llm_usage_tracker.recent_calls(limit=limit)


@router.get("/usage/summary", response_model=Dict[str, Any])
def usage_summary():
    """Return a lightweight summary based on recent in-memory buffer."""
    events = default_llm_usage_tracker.recent_calls(limit=5000)
    total_calls = len(events)
    total_cost_usd = sum(float(e.get("cost_usd", 0.0) or 0.0) for e in events)
    total_tokens = sum(int(e.get("total_tokens", 0) or 0) for e in events)

    return {
        "total_calls": total_calls,
        "total_cost_usd": round(total_cost_usd, 6),
        "total_tokens": total_tokens,
        "window": "in-memory-recent (up to last 5000 events)",
    }

