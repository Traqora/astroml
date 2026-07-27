"""Streaming Service Health API.

Endpoints:
  GET /api/v1/streaming/health — overall streaming health
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])


# In-memory storage for streaming state (will be updated by streaming service)
class StreamState:
    def __init__(self):
        self.streams: Dict[str, Dict[str, Any]] = {}
        # Back‑pressure placeholder
        self.queue_depth: int = 0
        self.max_queue_depth: int = 1000
        self.batch_size: int = 100
        self.memory_pressure: bool = False
        self.last_updated: Optional[float] = None


_stream_state = StreamState()


def update_stream_state(stream_id: str, state: Dict[str, Any]) -> None:
    """Update the state for a specific stream. Called by streaming service."""
    _stream_state.streams[stream_id] = state
    _stream_state.last_updated = time.time()

# Retry / back‑off placeholder
def _apply_retry_backoff(attempt: int) -> None:
    """Placeholder for retry back‑off logic.
    * `attempt` – the current retry count (starting at 1).
    * Sleeps for `base * 2**(attempt-1)` seconds where `base` is 0.5s.
    """
    import time
    base = 0.5
    delay = base * (2 ** (attempt - 1))
    time.sleep(delay)


class StreamHealth(BaseModel):
    stream_id: str
    stream_type: str
    horizon_url: str
    is_healthy: bool
    status: str  # "active", "inactive", "error"
    cursor: Optional[str] = None
    processed_count: int = 0
    consecutive_failures: int = 0
    current_backoff_seconds: float = 0.0
    lag_seconds: Optional[float] = None


class StreamingHealthOut(BaseModel):
    overall_status: str  # "healthy", "degraded", "unhealthy"
    last_updated: Optional[float]
    streams: List[StreamHealth]


@router.get("/health", response_model=StreamingHealthOut)
def get_streaming_health():
    """Return overall health of all streaming services."""
    streams = []
    healthy_count = 0
    total_count = len(_stream_state.streams)
    
    for stream_id, state in _stream_state.streams.items():
        is_healthy = state.get("is_healthy", False)
        if is_healthy:
            healthy_count += 1
        
        streams.append(StreamHealth(
            stream_id=stream_id,
            stream_type=state.get("stream_type", "unknown"),
            horizon_url=state.get("horizon_url", "unknown"),
            is_healthy=is_healthy,
            status=state.get("status", "inactive"),
            cursor=state.get("cursor"),
            processed_count=state.get("processed_count", 0),
            consecutive_failures=state.get("consecutive_failures", 0),
            current_backoff_seconds=state.get("current_backoff", 0.0),
            lag_seconds=state.get("lag_seconds")
        ))
    
    # Determine overall status
    if total_count == 0:
        overall_status = "degraded"
    elif healthy_count == total_count:
        overall_status = "healthy"
    elif healthy_count > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return StreamingHealthOut(
        overall_status=overall_status,
        last_updated=_stream_state.last_updated,
        streams=streams
    )


# ---------------------------------------------------------------------------
# LLM SSE Streaming Endpoint
# ---------------------------------------------------------------------------
import asyncio
from fastapi.responses import StreamingResponse
from fastapi import Depends, Query
from api.auth.dependencies import AuthContext, get_current_auth
from api.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from astroml.llm.streaming import StreamHandler, format_sse
from astroml.llm.cost import check_budget, track_request, route_request

@router.get("/llm")
async def stream_llm_response(
    prompt: str = Query(..., description="Prompt for the LLM"),
    model: str = Query("gpt-3.5-turbo", description="Model name"),
    feature: str = Query("chatbot", description="Feature category"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
):
    """
    Stream LLM response token-by-token using Server-Sent Events (SSE).
    Enforces cost tracking, dynamic model routing, and budgets.
    """
    user_id = str(auth.user_id or auth.subject)
    
    # 1. Dynamic routing based on budget and complexity
    routed_model = await route_request(db, user_id, model, prompt)
    
    # 2. Check budget limits and model permissions
    try:
        await check_budget(db, user_id, routed_model)
    except Exception as e:
        raise HTTPException(status_code=403, detail="Access denied")
        
    async def sse_generator():
        handler = StreamHandler(session_id=f"sse_{user_id}_{int(time.time())}")
        
        # Simulate some generation tokens
        mock_response = f"This is a progressive response for your query '{prompt}' using {routed_model}."
        words = mock_response.split(" ")
        
        total_tokens = 0
        latency_ms = 100.0  # mock first token latency
        start_time = time.perf_counter()
        
        async def mock_word_gen():
            for i, word in enumerate(words):
                await asyncio.sleep(0.05)  # 50ms streaming latency
                yield word + " " if i < len(words) - 1 else word
                
        async for token in handler.process_stream(mock_word_gen()):
            total_tokens += 1
            yield format_sse(token=token, finished=False)
            
        duration = (time.perf_counter() - start_time) * 1000
        
        # Track cost and update budget
        usage = {"prompt_tokens": len(prompt) // 4 + 1, "completion_tokens": total_tokens}
        await track_request(
            db=db,
            user_id=user_id,
            feature=feature,
            model_name=routed_model,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            latency_ms=duration,
        )
        
        yield format_sse(token=None, finished=True, usage={"total_tokens": usage["prompt_tokens"] + usage["completion_tokens"]})
        
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

