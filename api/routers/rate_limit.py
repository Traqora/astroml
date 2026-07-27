"""Rate limit metrics and configuration router (issue #331)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth.dependencies import get_current_user
from api.auth.rate_limit import RateLimitConfig, rate_limiter
from api.models.orm import User

router = APIRouter(prefix="/api/v1/rate-limit", tags=["rate-limit"])


class RateLimitMetrics(BaseModel):
    """Rate limiting metrics."""

    metrics: dict[str, int]


class RateLimitConfigUpdate(BaseModel):
    """Rate limit configuration update."""

    requests_per_minute: int
    burst_size: int


@router.get("/metrics")
async def get_rate_limit_metrics(
    current_user: User = Depends(get_current_user),
) -> RateLimitMetrics:
    """Get rate limiting metrics."""
    if "rate_limit:read" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return RateLimitMetrics(metrics=rate_limiter.get_metrics())


@router.post("/metrics/reset")
async def reset_rate_limit_metrics(
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Reset rate limiting metrics."""
    if "rate_limit:admin" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    rate_limiter.reset_metrics()
    return {"message": "Rate limit metrics reset"}


@router.post("/config/{path:path}")
async def update_rate_limit_config(
    path: str,
    config: RateLimitConfigUpdate,
    current_user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Update rate limit configuration for a specific endpoint."""
    if "rate_limit:admin" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    rate_limiter.set_endpoint_config(
        path,
        RateLimitConfig(
            requests_per_minute=config.requests_per_minute,
            burst_size=config.burst_size,
        ),
    )
    return {"message": f"Rate limit config updated for {path}"}
