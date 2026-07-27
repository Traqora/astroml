"""
Admin endpoints for managing rate limits (issue #299).
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.dependencies import get_current_admin_user
from api.auth.rate_limit import rate_limiter, RateLimitConfig
from api.auth.config import ADMIN_WHITELIST, ADMIN_BLACKLIST

router = APIRouter(prefix="/admin/rate-limit", tags=["admin"])


@router.get("/metrics")
async def get_rate_limit_metrics(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get rate limiting metrics."""
    return {
        "metrics": rate_limiter.get_metrics(),
        "whitelist": ADMIN_WHITELIST,
        "blacklist": ADMIN_BLACKLIST,
    }


@router.post("/reset")
async def reset_rate_limit_metrics(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, str]:
    """Reset rate limiting metrics."""
    rate_limiter.reset_metrics()
    return {"status": "metrics reset successfully"}


@router.get("/config/{path:path}")
async def get_endpoint_config(
    path: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Get rate limit config for a specific endpoint."""
    # Rate limit config is determined dynamically, so we just return the current limits
    # This is a placeholder for potential future config storage
    return {
        "path": path,
        "limits": {
            "default": {"requests_per_minute": 60, "burst_size": 10},
        },
    }


@router.post("/config/{path:path}")
async def set_endpoint_config(
    path: str,
    config: RateLimitConfig,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, Any]:
    """Set rate limit config for a specific endpoint."""
    rate_limiter.set_endpoint_config(path, config)
    return {
        "status": "config updated",
        "path": path,
        "config": {
            "requests_per_minute": config.requests_per_minute,
            "burst_size": config.burst_size,
        },
    }


@router.post("/whitelist")
async def add_to_whitelist(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, str]:
    """Add a key to the rate limit whitelist."""
    rate_limiter.add_to_whitelist(key)
    return {"status": f"Added {key} to whitelist"}


@router.delete("/whitelist")
async def remove_from_whitelist(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, str]:
    """Remove a key from the rate limit whitelist."""
    rate_limiter.remove_from_whitelist(key)
    return {"status": f"Removed {key} from whitelist"}


@router.post("/blacklist")
async def add_to_blacklist(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, str]:
    """Add a key to the rate limit blacklist."""
    rate_limiter.add_to_blacklist(key)
    return {"status": f"Added {key} to blacklist"}


@router.delete("/blacklist")
async def remove_from_blacklist(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, str]:
    """Remove a key from the rate limit blacklist."""
    rate_limiter.remove_from_blacklist(key)
    return {"status": f"Removed {key} from blacklist"}


@router.get("/whitelist")
async def get_whitelist(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, list]:
    """Get the current whitelist."""
    return {"whitelist": ADMIN_WHITELIST}


@router.get("/blacklist")
async def get_blacklist(
    current_user: Dict[str, Any] = Depends(get_current_admin_user),
) -> Dict[str, list]:
    """Get the current blacklist."""
    return {"blacklist": ADMIN_BLACKLIST}