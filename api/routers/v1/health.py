"""
Health check endpoints for the AstroML API.
"""

# NOTE: This file is a dormant duplicate of api/routers/health.py.
# It is NOT registered in app.py. Before registering it, consolidate
# or delete this file to avoid endpoint path conflicts.

from __future__ import annotations

import asyncio
from typing import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.features import AccountFeatureContext, require_account_feature
from api.database import check_database_connection, get_db, get_db_status, get_pool_metrics
from api.models.orm import ApiAccount

router = APIRouter(tags=["health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "service": "astroml-api"}


@router.get("/health/db", status_code=status.HTTP_200_OK)
async def database_health_check(
    _account: Union[ApiAccount, AccountFeatureContext] = Depends(
        require_account_feature("health")
    ),
):
    """Database connection health check with pool statistics."""
    result = await check_database_connection()

    if result["status"] == "unhealthy":
        # Return 503 Service Unavailable for unhealthy DB
        from fastapi import HTTPException

        code = result.get("error_code") or "DB_ERROR"
        title = "Database query timed out" if code == "DB_TIMEOUT" else "Database connection failed"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": code, "message": title, "details": {}},
        )

    return result


@router.get("/health/db/pool", status_code=status.HTTP_200_OK)
async def pool_health_check():
    """Get connection pool metrics and statistics."""
    return get_pool_metrics()


@router.get("/health/db/status", status_code=status.HTTP_200_OK)
async def db_status():
    """Get comprehensive database status including health and metrics."""
    return await get_db_status()


@router.get("/health/readiness", status_code=status.HTTP_200_OK)
async def readiness_check(db: AsyncSession = Depends(get_db)):
    """
    Readiness probe for Kubernetes/container orchestration.
    Checks that the API can connect to the database.
    """
    import logging

    from api.config import settings

    logger = logging.getLogger(__name__)

    async def _check():
        await db.execute(text("SELECT 1"))

    try:
        await asyncio.wait_for(_check(), timeout=settings.db_query_timeout_seconds)
        return {"status": "ready"}
    except asyncio.TimeoutError:
        from fastapi import HTTPException

        logger.error("Readiness check timed out after %ds", settings.db_query_timeout_seconds)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "DB_TIMEOUT", "message": "Database not ready", "details": {}},
        )
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "DB_ERROR", "message": "Database not ready", "details": {}},
        )


@router.get("/health/liveness", status_code=status.HTTP_200_OK)
async def liveness_check():
    """
    Liveness probe for Kubernetes/container orchestration.
    Simple endpoint that always returns OK if the API is running.
    """
    return {"status": "alive"}


@router.post("/health/db/reset-pool", status_code=status.HTTP_200_OK)
async def reset_pool():
    """
    Reset database connection pool (admin endpoint).
    Forces all connections to be recreated.
    """
    from api.database import reset_engines

    reset_engines()
    return {"status": "pool reset successfully"}
