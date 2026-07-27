"""Audit log router for searching and exporting audit logs (issue #332)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.audit import audit_logger
from api.auth.dependencies import get_current_user
from api.database import get_db
from api.models.orm import User

router = APIRouter(prefix="/api/v1/audit", tags=["audit"])


@router.get("/logs")
async def search_audit_logs(
    user_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Search audit logs with filters."""
    if "audit:read" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    logs = await audit_logger.search_logs(
        session,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
    )
    return {
        "logs": [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "user_id": log.user_id,
                "username": log.username,
                "auth_type": log.auth_type,
                "action": log.action,
                "resource_type": log.resource_type,
                "resource_id": log.resource_id,
                "ip_address": log.ip_address,
                "user_agent": log.user_agent,
                "request_path": log.request_path,
                "request_method": log.request_method,
                "status_code": log.status_code,
                "details": log.details,
            }
            for log in logs
        ],
        "count": len(logs),
    }


@router.get("/export")
async def export_audit_logs(
    user_id: Optional[int] = Query(None),
    action: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Export audit logs as JSON."""
    if "audit:export" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    export_data = await audit_logger.export_logs(
        session,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "data": export_data,
        "exported_at": datetime.utcnow().isoformat(),
    }


@router.post("/rotate")
async def rotate_audit_logs(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Manually trigger audit log rotation (delete old logs)."""
    if "audit:admin" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    deleted_count = await audit_logger.rotate_logs(session)
    return {
        "deleted_count": deleted_count,
        "message": f"Deleted {deleted_count} old audit log entries",
    }


@router.get("/stats")
async def get_audit_stats(
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    """Get audit log statistics."""
    if "audit:read" not in current_user.scopes:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    total_count = await audit_logger.get_log_count(session)
    return {
        "total_logs": total_count,
        "retention_days": audit_logger.retention_days,
        "max_records": audit_logger.max_records,
    }
