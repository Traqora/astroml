"""Audit logging service for sensitive API operations (issue #332)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.orm import AuditLog


class AuditLogger:
    """Service for recording and querying audit logs."""

    def __init__(
        self,
        retention_days: int = 90,
        max_records: int = 1000000,
    ) -> None:
        self.retention_days = retention_days
        self.max_records = max_records

    async def log_event(
        self,
        session: AsyncSession,
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        auth_type: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_path: Optional[str] = None,
        request_method: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[dict] = None,
    ) -> AuditLog:
        """Record an audit event."""
        audit_log = AuditLog(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            username=username,
            auth_type=auth_type,
            ip_address=ip_address,
            user_agent=user_agent,
            request_path=request_path,
            request_method=request_method,
            status_code=status_code,
            details=details,
        )
        session.add(audit_log)
        await session.commit()
        await session.refresh(audit_log)
        return audit_log

    async def search_logs(
        self,
        session: AsyncSession,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Search audit logs with filters."""
        query = select(AuditLog)

        conditions = []
        if user_id is not None:
            conditions.append(AuditLog.user_id == user_id)
        if action is not None:
            conditions.append(AuditLog.action == action)
        if resource_type is not None:
            conditions.append(AuditLog.resource_type == resource_type)
        if resource_id is not None:
            conditions.append(AuditLog.resource_id == resource_id)
        if start_date is not None:
            conditions.append(AuditLog.timestamp >= start_date)
        if end_date is not None:
            conditions.append(AuditLog.timestamp <= end_date)

        if conditions:
            query = query.where(and_(*conditions))

        query = query.order_by(AuditLog.timestamp.desc()).limit(limit).offset(offset)
        result = await session.execute(query)
        return list(result.scalars().all())

    async def export_logs(
        self,
        session: AsyncSession,
        user_id: Optional[int] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> str:
        """Export audit logs as JSON."""
        logs = await self.search_logs(
            session,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            start_date=start_date,
            end_date=end_date,
            limit=10000,  # Higher limit for exports
        )
        return json.dumps(
            [
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
            indent=2,
        )

    async def rotate_logs(self, session: AsyncSession) -> int:
        """Delete old audit logs beyond retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        query = select(AuditLog).where(AuditLog.timestamp < cutoff_date)
        result = await session.execute(query)
        old_logs = result.scalars().all()

        count = len(old_logs)
        for log in old_logs:
            await session.delete(log)
        await session.commit()
        return count

    async def get_log_count(self, session: AsyncSession) -> int:
        """Get total number of audit log records."""
        from sqlalchemy import func

        query = select(func.count()).select_from(AuditLog)
        result = await session.execute(query)
        return result.scalar() or 0


# Global audit logger instance
audit_logger = AuditLogger()
