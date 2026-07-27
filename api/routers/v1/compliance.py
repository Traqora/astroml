"""Compliance and audit endpoints for LLM interactions (issue #412)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_auth, AuthContext
from api.database import get_db
from astroml.llm.compliance_logger import compliance_logger

router = APIRouter(prefix="/api/v1/compliance", tags=["compliance"])


class AuditReportResponse(dict):
    """Audit report response."""
    pass


class ExportResponse(dict):
    """Export response."""
    pass


@router.get("/audit-report")
async def get_audit_report(
    start_date: Optional[str] = Query(None, description="ISO format start date"),
    end_date: Optional[str] = Query(None, description="ISO format end date"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> dict:
    """Get LLM compliance audit report.

    Returns summary of all LLM interactions with PII detection statistics.
    """
    try:
        parsed_start = None
        parsed_end = None

        if start_date:
            parsed_start = datetime.fromisoformat(start_date)
        if end_date:
            parsed_end = datetime.fromisoformat(end_date)

        report = await compliance_logger.get_audit_report(
            db,
            start_date=parsed_start,
            end_date=parsed_end,
        )
        return AuditReportResponse(report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export")
async def export_logs(
    format: str = Query("json", description="Export format: json or csv"),
    start_date: Optional[str] = Query(None, description="ISO format start date"),
    end_date: Optional[str] = Query(None, description="ISO format end date"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> Any:
    """Export compliance logs in specified format.

    Supports JSON and CSV formats for audit trail export.
    """
    try:
        if format not in ["json", "csv"]:
            raise ValueError("Format must be 'json' or 'csv'")

        parsed_start = None
        parsed_end = None

        if start_date:
            parsed_start = datetime.fromisoformat(start_date)
        if end_date:
            parsed_end = datetime.fromisoformat(end_date)

        exported = await compliance_logger.export_logs(
            db,
            start_date=parsed_start,
            end_date=parsed_end,
            format=format,
        )

        if format == "csv":
            from fastapi.responses import StreamingResponse
            from io import BytesIO

            csv_bytes = BytesIO(exported.encode())
            return StreamingResponse(
                iter([csv_bytes.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=compliance_logs.csv"},
            )
        else:
            return ExportResponse({"data": exported})
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid request")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/logs")
async def get_logs(
    user_id: Optional[int] = Query(None),
    interaction_type: Optional[str] = Query(None),
    feature: Optional[str] = Query(None),
    pii_detected: Optional[bool] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_current_auth),
) -> dict:
    """Get compliance logs with optional filters."""
    try:
        logs = await compliance_logger.search_logs(
            db,
            user_id=user_id,
            interaction_type=interaction_type,
            feature=feature,
            pii_detected=pii_detected,
            limit=limit,
            offset=offset,
        )

        return {
            "total": len(logs),
            "logs": [
                {
                    "id": log.id,
                    "timestamp": log.created_at.isoformat(),
                    "user_id": log.user_id,
                    "username": log.username,
                    "interaction_type": log.interaction_type,
                    "feature": log.feature,
                    "status": log.status,
                    "pii_detected": log.pii_detected,
                    "pii_types": log.pii_types,
                    "latency_ms": log.latency_ms,
                    "tokens_used": log.tokens_used,
                }
                for log in logs
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
