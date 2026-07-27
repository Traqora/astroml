"""Compliance and audit logging service for LLM interactions (issue #412)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.orm import LLMComplianceLog
from astroml.llm.pii_redactor import pii_redactor


class ComplianceLogger:
    """Service for logging LLM interactions with compliance and audit trail."""

    async def log_interaction(
        self,
        session: AsyncSession,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        interaction_type: str = "query",
        feature: str = "default",
        prompt: str = "",
        response: str = "",
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> LLMComplianceLog:
        """Log an LLM interaction with automatic PII redaction.

        Args:
            session: Database session
            user_id: User ID
            username: Username
            interaction_type: Type of interaction (query, explain, translate, etc.)
            feature: Feature name
            prompt: Original prompt text
            response: Original response text
            model_used: Model used for the interaction
            tokens_used: Number of tokens used
            latency_ms: Latency in milliseconds
            status: Interaction status (success, error, etc.)
            error_message: Error message if status is error
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Created LLMComplianceLog record
        """
        prompt_redacted, prompt_pii = pii_redactor.redact(prompt)
        response_redacted, response_pii = pii_redactor.redact(response)

        pii_types = {}
        for pii_type, detected in prompt_pii.items():
            if detected:
                pii_types.setdefault(pii_type, []).append("prompt")
        for pii_type, detected in response_pii.items():
            if detected:
                pii_types.setdefault(pii_type, []).append("response")

        pii_detected = bool(pii_types)

        log_record = LLMComplianceLog(
            user_id=user_id,
            username=username,
            interaction_type=interaction_type,
            feature=feature,
            prompt_redacted=prompt_redacted,
            response_redacted=response_redacted,
            model_used=model_used,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            status=status,
            error_message=error_message,
            pii_detected=pii_detected,
            pii_types=pii_types if pii_types else None,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        session.add(log_record)
        await session.commit()
        await session.refresh(log_record)
        return log_record

    async def search_logs(
        self,
        session: AsyncSession,
        user_id: Optional[int] = None,
        interaction_type: Optional[str] = None,
        feature: Optional[str] = None,
        pii_detected: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[LLMComplianceLog]:
        """Search compliance logs with filters.

        Args:
            session: Database session
            user_id: Filter by user ID
            interaction_type: Filter by interaction type
            feature: Filter by feature
            pii_detected: Filter by PII detection
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Result limit
            offset: Result offset

        Returns:
            List of matching compliance logs
        """
        query = select(LLMComplianceLog)

        conditions = []
        if user_id is not None:
            conditions.append(LLMComplianceLog.user_id == user_id)
        if interaction_type is not None:
            conditions.append(LLMComplianceLog.interaction_type == interaction_type)
        if feature is not None:
            conditions.append(LLMComplianceLog.feature == feature)
        if pii_detected is not None:
            conditions.append(LLMComplianceLog.pii_detected == pii_detected)
        if start_date is not None:
            conditions.append(LLMComplianceLog.created_at >= start_date)
        if end_date is not None:
            conditions.append(LLMComplianceLog.created_at <= end_date)

        if conditions:
            from sqlalchemy import and_
            query = query.where(and_(*conditions))

        query = (
            query.order_by(LLMComplianceLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(query)
        return list(result.scalars().all())

    async def get_audit_report(
        self,
        session: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict:
        """Generate audit report for LLM interactions.

        Args:
            session: Database session
            start_date: Report start date
            end_date: Report end date

        Returns:
            Audit report dictionary
        """
        logs = await self.search_logs(
            session,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        total_interactions = len(logs)
        successful = sum(1 for log in logs if log.status == "success")
        failed = sum(1 for log in logs if log.status == "error")
        pii_incidents = sum(1 for log in logs if log.pii_detected)

        by_feature = {}
        for log in logs:
            if log.feature not in by_feature:
                by_feature[log.feature] = {
                    "count": 0,
                    "avg_latency": 0,
                    "total_latency": 0,
                }
            by_feature[log.feature]["count"] += 1
            if log.latency_ms:
                by_feature[log.feature]["total_latency"] += log.latency_ms

        for feature_data in by_feature.values():
            if feature_data["count"] > 0:
                feature_data["avg_latency"] = (
                    feature_data["total_latency"] / feature_data["count"]
                )
            del feature_data["total_latency"]

        return {
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
            "summary": {
                "total_interactions": total_interactions,
                "successful": successful,
                "failed": failed,
                "pii_incidents": pii_incidents,
                "pii_percentage": (
                    round(100 * pii_incidents / total_interactions, 2)
                    if total_interactions > 0
                    else 0
                ),
            },
            "by_feature": by_feature,
        }

    async def export_logs(
        self,
        session: AsyncSession,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        format: str = "json",
    ) -> str:
        """Export compliance logs in specified format.

        Args:
            session: Database session
            start_date: Export start date
            end_date: Export end date
            format: Export format (json or csv)

        Returns:
            Exported logs as string
        """
        import json
        import csv
        from io import StringIO

        logs = await self.search_logs(
            session,
            start_date=start_date,
            end_date=end_date,
            limit=100000,
        )

        if format == "json":
            return json.dumps(
                [
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
                indent=2,
            )
        elif format == "csv":
            output = StringIO()
            writer = csv.DictWriter(
                output,
                fieldnames=[
                    "id",
                    "timestamp",
                    "user_id",
                    "username",
                    "interaction_type",
                    "feature",
                    "status",
                    "pii_detected",
                    "latency_ms",
                    "tokens_used",
                ],
            )
            writer.writeheader()
            for log in logs:
                writer.writerow(
                    {
                        "id": log.id,
                        "timestamp": log.created_at.isoformat(),
                        "user_id": log.user_id,
                        "username": log.username,
                        "interaction_type": log.interaction_type,
                        "feature": log.feature,
                        "status": log.status,
                        "pii_detected": log.pii_detected,
                        "latency_ms": log.latency_ms,
                        "tokens_used": log.tokens_used,
                    }
                )
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")


compliance_logger = ComplianceLogger()
