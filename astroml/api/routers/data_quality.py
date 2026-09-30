from astroml.utils.exceptions import AstroMLError
"""FastAPI router for data quality monitoring and reporting endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from astroml.validation.data_quality.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
)
from astroml.validation.data_quality.checks import DataQualityValidator
from astroml.validation.data_quality.monitor import DataQualityMonitor
from astroml.validation.data_quality.reporter import DataQualityReporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data-quality", tags=["data-quality"])

_alert_manager = AlertManager()
_reporter = DataQualityReporter()
_monitor = DataQualityMonitor(alert_manager=_alert_manager, reporter=_reporter)
_validator = DataQualityValidator()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class CheckRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1)
    required_fields: list[str] | None = None
    max_null_rate: float = 0.05


class MonitorBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(..., min_length=1)
    batch_id: str | None = None


class AddRuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    metric_name: str = Field(..., min_length=1)
    dimension: str = Field(default="completeness")
    operator: str = Field(..., pattern=r"^(<|<=|>|>=|==|!=)$")
    threshold: float
    severity: str = Field(default="warning", pattern=r"^(info|warning|error|critical)$")
    description: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/check", summary="Run immediate data quality checks on records")
async def check_data_quality(payload: CheckRequest) -> dict[str, Any]:
    """Run immediate data quality checks on provided transactions or records."""
    try:
        report = _validator.validate_batch(payload.records)
        return {
            "status": "success",
            "quality_score": report.quality_score,
            "total_records": report.total_records,
            "valid_records": report.valid_records,
            "dimension_scores": report.dimension_scores,
            "summary": report.summary,
            "check_results": [
                {
                    "check_name": c.check_name,
                    "dimension": c.dimension.value if hasattr(c.dimension, "value") else str(c.dimension),
                    "is_valid": c.is_valid,
                    "score": round(c.score * 100.0, 2),
                    "message": c.message,
                    "severity": c.severity.value if hasattr(c.severity, "value") else str(c.severity),
                }
                for c in report.check_results
            ],
        }
    except AstroMLError as e:
        logger.error("Data quality check failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/monitor", summary="Ingest batch into quality monitoring pipeline")
async def monitor_batch(payload: MonitorBatchRequest) -> dict[str, Any]:
    """Process a batch through continuous monitoring, alerting, and metric tracking."""
    try:
        report = _monitor.process_batch(payload.records, batch_id=payload.batch_id)
        active_alerts = _monitor.get_active_alerts()
        return {
            "status": "success",
            "batch_id": report.batch_id,
            "quality_score": report.quality_score,
            "dimension_scores": report.dimension_scores,
            "total_records": report.total_records,
            "active_alerts_count": len(active_alerts),
            "created_at": report.created_at,
        }
    except AstroMLError as e:
        logger.error("Monitoring batch processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/metrics", summary="Get monitored metrics history")
async def get_metrics_history(limit: int = Query(default=100, ge=1, le=1000)) -> dict[str, Any]:
    """Retrieve recorded historical data quality metrics."""
    return {
        "status": "success",
        "data": _monitor.get_metrics_history(limit=limit),
    }


@router.get("/trends", summary="Get quality score trends")
async def get_quality_trends() -> dict[str, Any]:
    """Get quality score trend analysis across monitored batches."""
    return {
        "status": "success",
        "data": _monitor.get_quality_trend(),
    }


@router.get("/reports/latest", summary="Get the latest generated data quality report")
async def get_latest_report(
    format: str = Query(default="json", pattern=r"^(json|markdown|html)$")
) -> Any:
    """Retrieve latest data quality report formatted as JSON, Markdown, or HTML."""
    report = _monitor.get_latest_report()
    if not report:
        raise HTTPException(status_code=404, detail="No quality reports generated yet.")

    alerts = _monitor.get_active_alerts()
    if format == "json":
        return {"status": "success", "data": _reporter.generate_report(report, alerts)}
    elif format == "markdown":
        return {"status": "success", "content": _reporter.to_markdown(report, alerts)}
    elif format == "html":
        return {"status": "success", "content": _reporter.to_html(report, alerts)}


@router.post("/reports", summary="Generate custom quality report for records")
async def generate_custom_report(payload: CheckRequest) -> dict[str, Any]:
    """Generate and return a full data quality report for provided records."""
    report = _validator.validate_batch(payload.records)
    alerts = _alert_manager.evaluate_metrics(
        {
            "completeness": report.dimension_scores.get("completeness", 100.0),
            "consistency": report.dimension_scores.get("consistency", 100.0),
            "accuracy": report.dimension_scores.get("accuracy", 100.0),
            "timeliness": report.dimension_scores.get("timeliness", 100.0),
            "overall_score": report.quality_score,
        }
    )
    return {
        "status": "success",
        "data": _reporter.generate_report(report, alerts),
    }


@router.get("/alerts", summary="List data quality alerts")
async def list_alerts(
    status: str | None = Query(default=None, pattern=r"^(active|resolved|silenced)$"),
    severity: str | None = Query(default=None, pattern=r"^(info|warning|error|critical)$"),
) -> dict[str, Any]:
    """List data quality alerts with optional filtering."""
    alerts = _alert_manager.list_alerts(status=status, severity=severity)
    return {
        "status": "success",
        "total": len(alerts),
        "alerts": [
            {
                "alert_id": a.alert_id,
                "rule_name": a.rule_name,
                "metric_name": a.metric_name,
                "dimension": a.dimension,
                "current_value": a.current_value,
                "threshold": a.threshold,
                "severity": a.severity.value if hasattr(a.severity, "value") else str(a.severity),
                "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                "message": a.message,
                "created_at": a.created_at,
                "resolved_at": a.resolved_at,
            }
            for a in alerts
        ],
    }


@router.post("/alerts/rules", summary="Add or update an alert rule")
async def add_alert_rule(payload: AddRuleRequest) -> dict[str, Any]:
    """Add a new alert threshold rule."""
    rule = AlertRule(
        rule_id=payload.rule_id,
        name=payload.name,
        metric_name=payload.metric_name,
        dimension=payload.dimension,
        operator=payload.operator,
        threshold=payload.threshold,
        severity=AlertSeverity(payload.severity),
        description=payload.description,
    )
    _alert_manager.add_rule(rule)
    return {"status": "success", "message": f"Rule '{rule.name}' saved.", "rule_id": rule.rule_id}


@router.get("/alerts/rules", summary="List configured alert rules")
async def list_alert_rules() -> dict[str, Any]:
    """List all configured alert rules."""
    rules = _alert_manager.list_rules()
    return {
        "status": "success",
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "metric_name": r.metric_name,
                "dimension": str(r.dimension),
                "operator": r.operator,
                "threshold": r.threshold,
                "severity": r.severity.value if hasattr(r.severity, "value") else str(r.severity),
                "description": r.description,
                "enabled": r.enabled,
            }
            for r in rules
        ],
    }


@router.post("/alerts/{alert_id}/resolve", summary="Resolve a quality alert")
async def resolve_alert(alert_id: str) -> dict[str, Any]:
    """Resolve an active alert by ID."""
    success = _alert_manager.resolve_alert(alert_id, resolved_by="api_user")
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return {"status": "success", "message": f"Alert '{alert_id}' resolved."}
