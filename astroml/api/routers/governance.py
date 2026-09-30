"""Governance API router for model governance operations.

Issue #637: REST API endpoints for audit logging, approval workflows,
compliance checks, model cards, and risk assessments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["governance"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class AuditQueryParams(BaseModel):
    """Query parameters for audit event search."""

    model_id: str | None = None
    event_type: str | None = None
    actor: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)

    model_config = ConfigDict(extra="forbid")


class ApprovalCreateRequest(BaseModel):
    """Request to create an approval workflow."""

    model_id: str = Field(..., min_length=1, max_length=256)
    reviewers: list[str] = Field(..., min_length=1)
    step_types: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class ApprovalActionRequest(BaseModel):
    """Request to approve or reject a workflow step."""

    step_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern=r"^(approve|reject)$")
    comments: str = ""


class ComplianceRunRequest(BaseModel):
    """Request to run compliance checks."""

    model_id: str = Field(..., min_length=1, max_length=256)
    standards: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RiskAssessmentRequest(BaseModel):
    """Request to create a risk assessment."""

    model_id: str = Field(..., min_length=1, max_length=256)
    findings: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ModelCardRequest(BaseModel):
    """Request to generate a model card."""

    model_name: str = Field(..., min_length=1, max_length=256)
    model_version: str = Field(..., min_length=1, max_length=128)
    model_type: str = "other"
    description: str = ""
    authors: list[str] = Field(default_factory=list)
    primary_use: str = ""
    primary_users: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    performance: dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RetirementCreateRequest(BaseModel):
    """Request to start a model retirement workflow."""

    model_id: str = Field(..., min_length=1, max_length=256)
    reason: str = Field(..., min_length=1)
    replacement_model_id: str | None = None
    deprecation_notice_days: int = Field(default=30, ge=1, le=365)

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Lazy service instances
# ---------------------------------------------------------------------------

_audit_logger = None
_workflow_manager = None
_compliance_checker = None
_card_generator = None


def _get_audit_logger() -> Any:
    global _audit_logger
    if _audit_logger is None:
        from astroml.governance.audit_logger import FileAuditStore, ModelAuditLogger

        _audit_logger = ModelAuditLogger(store=FileAuditStore())
    return _audit_logger


def _get_workflow_manager() -> Any:
    global _workflow_manager
    if _workflow_manager is None:
        from astroml.governance.approval_workflow import ApprovalWorkflowManager

        _workflow_manager = ApprovalWorkflowManager()
    return _workflow_manager


def _get_compliance_checker() -> Any:
    global _compliance_checker
    if _compliance_checker is None:
        from astroml.governance.compliance import ComplianceChecker

        _compliance_checker = ComplianceChecker()
    return _compliance_checker


def _get_card_generator() -> Any:
    global _card_generator
    if _card_generator is None:
        from astroml.governance.model_card import ModelCardGenerator

        _card_generator = ModelCardGenerator()
    return _card_generator


# ---------------------------------------------------------------------------
# Audit endpoints
# ---------------------------------------------------------------------------


@router.get("/governance/audit")
async def query_audit_log(
    model_id: str | None = Query(None),
    event_type: str | None = Query(None),
    actor: str | None = Query(None),
    start_time: str | None = Query(None),
    end_time: str | None = Query(None),
    limit: int = Query(default=100, ge=1, le=1000),
) -> dict[str, Any]:
    """Query audit events with optional filters."""
    from astroml.governance.audit_logger import AuditEventType

    audit = _get_audit_logger()

    audit_event_type = None
    if event_type:
        try:
            audit_event_type = AuditEventType[event_type.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")

    start_dt = datetime.fromisoformat(start_time) if start_time else None
    end_dt = datetime.fromisoformat(end_time) if end_time else None

    events = audit.query(
        model_id=model_id,
        event_type=audit_event_type,
        start_time=start_dt,
        end_time=end_dt,
        actor=actor,
        limit=limit,
    )

    return {
        "total": len(events),
        "events": [e.to_dict() for e in events],
    }


@router.get("/governance/audit/{model_id}")
async def get_model_audit_trail(model_id: str) -> dict[str, Any]:
    """Get the complete audit trail for a model."""
    audit = _get_audit_logger()
    events = audit.get_model_audit_trail(model_id)

    return {
        "model_id": model_id,
        "total_events": len(events),
        "audit_trail": [e.to_dict() for e in events],
    }


# ---------------------------------------------------------------------------
# Approval workflow endpoints
# ---------------------------------------------------------------------------


@router.post("/governance/approvals")
async def create_approval_workflow(body: ApprovalCreateRequest) -> dict[str, Any]:
    """Create a new approval workflow for a model."""
    from astroml.governance.approval_workflow import ApprovalStepType

    manager = _get_workflow_manager()

    step_types = None
    if body.step_types:
        try:
            step_types = [ApprovalStepType[st.upper()] for st in body.step_types]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid step type: {e}")

    workflow = manager.create_workflow(
        model_id=body.model_id,
        reviewers=body.reviewers,
        step_types=step_types,
        metadata=body.metadata,
    )

    # Audit log the creation
    audit = _get_audit_logger()
    from astroml.governance.audit_logger import AuditEventType

    audit.log(
        event_type=AuditEventType.APPROVAL_REQUESTED,
        model_id=body.model_id,
        details={"workflow_id": workflow.workflow_id, "reviewers": body.reviewers},
    )

    return workflow.to_dict()


@router.get("/governance/approvals/{workflow_id}")
async def get_approval_workflow(workflow_id: str) -> dict[str, Any]:
    """Get an approval workflow by ID."""
    manager = _get_workflow_manager()
    workflow = manager.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow.to_dict()


@router.get("/governance/approvals/model/{model_id}")
async def get_model_approvals(model_id: str) -> dict[str, Any]:
    """Get all approval workflows for a model."""
    manager = _get_workflow_manager()
    workflows = manager.get_model_workflows(model_id)
    return {
        "model_id": model_id,
        "total": len(workflows),
        "workflows": [w.to_dict() for w in workflows],
    }


@router.post("/governance/approvals/{workflow_id}/step")
async def action_approval_step(workflow_id: str, body: ApprovalActionRequest) -> dict[str, Any]:
    """Approve or reject a step in an approval workflow."""
    from astroml.governance.audit_logger import AuditEventType

    manager = _get_workflow_manager()
    workflow = manager.get_workflow(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Workflow not found")

    try:
        if body.action == "approve":
            workflow.approve_step(body.step_id, body.comments)
        else:
            workflow.reject_step(body.step_id, body.comments)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Audit log
    audit = _get_audit_logger()
    audit.log(
        event_type=(
            AuditEventType.APPROVAL_GRANTED if body.action == "approve"
            else AuditEventType.APPROVAL_DENIED
        ),
        model_id=workflow.model_id,
        details={
            "workflow_id": workflow_id,
            "step_id": body.step_id,
            "action": body.action,
            "comments": body.comments,
        },
    )

    return workflow.to_dict()


# ---------------------------------------------------------------------------
# Compliance endpoints
# ---------------------------------------------------------------------------


@router.post("/governance/compliance/check")
async def run_compliance_check(body: ComplianceRunRequest) -> dict[str, Any]:
    """Run compliance checks and return a report."""
    from astroml.governance.compliance import ComplianceStandard

    checker = _get_compliance_checker()

    standards = None
    if body.standards:
        try:
            standards = [ComplianceStandard[s.upper()] for s in body.standards]
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"Invalid standard: {e}")

    report = checker.run_checks(
        model_id=body.model_id,
        standards=standards,
        model_metadata=body.metadata,
    )

    # Audit log
    audit = _get_audit_logger()
    from astroml.governance.audit_logger import AuditEventType

    audit.log(
        event_type=AuditEventType.COMPLIANCE_CHECK_COMPLETED,
        model_id=body.model_id,
        details=report.to_dict(),
    )

    return report.to_dict()


# ---------------------------------------------------------------------------
# Risk assessment endpoints
# ---------------------------------------------------------------------------


@router.post("/governance/risk-assessment")
async def create_risk_assessment(body: RiskAssessmentRequest) -> dict[str, Any]:
    """Create a risk assessment for a model."""
    from astroml.governance.compliance import (
        RiskAssessment,
        RiskCategory,
        RiskFinding,
        RiskLevel,
    )

    assessment = RiskAssessment(model_id=body.model_id)

    for finding_data in body.findings:
        try:
            finding = RiskFinding(
                category=RiskCategory(finding_data["category"]),
                level=RiskLevel(finding_data["level"]),
                description=finding_data.get("description", ""),
                mitigation=finding_data.get("mitigation", ""),
                metadata=finding_data.get("metadata", {}),
            )
            assessment.add_finding(finding)
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid finding: {e}")

    # Audit log
    audit = _get_audit_logger()
    from astroml.governance.audit_logger import AuditEventType

    audit.log(
        event_type=AuditEventType.RISK_ASSESSMENT_COMPLETED,
        model_id=body.model_id,
        details=assessment.to_dict(),
    )

    return assessment.to_dict()


# ---------------------------------------------------------------------------
# Model card endpoints
# ---------------------------------------------------------------------------


@router.post("/governance/model-card")
async def generate_model_card(body: ModelCardRequest) -> dict[str, Any]:
    """Generate a model card for a model."""
    from astroml.governance.model_card import ModelType

    gen = _get_card_generator()

    try:
        model_type = ModelType(body.model_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid model_type: {body.model_type}")

    card = gen.generate(
        model_name=body.model_name,
        model_version=body.model_version,
        model_type=model_type,
        description=body.description,
        authors=body.authors,
        primary_use=body.primary_use,
        primary_users=body.primary_users,
        out_of_scope=body.out_of_scope,
        limitations=body.limitations,
        performance=body.performance,
    )

    return card.to_dict()


# ---------------------------------------------------------------------------
# Model retirement endpoints
# ---------------------------------------------------------------------------


@router.post("/governance/retire")
async def create_retirement(body: RetirementCreateRequest) -> dict[str, Any]:
    """Start a model retirement workflow."""
    manager = _get_workflow_manager()

    retirement = manager.create_retirement(
        model_id=body.model_id,
        reason=body.reason,
        replacement_model_id=body.replacement_model_id,
        deprecation_notice_days=body.deprecation_notice_days,
    )

    # Audit log
    audit = _get_audit_logger()
    from astroml.governance.audit_logger import AuditEventType

    audit.log(
        event_type=AuditEventType.MODEL_RETIRED,
        model_id=body.model_id,
        details=retirement.to_dict(),
    )

    return retirement.to_dict()


@router.get("/governance/retirements")
async def list_active_retirements() -> dict[str, Any]:
    """List all active (non-cancelled, non-archived) retirements."""
    manager = _get_workflow_manager()
    retirements = manager.get_active_retirements()

    return {
        "total": len(retirements),
        "retirements": [r.to_dict() for r in retirements],
    }


# ---------------------------------------------------------------------------
# Governance dashboard
# ---------------------------------------------------------------------------


@router.get("/governance/dashboard")
async def governance_dashboard() -> dict[str, Any]:
    """Get a summary of governance metrics for the dashboard."""
    audit = _get_audit_logger()
    manager = _get_workflow_manager()

    # Recent audit activity
    recent_events = audit.query(limit=50)

    return {
        "audit_stats": {
            "total_events": len(recent_events),
            "recent_events": [e.to_dict() for e in recent_events[:10]],
        },
        "active_retirements": len(manager.get_active_retirements()),
        "generated_at": datetime.utcnow().isoformat(),
    }