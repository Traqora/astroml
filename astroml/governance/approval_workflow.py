"""Approval workflow and model retirement for model governance.

Issue #637 Step 2 & 7: Implements approval workflow for model deployment
and model retirement workflow.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval step."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    EXPIRED = "expired"


class ApprovalStepType(Enum):
    """Types of approval steps."""

    TECHNICAL_REVIEW = auto()
    SECURITY_REVIEW = auto()
    COMPLIANCE_REVIEW = auto()
    BUSINESS_REVIEW = auto()
    ETHICS_REVIEW = auto()
    STAKEHOLDER_SIGN_OFF = auto()


@dataclass
class ApprovalStep:
    """A single step in an approval workflow.

    Attributes:
        step_id: Unique step identifier.
        step_type: Type of approval required.
        reviewer: Who must approve (user ID, role, or team).
        status: Current approval status.
        comments: Reviewer comments.
        created_at: When the step was created.
        resolved_at: When the step was resolved.
        deadline: Optional deadline for this step.
    """

    step_type: ApprovalStepType
    reviewer: str
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: ApprovalStatus = ApprovalStatus.PENDING
    comments: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    deadline: datetime | None = None

    def approve(self, comments: str = "") -> None:
        """Approve this step."""
        if self.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot approve step in status {self.status.value}")
        self.status = ApprovalStatus.APPROVED
        self.comments = comments
        self.resolved_at = datetime.now(timezone.utc)

    def reject(self, comments: str = "") -> None:
        """Reject this step."""
        if self.status != ApprovalStatus.PENDING:
            raise ValueError(f"Cannot reject step in status {self.status.value}")
        self.status = ApprovalStatus.REJECTED
        self.comments = comments
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.name,
            "reviewer": self.reviewer,
            "status": self.status.value,
            "comments": self.comments,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }


@dataclass
class ApprovalWorkflow:
    """A multi-step approval workflow for model deployment.

    Models must pass through all required approval steps before deployment.

    Example:
        workflow = ApprovalWorkflow(
            model_id="fraud-detector-v2",
            steps=[
                ApprovalStep(ApprovalStepType.TECHNICAL_REVIEW, reviewer="ml-eng-team"),
                ApprovalStep(ApprovalStepType.SECURITY_REVIEW, reviewer="security-team"),
                ApprovalStep(ApprovalStepType.COMPLIANCE_REVIEW, reviewer="compliance-team"),
            ],
        )

        workflow.approve_step(workflow.steps[0].step_id, "Code review passed")
        workflow.approve_step(workflow.steps[1].step_id, "Security scan clean")
    """

    model_id: str
    steps: list[ApprovalStep]
    workflow_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> ApprovalStatus:
        """Aggregate status across all steps."""
        if any(s.status == ApprovalStatus.REJECTED for s in self.steps):
            return ApprovalStatus.REJECTED
        if all(s.status == ApprovalStatus.APPROVED for s in self.steps):
            return ApprovalStatus.APPROVED
        return ApprovalStatus.PENDING

    @property
    def pending_steps(self) -> list[ApprovalStep]:
        """Return steps that are still pending."""
        return [s for s in self.steps if s.status == ApprovalStatus.PENDING]

    @property
    def approved_steps(self) -> list[ApprovalStep]:
        """Return steps that have been approved."""
        return [s for s in self.steps if s.status == ApprovalStatus.APPROVED]

    @property
    def progress(self) -> float:
        """Fraction of steps approved (0.0 to 1.0)."""
        if not self.steps:
            return 1.0
        return len(self.approved_steps) / len(self.steps)

    def get_step(self, step_id: str) -> ApprovalStep | None:
        """Find a step by its ID."""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def approve_step(self, step_id: str, comments: str = "") -> ApprovalStep:
        """Approve a specific step.

        Args:
            step_id: ID of the step to approve.
            comments: Approval comments.

        Returns:
            The updated step.

        Raises:
            ValueError: If step not found or already resolved.
        """
        step = self.get_step(step_id)
        if step is None:
            raise ValueError(f"Step {step_id} not found in workflow {self.workflow_id}")
        step.approve(comments)
        logger.info(f"Workflow {self.workflow_id}: step {step.step_type.name} approved by {step.reviewer}")
        return step

    def reject_step(self, step_id: str, comments: str = "") -> ApprovalStep:
        """Reject a specific step.

        Args:
            step_id: ID of the step to reject.
            comments: Rejection reason.

        Returns:
            The updated step.

        Raises:
            ValueError: If step not found or already resolved.
        """
        step = self.get_step(step_id)
        if step is None:
            raise ValueError(f"Step {step_id} not found in workflow {self.workflow_id}")
        step.reject(comments)
        logger.warning(f"Workflow {self.workflow_id}: step {step.step_type.name} REJECTED by {step.reviewer}: {comments}")
        return step

    def is_ready_for_deployment(self) -> bool:
        """Check if all steps are approved and model can be deployed."""
        return self.status == ApprovalStatus.APPROVED

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "model_id": self.model_id,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "metadata": self.metadata,
        }


class ModelRetirementStatus(Enum):
    """Status of model retirement process."""

    PROPOSED = "proposed"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    RETIRED = "retired"
    ARCHIVED = "archived"
    CANCELLED = "cancelled"


@dataclass
class ModelRetirementWorkflow:
    """Workflow for safely retiring a model from production.

    Handles the full retirement lifecycle: proposal → review → approval →
    deprecation → archival.

    Example:
        retirement = ModelRetirementWorkflow(
            model_id="fraud-detector-v1",
            replacement_model_id="fraud-detector-v2",
            reason="Replaced by v2 with improved accuracy",
        )
        retirement.start_review()
    """

    model_id: str
    replacement_model_id: str | None = None
    reason: str = ""
    workflow_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: ModelRetirementStatus = ModelRetirementStatus.PROPOSED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retired_at: datetime | None = None
    archived_at: datetime | None = None
    affected_endpoints: list[str] = field(default_factory=list)
    deprecation_notice_days: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)

    def start_review(self) -> None:
        """Begin the review process."""
        if self.status != ModelRetirementStatus.PROPOSED:
            raise ValueError(f"Cannot start review from status {self.status.value}")
        self.status = ModelRetirementStatus.UNDER_REVIEW
        logger.info(f"Retirement workflow {self.workflow_id}: review started for {self.model_id}")

    def approve(self) -> None:
        """Approve the retirement."""
        if self.status != ModelRetirementStatus.UNDER_REVIEW:
            raise ValueError(f"Cannot approve from status {self.status.value}")
        self.status = ModelRetirementStatus.APPROVED
        logger.info(f"Retirement workflow {self.workflow_id}: {self.model_id} retirement approved")

    def start_retirement(self) -> None:
        """Begin the technical retirement process."""
        if self.status != ModelRetirementStatus.APPROVED:
            raise ValueError(f"Cannot start retirement from status {self.status.value}")
        self.status = ModelRetirementStatus.IN_PROGRESS
        logger.info(f"Retirement workflow {self.workflow_id}: {self.model_id} retirement in progress")

    def complete_retirement(self) -> None:
        """Mark the model as retired."""
        if self.status != ModelRetirementStatus.IN_PROGRESS:
            raise ValueError(f"Cannot complete retirement from status {self.status.value}")
        self.status = ModelRetirementStatus.RETIRED
        self.retired_at = datetime.now(timezone.utc)
        logger.info(f"Retirement workflow {self.workflow_id}: {self.model_id} retired")

    def archive(self) -> None:
        """Archive the retired model artifacts."""
        if self.status != ModelRetirementStatus.RETIRED:
            raise ValueError(f"Cannot archive from status {self.status.value}")
        self.status = ModelRetirementStatus.ARCHIVED
        self.archived_at = datetime.now(timezone.utc)
        logger.info(f"Retirement workflow {self.workflow_id}: {self.model_id} archived")

    def cancel(self, reason: str = "") -> None:
        """Cancel the retirement process."""
        if self.status in (ModelRetirementStatus.RETIRED, ModelRetirementStatus.ARCHIVED):
            raise ValueError(f"Cannot cancel already retired model")
        self.status = ModelRetirementStatus.CANCELLED
        self.reason = reason or "Cancelled by administrator"
        logger.info(f"Retirement workflow {self.workflow_id}: {self.model_id} retirement cancelled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "model_id": self.model_id,
            "replacement_model_id": self.replacement_model_id,
            "status": self.status.value,
            "reason": self.reason,
            "created_at": self.created_at.isoformat(),
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "affected_endpoints": self.affected_endpoints,
            "deprecation_notice_days": self.deprecation_notice_days,
        }


class ApprovalWorkflowManager:
    """Manages multiple approval workflows for model governance."""

    def __init__(self) -> None:
        self._workflows: dict[str, ApprovalWorkflow] = {}
        self._retirements: dict[str, ModelRetirementWorkflow] = {}

    def create_workflow(
        self,
        model_id: str,
        reviewers: Sequence[str],
        step_types: Sequence[ApprovalStepType] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ApprovalWorkflow:
        """Create a new approval workflow.

        Args:
            model_id: Model to approve.
            reviewers: List of reviewers (one per step).
            step_types: Optional step types (defaults to standard pipeline).
            metadata: Optional metadata.

        Returns:
            Created ApprovalWorkflow.
        """
        if step_types is None:
            step_types = [
                ApprovalStepType.TECHNICAL_REVIEW,
                ApprovalStepType.SECURITY_REVIEW,
                ApprovalStepType.COMPLIANCE_REVIEW,
            ]

        if len(step_types) > len(reviewers):
            raise ValueError(f"Not enough reviewers: {len(reviewers)} for {len(step_types)} steps")

        steps = [
            ApprovalStep(step_type=st, reviewer=rev)
            for st, rev in zip(step_types, reviewers)
        ]

        workflow = ApprovalWorkflow(
            model_id=model_id,
            steps=steps,
            metadata=metadata or {},
        )
        self._workflows[workflow.workflow_id] = workflow
        logger.info(f"Created approval workflow {workflow.workflow_id} for {model_id}")
        return workflow

    def get_workflow(self, workflow_id: str) -> ApprovalWorkflow | None:
        """Get a workflow by ID."""
        return self._workflows.get(workflow_id)

    def get_model_workflows(self, model_id: str) -> list[ApprovalWorkflow]:
        """Get all workflows for a model."""
        return [w for w in self._workflows.values() if w.model_id == model_id]

    def create_retirement(
        self,
        model_id: str,
        reason: str,
        replacement_model_id: str | None = None,
        deprecation_notice_days: int = 30,
    ) -> ModelRetirementWorkflow:
        """Start a model retirement workflow.

        Args:
            model_id: Model to retire.
            reason: Reason for retirement.
            replacement_model_id: Model replacing this one.
            deprecation_notice_days: Days of deprecation notice.

        Returns:
            Created retirement workflow.
        """
        retirement = ModelRetirementWorkflow(
            model_id=model_id,
            replacement_model_id=replacement_model_id,
            reason=reason,
            deprecation_notice_days=deprecation_notice_days,
        )
        self._retirements[retirement.workflow_id] = retirement
        logger.info(f"Created retirement workflow {retirement.workflow_id} for {model_id}")
        return retirement

    def get_retirement(self, workflow_id: str) -> ModelRetirementWorkflow | None:
        """Get a retirement workflow by ID."""
        return self._retirements.get(workflow_id)

    def get_active_retirements(self) -> list[ModelRetirementWorkflow]:
        """Get all non-cancelled, non-archived retirements."""
        return [
            r for r in self._retirements.values()
            if r.status not in (ModelRetirementStatus.CANCELLED, ModelRetirementStatus.ARCHIVED)
        ]


__all__ = [
    "ApprovalStatus",
    "ApprovalStepType",
    "ApprovalStep",
    "ApprovalWorkflow",
    "ModelRetirementStatus",
    "ModelRetirementWorkflow",
    "ApprovalWorkflowManager",
]