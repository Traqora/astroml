"""Tests for astroml.governance.approval_workflow.

Covers the ApprovalStep/ApprovalWorkflow state machine, the
ModelRetirementWorkflow lifecycle, and an end-to-end scenario driven
through ApprovalWorkflowManager from creation through deployment
readiness and, separately, model retirement.
"""

from __future__ import annotations

import pytest

from astroml.governance.approval_workflow import (
    ApprovalStatus,
    ApprovalStep,
    ApprovalStepType,
    ApprovalWorkflow,
    ApprovalWorkflowManager,
    ModelRetirementStatus,
    ModelRetirementWorkflow,
)


class TestApprovalStep:
    def test_approve_transitions_status_and_sets_resolved_at(self):
        step = ApprovalStep(ApprovalStepType.TECHNICAL_REVIEW, reviewer="ml-eng")
        assert step.status == ApprovalStatus.PENDING
        assert step.resolved_at is None

        step.approve("looks good")

        assert step.status == ApprovalStatus.APPROVED
        assert step.comments == "looks good"
        assert step.resolved_at is not None

    def test_reject_transitions_status(self):
        step = ApprovalStep(ApprovalStepType.SECURITY_REVIEW, reviewer="sec-team")
        step.reject("found a vulnerability")
        assert step.status == ApprovalStatus.REJECTED
        assert step.comments == "found a vulnerability"

    def test_cannot_approve_a_resolved_step(self):
        step = ApprovalStep(ApprovalStepType.TECHNICAL_REVIEW, reviewer="ml-eng")
        step.approve()
        with pytest.raises(ValueError, match="Cannot approve"):
            step.approve()

    def test_cannot_reject_a_resolved_step(self):
        step = ApprovalStep(ApprovalStepType.TECHNICAL_REVIEW, reviewer="ml-eng")
        step.reject()
        with pytest.raises(ValueError, match="Cannot reject"):
            step.reject()

    def test_to_dict_serializes_all_fields(self):
        step = ApprovalStep(ApprovalStepType.COMPLIANCE_REVIEW, reviewer="compliance")
        step.approve("ok")
        data = step.to_dict()
        assert data["step_type"] == "COMPLIANCE_REVIEW"
        assert data["reviewer"] == "compliance"
        assert data["status"] == "approved"
        assert data["resolved_at"] is not None


class TestApprovalWorkflow:
    def _workflow(self) -> ApprovalWorkflow:
        return ApprovalWorkflow(
            model_id="fraud-detector-v2",
            steps=[
                ApprovalStep(ApprovalStepType.TECHNICAL_REVIEW, reviewer="ml-eng-team"),
                ApprovalStep(ApprovalStepType.SECURITY_REVIEW, reviewer="security-team"),
                ApprovalStep(ApprovalStepType.COMPLIANCE_REVIEW, reviewer="compliance-team"),
            ],
        )

    def test_status_is_pending_until_all_steps_approved(self):
        workflow = self._workflow()
        assert workflow.status == ApprovalStatus.PENDING

        workflow.approve_step(workflow.steps[0].step_id)
        workflow.approve_step(workflow.steps[1].step_id)
        assert workflow.status == ApprovalStatus.PENDING

        workflow.approve_step(workflow.steps[2].step_id)
        assert workflow.status == ApprovalStatus.APPROVED

    def test_any_rejection_makes_workflow_rejected(self):
        workflow = self._workflow()
        workflow.approve_step(workflow.steps[0].step_id)
        workflow.reject_step(workflow.steps[1].step_id, "security concerns")

        assert workflow.status == ApprovalStatus.REJECTED
        # A rejection elsewhere still counts even if other steps later approve.
        workflow.approve_step(workflow.steps[2].step_id)
        assert workflow.status == ApprovalStatus.REJECTED

    def test_is_ready_for_deployment_reflects_status(self):
        workflow = self._workflow()
        assert workflow.is_ready_for_deployment() is False

        for step in workflow.steps:
            workflow.approve_step(step.step_id)

        assert workflow.is_ready_for_deployment() is True

    def test_progress_tracks_fraction_approved(self):
        workflow = self._workflow()
        assert workflow.progress == 0.0

        workflow.approve_step(workflow.steps[0].step_id)
        assert workflow.progress == pytest.approx(1 / 3)

    def test_progress_is_one_when_no_steps(self):
        workflow = ApprovalWorkflow(model_id="m", steps=[])
        assert workflow.progress == 1.0

    def test_pending_and_approved_steps_partition_correctly(self):
        workflow = self._workflow()
        workflow.approve_step(workflow.steps[0].step_id)

        assert workflow.approved_steps == [workflow.steps[0]]
        assert workflow.pending_steps == [workflow.steps[1], workflow.steps[2]]

    def test_get_step_returns_none_for_unknown_id(self):
        workflow = self._workflow()
        assert workflow.get_step("unknown") is None

    def test_approve_step_raises_for_unknown_step_id(self):
        workflow = self._workflow()
        with pytest.raises(ValueError, match="not found"):
            workflow.approve_step("unknown")

    def test_to_dict_reflects_aggregate_state(self):
        workflow = self._workflow()
        workflow.approve_step(workflow.steps[0].step_id)
        data = workflow.to_dict()
        assert data["model_id"] == "fraud-detector-v2"
        assert data["status"] == "pending"
        assert data["progress"] == pytest.approx(1 / 3)
        assert len(data["steps"]) == 3


class TestModelRetirementWorkflow:
    def test_full_lifecycle_happy_path(self):
        retirement = ModelRetirementWorkflow(
            model_id="fraud-detector-v1",
            replacement_model_id="fraud-detector-v2",
            reason="Replaced by v2 with improved accuracy",
        )
        assert retirement.status == ModelRetirementStatus.PROPOSED

        retirement.start_review()
        assert retirement.status == ModelRetirementStatus.UNDER_REVIEW

        retirement.approve()
        assert retirement.status == ModelRetirementStatus.APPROVED

        retirement.start_retirement()
        assert retirement.status == ModelRetirementStatus.IN_PROGRESS

        retirement.complete_retirement()
        assert retirement.status == ModelRetirementStatus.RETIRED
        assert retirement.retired_at is not None

        retirement.archive()
        assert retirement.status == ModelRetirementStatus.ARCHIVED
        assert retirement.archived_at is not None

    def test_cannot_skip_lifecycle_steps(self):
        retirement = ModelRetirementWorkflow(model_id="m")
        with pytest.raises(ValueError, match="Cannot approve"):
            retirement.approve()
        with pytest.raises(ValueError, match="Cannot start retirement"):
            retirement.start_retirement()

    def test_cancel_sets_reason_and_status(self):
        retirement = ModelRetirementWorkflow(model_id="m")
        retirement.start_review()
        retirement.cancel("no longer needed")

        assert retirement.status == ModelRetirementStatus.CANCELLED
        assert retirement.reason == "no longer needed"

    def test_cancel_defaults_reason_when_not_given(self):
        retirement = ModelRetirementWorkflow(model_id="m")
        retirement.cancel()
        assert retirement.reason == "Cancelled by administrator"

    def test_cannot_cancel_already_retired_model(self):
        retirement = ModelRetirementWorkflow(model_id="m")
        retirement.start_review()
        retirement.approve()
        retirement.start_retirement()
        retirement.complete_retirement()

        with pytest.raises(ValueError, match="already retired"):
            retirement.cancel()

    def test_to_dict_serializes_lifecycle_timestamps(self):
        retirement = ModelRetirementWorkflow(model_id="m")
        data = retirement.to_dict()
        assert data["status"] == "proposed"
        assert data["retired_at"] is None
        assert data["archived_at"] is None


class TestApprovalWorkflowManagerEndToEnd:
    """End-to-end: create a workflow, drive it through approval to
    deployment-readiness, then separately retire the model — all through
    the manager, the way a real caller would use this module.
    """

    def test_default_workflow_uses_standard_three_step_pipeline(self):
        manager = ApprovalWorkflowManager()
        workflow = manager.create_workflow(
            model_id="fraud-detector-v2",
            reviewers=["ml-eng-team", "security-team", "compliance-team"],
        )

        assert [s.step_type for s in workflow.steps] == [
            ApprovalStepType.TECHNICAL_REVIEW,
            ApprovalStepType.SECURITY_REVIEW,
            ApprovalStepType.COMPLIANCE_REVIEW,
        ]
        assert manager.get_workflow(workflow.workflow_id) is workflow
        assert manager.get_model_workflows("fraud-detector-v2") == [workflow]

    def test_create_workflow_rejects_insufficient_reviewers(self):
        manager = ApprovalWorkflowManager()
        with pytest.raises(ValueError, match="Not enough reviewers"):
            manager.create_workflow(model_id="m", reviewers=["only-one"])

    def test_full_deployment_approval_flow(self):
        manager = ApprovalWorkflowManager()
        workflow = manager.create_workflow(
            model_id="fraud-detector-v2",
            reviewers=["ml-eng-team", "security-team", "compliance-team"],
        )

        for step in workflow.steps:
            manager.get_workflow(workflow.workflow_id).approve_step(
                step.step_id, comments="reviewed"
            )

        refreshed = manager.get_workflow(workflow.workflow_id)
        assert refreshed.is_ready_for_deployment() is True
        assert refreshed.status == ApprovalStatus.APPROVED

    def test_deployment_blocked_after_any_rejection(self):
        manager = ApprovalWorkflowManager()
        workflow = manager.create_workflow(
            model_id="fraud-detector-v2",
            reviewers=["ml-eng-team", "security-team", "compliance-team"],
        )

        workflow.approve_step(workflow.steps[0].step_id)
        workflow.reject_step(workflow.steps[1].step_id, "found critical bug")

        assert workflow.is_ready_for_deployment() is False
        assert workflow.status == ApprovalStatus.REJECTED

    def test_retirement_lifecycle_through_manager(self):
        manager = ApprovalWorkflowManager()
        retirement = manager.create_retirement(
            model_id="fraud-detector-v1",
            reason="superseded by v2",
            replacement_model_id="fraud-detector-v2",
        )

        assert manager.get_retirement(retirement.workflow_id) is retirement
        assert retirement in manager.get_active_retirements()

        retirement.start_review()
        retirement.approve()
        retirement.start_retirement()
        retirement.complete_retirement()
        retirement.archive()

        assert retirement not in manager.get_active_retirements()

    def test_cancelled_retirement_excluded_from_active(self):
        manager = ApprovalWorkflowManager()
        retirement = manager.create_retirement(model_id="m", reason="test")
        retirement.cancel()

        assert retirement not in manager.get_active_retirements()

    def test_custom_step_types_respected(self):
        manager = ApprovalWorkflowManager()
        workflow = manager.create_workflow(
            model_id="m",
            reviewers=["ethics-team", "stakeholders"],
            step_types=[ApprovalStepType.ETHICS_REVIEW, ApprovalStepType.STAKEHOLDER_SIGN_OFF],
        )
        assert [s.step_type for s in workflow.steps] == [
            ApprovalStepType.ETHICS_REVIEW,
            ApprovalStepType.STAKEHOLDER_SIGN_OFF,
        ]
