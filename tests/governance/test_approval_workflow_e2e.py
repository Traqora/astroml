"""End-to-end tests for the model approval and retirement workflows (issue #966).

Exercises the full lifecycle through `ApprovalWorkflowManager`, the entry
point production code actually uses, rather than only unit-testing
`ApprovalWorkflow`/`ModelRetirementWorkflow` in isolation.
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


@pytest.fixture
def manager() -> ApprovalWorkflowManager:
    return ApprovalWorkflowManager()


def test_full_approval_happy_path_reaches_deployment_ready(manager):
    workflow = manager.create_workflow(
        model_id="fraud-detector-v2",
        reviewers=["ml-eng-team", "security-team", "compliance-team"],
    )

    assert workflow.status == ApprovalStatus.PENDING
    assert workflow.is_ready_for_deployment() is False
    assert workflow.progress == 0.0

    for step in list(workflow.steps):
        workflow.approve_step(step.step_id, comments=f"{step.step_type.name} passed")

    assert workflow.status == ApprovalStatus.APPROVED
    assert workflow.is_ready_for_deployment() is True
    assert workflow.progress == 1.0
    assert workflow.pending_steps == []
    assert len(workflow.approved_steps) == 3


def test_a_single_rejection_blocks_deployment_even_if_other_steps_approve(manager):
    workflow = manager.create_workflow(
        model_id="fraud-detector-v2",
        reviewers=["ml-eng-team", "security-team", "compliance-team"],
    )
    technical, security, compliance = workflow.steps

    workflow.approve_step(technical.step_id, "looks good")
    workflow.reject_step(security.step_id, "fails adversarial robustness check")
    workflow.approve_step(compliance.step_id, "no compliance concerns")

    assert workflow.status == ApprovalStatus.REJECTED
    assert workflow.is_ready_for_deployment() is False
    # A rejection anywhere makes the aggregate REJECTED even though two of
    # three steps individually approved; deployment readiness must not be
    # inferrable from progress alone.
    assert workflow.progress == 2 / 3


def test_approving_an_already_resolved_step_raises(manager):
    workflow = manager.create_workflow(
        model_id="m1", reviewers=["team-a"], step_types=[ApprovalStepType.TECHNICAL_REVIEW]
    )
    step = workflow.steps[0]
    workflow.approve_step(step.step_id)

    with pytest.raises(ValueError, match="Cannot approve"):
        workflow.approve_step(step.step_id)


def test_rejecting_an_already_approved_step_raises(manager):
    workflow = manager.create_workflow(
        model_id="m1", reviewers=["team-a"], step_types=[ApprovalStepType.TECHNICAL_REVIEW]
    )
    step = workflow.steps[0]
    workflow.approve_step(step.step_id)

    with pytest.raises(ValueError, match="Cannot reject"):
        workflow.reject_step(step.step_id)


def test_approving_an_unknown_step_id_raises(manager):
    workflow = manager.create_workflow(
        model_id="m1", reviewers=["team-a"], step_types=[ApprovalStepType.TECHNICAL_REVIEW]
    )
    with pytest.raises(ValueError, match="not found"):
        workflow.approve_step("does-not-exist")


def test_create_workflow_rejects_more_steps_than_reviewers(manager):
    with pytest.raises(ValueError, match="Not enough reviewers"):
        manager.create_workflow(
            model_id="m1",
            reviewers=["only-one-team"],
            step_types=[ApprovalStepType.TECHNICAL_REVIEW, ApprovalStepType.SECURITY_REVIEW],
        )


def test_a_workflow_with_no_steps_is_vacuously_ready_for_deployment():
    # Pinning documented-but-surprising behaviour: ApprovalWorkflow.status
    # and .progress both treat an empty step list as fully satisfied
    # (`all()` over an empty sequence is True). A caller building a
    # workflow from a possibly-empty step_types/reviewers pairing must not
    # be surprised that this is "ready to deploy" with zero actual review.
    workflow = ApprovalWorkflow(model_id="m1", steps=[])
    assert workflow.status == ApprovalStatus.APPROVED
    assert workflow.is_ready_for_deployment() is True
    assert workflow.progress == 1.0


def test_manager_tracks_multiple_workflows_per_model(manager):
    first = manager.create_workflow(model_id="m1", reviewers=["a", "b", "c"])
    second = manager.create_workflow(model_id="m1", reviewers=["a", "b", "c"])
    other_model = manager.create_workflow(model_id="m2", reviewers=["a", "b", "c"])

    model_workflows = manager.get_model_workflows("m1")
    assert {w.workflow_id for w in model_workflows} == {first.workflow_id, second.workflow_id}
    assert other_model.workflow_id not in {w.workflow_id for w in model_workflows}


def test_get_workflow_returns_none_for_unknown_id(manager):
    assert manager.get_workflow("nonexistent") is None


# --- Retirement lifecycle -----------------------------------------------


def test_full_retirement_lifecycle_proposed_to_archived(manager):
    retirement = manager.create_retirement(
        model_id="fraud-detector-v1",
        reason="Replaced by v2 with improved accuracy",
        replacement_model_id="fraud-detector-v2",
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


def test_retirement_steps_cannot_be_skipped_out_of_order():
    retirement = ModelRetirementWorkflow(model_id="m1", reason="test")

    with pytest.raises(ValueError, match="Cannot approve"):
        retirement.approve()

    with pytest.raises(ValueError, match="Cannot start retirement"):
        retirement.start_retirement()

    with pytest.raises(ValueError, match="Cannot complete retirement"):
        retirement.complete_retirement()

    with pytest.raises(ValueError, match="Cannot archive"):
        retirement.archive()


def test_retirement_can_be_cancelled_before_completion():
    retirement = ModelRetirementWorkflow(model_id="m1", reason="test")
    retirement.start_review()

    retirement.cancel(reason="Model performance recovered after all")

    assert retirement.status == ModelRetirementStatus.CANCELLED
    assert retirement.reason == "Model performance recovered after all"


def test_retirement_cancel_uses_a_default_reason_when_none_given():
    retirement = ModelRetirementWorkflow(model_id="m1", reason="test")
    retirement.cancel()
    assert retirement.reason == "Cancelled by administrator"


def test_a_retired_model_cannot_be_cancelled(manager):
    retirement = manager.create_retirement(model_id="m1", reason="test")
    retirement.start_review()
    retirement.approve()
    retirement.start_retirement()
    retirement.complete_retirement()

    with pytest.raises(ValueError, match="already retired"):
        retirement.cancel()


def test_an_archived_model_cannot_be_cancelled(manager):
    retirement = manager.create_retirement(model_id="m1", reason="test")
    retirement.start_review()
    retirement.approve()
    retirement.start_retirement()
    retirement.complete_retirement()
    retirement.archive()

    with pytest.raises(ValueError, match="already retired"):
        retirement.cancel()


def test_get_active_retirements_excludes_cancelled_and_archived(manager):
    active = manager.create_retirement(model_id="m-active", reason="test")

    cancelled = manager.create_retirement(model_id="m-cancelled", reason="test")
    cancelled.cancel()

    archived = manager.create_retirement(model_id="m-archived", reason="test")
    archived.start_review()
    archived.approve()
    archived.start_retirement()
    archived.complete_retirement()
    archived.archive()

    active_ids = {r.workflow_id for r in manager.get_active_retirements()}
    assert active.workflow_id in active_ids
    assert cancelled.workflow_id not in active_ids
    assert archived.workflow_id not in active_ids


def test_workflow_to_dict_round_trips_key_fields(manager):
    workflow = manager.create_workflow(model_id="m1", reviewers=["a", "b", "c"])
    workflow.approve_step(workflow.steps[0].step_id, "ok")

    payload = workflow.to_dict()

    assert payload["model_id"] == "m1"
    assert payload["status"] == ApprovalStatus.PENDING.value
    assert payload["progress"] == pytest.approx(1 / 3)
    assert len(payload["steps"]) == 3
    assert payload["steps"][0]["status"] == ApprovalStatus.APPROVED.value
    assert payload["steps"][0]["comments"] == "ok"


def test_step_to_dict_reports_none_for_unresolved_timestamps():
    step = ApprovalStep(step_type=ApprovalStepType.TECHNICAL_REVIEW, reviewer="team-a")
    payload = step.to_dict()
    assert payload["resolved_at"] is None
    assert payload["status"] == ApprovalStatus.PENDING.value
