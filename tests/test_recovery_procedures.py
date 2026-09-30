"""Tests for recovery procedures and step execution."""

import pytest

from astroml.deployment.recovery_procedures import RecoveryProcedure, RecoveryStep


def test_recovery_procedure_validate_no_steps() -> None:
    procedure = RecoveryProcedure(name="empty", description="No steps")
    with pytest.raises(ValueError, match="must include at least one step"):
        procedure.execute()


def test_recovery_procedure_validate_duplicate_names() -> None:
    procedure = RecoveryProcedure(
        name="duplicate",
        description="Duplicates",
        steps=[
            RecoveryStep(name="step1", description="first", action=lambda: True),
            RecoveryStep(name="step1", description="duplicate", action=lambda: True),
        ],
    )
    with pytest.raises(ValueError, match="step names must be unique"):
        procedure.execute()


def test_recovery_procedure_execute_success() -> None:
    procedure = RecoveryProcedure(
        name="success",
        description="All pass",
        steps=[
            RecoveryStep(name="check-1", description="A", action=lambda: True),
            RecoveryStep(name="check-2", description="B", action=lambda: True),
        ],
    )
    result = procedure.execute()

    assert result["success"] is True
    assert result["errors"] == []
    assert result["step_results"][0]["result"] == "success"
    assert result["step_results"][1]["result"] == "success"


def test_recovery_procedure_execute_failure_on_step() -> None:
    procedure = RecoveryProcedure(
        name="partial",
        description="One fails",
        steps=[
            RecoveryStep(name="check-1", description="A", action=lambda: True),
            RecoveryStep(name="check-2", description="B", action=lambda: False),
            RecoveryStep(name="check-3", description="C", action=lambda: True),
        ],
    )
    result = procedure.execute()

    assert result["success"] is False
    assert any("Step failed" in error for error in result["errors"])
    assert result["step_results"][2]["completed"] is False


def test_recovery_procedure_step_exception() -> None:
    def bad_action() -> bool:
        raise RuntimeError("unexpected")

    procedure = RecoveryProcedure(
        name="exception",
        description="Raises",
        steps=[RecoveryStep(name="check", description="Fail", action=bad_action)],
    )
    result = procedure.execute()

    assert result["success"] is False
    assert any("Step error" in error for error in result["errors"])
