from astroml.utils.exceptions import AstroMLError
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class RecoveryStep:
    """A single step in a recovery procedure."""

    name: str
    description: str
    action: Callable[[], bool]
    completed: bool = False
    result: str | None = None


@dataclass
class RecoveryProcedure:
    """A sequence of steps used to recover from an incident."""

    name: str
    description: str
    steps: list[RecoveryStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool | None = None
    errors: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate the procedure configuration before execution."""
        if not self.steps:
            raise ValueError("Recovery procedure must include at least one step")
        names = [step.name for step in self.steps]
        if len(names) != len(set(names)):
            raise ValueError("Recovery procedure step names must be unique")

    def execute(self) -> dict[str, Any]:
        """Execute recovery steps in order."""
        self.validate()
        self.completed_at = None
        self.success = True
        self.errors.clear()

        for step in self.steps:
            try:
                outcome = step.action()
                step.completed = True
                step.result = "success" if outcome else "failed"
                if not outcome:
                    self.success = False
                    self.errors.append(f"Step failed: {step.name}")
                    logger.warning("Recovery step failed: %s", step.name)
                    break
            except Exception as exc:
                step.completed = False
                step.result = str(exc)
                self.success = False
                self.errors.append(f"Step error: {step.name} - {exc}")
                logger.exception("Recovery step error: %s", step.name)
                break

        self.completed_at = datetime.now(timezone.utc)
        return {
            "name": self.name,
            "description": self.description,
            "success": self.success,
            "completed_at": self.completed_at.isoformat(),
            "errors": self.errors.copy(),
            "step_results": [
                {
                    "name": step.name,
                    "completed": step.completed,
                    "result": step.result,
                }
                for step in self.steps
            ],
        }
