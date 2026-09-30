"""Base planning system for task decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(Enum):
    """Status of a task in the plan."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """Represents a single task in a plan."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    def mark_completed(self) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.utcnow()

    def mark_failed(self) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED

    def is_ready(self, completed_tasks: set[str]) -> bool:
        """Check if task is ready to execute (dependencies satisfied).

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            True if ready to execute
        """
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class TaskPlan:
    """Represents a complete plan with multiple tasks."""

    description: str
    tasks: List[Task] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_task(self, task: Task) -> None:
        """Add a task to the plan.

        Args:
            task: Task to add
        """
        self.tasks.append(task)

    def get_next_task(self) -> Optional[Task]:
        """Get the next task that is ready to execute.

        Returns:
            Next ready task or None
        """
        completed = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        for task in self.tasks:
            if task.status == TaskStatus.PENDING and task.is_ready(completed):
                return task
        return None

    def is_complete(self) -> bool:
        """Check if all tasks are complete.

        Returns:
            True if all tasks completed
        """
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED) for t in self.tasks)

    def get_progress(self) -> float:
        """Get plan progress as percentage.

        Returns:
            Progress percentage (0-100)
        """
        if not self.tasks:
            return 100.0

        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        return (completed / len(self.tasks)) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "description": self.description,
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "status": t.status.value,
                    "dependencies": t.dependencies,
                    "metadata": t.metadata,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "progress": self.get_progress(),
        }


class Planner:
    """Base planner for task decomposition."""

    def plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """Generate a plan for the given task.

        Args:
            task: Task description
            context: Additional context

        Returns:
            TaskPlan with decomposed tasks
        """
        raise NotImplementedError("Subclasses must implement plan method")

    def refine_plan(
        self,
        plan: TaskPlan,
        feedback: str,
    ) -> TaskPlan:
        """Refine an existing plan based on feedback.

        Args:
            plan: Existing plan
            feedback: Feedback for refinement

        Returns:
            Refined TaskPlan
        """
        raise NotImplementedError("Subclasses must implement refine_plan method")
