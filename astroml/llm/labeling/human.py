"""Human-in-the-loop integration for LLM-based data labeling (issue #475)."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .schemas import Label, LabelDefinition

logger = logging.getLogger(__name__)


class ReviewStatus(Enum):
    """Status of a review task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class ReviewTask:
    """A human review task for low-confidence labels.

    Attributes:
        id: Unique identifier
        item_id: ID of the item to review
        data: Data to review
        suggested_labels: Suggested labels from LLM
        definition: Label definition
        status: Review status
        assigned_to: User assigned to review
        reviewed_by: User who reviewed
        reviewed_at: When review was completed
        accepted_labels: Labels accepted by reviewer
        rejected_labels: Labels rejected by reviewer
        corrected_labels: Labels corrected by reviewer
        notes: Reviewer notes
        created_at: When task was created
    """

    id: str
    item_id: str
    data: Any
    suggested_labels: List[Label]
    definition: LabelDefinition
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_to: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    accepted_labels: List[Label] = field(default_factory=list)
    rejected_labels: List[Label] = field(default_factory=list)
    corrected_labels: List[Label] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "item_id": self.item_id,
            "suggested_labels": [l.to_dict() for l in self.suggested_labels],
            "definition": self.definition.to_dict(),
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "accepted_labels": [l.to_dict() for l in self.accepted_labels],
            "rejected_labels": [l.to_dict() for l in self.rejected_labels],
            "corrected_labels": [l.to_dict() for l in self.corrected_labels],
            "notes": self.notes,
            "created_at": self.created_at.isoformat(),
        }


class HumanReviewQueue:
    """Queue for human review of low-confidence labels."""

    def __init__(self) -> Any:
        """Initialize human review queue."""
        self.tasks: Dict[str, ReviewTask] = {}
        self.user_assignments: Dict[str, List[str]] = {}  # user -> task_ids

    def add_task(
        self,
        item_id: str,
        data: Any,
        suggested_labels: List[Label],
        definition: LabelDefinition,
        priority: int = 0,
    ) -> ReviewTask:
        """Add a review task to the queue.

        Args:
            item_id: ID of the item to review
            data: Data to review
            suggested_labels: Suggested labels from LLM
            definition: Label definition
            priority: Task priority (higher = more urgent)

        Returns:
            ReviewTask
        """
        task = ReviewTask(
            id=str(uuid.uuid4()),
            item_id=item_id,
            data=data,
            suggested_labels=suggested_labels,
            definition=definition,
        )

        self.tasks[task.id] = task
        logger.info(f"Added review task: {task.id} for item {item_id}")

        return task

    def assign_task(self, task_id: str, user_id: str) -> bool:
        """Assign a task to a user.

        Args:
            task_id: Task ID
            user_id: User ID

        Returns:
            True if assignment successful
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        if task.status != ReviewStatus.PENDING:
            return False

        task.assigned_to = user_id
        task.status = ReviewStatus.IN_PROGRESS

        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = []
        self.user_assignments[user_id].append(task_id)

        logger.info(f"Assigned task {task_id} to user {user_id}")
        return True

    def complete_review(
        self,
        task_id: str,
        user_id: str,
        accepted_labels: List[Label],
        rejected_labels: List[Label],
        corrected_labels: List[Label],
        notes: Optional[str] = None,
    ) -> bool:
        """Complete a review task.

        Args:
            task_id: Task ID
            user_id: User ID
            accepted_labels: Labels accepted by reviewer
            rejected_labels: Labels rejected by reviewer
            corrected_labels: Labels corrected by reviewer
            notes: Reviewer notes

        Returns:
            True if completion successful
        """
        if task_id not in self.tasks:
            return False

        task = self.tasks[task_id]
        task.reviewed_by = user_id
        task.reviewed_at = datetime.utcnow()
        task.status = ReviewStatus.COMPLETED
        task.accepted_labels = accepted_labels
        task.rejected_labels = rejected_labels
        task.corrected_labels = corrected_labels
        task.notes = notes

        # Remove from user assignments
        if user_id in self.user_assignments:
            self.user_assignments[user_id] = [
                tid for tid in self.user_assignments[user_id] if tid != task_id
            ]

        logger.info(f"Completed review task {task_id} by user {user_id}")
        return True

    def get_task(self, task_id: str) -> Optional[ReviewTask]:
        """Get a review task by ID.

        Args:
            task_id: Task ID

        Returns:
            ReviewTask if found
        """
        return self.tasks.get(task_id)

    def get_user_tasks(self, user_id: str) -> List[ReviewTask]:
        """Get tasks assigned to a user.

        Args:
            user_id: User ID

        Returns:
            List of ReviewTasks
        """
        task_ids = self.user_assignments.get(user_id, [])
        return [self.tasks[tid] for tid in task_ids if tid in self.tasks]

    def get_pending_tasks(self, limit: int = 50) -> List[ReviewTask]:
        """Get pending tasks.

        Args:
            limit: Maximum number of tasks to return

        Returns:
            List of pending ReviewTasks
        """
        pending = [task for task in self.tasks.values() if task.status == ReviewStatus.PENDING]
        return pending[:limit]

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with statistics
        """
        total = len(self.tasks)
        pending = sum(1 for t in self.tasks.values() if t.status == ReviewStatus.PENDING)
        in_progress = sum(1 for t in self.tasks.values() if t.status == ReviewStatus.IN_PROGRESS)
        completed = sum(1 for t in self.tasks.values() if t.status == ReviewStatus.COMPLETED)

        return {
            "total_tasks": total,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "active_users": len(self.user_assignments),
        }
