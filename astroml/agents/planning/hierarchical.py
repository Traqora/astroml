"""Hierarchical planner for complex multi-level task decomposition."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Planner, Task, TaskPlan


class HierarchicalPlanner(Planner):
    """Planner that decomposes tasks hierarchically into subtasks."""

    def __init__(self, max_depth: int = 3) -> None:
        """Initialize hierarchical planner.

        Args:
            max_depth: Maximum depth of task decomposition
        """
        self.max_depth = max_depth

    def plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """Generate a hierarchical plan.

        Args:
            task: Task description
            context: Additional context

        Returns:
            TaskPlan with hierarchical tasks
        """
        context = context or {}
        tasks = self._decompose_recursive(task, context, depth=0)

        return TaskPlan(
            description=task,
            tasks=tasks,
            metadata={"planner_type": "hierarchical", "max_depth": self.max_depth},
        )

    def _decompose_recursive(
        self,
        task: str,
        context: Dict[str, Any],
        depth: int,
        parent_id: Optional[str] = None,
    ) -> List[Task]:
        """Recursively decompose tasks.

        Args:
            task: Task description
            context: Context
            depth: Current depth
            parent_id: Parent task ID

        Returns:
            List of decomposed tasks
        """
        if depth >= self.max_depth:
            # Leaf task - no further decomposition
            task_id = parent_id + "_leaf" if parent_id else "task_0"
            return [Task(id=task_id, description=task)]

        # Decompose into subtasks
        subtasks = self._get_subtasks(task, context, depth)

        for i, subtask_desc in enumerate(subtasks):
            task_id = f"{parent_id}_{i}" if parent_id else f"task_{i}"
            task = Task(id=task_id, description=subtask_desc)

            # Recursively decompose if needed
            if depth < self.max_depth - 1:
                # Check if this subtask should be further decomposed
                if self._should_decompose(subtask_desc):
                    # Add placeholder for recursive decomposition
                    task.metadata["needs_decomposition"] = True

        # Create Task objects
        tasks = []
        for i, subtask_desc in enumerate(subtasks):
            task_id = f"{parent_id}_{i}" if parent_id else f"task_{i}"
            dependencies = [f"{parent_id}_{i-1}"] if i > 0 and parent_id else []

            task = Task(
                id=task_id,
                description=subtask_desc,
                dependencies=dependencies,
            )

            # Recursively decompose
            if self._should_decompose(subtask_desc) and depth < self.max_depth - 1:
                sub_subtasks = self._decompose_recursive(
                    subtask_desc, context, depth + 1, task_id
                )
                # In a full implementation, we'd handle subtask hierarchies
                # For now, we keep it flat with dependencies

            tasks.append(task)

        return tasks

    def _get_subtasks(
        self,
        task: str,
        context: Dict[str, Any],
        depth: int,
    ) -> List[str]:
        """Get subtasks for a given task.

        Args:
            task: Task description
            context: Context
            depth: Current depth

        Returns:
            List of subtask descriptions
        """
        # Simple rule-based decomposition
        # In production, this would use LLM-based decomposition

        task_lower = task.lower()

        if "fraud" in task_lower and "detect" in task_lower:
            return [
                "Load transaction data",
                "Build transaction graph",
                "Extract graph features",
                "Run anomaly detection model",
                "Validate results",
                "Generate report",
            ]

        if "train" in task_lower and "model" in task_lower:
            return [
                "Prepare training data",
                "Split data into train/validation",
                "Engineer features",
                "Train model",
                "Evaluate model",
                "Save model",
            ]

        if "analyze" in task_lower and "data" in task_lower:
            return [
                "Load data",
                "Explore data",
                "Clean data",
                "Perform analysis",
                "Generate report",
            ]

        if "graph" in task_lower or "network" in task_lower:
            return [
                "Load graph data",
                "Compute graph metrics",
                "Identify communities",
                "Visualize graph",
                "Analyze patterns",
            ]

        # Default generic decomposition
        return [
            f"Understand requirements for: {task}",
            f"Gather necessary data for: {task}",
            f"Execute main task: {task}",
            f"Validate results for: {task}",
        ]

    def _should_decompose(self, task: str) -> bool:
        """Determine if a task should be further decomposed.

        Args:
            task: Task description

        Returns:
            True if should decompose
        """
        # Simple heuristic: decompose if task is complex
        # Complex tasks have multiple action verbs or are long
        task_lower = task.lower()
        action_verbs = ["analyze", "train", "build", "create", "detect", "identify"]
        has_action = any(verb in task_lower for verb in action_verbs)
        is_long = len(task) > 50

        return has_action or is_long

    def refine_plan(
        self,
        plan: TaskPlan,
        feedback: str,
    ) -> TaskPlan:
        """Refine plan based on feedback.

        Args:
            plan: Existing plan
            feedback: Feedback for refinement

        Returns:
            Refined plan
        """
        # Add a task to address feedback
        new_task = Task(
            id=f"refinement_{len(plan.tasks)}",
            description=f"Address feedback: {feedback}",
            dependencies=[plan.tasks[-1].id] if plan.tasks else [],
        )
        plan.add_task(new_task)
        return plan
