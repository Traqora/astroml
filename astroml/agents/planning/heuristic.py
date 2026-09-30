"""Heuristic-based planner using rule-based task decomposition."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import Planner, Task, TaskPlan


class HeuristicPlanner(Planner):
    """Planner that uses heuristic rules to decompose tasks."""

    def __init__(self) -> None:
        """Initialize heuristic planner with decomposition rules."""
        self.rules = [
            # Data analysis tasks
            (
                r"(analyze|investigate|examine).+(data|dataset|transactions)",
                self._decompose_analysis_task,
            ),
            # Model training tasks
            (
                r"(train|build|create).+(model|classifier|predictor)",
                self._decompose_training_task,
            ),
            # Fraud detection tasks
            (
                r"(detect|find|identify).+(fraud|anomaly|suspicious)",
                self._decompose_fraud_detection_task,
            ),
            # Graph analysis tasks
            (
                r"(analyze|explore|study).+(graph|network|connections)",
                self._decompose_graph_task,
            ),
            # Feature engineering tasks
            (
                r"(create|extract|engineer).+(features|attributes)",
                self._decompose_feature_task,
            ),
        ]

    def plan(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskPlan:
        """Generate a plan using heuristic rules.

        Args:
            task: Task description
            context: Additional context

        Returns:
            TaskPlan with decomposed tasks
        """
        context = context or {}

        # Try to match against rules
        for pattern, decomposer in self.rules:
            if re.search(pattern, task, re.IGNORECASE):
                return decomposer(task, context)

        # Default: simple single-task plan
        return TaskPlan(
            description=task,
            tasks=[
                Task(
                    id="task_1",
                    description=task,
                )
            ],
        )

    def _decompose_analysis_task(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Decompose data analysis task.

        Args:
            task: Original task
            context: Context

        Returns:
            TaskPlan with analysis subtasks
        """
        return TaskPlan(
            description=task,
            tasks=[
                Task(id="load_data", description="Load and validate data"),
                Task(
                    id="explore_data",
                    description="Explore data structure and statistics",
                    dependencies=["load_data"],
                ),
                Task(
                    id="clean_data",
                    description="Clean and preprocess data",
                    dependencies=["explore_data"],
                ),
                Task(
                    id="analyze_data",
                    description="Perform analysis",
                    dependencies=["clean_data"],
                ),
                Task(
                    id="report_results",
                    description="Generate analysis report",
                    dependencies=["analyze_data"],
                ),
            ],
        )

    def _decompose_training_task(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Decompose model training task.

        Args:
            task: Original task
            context: Context

        Returns:
            TaskPlan with training subtasks
        """
        return TaskPlan(
            description=task,
            tasks=[
                Task(id="prepare_data", description="Prepare training data"),
                Task(
                    id="split_data",
                    description="Split data into train/validation/test",
                    dependencies=["prepare_data"],
                ),
                Task(
                    id="engineer_features",
                    description="Engineer features",
                    dependencies=["split_data"],
                ),
                Task(
                    id="train_model",
                    description="Train model",
                    dependencies=["engineer_features"],
                ),
                Task(
                    id="evaluate_model",
                    description="Evaluate model performance",
                    dependencies=["train_model"],
                ),
                Task(
                    id="save_model",
                    description="Save trained model",
                    dependencies=["evaluate_model"],
                ),
            ],
        )

    def _decompose_fraud_detection_task(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Decompose fraud detection task.

        Args:
            task: Original task
            context: Context

        Returns:
            TaskPlan with fraud detection subtasks
        """
        return TaskPlan(
            description=task,
            tasks=[
                Task(id="load_transactions", description="Load transaction data"),
                Task(
                    id="build_graph",
                    description="Build transaction graph",
                    dependencies=["load_transactions"],
                ),
                Task(
                    id="extract_features",
                    description="Extract graph features",
                    dependencies=["build_graph"],
                ),
                Task(
                    id="detect_anomalies",
                    description="Detect anomalies using model",
                    dependencies=["extract_features"],
                ),
                Task(
                    id="validate_results",
                    description="Validate detection results",
                    dependencies=["detect_anomalies"],
                ),
                Task(
                    id="report_findings",
                    description="Report fraud findings",
                    dependencies=["validate_results"],
                ),
            ],
        )

    def _decompose_graph_task(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Decompose graph analysis task.

        Args:
            task: Original task
            context: Context

        Returns:
            TaskPlan with graph analysis subtasks
        """
        return TaskPlan(
            description=task,
            tasks=[
                Task(id="load_graph", description="Load graph data"),
                Task(
                    id="compute_metrics",
                    description="Compute graph metrics",
                    dependencies=["load_graph"],
                ),
                Task(
                    id="identify_patterns",
                    description="Identify patterns and communities",
                    dependencies=["compute_metrics"],
                ),
                Task(
                    id="visualize_graph",
                    description="Visualize graph structure",
                    dependencies=["identify_patterns"],
                ),
                Task(
                    id="analyze_results",
                    description="Analyze graph analysis results",
                    dependencies=["visualize_graph"],
                ),
            ],
        )

    def _decompose_feature_task(
        self,
        task: str,
        context: Dict[str, Any],
    ) -> TaskPlan:
        """Decompose feature engineering task.

        Args:
            task: Original task
            context: Context

        Returns:
            TaskPlan with feature engineering subtasks
        """
        return TaskPlan(
            description=task,
            tasks=[
                Task(id="analyze_data", description="Analyze data for feature opportunities"),
                Task(
                    id="create_features",
                    description="Create new features",
                    dependencies=["analyze_data"],
                ),
                Task(
                    id="transform_features",
                    description="Transform and normalize features",
                    dependencies=["create_features"],
                ),
                Task(
                    id="select_features",
                    description="Select best features",
                    dependencies=["transform_features"],
                ),
                Task(
                    id="validate_features",
                    description="Validate feature quality",
                    dependencies=["select_features"],
                ),
            ],
        )

    def refine_plan(
        self,
        plan: TaskPlan,
        feedback: str,
    ) -> TaskPlan:
        """Refine plan based on feedback (simple implementation).

        Args:
            plan: Existing plan
            feedback: Feedback string

        Returns:
            Refined plan
        """
        # Simple implementation: add a task based on feedback
        new_task = Task(
            id=f"additional_task_{len(plan.tasks) + 1}",
            description=f"Address feedback: {feedback}",
        )
        plan.add_task(new_task)
        return plan
