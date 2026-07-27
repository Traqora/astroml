"""A/B testing framework for comparing models and prompts."""
from __future__ import annotations

import hashlib
import logging
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from astroml.db.schema import Experiment, ExperimentResult, ModelVersion, Variant
from astroml.db.session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment Status State Machine
# ---------------------------------------------------------------------------

VALID_EXPERIMENT_TRANSITIONS = {
    "draft": ["running", "archived"],
    "running": ["paused", "completed", "archived"],
    "paused": ["running", "archived"],
    "completed": ["archived"],
    "archived": [],  # Terminal state
}

VALID_EXPERIMENT_STATUSES = set(VALID_EXPERIMENT_TRANSITIONS.keys())


class InvalidExperimentStatusError(ValueError):
    """Raised when an invalid experiment status transition is attempted."""

    pass


class ABTestingFramework:
    """Core class for managing A/B tests for models and prompts.

    Provides experiment management, variant assignment, and result tracking
    with statistical analysis capabilities.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize the A/B testing framework.

        Args:
            session: Optional SQLAlchemy session. If not provided, creates a new session.
        """
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> Session:
        """Get the SQLAlchemy session, creating one if needed."""
        if self._session is None:
            self._session = get_session()
        return self._session

    def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "ABTestingFramework":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Experiment CRUD operations
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        experiment_type: str,
        description: Optional[str] = None,
        traffic_allocation: float = 1.0,
    ) -> Experiment:
        """Create a new A/B test experiment.

        Args:
            name: Unique experiment name
            experiment_type: Type of experiment ('model' or 'prompt')
            description: Optional experiment description
            traffic_allocation: Fraction of traffic to allocate (0.0 to 1.0)

        Returns:
            Created Experiment instance

        Raises:
            ValueError: If experiment with same name exists or invalid parameters
        """
        if experiment_type not in ("model", "prompt"):
            raise ValueError(f"experiment_type must be 'model' or 'prompt', got '{experiment_type}'")

        if not 0.0 <= traffic_allocation <= 1.0:
            raise ValueError(f"traffic_allocation must be between 0.0 and 1.0, got {traffic_allocation}")

        existing = self.get_experiment_by_name(name)
        if existing:
            raise ValueError(f"Experiment with name '{name}' already exists")

        experiment = Experiment(
            name=name,
            description=description,
            experiment_type=experiment_type,
            traffic_allocation=traffic_allocation,
        )
        self.session.add(experiment)
        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Created experiment: %s (id=%d, type=%s)", name, experiment.id, experiment_type)
        return experiment

    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment instance or None if not found
        """
        return self.session.get(Experiment, experiment_id)

    def get_experiment_by_name(self, name: str) -> Optional[Experiment]:
        """Get an experiment by name.

        Args:
            name: Experiment name

        Returns:
            Experiment instance or None if not found
        """
        stmt = select(Experiment).where(Experiment.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Experiment]:
        """List experiments with optional filters.

        Args:
            experiment_type: Filter by experiment type
            status: Filter by status

        Returns:
            List of Experiment instances
        """
        stmt = select(Experiment)
        if experiment_type:
            stmt = stmt.where(Experiment.experiment_type == experiment_type)
        if status:
            stmt = stmt.where(Experiment.status == status)
        stmt = stmt.order_by(Experiment.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_experiment(
        self,
        experiment_id: int,
        description: Optional[str] = None,
        traffic_allocation: Optional[float] = None,
        start_at: Optional[datetime] = None,
        end_at: Optional[datetime] = None,
    ) -> Optional[Experiment]:
        """Update an experiment.

        Args:
            experiment_id: Experiment ID
            description: New description
            traffic_allocation: New traffic allocation
            start_at: Start timestamp
            end_at: End timestamp

        Returns:
            Updated Experiment instance or None if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        if description is not None:
            experiment.description = description
        if traffic_allocation is not None:
            if not 0.0 <= traffic_allocation <= 1.0:
                raise ValueError(f"traffic_allocation must be between 0.0 and 1.0, got {traffic_allocation}")
            experiment.traffic_allocation = traffic_allocation
        if start_at is not None:
            experiment.start_at = start_at
        if end_at is not None:
            experiment.end_at = end_at

        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Updated experiment: %s (id=%d)", experiment.name, experiment_id)
        return experiment

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment and all its variants and results.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return False

        self.session.delete(experiment)
        self.session.commit()
        logger.info("Deleted experiment: %s (id=%d)", experiment.name, experiment_id)
        return True

    # ------------------------------------------------------------------
    # Variant CRUD operations
    # ------------------------------------------------------------------

    def create_variant(
        self,
        experiment_id: int,
        name: str,
        traffic_weight: float = 0.5,
        is_control: bool = False,
        model_version_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Variant:
        """Create a new variant for an experiment.

        Args:
            experiment_id: Parent experiment ID
            name: Variant name
            traffic_weight: Traffic weight (0.0 to 1.0)
            is_control: Whether this is the control variant
            model_version_id: Optional model version ID for model experiments
            config: Configuration dict (for prompts or model config)
            description: Optional variant description

        Returns:
            Created Variant instance

        Raises:
            ValueError: If variant with same name exists or invalid parameters
        """
        if not 0.0 <= traffic_weight <= 1.0:
            raise ValueError(f"traffic_weight must be between 0.0 and 1.0, got {traffic_weight}")

        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment with id {experiment_id} not found")

        existing = self.get_variant(experiment_id, name)
        if existing:
            raise ValueError(f"Variant '{name}' already exists for experiment {experiment_id}")

        variant = Variant(
            experiment_id=experiment_id,
            name=name,
            description=description,
            traffic_weight=traffic_weight,
            is_control=is_control,
            model_version_id=model_version_id,
            config=config,
        )
        self.session.add(variant)
        self.session.commit()
        self.session.refresh(variant)
        logger.info(
            "Created variant: %s (id=%d, experiment_id=%d)",
            name,
            variant.id,
            experiment_id,
        )
        return variant

    def get_variant(self, experiment_id: int, name: str) -> Optional[Variant]:
        """Get a variant by experiment ID and name.

        Args:
            experiment_id: Experiment ID
            name: Variant name

        Returns:
            Variant instance or None if not found
        """
        stmt = select(Variant).where(
            Variant.experiment_id == experiment_id, Variant.name == name
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def get_variant_by_id(self, variant_id: int) -> Optional[Variant]:
        """Get a variant by ID.

        Args:
            variant_id: Variant ID

        Returns:
            Variant instance or None if not found
        """
        return self.session.get(Variant, variant_id)

    def list_variants(self, experiment_id: int) -> List[Variant]:
        """List all variants for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            List of Variant instances
        """
        stmt = select(Variant).where(Variant.experiment_id == experiment_id)
        return list(self.session.execute(stmt).scalars().all())

    def delete_variant(self, variant_id: int) -> bool:
        """Delete a variant and all its results.

        Args:
            variant_id: Variant ID

        Returns:
            True if deleted, False if not found
        """
        variant = self.get_variant_by_id(variant_id)
        if not variant:
            return False

        self.session.delete(variant)
        self.session.commit()
        logger.info("Deleted variant: %s (id=%d)", variant.name, variant_id)
        return True

    # ------------------------------------------------------------------
    # Experiment lifecycle management
    # ------------------------------------------------------------------

    def start_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Start an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated Experiment or None if not found

        Raises:
            InvalidExperimentStatusError: If experiment cannot be started
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        self._validate_experiment_status_transition(experiment.status, "running")
        experiment.status = "running"
        experiment.start_at = datetime.now(datetime.UTC)

        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Started experiment: %s (id=%d)", experiment.name, experiment_id)
        return experiment

    def pause_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Pause an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated Experiment or None if not found

        Raises:
            InvalidExperimentStatusError: If experiment cannot be paused
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        self._validate_experiment_status_transition(experiment.status, "paused")
        experiment.status = "paused"

        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Paused experiment: %s (id=%d)", experiment.name, experiment_id)
        return experiment

    def complete_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Complete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated Experiment or None if not found

        Raises:
            InvalidExperimentStatusError: If experiment cannot be completed
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        self._validate_experiment_status_transition(experiment.status, "completed")
        experiment.status = "completed"
        experiment.end_at = datetime.now(datetime.UTC)

        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Completed experiment: %s (id=%d)", experiment.name, experiment_id)
        return experiment

    def archive_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Archive an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Updated Experiment or None if not found

        Raises:
            InvalidExperimentStatusError: If experiment cannot be archived
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            return None

        self._validate_experiment_status_transition(experiment.status, "archived")
        experiment.status = "archived"

        self.session.commit()
        self.session.refresh(experiment)
        logger.info("Archived experiment: %s (id=%d)", experiment.name, experiment_id)
        return experiment

    @staticmethod
    def _validate_experiment_status_transition(from_status: str, to_status: str) -> None:
        """Validate that an experiment status transition is allowed.

        Args:
            from_status: Current status
            to_status: Target status

        Raises:
            InvalidExperimentStatusError: If transition is not allowed
        """
        if to_status not in VALID_EXPERIMENT_STATUSES:
            raise InvalidExperimentStatusError(f"Invalid target status: '{to_status}'")

        if from_status == to_status:
            return  # No-op transition is allowed

        allowed_transitions = VALID_EXPERIMENT_TRANSITIONS.get(from_status, [])
        if to_status not in allowed_transitions:
            raise InvalidExperimentStatusError(
                f"Cannot transition from '{from_status}' to '{to_status}'. "
                f"Allowed transitions from '{from_status}': {allowed_transitions}"
            )

    # ------------------------------------------------------------------
    # Variant assignment
    # ------------------------------------------------------------------

    def assign_variant(
        self,
        experiment_id: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Variant]:
        """Assign a variant to a user/session based on traffic weights.

        Uses deterministic hashing for consistent assignment across requests.

        Args:
            experiment_id: Experiment ID
            user_id: Optional user identifier
            session_id: Optional session identifier

        Returns:
            Assigned Variant or None if experiment not found/not running

        Raises:
            ValueError: If neither user_id nor session_id provided
        """
        if not user_id and not session_id:
            raise ValueError("Either user_id or session_id must be provided")

        experiment = self.get_experiment(experiment_id)
        if not experiment or experiment.status != "running":
            return None

        variants = self.list_variants(experiment_id)
        if not variants:
            return None

        # Normalize traffic weights to sum to 1
        total_weight = sum(v.traffic_weight for v in variants)
        if total_weight == 0:
            return None

        # Use deterministic hashing for consistent assignment
        identifier = user_id or session_id
        hash_value = int(hashlib.md5(f"{experiment_id}:{identifier}".encode()).hexdigest(), 16)
        hash_float = (hash_value % 10000) / 10000.0

        # Select variant based on cumulative weights
        cumulative = 0.0
        for variant in variants:
            cumulative += variant.traffic_weight / total_weight
            if hash_float < cumulative:
                return variant

        return variants[-1]  # Fallback to last variant

    # ------------------------------------------------------------------
    # Result tracking
    # ------------------------------------------------------------------

    def record_result(
        self,
        variant_id: int,
        metrics: Dict[str, float],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentResult:
        """Record a result for a variant.

        Args:
            variant_id: Variant ID
            metrics: Dictionary of metric values (e.g., {"accuracy": 0.95})
            user_id: Optional user identifier
            session_id: Optional session identifier
            metadata: Optional additional context

        Returns:
            Created ExperimentResult instance

        Raises:
            ValueError: If variant not found
        """
        variant = self.get_variant_by_id(variant_id)
        if not variant:
            raise ValueError(f"Variant with id {variant_id} not found")

        result = ExperimentResult(
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            metrics=metrics,
            metadata=metadata,
        )
        self.session.add(result)
        self.session.commit()
        self.session.refresh(result)
        logger.debug("Recorded result for variant: %s (id=%d)", variant.name, variant_id)
        return result

    def get_variant_results(
        self,
        variant_id: int,
        metric_name: Optional[str] = None,
    ) -> List[ExperimentResult]:
        """Get results for a variant, optionally filtered by metric.

        Args:
            variant_id: Variant ID
            metric_name: Optional metric name to filter

        Returns:
            List of ExperimentResult instances
        """
        stmt = select(ExperimentResult).where(ExperimentResult.variant_id == variant_id)
        results = list(self.session.execute(stmt).scalars().all())

        if metric_name:
            results = [r for r in results if metric_name in r.metrics]

        return results

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------

    def compare_variants(
        self,
        experiment_id: int,
        metric_name: str,
        control_variant_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare variants using statistical tests.

        Args:
            experiment_id: Experiment ID
            metric_name: Metric to compare
            control_variant_name: Optional control variant name (auto-detect if not provided)

        Returns:
            Dictionary with comparison results including:
            - variant_stats: Statistics for each variant
            - pairwise_tests: Statistical test results between variants
            - winner: Best performing variant
        """
        variants = self.list_variants(experiment_id)
        if len(variants) < 2:
            raise ValueError("Experiment must have at least 2 variants to compare")

        # Identify control variant
        control = None
        if control_variant_name:
            control = next((v for v in variants if v.name == control_variant_name), None)
        else:
            control = next((v for v in variants if v.is_control), None)

        if not control:
            control = variants[0]  # Use first variant as control

        # Collect metrics for each variant
        variant_data = {}
        for variant in variants:
            results = self.get_variant_results(variant.id, metric_name)
            values = [r.metrics[metric_name] for r in results]
            variant_data[variant.name] = values

        # Calculate statistics for each variant
        variant_stats = {}
        for name, values in variant_data.items():
            if values:
                variant_stats[name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }
            else:
                variant_stats[name] = {
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                }

        # Perform pairwise tests
        pairwise_tests = []
        control_values = variant_data.get(control.name, [])

        for variant in variants:
            if variant.name == control.name:
                continue

            variant_values = variant_data.get(variant.name, [])
            if len(control_values) > 1 and len(variant_values) > 1:
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(control_values, variant_values)

                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(
                    (np.std(control_values) ** 2 + np.std(variant_values) ** 2) / 2
                )
                effect_size = (np.mean(variant_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0

                pairwise_tests.append(
                    {
                        "control": control.name,
                        "treatment": variant.name,
                        "t_statistic": t_stat,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "significant": p_value < 0.05,
                    }
                )

        # Determine winner (highest mean)
        winner = None
        best_mean = -float("inf")
        for name, stats in variant_stats.items():
            if stats["mean"] is not None and stats["mean"] > best_mean:
                best_mean = stats["mean"]
                winner = name

        return {
            "metric": metric_name,
            "variant_stats": variant_stats,
            "pairwise_tests": pairwise_tests,
            "winner": winner,
        }
