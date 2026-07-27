"""
A/B test runner for LLM prompt and model comparison.

Manages traffic allocation, randomization, and experiment lifecycle.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from datetime import datetime
import uuid


class TrafficAllocation(str, Enum):
    """Traffic allocation strategies."""
    EQUAL = "50/50"
    NINETY_TEN = "90/10"
    CUSTOM = "custom"


@dataclass
class Variant:
    """A/B test variant."""
    name: str
    prompt_override: Optional[str] = None
    model_override: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ABTestConfig:
    """Configuration for A/B test."""
    test_id: str
    name: str
    description: str
    control_variant: Variant
    treatment_variant: Variant
    traffic_allocation: TrafficAllocation = TrafficAllocation.EQUAL
    custom_allocation: Optional[float] = None  # For CUSTOM allocation
    min_sample_size: int = 1000
    target_significance: float = 0.05  # p-value threshold
    primary_metric: str = "task_success_rate"
    secondary_metrics: List[str] = field(default_factory=list)
    duration_hours: int = 24
    auto_winner_declaration: bool = True
    auto_rollback_on_regression: bool = True


class ExperimentVariant:
    """Represents a variant in an experiment with accumulated results."""

    def __init__(self, variant: Variant):
        """Initialize variant."""
        self.variant = variant
        self.samples_seen = 0
        self.successes = 0
        self.metric_values: Dict[str, List[float]] = {}
        self.user_ids: List[str] = []

    def add_sample(
        self,
        user_id: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a sample to variant."""
        self.samples_seen += 1
        self.user_ids.append(user_id)

        if success:
            self.successes += 1

        if metrics:
            for key, value in metrics.items():
                if key not in self.metric_values:
                    self.metric_values[key] = []
                self.metric_values[key].append(value)

    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.samples_seen == 0:
            return 0.0
        return self.successes / self.samples_seen


class ABTest:
    """
    A/B test runner for comparing prompts and models.

    Manages experiment lifecycle, randomization, and result analysis.
    """

    def __init__(self, config: ABTestConfig):
        """Initialize A/B test."""
        self.config = config
        self.test_id = config.test_id or str(uuid.uuid4())
        self.started_at = datetime.now()
        self.ended_at: Optional[datetime] = None
        self.status = "running"  # running, completed, paused

        self.control = ExperimentVariant(config.control_variant)
        self.treatment = ExperimentVariant(config.treatment_variant)

        self.winner: Optional[str] = None
        self.is_significant = False

    def assign_variant(self, user_id: str) -> Variant:
        """
        Assign user to control or treatment variant.

        Args:
            user_id: User identifier

        Returns:
            Assigned variant
        """
        # Simulate deterministic assignment based on user_id
        hash_val = hash(user_id) % 100

        if self.config.traffic_allocation == TrafficAllocation.EQUAL:
            threshold = 50
        elif self.config.traffic_allocation == TrafficAllocation.NINETY_TEN:
            threshold = 90  # 90% control, 10% treatment
        else:  # CUSTOM
            threshold = int((self.config.custom_allocation or 50) * 100)

        if hash_val < threshold:
            return self.config.control_variant
        else:
            return self.config.treatment_variant

    def record_observation(
        self,
        user_id: str,
        variant_name: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Record an observation/result in the test.

        Args:
            user_id: User identifier
            variant_name: Name of variant
            success: Whether the task succeeded
            metrics: Additional metric values
        """
        if variant_name == self.config.control_variant.name:
            self.control.add_sample(user_id, success, metrics)
        elif variant_name == self.config.treatment_variant.name:
            self.treatment.add_sample(user_id, success, metrics)

    def get_test_status(self) -> Dict[str, Any]:
        """Get current test status and results."""
        control_rate = self.control.get_success_rate()
        treatment_rate = self.treatment.get_success_rate()

        return {
            "test_id": self.test_id,
            "name": self.config.name,
            "status": self.status,
            "duration_hours": (datetime.now() - self.started_at).total_seconds() / 3600,
            "control": {
                "name": self.config.control_variant.name,
                "samples": self.control.samples_seen,
                "success_rate": control_rate,
            },
            "treatment": {
                "name": self.config.treatment_variant.name,
                "samples": self.treatment.samples_seen,
                "success_rate": treatment_rate,
            },
            "min_samples_reached": (
                self.control.samples_seen >= self.config.min_sample_size
                and self.treatment.samples_seen >= self.config.min_sample_size
            ),
            "winner": self.winner,
            "is_significant": self.is_significant,
        }

    def check_significance(self) -> Dict[str, Any]:
        """
        Check statistical significance of results.

        Returns:
            Dictionary with significance testing results
        """
        # Simulate t-test
        p_value = 0.02  # Example p-value
        is_significant = p_value < self.config.target_significance

        control_rate = self.control.get_success_rate()
        treatment_rate = self.treatment.get_success_rate()

        self.is_significant = is_significant

        if is_significant:
            if treatment_rate > control_rate:
                self.winner = self.config.treatment_variant.name
            else:
                self.winner = self.config.control_variant.name

        return {
            "p_value": p_value,
            "is_significant": is_significant,
            "control_success_rate": control_rate,
            "treatment_success_rate": treatment_rate,
            "winner": self.winner,
            "confidence_level": 1 - p_value,
        }

    def should_auto_rollback(self) -> bool:
        """
        Check if automatic rollback is needed due to regression.

        Returns:
            True if rollback is recommended
        """
        if not self.config.auto_rollback_on_regression:
            return False

        control_rate = self.control.get_success_rate()
        treatment_rate = self.treatment.get_success_rate()

        # Rollback if treatment has >10% regression
        return (control_rate - treatment_rate) > 0.10

    def end_test(self) -> Dict[str, Any]:
        """End the test and declare winner if applicable."""
        self.ended_at = datetime.now()
        self.status = "completed"

        significance = self.check_significance()

        result = {
            "test_id": self.test_id,
            "name": self.config.name,
            "duration_hours": (self.ended_at - self.started_at).total_seconds() / 3600,
            "control_samples": self.control.samples_seen,
            "treatment_samples": self.treatment.samples_seen,
            **significance,
        }

        if self.config.auto_winner_declaration and self.is_significant:
            result["action"] = f"Deploy {self.winner}"
        else:
            result["action"] = "No clear winner - continue testing or manual review"

        return result

    def pause_test(self) -> None:
        """Pause the test."""
        self.status = "paused"

    def resume_test(self) -> None:
        """Resume a paused test."""
        if self.status == "paused":
            self.status = "running"
