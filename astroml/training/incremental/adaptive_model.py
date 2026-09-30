"""Adaptive model wrapper with catastrophic forgetting prevention."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .online_learner import (
    BaseOnlineLearner,
    OnlineLearnerConfig,
    OnlineSGDClassifier,
    OnlineSGDRegressor,
)


@dataclass
class AdaptiveModelConfig:
    """Configuration for adaptive continuous learning model."""

    base_estimator_type: str = "sgd_classifier"  # "sgd_classifier", "sgd_regressor", "pa_classifier"
    learner_config: OnlineLearnerConfig = field(default_factory=OnlineLearnerConfig)
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.2  # Fraction of replay samples mixed per batch
    enable_replay: bool = True
    enable_ewc: bool = False
    ewc_lambda: float = 100.0  # Weight for EWC quadratic penalty
    drift_threshold: float = 0.15  # Metric drop threshold to flag drift
    drift_window_size: int = 50
    adaptive_lr_boost: float = 2.0  # Multiplier for learning rate upon drift detection
    random_state: int | None = 42


class ExperienceReplayBuffer:
    """Experience replay buffer utilizing reservoir sampling for bounded memory."""

    def __init__(self, max_size: int = 1000, random_state: int | None = None) -> None:
        self.max_size = max_size
        self.buffer_X: list[list[float]] = []
        self.buffer_y: list[Any] = []
        self.total_seen: int = 0
        if random_state is not None:
            random.seed(random_state)

    def add(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> None:
        """Add new samples using reservoir sampling."""
        for xi, yi in zip(X, y):
            self.total_seen += 1
            if len(self.buffer_X) < self.max_size:
                self.buffer_X.append(list(xi))
                self.buffer_y.append(yi)
            else:
                # Reservoir sampling: replace with probability max_size / total_seen
                idx = random.randint(0, self.total_seen - 1)
                if idx < self.max_size:
                    self.buffer_X[idx] = list(xi)
                    self.buffer_y[idx] = yi

    def sample(self, n: int) -> tuple[list[list[float]], list[Any]]:
        """Uniformly sample n exemplars from buffer."""
        if not self.buffer_X or n <= 0:
            return [], []
        sample_size = min(n, len(self.buffer_X))
        indices = random.sample(range(len(self.buffer_X)), sample_size)
        return [self.buffer_X[i] for i in indices], [self.buffer_y[i] for i in indices]

    def size(self) -> int:
        """Return current number of items in buffer."""
        return len(self.buffer_X)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer_X.clear()
        self.buffer_y.clear()
        self.total_seen = 0


class EWCRegularizer:
    """Elastic Weight Consolidation (EWC) regularizer for preventing catastrophic forgetting."""

    def __init__(self, ewc_lambda: float = 100.0) -> None:
        self.ewc_lambda = ewc_lambda
        self.optimal_weights: list[float] = []
        self.fisher_information: list[float] = []

    def update_task_anchors(
        self,
        current_weights: Sequence[float],
        X: Sequence[Sequence[float]],
        y: Sequence[Any],
    ) -> None:
        """Compute empirical Fisher Information diagonal and store optimal weights."""
        if not current_weights or not X:
            return
        self.optimal_weights = list(current_weights)
        n_features = len(current_weights)
        # Approximate diagonal Fisher Information Matrix
        fisher = [0.0] * n_features
        n_samples = len(X)
        for xi, yi in zip(X, y):
            for j in range(min(n_features, len(xi))):
                grad_approx = xi[j] * (1.0 if yi == 1 else -1.0)
                fisher[j] += (grad_approx ** 2) / max(1, n_samples)
        self.fisher_information = fisher

    def penalty_gradient(self, current_weights: Sequence[float]) -> list[float]:
        """Compute gradient of EWC quadratic penalty: lambda * sum(F_i * (w_i - w*_i))."""
        if not self.optimal_weights or not self.fisher_information:
            return [0.0] * len(current_weights)
        grad = []
        for w, w_opt, f_i in zip(current_weights, self.optimal_weights, self.fisher_information):
            grad.append(self.ewc_lambda * f_i * (w - w_opt))
        return grad


class AdaptiveModel:
    """Adaptive model coordinator combining online estimators, replay, and drift detection."""

    def __init__(
        self,
        config: AdaptiveModelConfig | None = None,
        estimator: BaseOnlineLearner | None = None,
    ) -> None:
        self.config = config or AdaptiveModelConfig()
        self.estimator = estimator or self._create_estimator()
        self.replay_buffer = ExperienceReplayBuffer(
            max_size=self.config.replay_buffer_size,
            random_state=self.config.random_state,
        )
        self.ewc = EWCRegularizer(ewc_lambda=self.config.ewc_lambda)
        
        # Drift tracking
        self.recent_errors: list[float] = []
        self.baseline_error: float | None = None
        self.drift_detected_count: int = 0
        self.total_updates: int = 0

    def _create_estimator(self) -> BaseOnlineLearner:
        est_type = self.config.base_estimator_type.lower()
        if est_type == "sgd_regressor":
            return OnlineSGDRegressor(self.config.learner_config)
        return OnlineSGDClassifier(self.config.learner_config)

    def adapt_and_update(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[Any],
        classes: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Perform adaptive update with replay mixing and drift monitoring."""
        if not X or not y:
            return {"loss": 0.0, "drift_detected": False, "samples_processed": 0}

        # 1. Prequential evaluation for drift detection
        preds = self.estimator.predict(X)
        errors = [1.0 if p != yi else 0.0 for p, yi in zip(preds, y)]
        batch_error = sum(errors) / max(1, len(errors))
        self.recent_errors.extend(errors)
        if len(self.recent_errors) > self.config.drift_window_size:
            self.recent_errors = self.recent_errors[-self.config.drift_window_size:]

        window_error = sum(self.recent_errors) / max(1, len(self.recent_errors))
        drift_detected = False

        if self.baseline_error is None:
            self.baseline_error = window_error
        elif window_error - self.baseline_error > self.config.drift_threshold:
            drift_detected = True
            self.drift_detected_count += 1
            # Adapt learning rate upon drift
            self.estimator.config.learning_rate *= self.config.adaptive_lr_boost
            self.baseline_error = window_error

        # 2. Mix with experience replay to prevent catastrophic forgetting
        fit_X = [list(xi) for xi in X]
        fit_y = list(y)

        if self.config.enable_replay and self.replay_buffer.size() > 0:
            n_replay = max(1, int(len(X) * self.config.replay_ratio))
            replay_X, replay_y = self.replay_buffer.sample(n_replay)
            fit_X.extend(replay_X)
            fit_y.extend(replay_y)

        # 3. Incremental update
        if isinstance(self.estimator, OnlineSGDClassifier):
            self.estimator.partial_fit(fit_X, fit_y, classes=classes)
        elif hasattr(self.estimator, "partial_fit"):
            self.estimator.partial_fit(fit_X, fit_y)

        # 4. Update replay buffer with new samples
        if self.config.enable_replay:
            self.replay_buffer.add(X, y)

        # 5. Update EWC if enabled
        if self.config.enable_ewc and self.total_updates % 10 == 0:
            self.ewc.update_task_anchors(self.estimator.weights, X, y)

        self.total_updates += 1

        return {
            "batch_error": batch_error,
            "window_error": window_error,
            "drift_detected": drift_detected,
            "samples_processed": len(fit_X),
            "total_updates": self.total_updates,
        }

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        """Predict outcomes for input samples."""
        return self.estimator.predict(X)

    def get_state(self) -> dict[str, Any]:
        """Serialize state for checkpointing."""
        return {
            "estimator_params": self.estimator.get_params(),
            "config": self.config,
            "total_updates": self.total_updates,
            "baseline_error": self.baseline_error,
            "drift_detected_count": self.drift_detected_count,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore state from checkpoint."""
        if "estimator_params" in state:
            self.estimator.set_params(state["estimator_params"])
        if "total_updates" in state:
            self.total_updates = state["total_updates"]
        if "baseline_error" in state:
            self.baseline_error = state["baseline_error"]
        if "drift_detected_count" in state:
            self.drift_detected_count = state["drift_detected_count"]
