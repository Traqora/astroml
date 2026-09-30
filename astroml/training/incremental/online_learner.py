"""Online learning algorithms for continuous model updating."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence


@dataclass
class OnlineLearnerConfig:
    """Configuration for online learners."""

    learning_rate: float = 0.01
    lr_schedule: str = "optimal"  # "constant", "optimal", "invscaling", "adaptive"
    loss: str = "log_loss"  # "log_loss", "hinge", "squared_error", "huber"
    penalty: str = "l2"  # "l2", "l1", "elasticnet", "none"
    alpha: float = 0.0001
    l1_ratio: float = 0.15
    fit_intercept: bool = True
    c_param: float = 1.0  # For Passive-Aggressive
    epsilon: float = 0.1  # For epsilon-insensitive regression
    power_t: float = 0.5
    eta0: float = 0.01
    random_state: int | None = 42


class BaseOnlineLearner:
    """Base class for online learning estimators."""

    def __init__(self, config: OnlineLearnerConfig | None = None) -> None:
        self.config = config or OnlineLearnerConfig()
        self.weights: list[float] = []
        self.intercept: float = 0.0
        self.t_: int = 0
        self.classes_: list[int] = []
        self.n_features_: int = 0
        if self.config.random_state is not None:
            random.seed(self.config.random_state)

    def _get_eta(self) -> float:
        """Compute learning rate based on schedule and step t."""
        sched = self.config.lr_schedule
        t = max(1, self.t_)
        if sched == "constant":
            return self.config.learning_rate
        elif sched == "optimal":
            # Heuristic optimal schedule: eta0 / (1.0 + alpha * eta0 * t)
            return self.config.eta0 / (1.0 + self.config.alpha * self.config.eta0 * t)
        elif sched == "invscaling":
            return self.config.eta0 / math.pow(t, self.config.power_t)
        elif sched == "adaptive":
            return self.config.learning_rate / math.sqrt(t)
        return self.config.learning_rate

    def _apply_regularization(self, weight: float, eta: float) -> float:
        """Apply L1/L2 penalty to weight."""
        penalty = self.config.penalty
        alpha = self.config.alpha
        if penalty == "l2":
            return weight * (1.0 - eta * alpha)
        elif penalty == "l1":
            shrinkage = eta * alpha
            if weight > shrinkage:
                return weight - shrinkage
            elif weight < -shrinkage:
                return weight + shrinkage
            else:
                return 0.0
        elif penalty == "elasticnet":
            l1_ratio = self.config.l1_ratio
            # L2 step
            w_l2 = weight * (1.0 - eta * alpha * (1.0 - l1_ratio))
            # L1 step
            l1_shrink = eta * alpha * l1_ratio
            if w_l2 > l1_shrink:
                return w_l2 - l1_shrink
            elif w_l2 < -l1_shrink:
                return w_l2 + l1_shrink
            else:
                return 0.0
        return weight

    def _dot(self, x: Sequence[float]) -> float:
        """Dot product of features and weights plus intercept."""
        val = self.intercept if self.config.fit_intercept else 0.0
        for w, xi in zip(self.weights, x):
            val += w * xi
        return val

    def get_params(self) -> dict[str, Any]:
        """Return model parameters."""
        return {
            "weights": list(self.weights),
            "intercept": self.intercept,
            "t_": self.t_,
            "n_features_": self.n_features_,
            "config": self.config,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set model parameters."""
        if "weights" in params:
            self.weights = list(params["weights"])
        if "intercept" in params:
            self.intercept = params["intercept"]
        if "t_" in params:
            self.t_ = params["t_"]
        if "n_features_" in params:
            self.n_features_ = params["n_features_"]
        if "config" in params and isinstance(params["config"], OnlineLearnerConfig):
            self.config = params["config"]


class OnlineSGDClassifier(BaseOnlineLearner):
    """Online linear classifier trained with Stochastic Gradient Descent."""

    def __init__(self, config: OnlineLearnerConfig | None = None) -> None:
        super().__init__(config)
        self.classes_: list[int] = [0, 1]

    def _initialize(self, n_features: int) -> None:
        if self.n_features_ == 0:
            self.n_features_ = n_features
            self.weights = [0.0] * n_features
            self.intercept = 0.0

    def partial_fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        classes: Sequence[int] | None = None,
    ) -> OnlineSGDClassifier:
        """Incremental fit on a batch of samples."""
        if not X:
            return self
        if classes is not None:
            self.classes_ = sorted(list(set(classes)))
        
        n_features = len(X[0])
        self._initialize(n_features)

        for xi, yi in zip(X, y):
            self.t_ += 1
            eta = self._get_eta()
            
            # Map binary {0, 1} to {-1, +1}
            target = 1.0 if yi == 1 else -1.0
            margin = self._dot(xi)

            if self.config.loss == "log_loss":
                # Logistic loss: dL/dz = -target / (1 + exp(target * margin))
                # Stable sigmoid computation
                z = target * margin
                if z >= 0:
                    prob_neg = 1.0 / (1.0 + math.exp(z))
                else:
                    exp_z = math.exp(z)
                    prob_neg = 1.0 - (exp_z / (1.0 + exp_z))
                grad = -target * prob_neg
            elif self.config.loss == "hinge":
                # Hinge loss: max(0, 1 - target * margin)
                grad = -target if target * margin < 1.0 else 0.0
            else:
                # Default logistic loss
                z = target * margin
                prob_neg = 1.0 / (1.0 + math.exp(min(max(z, -500), 500)))
                grad = -target * prob_neg

            # Update weights
            for j in range(self.n_features_):
                self.weights[j] = self._apply_regularization(self.weights[j], eta)
                self.weights[j] -= eta * grad * xi[j]

            if self.config.fit_intercept:
                self.intercept -= eta * grad

        return self

    def decision_function(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Compute decision function values."""
        return [self._dot(xi) for xi in X]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        """Predict class probabilities."""
        probas: list[list[float]] = []
        for xi in X:
            margin = self._dot(xi)
            # Sigmoid with overflow protection
            if margin >= 0:
                p1 = 1.0 / (1.0 + math.exp(-margin))
            else:
                p1 = math.exp(margin) / (1.0 + math.exp(margin))
            p0 = 1.0 - p1
            probas.append([p0, p1])
        return probas

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Predict binary class labels."""
        return [1 if self._dot(xi) >= 0.0 else 0 for xi in X]


class OnlineSGDRegressor(BaseOnlineLearner):
    """Online linear regressor trained with Stochastic Gradient Descent."""

    def _initialize(self, n_features: int) -> None:
        if self.n_features_ == 0:
            self.n_features_ = n_features
            self.weights = [0.0] * n_features
            self.intercept = 0.0

    def partial_fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
    ) -> OnlineSGDRegressor:
        """Incremental fit on a batch of samples."""
        if not X:
            return self

        n_features = len(X[0])
        self._initialize(n_features)

        for xi, yi in zip(X, y):
            self.t_ += 1
            eta = self._get_eta()
            pred = self._dot(xi)
            diff = pred - yi

            if self.config.loss == "squared_error":
                grad = diff
            elif self.config.loss == "huber":
                delta = self.config.epsilon
                if abs(diff) <= delta:
                    grad = diff
                else:
                    grad = delta if diff > 0 else -delta
            else:  # squared error default
                grad = diff

            # Update weights
            for j in range(self.n_features_):
                self.weights[j] = self._apply_regularization(self.weights[j], eta)
                self.weights[j] -= eta * grad * xi[j]

            if self.config.fit_intercept:
                self.intercept -= eta * grad

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict continuous targets."""
        return [self._dot(xi) for xi in X]


class OnlinePassiveAggressiveClassifier(BaseOnlineLearner):
    """Online Passive-Aggressive Classifier (Crammer & Singer)."""

    def __init__(self, config: OnlineLearnerConfig | None = None) -> None:
        super().__init__(config)
        self.classes_: list[int] = [0, 1]

    def _initialize(self, n_features: int) -> None:
        if self.n_features_ == 0:
            self.n_features_ = n_features
            self.weights = [0.0] * n_features
            self.intercept = 0.0

    def partial_fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        classes: Sequence[int] | None = None,
    ) -> OnlinePassiveAggressiveClassifier:
        """Incremental fit using Passive-Aggressive update rules."""
        if not X:
            return self
        if classes is not None:
            self.classes_ = sorted(list(set(classes)))

        n_features = len(X[0])
        self._initialize(n_features)
        C = self.config.c_param

        for xi, yi in zip(X, y):
            self.t_ += 1
            target = 1.0 if yi == 1 else -1.0
            margin = self._dot(xi)
            loss = max(0.0, 1.0 - target * margin)

            if loss > 0:
                sq_norm = sum(x_ij * x_ij for x_ij in xi) + (1.0 if self.config.fit_intercept else 0.0)
                if sq_norm > 1e-12:
                    # PA-I rule: tau = min(C, loss / ||x||^2)
                    tau = min(C, loss / sq_norm)
                    step = tau * target
                    for j in range(self.n_features_):
                        self.weights[j] += step * xi[j]
                    if self.config.fit_intercept:
                        self.intercept += step

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        """Predict binary class labels."""
        return [1 if self._dot(xi) >= 0.0 else 0 for xi in X]


class OnlinePassiveAggressiveRegressor(BaseOnlineLearner):
    """Online Passive-Aggressive Regressor."""

    def _initialize(self, n_features: int) -> None:
        if self.n_features_ == 0:
            self.n_features_ = n_features
            self.weights = [0.0] * n_features
            self.intercept = 0.0

    def partial_fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
    ) -> OnlinePassiveAggressiveRegressor:
        """Incremental fit using Passive-Aggressive epsilon-insensitive updates."""
        if not X:
            return self

        n_features = len(X[0])
        self._initialize(n_features)
        C = self.config.c_param
        eps = self.config.epsilon

        for xi, yi in zip(X, y):
            self.t_ += 1
            pred = self._dot(xi)
            diff = pred - yi
            loss = max(0.0, abs(diff) - eps)

            if loss > 0:
                sign = 1.0 if diff > 0 else -1.0
                sq_norm = sum(x_ij * x_ij for x_ij in xi) + (1.0 if self.config.fit_intercept else 0.0)
                if sq_norm > 1e-12:
                    tau = min(C, loss / sq_norm)
                    step = tau * sign
                    for j in range(self.n_features_):
                        self.weights[j] -= step * xi[j]
                    if self.config.fit_intercept:
                        self.intercept -= step

        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Predict continuous values."""
        return [self._dot(xi) for xi in X]
