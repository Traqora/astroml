"""Model robustness and adversarial resilience evaluation suite.

Assesses model stability against Gaussian jittering, uniform noise, feature dropout,
adversarial perturbations (FGSM), and distribution drift.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from enum import Enum
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


class PerturbationType(str, Enum):
    """Types of data and input perturbations."""

    GAUSSIAN_NOISE = "gaussian_noise"
    UNIFORM_NOISE = "uniform_noise"
    ADVERSARIAL_FGSM = "adversarial_fgsm"
    FEATURE_DROPOUT = "feature_dropout"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass
class RobustnessMetricResult:
    """Outcome of a robustness stress test."""

    perturbation_type: str
    baseline_score: float
    perturbed_score: float
    performance_drop: float
    passed: bool
    threshold: float
    details: dict[str, Any] = dc_field(default_factory=dict)
    message: str = ""


class ModelRobustnessEvaluator:
    """Stress tests ML models under noisy, adversarial, and drifted inputs."""

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

    def _predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Helper to invoke model prediction across varying model types."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] == 2:
                return probs[:, 1]
            return probs
        elif hasattr(model, "predict"):
            return model.predict(X)
        elif callable(model):
            return model(X)
        elif isinstance(model, dict) and "weight" in model:
            w = model["weight"]
            b = model.get("bias", 0.0)
            logits = np.dot(X, w) + b
            return (logits >= 0).astype(float).flatten()
        else:
            raise ValueError(f"Unable to invoke prediction on model of type {type(model)}")

    def _score_accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate binary / multi-class accuracy score."""
        y_true_flat = y_true.flatten()
        if y_pred.dtype == float and np.all((y_pred >= 0.0) & (y_pred <= 1.0)):
            preds_bin = (y_pred >= 0.5).astype(int).flatten()
        else:
            preds_bin = (y_pred >= 0.5).astype(int).flatten()
        return float(np.mean(preds_bin == y_true_flat))

    def evaluate_noise_perturbation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        noise_levels: list[float] | None = None,
        noise_type: str = "gaussian",
        max_allowed_drop: float = 0.15,
    ) -> RobustnessMetricResult:
        """Test model resilience against Gaussian or uniform feature noise."""
        levels = noise_levels or [0.01, 0.05, 0.1, 0.2]
        base_preds = self._predict(model, X)
        base_acc = self._score_accuracy(base_preds, y)

        curve: dict[float, float] = {}
        for lvl in levels:
            if noise_type == "gaussian":
                noise = self.rng.normal(0, lvl, size=X.shape)
            else:
                noise = self.rng.uniform(-lvl, lvl, size=X.shape)

            X_noisy = X + noise
            noisy_preds = self._predict(model, X_noisy)
            noisy_acc = self._score_accuracy(noisy_preds, y)
            curve[lvl] = round(noisy_acc, 4)

        # Average drop across noise levels
        worst_acc = min(curve.values())
        drop = max(0.0, base_acc - worst_acc)
        passed = drop <= max_allowed_drop

        return RobustnessMetricResult(
            perturbation_type=f"{noise_type}_noise",
            baseline_score=round(base_acc, 4),
            perturbed_score=round(worst_acc, 4),
            performance_drop=round(drop, 4),
            passed=passed,
            threshold=max_allowed_drop,
            details={"accuracy_by_noise_level": curve},
            message=(
                f"Noise resilience passed (drop={drop:.2%} <= {max_allowed_drop:.2%})"
                if passed
                else f"Noise resilience failed (drop={drop:.2%} > {max_allowed_drop:.2%})"
            ),
        )

    def evaluate_feature_dropout(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        drop_rates: list[float] | None = None,
        max_allowed_drop: float = 0.20,
    ) -> RobustnessMetricResult:
        """Test model tolerance to missing/dropped feature values."""
        rates = drop_rates or [0.05, 0.1, 0.2]
        base_preds = self._predict(model, X)
        base_acc = self._score_accuracy(base_preds, y)

        curve: dict[float, float] = {}
        for rate in rates:
            X_dropped = np.copy(X)
            mask = self.rng.binomial(1, rate, size=X.shape).astype(bool)
            X_dropped[mask] = 0.0  # Impute dropped with zero

            d_preds = self._predict(model, X_dropped)
            d_acc = self._score_accuracy(d_preds, y)
            curve[rate] = round(d_acc, 4)

        worst_acc = min(curve.values())
        drop = max(0.0, base_acc - worst_acc)
        passed = drop <= max_allowed_drop

        return RobustnessMetricResult(
            perturbation_type=PerturbationType.FEATURE_DROPOUT.value,
            baseline_score=round(base_acc, 4),
            perturbed_score=round(worst_acc, 4),
            performance_drop=round(drop, 4),
            passed=passed,
            threshold=max_allowed_drop,
            details={"accuracy_by_dropout_rate": curve},
            message=(
                f"Feature dropout tolerance passed (drop={drop:.2%} <= {max_allowed_drop:.2%})"
                if passed
                else f"Feature dropout tolerance failed (drop={drop:.2%} > {max_allowed_drop:.2%})"
            ),
        )

    def evaluate_adversarial_fgsm(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.05,
        max_allowed_drop: float = 0.25,
    ) -> RobustnessMetricResult:
        """Fast Gradient Sign Method (FGSM) adversarial vulnerability check."""
        base_preds = self._predict(model, X)
        base_acc = self._score_accuracy(base_preds, y)

        # Estimate empirical gradient sign for black-box/linear model
        h = 1e-4
        grad_sign = np.zeros_like(X)

        for j in range(X.shape[1]):
            X_plus = np.copy(X)
            X_plus[:, j] += h
            p_plus = self._predict(model, X_plus)

            X_minus = np.copy(X)
            X_minus[:, j] -= h
            p_minus = self._predict(model, X_minus)

            diff = (p_plus - p_minus) / (2 * h)
            grad_sign[:, j] = np.sign(diff)

        # Adversarial attack: X_adv = X + eps * sign(grad)
        X_adv = X + epsilon * grad_sign
        adv_preds = self._predict(model, X_adv)
        adv_acc = self._score_accuracy(adv_preds, y)

        drop = max(0.0, base_acc - adv_acc)
        passed = drop <= max_allowed_drop

        return RobustnessMetricResult(
            perturbation_type=PerturbationType.ADVERSARIAL_FGSM.value,
            baseline_score=round(base_acc, 4),
            perturbed_score=round(adv_acc, 4),
            performance_drop=round(drop, 4),
            passed=passed,
            threshold=max_allowed_drop,
            details={"epsilon": epsilon},
            message=(
                f"Adversarial FGSM check passed (drop={drop:.2%} <= {max_allowed_drop:.2%})"
                if passed
                else f"Adversarial FGSM check failed (drop={drop:.2%} > {max_allowed_drop:.2%})"
            ),
        )

    def evaluate_distribution_shift(
        self,
        model: Any,
        X_baseline: np.ndarray,
        y_baseline: np.ndarray,
        X_shifted: np.ndarray,
        y_shifted: np.ndarray,
        max_allowed_drop: float = 0.15,
    ) -> RobustnessMetricResult:
        """Evaluate performance retention under covariates or concept drift."""
        base_preds = self._predict(model, X_baseline)
        base_acc = self._score_accuracy(base_preds, y_baseline)

        shift_preds = self._predict(model, X_shifted)
        shift_acc = self._score_accuracy(shift_preds, y_shifted)

        drop = max(0.0, base_acc - shift_acc)
        passed = drop <= max_allowed_drop

        return RobustnessMetricResult(
            perturbation_type=PerturbationType.DISTRIBUTION_SHIFT.value,
            baseline_score=round(base_acc, 4),
            perturbed_score=round(shift_acc, 4),
            performance_drop=round(drop, 4),
            passed=passed,
            threshold=max_allowed_drop,
            details={"baseline_samples": len(X_baseline), "shifted_samples": len(X_shifted)},
            message=(
                f"Distribution shift test passed (drop={drop:.2%} <= {max_allowed_drop:.2%})"
                if passed
                else f"Distribution shift test failed (drop={drop:.2%} > {max_allowed_drop:.2%})"
            ),
        )

    def run_comprehensive_suite(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        max_drop: float = 0.20,
    ) -> dict[str, Any]:
        """Execute all robustness tests and compute an overall robustness score."""
        g_res = self.evaluate_noise_perturbation(model, X, y, noise_type="gaussian", max_allowed_drop=max_drop)
        u_res = self.evaluate_noise_perturbation(model, X, y, noise_type="uniform", max_allowed_drop=max_drop)
        f_res = self.evaluate_feature_dropout(model, X, y, max_allowed_drop=max_drop)
        adv_res = self.evaluate_adversarial_fgsm(model, X, y, max_allowed_drop=max_drop + 0.1)

        tests = [g_res, u_res, f_res, adv_res]
        passed_count = sum(1 for t in tests if t.passed)
        avg_drop = float(np.mean([t.performance_drop for t in tests]))
        robustness_score = max(0.0, round((1.0 - avg_drop) * 100.0, 2))

        return {
            "robustness_score": robustness_score,
            "all_passed": passed_count == len(tests),
            "passed_tests": passed_count,
            "total_tests": len(tests),
            "results": {
                t.perturbation_type: {
                    "passed": t.passed,
                    "baseline_score": t.baseline_score,
                    "perturbed_score": t.perturbed_score,
                    "performance_drop": t.performance_drop,
                    "threshold": t.threshold,
                    "message": t.message,
                    "details": t.details,
                }
                for t in tests
            },
        }
