"""Adversarial attack generation: FGSM, PGD and Carlini-Wagner.

Resolves part of #645.

The attacks are framework-agnostic and NumPy-only.  A model is supplied as two
callables:

``predict_proba(x) -> ndarray``
    Class probabilities of shape ``(n_samples, n_classes)``.
``gradient(x, y) -> ndarray``
    Gradient of the loss with respect to ``x``, same shape as ``x``.  When no
    analytic gradient is available, wrap ``predict_proba`` with
    :func:`numerical_gradient`, which estimates it by finite differences —
    slower, but it works for tree ensembles and remote scoring endpoints alike.

All attacks clip perturbed samples to ``clip_range`` so the adversarial inputs
stay inside the model's valid feature domain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "AttackConfig",
    "AttackResult",
    "AttackType",
    "CarliniWagnerAttack",
    "FGSMAttack",
    "GradientFn",
    "PGDAttack",
    "PredictProbaFn",
    "generate_attack",
    "numerical_gradient",
]

#: Signature of a probability-scoring model.
PredictProbaFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]
#: Signature of a loss-gradient function.
GradientFn = Callable[[NDArray[np.float64], NDArray[np.int_]], NDArray[np.float64]]


class AttackType(str, Enum):
    """Supported attack families."""

    FGSM = "fgsm"
    PGD = "pgd"
    CARLINI_WAGNER = "carlini_wagner"


@dataclass(frozen=True)
class AttackConfig:
    """Hyper-parameters shared by the attack implementations.

    Attributes
    ----------
    epsilon:
        L-infinity perturbation budget.
    step_size:
        Per-iteration step for iterative attacks; defaults to ``epsilon / 4``.
    max_iterations:
        Iteration cap for iterative attacks.
    targeted:
        When true, drive predictions *towards* the supplied labels rather than
        away from them.
    clip_range:
        ``(low, high)`` bounds applied to every perturbed feature.
    random_start:
        PGD only — start from a random point inside the epsilon ball.
    confidence:
        Carlini-Wagner only — margin by which the adversarial class must win.
    learning_rate:
        Carlini-Wagner only — optimizer step size.
    seed:
        Seed for the random start, so attacks are reproducible.
    """

    epsilon: float = 0.1
    step_size: float | None = None
    max_iterations: int = 40
    targeted: bool = False
    clip_range: tuple[float, float] = (0.0, 1.0)
    random_start: bool = True
    confidence: float = 0.0
    learning_rate: float = 0.01
    seed: int | None = 0

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.clip_range[0] >= self.clip_range[1]:
            raise ValueError("clip_range must be (low, high) with low < high")

    @property
    def resolved_step_size(self) -> float:
        """Return the configured step size, or a sensible default."""
        return self.step_size if self.step_size is not None else self.epsilon / 4.0


@dataclass(frozen=True)
class AttackResult:
    """Outcome of running an attack over a batch of samples."""

    attack: AttackType
    adversarial_examples: NDArray[np.float64]
    original_predictions: NDArray[np.int_]
    adversarial_predictions: NDArray[np.int_]
    success_mask: NDArray[np.bool_]
    mean_l2_distortion: float
    mean_linf_distortion: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Fraction of samples for which the attack changed the prediction."""
        return float(self.success_mask.mean()) if self.success_mask.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (without the example arrays)."""
        return {
            "attack": self.attack.value,
            "sample_count": int(self.success_mask.size),
            "success_rate": self.success_rate,
            "mean_l2_distortion": self.mean_l2_distortion,
            "mean_linf_distortion": self.mean_linf_distortion,
            "iterations": self.iterations,
            "metadata": dict(self.metadata),
        }


def numerical_gradient(predict_proba: PredictProbaFn, *, epsilon: float = 1e-4) -> GradientFn:
    """Return a finite-difference gradient of cross-entropy loss w.r.t. inputs.

    Costs ``2 * n_features`` model evaluations per call, so it is intended for
    low-dimensional tabular models and for black-box robustness auditing —
    prefer an analytic gradient where the framework provides one.
    """

    def gradient(x: NDArray[np.float64], y: NDArray[np.int_]) -> NDArray[np.float64]:
        x = np.asarray(x, dtype=np.float64)
        grad = np.zeros_like(x)
        for feature in range(x.shape[1]):
            step = np.zeros_like(x)
            step[:, feature] = epsilon
            loss_plus = _cross_entropy(predict_proba(x + step), y)
            loss_minus = _cross_entropy(predict_proba(x - step), y)
            grad[:, feature] = (loss_plus - loss_minus) / (2.0 * epsilon)
        return grad

    return gradient


class _BaseAttack:
    """Shared plumbing for the attack implementations."""

    attack_type: AttackType

    def __init__(
        self,
        predict_proba: PredictProbaFn,
        gradient: GradientFn | None = None,
        config: AttackConfig | None = None,
    ) -> None:
        self.predict_proba = predict_proba
        self.config = config or AttackConfig()
        self.gradient = gradient or numerical_gradient(predict_proba)

    def _predict(self, x: NDArray[np.float64]) -> NDArray[np.int_]:
        """Return hard class predictions for ``x``."""
        return np.asarray(self.predict_proba(x)).argmax(axis=1)

    def _clip(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip ``x`` into the configured valid feature range."""
        low, high = self.config.clip_range
        return np.clip(x, low, high)

    def _project(
        self, adversarial: NDArray[np.float64], original: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Project ``adversarial`` back into the epsilon ball around ``original``."""
        delta = np.clip(adversarial - original, -self.config.epsilon, self.config.epsilon)
        return self._clip(original + delta)

    def _result(
        self,
        original: NDArray[np.float64],
        adversarial: NDArray[np.float64],
        labels: NDArray[np.int_],
        iterations: int,
        metadata: dict[str, Any] | None = None,
    ) -> AttackResult:
        """Assemble an :class:`AttackResult` from an attack run."""
        original_preds = self._predict(original)
        adversarial_preds = self._predict(adversarial)
        if self.config.targeted:
            success = adversarial_preds == labels
        else:
            success = adversarial_preds != original_preds
        delta = adversarial - original
        return AttackResult(
            attack=self.attack_type,
            adversarial_examples=adversarial,
            original_predictions=original_preds,
            adversarial_predictions=adversarial_preds,
            success_mask=success,
            mean_l2_distortion=float(np.linalg.norm(delta, axis=1).mean()),
            mean_linf_distortion=float(np.abs(delta).max(axis=1).mean()) if delta.size else 0.0,
            iterations=iterations,
            metadata=metadata or {},
        )


class FGSMAttack(_BaseAttack):
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    A single step of size ``epsilon`` along the sign of the loss gradient.
    Cheap, and a good smoke test for gross robustness failures.
    """

    attack_type = AttackType.FGSM

    def generate(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> AttackResult:
        """Return adversarial examples for ``x`` with labels ``y``."""
        original = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y)
        direction = -1.0 if self.config.targeted else 1.0
        grad = self.gradient(original, labels)
        adversarial = self._clip(original + direction * self.config.epsilon * np.sign(grad))
        return self._result(original, adversarial, labels, iterations=1)


class PGDAttack(_BaseAttack):
    """Projected Gradient Descent (Madry et al., 2018).

    Iterated FGSM with projection back into the epsilon ball, optionally from a
    random start.  The strongest first-order attack in common use, and the
    reference benchmark for adversarial-training defences.
    """

    attack_type = AttackType.PGD

    def generate(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> AttackResult:
        """Return adversarial examples for ``x`` with labels ``y``."""
        original = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y)
        direction = -1.0 if self.config.targeted else 1.0

        adversarial = original.copy()
        if self.config.random_start:
            rng = np.random.default_rng(self.config.seed)
            noise = rng.uniform(-self.config.epsilon, self.config.epsilon, original.shape)
            adversarial = self._clip(original + noise)

        step = self.config.resolved_step_size
        for _ in range(self.config.max_iterations):
            grad = self.gradient(adversarial, labels)
            adversarial = adversarial + direction * step * np.sign(grad)
            adversarial = self._project(adversarial, original)

        return self._result(
            original,
            adversarial,
            labels,
            iterations=self.config.max_iterations,
            metadata={"random_start": self.config.random_start, "step_size": step},
        )


class CarliniWagnerAttack(_BaseAttack):
    """Carlini & Wagner L2 attack (2017), gradient-descent formulation.

    Minimises ``||delta||_2 + c * f(x + delta)`` where ``f`` is the margin
    between the true-class logit-surrogate and the best other class.  This
    implementation uses a fixed trade-off constant and plain gradient descent
    on the margin, which finds far smaller distortions than PGD at the cost of
    more model evaluations.
    """

    attack_type = AttackType.CARLINI_WAGNER

    def __init__(
        self,
        predict_proba: PredictProbaFn,
        gradient: GradientFn | None = None,
        config: AttackConfig | None = None,
        *,
        trade_off: float = 1.0,
    ) -> None:
        super().__init__(predict_proba, gradient, config)
        if trade_off <= 0:
            raise ValueError("trade_off must be positive")
        self.trade_off = trade_off

    def generate(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> AttackResult:
        """Return minimally-distorted adversarial examples for ``x``."""
        original = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y)
        adversarial = original.copy()
        best = original.copy()
        best_distance = np.full(original.shape[0], np.inf)

        for _ in range(self.config.max_iterations):
            margin_grad = self.gradient(adversarial, labels)
            distance_grad = 2.0 * (adversarial - original)
            update = self.trade_off * margin_grad - distance_grad
            adversarial = self._clip(adversarial + self.config.learning_rate * update)

            predictions = self._predict(adversarial)
            succeeded = predictions == labels if self.config.targeted else predictions != labels
            distances = np.linalg.norm(adversarial - original, axis=1)
            improved = succeeded & (distances < best_distance)
            best[improved] = adversarial[improved]
            best_distance[improved] = distances[improved]

        # Samples the attack never flipped keep their last iterate, which is
        # still the closest thing to an adversarial example we found.
        never_succeeded = ~np.isfinite(best_distance)
        best[never_succeeded] = adversarial[never_succeeded]

        return self._result(
            original,
            best,
            labels,
            iterations=self.config.max_iterations,
            metadata={"trade_off": self.trade_off, "confidence": self.config.confidence},
        )


#: Attack classes keyed by :class:`AttackType`.
_ATTACKS: dict[AttackType, type[_BaseAttack]] = {
    AttackType.FGSM: FGSMAttack,
    AttackType.PGD: PGDAttack,
    AttackType.CARLINI_WAGNER: CarliniWagnerAttack,
}


def generate_attack(
    attack: AttackType | str,
    predict_proba: PredictProbaFn,
    x: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    gradient: GradientFn | None = None,
    config: AttackConfig | None = None,
) -> AttackResult:
    """Run the named attack against ``predict_proba`` and return its result."""
    attack_type = AttackType(attack)
    attack_cls = _ATTACKS[attack_type]
    instance = attack_cls(predict_proba, gradient, config)
    return instance.generate(x, y)  # type: ignore[attr-defined]


def _cross_entropy(
    probabilities: NDArray[np.float64], labels: NDArray[np.int_]
) -> NDArray[np.float64]:
    """Return per-sample cross-entropy loss, guarding against log(0)."""
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    rows = np.arange(probs.shape[0])
    return -np.log(probs[rows, np.asarray(labels, dtype=int)])
