"""Adversarial defence mechanisms and robustness evaluation.

Resolves part of #645.

Provides input-space defences that need no retraining (feature squeezing,
Gaussian noise smoothing, input reconstruction), a detector that flags likely
adversarial inputs by prediction disagreement under those transformations,
adversarial-training data augmentation, and a robustness evaluator that scores
a model against the attacks in :mod:`astroml.security.adversarial.attacks`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroml.security.adversarial.attacks import (
    AttackConfig,
    AttackResult,
    AttackType,
    GradientFn,
    PredictProbaFn,
    generate_attack,
)

__all__ = [
    "AdversarialDetector",
    "DefenseType",
    "DetectionResult",
    "FeatureSqueezing",
    "GaussianSmoothing",
    "InputClipping",
    "RobustnessEvaluator",
    "RobustnessReport",
    "adversarial_training_set",
]


class DefenseType(str, Enum):
    """Supported input-space defences."""

    FEATURE_SQUEEZING = "feature_squeezing"
    GAUSSIAN_SMOOTHING = "gaussian_smoothing"
    INPUT_CLIPPING = "input_clipping"


class _Defense:
    """Base class for input transformations applied before inference."""

    defense_type: DefenseType

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the defended version of ``x``."""
        raise NotImplementedError

    def wrap(self, predict_proba: PredictProbaFn) -> PredictProbaFn:
        """Return ``predict_proba`` with this defence applied to its input."""

        def defended(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return predict_proba(self.transform(x))

        return defended


@dataclass
class FeatureSqueezing(_Defense):
    """Reduce input precision to collapse small adversarial perturbations.

    Quantises each feature onto ``2 ** bit_depth`` levels within
    ``value_range``.  Effective against low-amplitude L-infinity attacks and
    essentially free at inference time.
    """

    bit_depth: int = 4
    value_range: tuple[float, float] = (0.0, 1.0)
    defense_type: DefenseType = field(default=DefenseType.FEATURE_SQUEEZING, init=False)

    def __post_init__(self) -> None:
        """Validate the quantisation settings."""
        if not 1 <= self.bit_depth <= 16:
            raise ValueError("bit_depth must be within [1, 16]")
        if self.value_range[0] >= self.value_range[1]:
            raise ValueError("value_range must be (low, high) with low < high")

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Quantise ``x`` onto the configured grid."""
        low, high = self.value_range
        span = high - low
        levels = float(2**self.bit_depth - 1)
        normalised = np.clip((np.asarray(x, dtype=np.float64) - low) / span, 0.0, 1.0)
        return low + span * (np.round(normalised * levels) / levels)


@dataclass
class GaussianSmoothing(_Defense):
    """Average predictions over Gaussian-noised copies of the input.

    This is the randomised-smoothing defence (Cohen et al., 2019) in its
    simplest form: it trades a little clean accuracy for a much flatter
    decision surface around each input.
    """

    sigma: float = 0.05
    samples: int = 8
    seed: int | None = 0
    defense_type: DefenseType = field(default=DefenseType.GAUSSIAN_SMOOTHING, init=False)

    def __post_init__(self) -> None:
        """Validate the smoothing settings."""
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.samples < 1:
            raise ValueError("samples must be at least 1")

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``x`` with a single draw of Gaussian noise added."""
        rng = np.random.default_rng(self.seed)
        return np.asarray(x, dtype=np.float64) + rng.normal(0.0, self.sigma, np.shape(x))

    def wrap(self, predict_proba: PredictProbaFn) -> PredictProbaFn:
        """Return ``predict_proba`` averaged over ``samples`` noisy copies."""

        def defended(x: NDArray[np.float64]) -> NDArray[np.float64]:
            base = np.asarray(x, dtype=np.float64)
            rng = np.random.default_rng(self.seed)
            accumulated: NDArray[np.float64] | None = None
            for _ in range(self.samples):
                noisy = base + rng.normal(0.0, self.sigma, base.shape)
                probs = np.asarray(predict_proba(noisy), dtype=np.float64)
                accumulated = probs if accumulated is None else accumulated + probs
            assert accumulated is not None  # samples >= 1 enforced in __post_init__
            return accumulated / float(self.samples)

        return defended


@dataclass
class InputClipping(_Defense):
    """Clip inputs to the range observed during training.

    Cheap, and the only defence that reliably stops out-of-domain feature
    values from reaching a tabular fraud model at all.
    """

    lower: NDArray[np.float64] | float = 0.0
    upper: NDArray[np.float64] | float = 1.0
    defense_type: DefenseType = field(default=DefenseType.INPUT_CLIPPING, init=False)

    @classmethod
    def from_training_data(
        cls, x_train: NDArray[np.float64], *, quantile: float = 0.0
    ) -> InputClipping:
        """Fit clipping bounds from training data, optionally trimming tails."""
        if not 0.0 <= quantile < 0.5:
            raise ValueError("quantile must be within [0, 0.5)")
        data = np.asarray(x_train, dtype=np.float64)
        if quantile == 0.0:
            return cls(lower=data.min(axis=0), upper=data.max(axis=0))
        return cls(
            lower=np.quantile(data, quantile, axis=0),
            upper=np.quantile(data, 1.0 - quantile, axis=0),
        )

    def transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip ``x`` to the fitted bounds."""
        return np.clip(np.asarray(x, dtype=np.float64), self.lower, self.upper)


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of running :class:`AdversarialDetector` over a batch."""

    is_adversarial: NDArray[np.bool_]
    scores: NDArray[np.float64]
    threshold: float

    @property
    def flagged_count(self) -> int:
        """Number of samples flagged as adversarial."""
        return int(self.is_adversarial.sum())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "sample_count": int(self.is_adversarial.size),
            "flagged_count": self.flagged_count,
            "flagged_rate": float(self.is_adversarial.mean()) if self.is_adversarial.size else 0.0,
            "threshold": self.threshold,
            "mean_score": float(self.scores.mean()) if self.scores.size else 0.0,
        }


class AdversarialDetector:
    """Flags adversarial inputs via prediction disagreement under defences.

    An input whose predicted distribution shifts sharply when squeezed or
    smoothed sits close to a decision boundary in a way natural data rarely
    does — the feature-squeezing detection criterion of Xu et al. (2018).
    """

    def __init__(
        self,
        predict_proba: PredictProbaFn,
        defenses: Sequence[_Defense] | None = None,
        *,
        threshold: float = 0.3,
    ) -> None:
        if not 0.0 < threshold <= 2.0:
            raise ValueError("threshold must be within (0, 2]")
        self.predict_proba = predict_proba
        self.defenses: Sequence[_Defense] = defenses or (
            FeatureSqueezing(bit_depth=4),
            FeatureSqueezing(bit_depth=2),
        )
        self.threshold = threshold

    def scores(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the maximum L1 prediction shift across defences, per sample."""
        base = np.asarray(self.predict_proba(x), dtype=np.float64)
        worst = np.zeros(base.shape[0])
        for defense in self.defenses:
            defended = np.asarray(self.predict_proba(defense.transform(x)), dtype=np.float64)
            worst = np.maximum(worst, np.abs(base - defended).sum(axis=1))
        return worst

    def detect(self, x: NDArray[np.float64]) -> DetectionResult:
        """Return which samples in ``x`` look adversarial."""
        scores = self.scores(x)
        return DetectionResult(
            is_adversarial=scores > self.threshold,
            scores=scores,
            threshold=self.threshold,
        )

    def calibrate(
        self, x_clean: NDArray[np.float64], *, false_positive_rate: float = 0.05
    ) -> float:
        """Set the threshold from clean data at a target false-positive rate."""
        if not 0.0 < false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be within (0, 1)")
        scores = self.scores(x_clean)
        self.threshold = float(np.quantile(scores, 1.0 - false_positive_rate))
        # A degenerate all-zero score distribution would flag everything.
        self.threshold = max(self.threshold, 1e-6)
        return self.threshold


def adversarial_training_set(
    predict_proba: PredictProbaFn,
    x: NDArray[np.float64],
    y: NDArray[np.int_],
    *,
    gradient: GradientFn | None = None,
    attack: AttackType | str = AttackType.PGD,
    config: AttackConfig | None = None,
    ratio: float = 0.5,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """Return ``(x, y)`` augmented with adversarial examples for retraining.

    ``ratio`` is the fraction of the clean set to attack.  Labels of the
    adversarial examples stay the *true* labels — that is what makes retraining
    on them a defence rather than a poisoning.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be within (0, 1]")
    clean_x = np.asarray(x, dtype=np.float64)
    clean_y = np.asarray(y)
    take = max(1, int(round(clean_x.shape[0] * ratio)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(clean_x.shape[0], size=take, replace=False)

    result = generate_attack(
        attack,
        predict_proba,
        clean_x[indices],
        clean_y[indices],
        gradient=gradient,
        config=config,
    )
    return (
        np.vstack([clean_x, result.adversarial_examples]),
        np.concatenate([clean_y, clean_y[indices]]),
    )


@dataclass(frozen=True)
class RobustnessReport:
    """Per-attack robustness scores for one model."""

    clean_accuracy: float
    attack_results: dict[str, AttackResult]
    robustness_score: float

    def robust_accuracy(self, attack: AttackType | str) -> float:
        """Return accuracy under the named attack."""
        result = self.attack_results[AttackType(attack).value]
        return 1.0 - result.success_rate

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "clean_accuracy": self.clean_accuracy,
            "robustness_score": self.robustness_score,
            "attacks": {
                name: {
                    **result.to_dict(),
                    "robust_accuracy": 1.0 - result.success_rate,
                }
                for name, result in self.attack_results.items()
            },
        }


class RobustnessEvaluator:
    """Scores a model's adversarial robustness across several attacks."""

    def __init__(
        self,
        predict_proba: PredictProbaFn,
        *,
        gradient: GradientFn | None = None,
        config: AttackConfig | None = None,
        attacks: Iterable[AttackType | str] = (AttackType.FGSM, AttackType.PGD),
    ) -> None:
        self.predict_proba = predict_proba
        self.gradient = gradient
        self.config = config or AttackConfig()
        self.attacks = [AttackType(attack) for attack in attacks]

    def evaluate(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> RobustnessReport:
        """Run every configured attack and return a :class:`RobustnessReport`."""
        clean_x = np.asarray(x, dtype=np.float64)
        clean_y = np.asarray(y)
        clean_predictions = np.asarray(self.predict_proba(clean_x)).argmax(axis=1)
        clean_accuracy = float((clean_predictions == clean_y).mean())

        results: dict[str, AttackResult] = {}
        for attack in self.attacks:
            results[attack.value] = generate_attack(
                attack,
                self.predict_proba,
                clean_x,
                clean_y,
                gradient=self.gradient,
                config=self.config,
            )

        # The weakest link governs robustness: report the worst-case accuracy.
        worst = min((1.0 - r.success_rate for r in results.values()), default=1.0)
        return RobustnessReport(
            clean_accuracy=clean_accuracy,
            attack_results=results,
            robustness_score=worst * clean_accuracy,
        )
