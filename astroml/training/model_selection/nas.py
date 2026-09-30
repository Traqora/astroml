from astroml.utils.exceptions import AstroMLError
"""Neural architecture search (NAS) framework.

Samples feed-forward network architectures and evaluates them with
cross-validation, optionally under a time budget, returning the best
architecture found.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier

Activation = str  # one of "relu", "tanh", "logistic"

_UNIT_POOL = [16, 32, 64, 128]
_ACTIVATION_POOL: list[Activation] = ["relu", "tanh", "logistic"]
_MAX_LAYERS = 3


@dataclass
class ArchitectureSpec:
    """A candidate neural network architecture.

    Attributes:
        n_layers: Number of hidden layers.
        units: Hidden layer sizes (length equals ``n_layers``).
        activation: Activation function for hidden layers.
        learning_rate_init: Initial learning rate.
    """

    n_layers: int
    units: list[int]
    activation: Activation
    learning_rate_init: float = 1e-3

    def to_dict(self) -> dict[str, object]:
        """Serialize the architecture to a dictionary.

        Returns:
            A JSON-friendly dictionary.
        """
        return {
            "n_layers": self.n_layers,
            "units": self.units,
            "activation": self.activation,
            "learning_rate_init": self.learning_rate_init,
        }


@dataclass
class NASResult:
    """Result of a neural architecture search.

    Attributes:
        best_architecture: The best architecture found.
        best_score: CV score of the best architecture.
        candidates: Evaluated (architecture, score) pairs.
        evaluated: Number of candidates evaluated.
    """

    best_architecture: ArchitectureSpec
    best_score: float
    candidates: list[tuple[ArchitectureSpec, float]] = field(default_factory=list)
    evaluated: int = 0


class NeuralArchitectureSearch:
    """Randomly search feed-forward architectures for a dataset."""

    def __init__(
        self,
        build_fn: Callable[[ArchitectureSpec], Any] | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the search.

        Args:
            build_fn: Optional callable mapping an :class:`ArchitectureSpec`
                to a scikit-learn estimator. Defaults to an MLP classifier.
            random_state: Seed for reproducible sampling.
        """
        self.build_fn = build_fn or self._default_build
        self.random_state = random_state

    def sample_candidates(
        self, n: int, rng: np.random.Generator | None = None
    ) -> list[ArchitectureSpec]:
        """Sample a set of random architectures.

        Args:
            n: Number of architectures to sample.
            rng: Optional random generator; defaults to a seeded generator.

        Returns:
            A list of :class:`ArchitectureSpec` objects.

        Raises:
            ValueError: If ``n`` is not positive.
        """
        if n < 1:
            raise ValueError("n must be positive")
        rng = rng or np.random.default_rng(self.random_state)
        candidates: list[ArchitectureSpec] = []
        seen: set[tuple[object, ...]] = set()
        while len(candidates) < n:
            n_layers = int(rng.integers(1, _MAX_LAYERS + 1))
            units = [int(rng.choice(_UNIT_POOL)) for _ in range(n_layers)]
            activation = str(rng.choice(_ACTIVATION_POOL))
            learning_rate = float(rng.choice([1e-3, 1e-2, 5e-2]))
            key = (n_layers, tuple(units), activation, learning_rate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                ArchitectureSpec(
                    n_layers=n_layers,
                    units=units,
                    activation=activation,
                    learning_rate_init=learning_rate,
                )
            )
        return candidates

    def enumerate_small_space(self) -> list[ArchitectureSpec]:
        """Enumerate a small deterministic architecture space.

        Useful for exhaustive searches on small problems.

        Returns:
            All architectures combining the default unit pool, activations
            and layer counts up to :data:`_MAX_LAYERS`.
        """
        candidates: list[ArchitectureSpec] = []
        for n_layers in range(1, _MAX_LAYERS + 1):
            for units in itertools.product(_UNIT_POOL, repeat=n_layers):
                for activation in _ACTIVATION_POOL:
                    candidates.append(
                        ArchitectureSpec(
                            n_layers=n_layers,
                            units=list(units),
                            activation=activation,
                        )
                    )
        return candidates

    def evaluate(
        self,
        spec: ArchitectureSpec,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 3,
        scoring: str = "accuracy",
    ) -> float:
        """Cross-validate a single architecture.

        Args:
            spec: The architecture to evaluate.
            X: Feature matrix.
            y: Target labels.
            cv: Number of folds.
            scoring: Scoring metric.

        Returns:
            Mean cross-validation score.

        Raises:
            ValueError: If the data is invalid.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have matching lengths")
        if len(X) < max(2, cv):
            raise ValueError("Not enough samples for the requested number of folds")
        model = self.build_fn(spec)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, error_score="raise")
        return float(np.mean(scores))

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_candidates: int = 10,
        cv: int = 3,
        scoring: str = "accuracy",
        time_budget: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> NASResult:
        """Search for the best architecture.

        Args:
            X: Feature matrix.
            y: Target labels.
            n_candidates: Number of architectures to sample and evaluate.
            cv: Number of folds.
            scoring: Scoring metric.
            time_budget: Maximum search time in seconds (0 disables).
            rng: Optional random generator.

        Returns:
            A :class:`NASResult`.

        Raises:
            ValueError: If no candidates can be evaluated.
        """
        candidates = self.sample_candidates(n_candidates, rng)
        evaluated: list[tuple[ArchitectureSpec, float]] = []
        start = time.perf_counter()
        for spec in candidates:
            if time_budget > 0 and (time.perf_counter() - start) >= time_budget:
                break
            try:
                score = self.evaluate(spec, X, y, cv=cv, scoring=scoring)
                evaluated.append((spec, score))
            except AstroMLError:
                continue
        if not evaluated:
            raise ValueError("No architectures could be evaluated")
        best_spec, best_score = max(evaluated, key=lambda pair: pair[1])
        return NASResult(
            best_architecture=best_spec,
            best_score=best_score,
            candidates=evaluated,
            evaluated=len(evaluated),
        )

    def _default_build(self, spec: ArchitectureSpec) -> MLPClassifier:
        """Build an MLP classifier from an architecture spec.

        Args:
            spec: The architecture to build.

        Returns:
            An :class:`MLPClassifier` with the spec's hyperparameters.
        """
        return MLPClassifier(
            hidden_layer_sizes=tuple(spec.units),
            activation=spec.activation,
            learning_rate_init=spec.learning_rate_init,
            max_iter=300,
            random_state=self.random_state,
        )
