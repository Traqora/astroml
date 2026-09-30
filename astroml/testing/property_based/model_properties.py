from astroml.utils.exceptions import AstroMLError
"""Model invariant definitions and property-based test runner for ML models.

Implements Procedure steps 1 and 3:
  1. Define model invariants and properties.
  3. Build property-based test runner.

A *model property* is a callable ``(X, output) -> bool`` that asserts a
behavioural invariant that must hold for **every** input/output pair produced
by a model.  The :class:`ModelPropertyRunner` wires properties together with
Hypothesis and reports any violations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from hypothesis import HealthCheck, Phase, given, settings
from hypothesis import strategies as st

from astroml.testing.property_based.strategies import (
    feature_matrix_strategy,
    probability_output_strategy,
)

# ── Type aliases ──────────────────────────────────────────────────────────────

ModelCallable = Callable[[np.ndarray], np.ndarray]
PropertyFn = Callable[[np.ndarray, np.ndarray], bool]


# ── Built-in invariants ───────────────────────────────────────────────────────


def outputs_are_finite(X: np.ndarray, output: np.ndarray) -> bool:
    """Check that all model outputs are finite (no NaN or Inf).

    Args:
        X: Input feature matrix (unused, kept for uniform signature).
        output: Model output array.

    Returns:
        ``True`` if every element of *output* is finite.
    """
    return bool(np.all(np.isfinite(output)))


def output_shape_matches_samples(X: np.ndarray, output: np.ndarray) -> bool:
    """Check that the output leading dimension equals the number of samples.

    Args:
        X: Input feature matrix of shape ``(n_samples, n_features)``.
        output: Model output array; its first dimension must equal
            ``X.shape[0]``.

    Returns:
        ``True`` if ``output.shape[0] == X.shape[0]``.
    """
    return output.shape[0] == X.shape[0]


def binary_predictions_in_set(X: np.ndarray, output: np.ndarray) -> bool:
    """Check that every element of a binary prediction array is 0 or 1.

    Args:
        X: Input feature matrix (unused).
        output: 1-D integer prediction array.

    Returns:
        ``True`` if all values are in ``{0, 1}``.
    """
    return bool(np.all(np.isin(output, [0, 1])))


def probability_rows_sum_to_one(X: np.ndarray, output: np.ndarray) -> bool:
    """Check that each row of a probability matrix sums to approximately 1.

    Args:
        X: Input feature matrix (unused).
        output: 2-D probability matrix of shape ``(n_samples, n_classes)``.

    Returns:
        ``True`` if every row sum is within 1e-5 of 1.0.
    """
    if output.ndim != 2:
        return False
    row_sums = output.sum(axis=1)
    return bool(np.all(np.abs(row_sums - 1.0) < 1e-5))


def scores_in_unit_interval(X: np.ndarray, output: np.ndarray) -> bool:
    """Check that all scalar scores are in [0, 1].

    Args:
        X: Input feature matrix (unused).
        output: Array of scalar scores.

    Returns:
        ``True`` if every score is in ``[0, 1]``.
    """
    return bool(np.all((output >= 0.0) & (output <= 1.0)))


# ── Registry of built-in properties ──────────────────────────────────────────

BUILTIN_PROPERTIES: Dict[str, PropertyFn] = {
    "outputs_are_finite": outputs_are_finite,
    "output_shape_matches_samples": output_shape_matches_samples,
    "binary_predictions_in_set": binary_predictions_in_set,
    "probability_rows_sum_to_one": probability_rows_sum_to_one,
    "scores_in_unit_interval": scores_in_unit_interval,
}


# ── Result data class ─────────────────────────────────────────────────────────


@dataclass
class PropertyCheckResult:
    """Result of a single property check against one model.

    Attributes:
        property_name: Human-readable name of the property.
        passed: ``True`` if the property held for all tested inputs.
        violation_input: The input array that triggered a violation, or
            ``None`` if the property passed.
        violation_output: The output array produced for the violating input,
            or ``None`` if the property passed.
        error_message: Descriptive error string on failure, or ``None``.
    """

    property_name: str
    passed: bool
    violation_input: Optional[np.ndarray] = None
    violation_output: Optional[np.ndarray] = None
    error_message: Optional[str] = None


# ── Runner ────────────────────────────────────────────────────────────────────


@dataclass
class ModelPropertyRunner:
    """Run a set of property-based checks against a model callable.

    Usage::

        runner = ModelPropertyRunner(
            model=my_model.predict,
            properties=[outputs_are_finite, output_shape_matches_samples],
            n_features=10,
        )
        results = runner.run(max_examples=100)

    Attributes:
        model: Callable ``(X: np.ndarray) -> np.ndarray`` representing the
            model under test.
        properties: Sequence of property functions to check.
        n_features: Number of input features expected by *model*.
        max_samples: Maximum batch size to generate per example.
        _results: Internal list populated after :meth:`run` is called.
    """

    model: ModelCallable
    properties: Sequence[PropertyFn]
    n_features: int
    max_samples: int = 30
    _results: List[PropertyCheckResult] = field(default_factory=list, init=False)

    def run(self, max_examples: int = 50) -> List[PropertyCheckResult]:
        """Execute all registered properties via Hypothesis.

        For each property, a ``@given`` test is constructed inline and run.
        Violations are captured and returned rather than raised.

        Args:
            max_examples: Maximum Hypothesis examples per property.

        Returns:
            A list of :class:`PropertyCheckResult` instances—one per property.
        """
        self._results = []
        for prop in self.properties:
            result = self._check_property(prop, max_examples=max_examples)
            self._results.append(result)
        return list(self._results)

    def _check_property(
        self,
        prop: PropertyFn,
        max_examples: int,
    ) -> PropertyCheckResult:
        """Run a single property against the model.

        Args:
            prop: The property function to verify.
            max_examples: Maximum Hypothesis examples to generate.

        Returns:
            A :class:`PropertyCheckResult` describing the outcome.
        """
        n_features = self.n_features
        max_samples = self.max_samples
        model = self.model
        prop_name = prop.__name__

        violation: List[tuple[np.ndarray, np.ndarray, str]] = []

        @given(
            X=feature_matrix_strategy(
                min_samples=1,
                max_samples=max_samples,
                min_features=n_features,
                max_features=n_features,
            )
        )
        @settings(
            max_examples=max_examples,
            deadline=None,
            phases=[Phase.generate, Phase.shrink],
            suppress_health_check=[HealthCheck.too_slow],
        )
        def _inner(X: np.ndarray) -> None:
            try:
                output = model(X)
            except Exception as exc:
                violation.append((X, np.array([]), f"Model raised: {exc}"))
                return
            if not prop(X, output):
                violation.append(
                    (
                        X,
                        output,
                        f"Property '{prop_name}' violated on output {output!r}",
                    )
                )

        try:
            _inner()
        except Exception as exc:
            # Hypothesis itself raises AssertionError on shrunk counterexample
            if violation:
                inp, out, msg = violation[0]
                return PropertyCheckResult(
                    property_name=prop_name,
                    passed=False,
                    violation_input=inp,
                    violation_output=out,
                    error_message=msg,
                )
            return PropertyCheckResult(
                property_name=prop_name,
                passed=False,
                error_message=str(exc),
            )

        if violation:
            inp, out, msg = violation[0]
            return PropertyCheckResult(
                property_name=prop_name,
                passed=False,
                violation_input=inp,
                violation_output=out,
                error_message=msg,
            )

        return PropertyCheckResult(property_name=prop_name, passed=True)

    @property
    def results(self) -> List[PropertyCheckResult]:
        """Return the results from the most recent :meth:`run` call."""
        return list(self._results)
