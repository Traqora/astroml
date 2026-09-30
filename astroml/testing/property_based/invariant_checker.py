from astroml.utils.exceptions import AstroMLError
"""Invariant checker for ML model outputs.

Implements Procedure step 4:
  4. Add invariant checking for model outputs.

An *invariant* is a condition that must hold for **every** prediction a model
produces, regardless of the specific input.  This module provides:

* :class:`InvariantViolation` — structured description of a failed invariant.
* :class:`InvariantChecker` — registers named invariants and checks them
  against a batch of model outputs, collecting all violations.
* :func:`create_standard_invariant_checker` — convenience factory that
  pre-registers the most common ML output invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# ── Type alias ────────────────────────────────────────────────────────────────

InvariantFn = Callable[[np.ndarray, np.ndarray], bool]


# ── Violation record ──────────────────────────────────────────────────────────


@dataclass
class InvariantViolation:
    """Structured record of a single invariant violation.

    Attributes:
        invariant_name: Identifier of the invariant that was violated.
        sample_index: Row index in the input batch that triggered the
            violation, or ``None`` when the invariant operates on the
            whole output tensor.
        input_sample: The specific input row (or full matrix) that triggered
            the violation.
        model_output: The model output that violated the invariant.
        description: Human-readable explanation of the violation.
    """

    invariant_name: str
    sample_index: Optional[int]
    input_sample: np.ndarray
    model_output: np.ndarray
    description: str


# ── Invariant checker ─────────────────────────────────────────────────────────


@dataclass
class InvariantChecker:
    """Register and evaluate invariants against batches of model outputs.

    Usage::

        checker = InvariantChecker()
        checker.register("finite", lambda X, y: np.all(np.isfinite(y)))
        violations = checker.check(X_batch, predictions)

    Attributes:
        _invariants: Internal mapping from invariant name to its callable.
        _violations: Accumulated violations from the most recent
            :meth:`check` call.
    """

    _invariants: Dict[str, InvariantFn] = field(default_factory=dict, init=False)
    _violations: List[InvariantViolation] = field(default_factory=list, init=False)

    def register(self, name: str, fn: InvariantFn) -> None:
        """Register a named invariant function.

        Args:
            name: Unique identifier for the invariant.
            fn: Callable ``(X, output) -> bool``; return ``True`` if the
                invariant holds.

        Raises:
            ValueError: If *name* is already registered.
        """
        if name in self._invariants:
            raise ValueError(
                f"Invariant '{name}' is already registered. "
                "Use a different name or remove it first."
            )
        self._invariants[name] = fn

    def remove(self, name: str) -> None:
        """Remove a previously registered invariant.

        Args:
            name: Name of the invariant to remove.

        Raises:
            KeyError: If *name* is not registered.
        """
        if name not in self._invariants:
            raise KeyError(f"Invariant '{name}' is not registered.")
        del self._invariants[name]

    def check(
        self,
        X: np.ndarray,
        output: np.ndarray,
    ) -> List[InvariantViolation]:
        """Evaluate all registered invariants against *output*.

        Each invariant is called with the full *(X, output)* pair.  If an
        invariant returns ``False``, an :class:`InvariantViolation` is
        recorded.

        Args:
            X: Input feature matrix of shape ``(n_samples, n_features)``.
            output: Model output array of shape ``(n_samples, ...)``.

        Returns:
            A list of :class:`InvariantViolation` instances for every
            invariant that failed.  An empty list means all invariants passed.
        """
        self._violations = []
        for name, fn in self._invariants.items():
            try:
                passed = fn(X, output)
            except Exception as exc:
                self._violations.append(
                    InvariantViolation(
                        invariant_name=name,
                        sample_index=None,
                        input_sample=X,
                        model_output=output,
                        description=f"Invariant '{name}' raised an exception: {exc}",
                    )
                )
                continue

            if not passed:
                self._violations.append(
                    InvariantViolation(
                        invariant_name=name,
                        sample_index=None,
                        input_sample=X,
                        model_output=output,
                        description=(
                            f"Invariant '{name}' failed for output of " f"shape {output.shape}."
                        ),
                    )
                )
        return list(self._violations)

    def check_per_sample(
        self,
        X: np.ndarray,
        output: np.ndarray,
    ) -> List[InvariantViolation]:
        """Evaluate invariants row-by-row, reporting the first failing sample.

        This is slower than :meth:`check` but provides precise
        ``sample_index`` attribution in violations.

        Args:
            X: Input feature matrix of shape ``(n_samples, n_features)``.
            output: Model output array of shape ``(n_samples, ...)``.

        Returns:
            A list of :class:`InvariantViolation` instances; ``sample_index``
            is set to the index of the first row that violated each invariant.
        """
        self._violations = []
        n_samples = X.shape[0]

        for name, fn in self._invariants.items():
            for i in range(n_samples):
                xi = X[i: i + 1]
                yi = output[i: i + 1]
                try:
                    passed = fn(xi, yi)
                except Exception as exc:
                    self._violations.append(
                        InvariantViolation(
                            invariant_name=name,
                            sample_index=i,
                            input_sample=xi,
                            model_output=yi,
                            description=(f"Invariant '{name}' raised at sample {i}: {exc}"),
                        )
                    )
                    break

                if not passed:
                    self._violations.append(
                        InvariantViolation(
                            invariant_name=name,
                            sample_index=i,
                            input_sample=xi,
                            model_output=yi,
                            description=(
                                f"Invariant '{name}' violated at sample " f"{i}: output={yi!r}"
                            ),
                        )
                    )
                    break  # Report only the first violating sample per invariant

        return list(self._violations)

    @property
    def violations(self) -> List[InvariantViolation]:
        """Return violations recorded by the most recent :meth:`check` call."""
        return list(self._violations)

    @property
    def registered_names(self) -> List[str]:
        """Return the names of all registered invariants."""
        return list(self._invariants.keys())


# ── Convenience factory ───────────────────────────────────────────────────────


def create_standard_invariant_checker() -> InvariantChecker:
    """Return an :class:`InvariantChecker` pre-loaded with common invariants.

    The following invariants are registered:

    * ``"finite_outputs"`` — all output values are finite.
    * ``"non_negative_probabilities"`` — outputs are >= 0 (suitable for
      probability or score models).
    * ``"output_shape_consistent"`` — output has at least as many rows as *X*.

    Returns:
        A fully configured :class:`InvariantChecker`.
    """
    checker = InvariantChecker()

    checker.register(
        "finite_outputs",
        lambda X, output: bool(np.all(np.isfinite(output))),
    )
    checker.register(
        "non_negative_probabilities",
        lambda X, output: bool(np.all(output >= 0.0)),
    )
    checker.register(
        "output_shape_consistent",
        lambda X, output: output.shape[0] == X.shape[0],
    )

    return checker
