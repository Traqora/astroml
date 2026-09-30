"""Hybrid feature selection combining multiple strategies.

Supports ensemble-based selection, sequential strategy chaining,
and a FeatureSelectionPipeline for composing selection steps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astroml.preprocessing.feature_selection.embedded import EmbeddedSelector
from astroml.preprocessing.feature_selection.filter import FilterSelector, SelectionResult
from astroml.preprocessing.feature_selection.wrapper import WrapperSelector

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    """A single step in a feature selection pipeline.

    Attributes:
        name: Step name.
        selector: Configured selector instance (Filter, Wrapper, or Embedded).
        description: Optional description.
    """

    name: str
    selector: FilterSelector | WrapperSelector | EmbeddedSelector
    description: str = ""


class FeatureSelectionPipeline:
    """Sequential feature selection pipeline.

    Chains multiple selection strategies: e.g., filter -> wrapper -> embedded.

    Usage::

        pipe = FeatureSelectionPipeline([
            ("filter", FilterSelector(method="mutual_info", k=50)),
            ("embedded", EmbeddedSelector(method="tree", k=20)),
            ("wrapper", WrapperSelector(estimator=model, method="rfe", n_features_to_select=10)),
        ])
        X_selected = pipe.fit_transform(X, y)
    """

    def __init__(
        self,
        steps: list[tuple[str, FilterSelector | WrapperSelector | EmbeddedSelector]],
    ) -> None:
        """Initialize the pipeline.

        Args:
            steps: List of (name, selector) pairs executed in order.
        """
        self.steps: list[PipelineStep] = []
        for name, selector in steps:
            self.steps.append(PipelineStep(name=name, selector=selector))

        self._results: list[SelectionResult] = []

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> FeatureSelectionPipeline:
        """Fit all pipeline steps sequentially.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.

        Returns:
            Self.
        """
        self._results = []
        current_X = X
        current_names = feature_names

        for step in self.steps:
            logger.info("Fitting step: %s", step.name)

            if current_names is not None:
                try:
                    step.selector.fit(current_X, y, current_names)
                except TypeError:
                    step.selector.fit(current_X, y)
            else:
                step.selector.fit(current_X, y)

            result = step.selector.get_selection_result()
            self._results.append(result)

            current_X = step.selector.transform(current_X, y)

            # Update feature names for next step
            if current_names is not None and result.feature_names:
                current_names = result.feature_names

            logger.info(
                "Step '%s': %d/%d features selected",
                step.name,
                result.num_features_selected,
                result.num_features_total,
            )

        return self

    def transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Transform through all pipeline steps.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Reduced feature matrix.
        """
        current_X = X
        for step in self.steps:
            current_X = step.selector.transform(current_X)
        return current_X

    def fit_transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Fit and transform.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional names.

        Returns:
            Reduced feature matrix.
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_results(self) -> list[SelectionResult]:
        """Return per-step selection results.

        Returns:
            List of SelectionResult for each step.
        """
        return self._results

    def summary(self) -> str:
        """Human-readable pipeline summary.

        Returns:
            Summary string.
        """
        lines = [f"Feature Selection Pipeline ({len(self.steps)} steps):"]
        for step, result in zip(self.steps, self._results):
            lines.append(
                f"  {step.name}: {result.num_features_selected}/{result.num_features_total} "
                f"features ({step.selector.__class__.__name__})"
            )
        return "\n".join(lines)


class HybridSelector:
    """Hybrid feature selection combining multiple strategies via voting / ensemble.

    Runs multiple selectors and aggregates their results through
    voting, rank aggregation, or intersection/union strategies.

    Attributes:
        selectors: List of (name, selector) pairs.
        strategy: Aggregation strategy (``vote``, ``rank_aggregation``,
                  ``intersection``, ``union``).
        min_votes: Minimum number of selectors that must select a feature
                   (for ``vote`` strategy).
        k: Final number of features to keep.
    """

    SUPPORTED_STRATEGIES = ("vote", "rank_aggregation", "intersection", "union")

    def __init__(
        self,
        selectors: list[tuple[str, FilterSelector | WrapperSelector | EmbeddedSelector]],
        strategy: str = "vote",
        min_votes: int = 2,
        k: int | None = None,
    ) -> None:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported hybrid strategy '{strategy}'. "
                f"Choose from: {self.SUPPORTED_STRATEGIES}"
            )

        self.selectors = selectors
        self.strategy = strategy
        self.min_votes = min_votes
        self.k = k
        self._support: NDArray[np.bool_] | None = None
        self._importance_scores: NDArray[np.float64] | None = None
        self._feature_names: list[str] | None = None

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> HybridSelector:
        """Fit all selectors and aggregate results.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.

        Returns:
            Self.
        """
        n_features = X.shape[1]
        self._feature_names = feature_names

        votes = np.zeros(n_features, dtype=np.float64)
        rank_scores = np.zeros(n_features, dtype=np.float64)

        for name, selector in self.selectors:
            logger.info("Hybrid: fitting %s", name)
            try:
                if feature_names:
                    selector.fit(X, y, feature_names)
                else:
                    selector.fit(X, y)
            except TypeError:
                selector.fit(X, y)

            support = selector.get_support()
            votes += support.astype(np.float64)

            # Rank-based scoring (higher rank = better)
            result = selector.get_selection_result()
            for rank, idx in enumerate(result.selected_indices):
                rank_scores[idx] += 1.0 / (rank + 1)

        # Normalize rank scores
        mx = rank_scores.max()
        if mx > 0:
            rank_scores = rank_scores / mx

        self._importance_scores = rank_scores

        if self.strategy == "vote":
            self._support = votes >= self.min_votes
        elif self.strategy == "rank_aggregation":
            # Top k by rank score, then optionally threshold
            if self.k:
                indices = np.argsort(-rank_scores)[: self.k]
                self._support = np.zeros(n_features, dtype=bool)
                self._support[indices] = True
            else:
                self._support = rank_scores >= 0.1
        elif self.strategy == "intersection":
            # All selectors must agree
            self._support = votes >= len(self.selectors)
        elif self.strategy == "union":
            # Any selector can include a feature
            self._support = votes >= 1

        # Apply k limit if set
        if self.k is not None and self._support.sum() > self.k:
            top_indices = np.argsort(-rank_scores * self._support.astype(np.float64))[
                : self.k
            ]
            new_support = np.zeros(n_features, dtype=bool)
            new_support[top_indices] = True
            self._support = new_support

        return self

    def transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Return selected features.

        Args:
            X: Feature matrix.
            y: Ignored.

        Returns:
            Reduced matrix.
        """
        if self._support is None:
            raise RuntimeError("HybridSelector must be fit before transform")
        return X[:, self._support]

    def fit_transform(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64] | NDArray[np.int64],
        feature_names: list[str] | None = None,
    ) -> NDArray[np.float64]:
        """Fit and transform.

        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional names.

        Returns:
            Reduced matrix.
        """
        self.fit(X, y, feature_names)
        return self.transform(X)

    def get_support(self) -> NDArray[np.bool_]:
        """Boolean mask of selected features.

        Returns:
            Boolean array.
        """
        if self._support is None:
            raise RuntimeError("HybridSelector must be fit first")
        return self._support

    def get_selection_result(self) -> SelectionResult:
        """Structured selection result.

        Returns:
            SelectionResult.
        """
        if self._support is None:
            raise RuntimeError("HybridSelector must be fit first")

        indices = np.where(self._support)[0]
        scores = (
            [float(self._importance_scores[i]) for i in indices]
            if self._importance_scores is not None
            else []
        )

        return SelectionResult(
            selector_name=f"hybrid-{self.strategy}",
            num_features_selected=len(indices),
            num_features_total=len(self._support),
            selected_indices=indices.tolist(),
            scores=scores,
            feature_names=(
                [self._feature_names[i] for i in indices]
                if self._feature_names
                else None
            ),
            metadata={
                "strategy": self.strategy,
                "selectors": [name for name, _ in self.selectors],
                "min_votes": self.min_votes,
                "k": self.k,
            },
        )