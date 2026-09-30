from astroml.utils.exceptions import AstroMLError
"""Run comparison engine for experiment tracking.

Provides parallel coordinate plotting, hyperparameter importance analysis,
and pairwise comparison of training runs.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    """Metrics and metadata for a single experiment run."""

    run_id: str
    run_name: str
    experiment_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    metric_history: dict[str, list[float]] = field(default_factory=dict)
    start_time: str | None = None
    end_time: str | None = None
    status: str = "unknown"


@dataclass
class ComparisonResult:
    """Result of comparing two or more runs."""

    runs: list[RunMetrics]
    metric_diffs: dict[str, list[tuple[str, str, float]]]
    best_run: str | None = None
    worst_run: str | None = None
    correlation_scores: dict[str, float] = field(default_factory=dict)
    param_importance: list[tuple[str, float]] = field(default_factory=list)
    summary: str = ""


class RunComparator:
    """Compare experiment runs with parallel coordinates and hyperparameter analysis.

    Supports pairwise and multi-run comparison with parameter importance
    estimation and metric differential reporting.
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunMetrics] = {}

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def add_run(self, run: RunMetrics) -> None:
        """Register a run for comparison.

        Args:
            run: RunMetrics object describing the run.
        """
        self._runs[run.run_id] = run
        logger.debug("Registered run %s for comparison", run.run_id)

    def remove_run(self, run_id: str) -> None:
        """Remove a run from the comparison set.

        Args:
            run_id: ID of the run to remove.
        """
        self._runs.pop(run_id, None)

    def get_run(self, run_id: str) -> RunMetrics | None:
        """Retrieve a registered run by ID.

        Args:
            run_id: Run identifier.

        Returns:
            RunMetrics if found, None otherwise.
        """
        return self._runs.get(run_id)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        run_ids: list[str],
        target_metric: str = "accuracy",
        higher_is_better: bool = True,
    ) -> ComparisonResult:
        """Compare a set of runs and produce a structured result.

        Args:
            run_ids: List of run IDs to compare.
            target_metric: Primary metric for ranking.
            higher_is_better: Whether a higher target metric is desirable.

        Returns:
            ComparisonResult with diffs, best/worst run, and param importance.
        """
        runs = [self._runs[rid] for rid in run_ids if rid in self._runs]
        if len(runs) < 2:
            raise ValueError("At least two runs are required for comparison")

        metric_diffs = self._compute_metric_diffs(runs)
        best, worst = self._rank_runs(runs, target_metric, higher_is_better)
        importance = self._param_importance(runs, target_metric)
        correlation_scores = self._metric_correlations(runs, target_metric)

        summary_lines = [f"Compared {len(runs)} runs on metric: {target_metric}"]
        if best:
            summary_lines.append(f"Best: {best.run_name} ({best.run_id}) " f"with {target_metric}={best.metrics.get(target_metric, 'N/A')}")

        return ComparisonResult(
            runs=runs,
            metric_diffs=metric_diffs,
            best_run=best.run_id if best else None,
            worst_run=worst.run_id if worst else None,
            correlation_scores=correlation_scores,
            param_importance=importance,
            summary=". ".join(summary_lines),
        )

    def compare_pair(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> ComparisonResult:
        """Shortcut to compare exactly two runs.

        Args:
            run_id_a: First run ID.
            run_id_b: Second run ID.

        Returns:
            ComparisonResult for the pair.
        """
        return self.compare([run_id_a, run_id_b])

    # ------------------------------------------------------------------
    # Parallel coordinates
    # ------------------------------------------------------------------

    def parallel_coordinates_data(
        self,
        run_ids: list[str],
        metric_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build data suitable for parallel-coordinates plotting.

        Args:
            run_ids: Runs to include.
            metric_names: Subset of metrics to plot (defaults to all shared metrics).

        Returns:
            Dict with ``dimensions`` and ``data`` keys for chart rendering.
        """
        runs = [self._runs[rid] for rid in run_ids if rid in self._runs]
        if not runs:
            return {"dimensions": [], "data": []}

        all_metrics = set()
        for r in runs:
            all_metrics.update(r.metrics.keys())
        if metric_names is not None:
            all_metrics = all_metrics & set(metric_names)

        dimensions = sorted(all_metrics)
        data = []
        for r in runs:
            row: dict[str, Any] = {"run_name": r.run_name, "run_id": r.run_id}
            for dim in dimensions:
                row[dim] = r.metrics.get(dim)
            data.append(row)

        return {"dimensions": dimensions, "data": data}

    # ------------------------------------------------------------------
    # Hyperparameter importance
    # ------------------------------------------------------------------

    def hyperparameter_importance(
        self,
        run_ids: list[str],
        target_metric: str = "accuracy",
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Estimate hyperparameter importance via correlation with target metric.

        Uses absolute Spearman correlation between each numeric hyperparameter
        and the target metric across the given runs.

        Args:
            run_ids: Runs to analyze.
            target_metric: Metric to correlate against.
            top_k: Maximum number of parameters to return.

        Returns:
            Sorted list of (param_name, importance_score) tuples, descending.
        """
        runs = [self._runs[rid] for rid in run_ids if rid in self._runs]
        if len(runs) < 2:
            return []

        # Collect common numeric params
        param_values: dict[str, list[float]] = defaultdict(list)
        metric_values: list[float] = []

        for r in runs:
            m = r.metrics.get(target_metric)
            if m is None:
                continue
            metric_values.append(m)
            for k, v in r.params.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    param_values[k].append(float(v))

        if len(metric_values) < 3:
            return []

        # Compute correlations
        scores: list[tuple[str, float]] = []
        metric_arr = np.array(metric_values, dtype=np.float64)
        for pname, vals in param_values.items():
            if len(vals) < 3:
                continue
            vals_arr = np.array(vals, dtype=np.float64)
            try:
                corr = _spearman_r(vals_arr, metric_arr)
                if not np.isnan(corr):
                    scores.append((pname, abs(corr)))
            except AstroMLError:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_metric_diffs(
        self, runs: list[RunMetrics]
    ) -> dict[str, list[tuple[str, str, float]]]:
        """Compute pairwise metric differences."""
        diffs: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for i, ra in enumerate(runs):
            for j, rb in enumerate(runs):
                if i >= j:
                    continue
                for k in set(ra.metrics) & set(rb.metrics):
                    diff = ra.metrics[k] - rb.metrics[k]
                    if abs(diff) > 1e-9:
                        diffs[k].append((ra.run_name, rb.run_name, diff))
        return dict(diffs)

    def _rank_runs(
        self,
        runs: list[RunMetrics],
        target_metric: str,
        higher_is_better: bool,
    ) -> tuple[RunMetrics | None, RunMetrics | None]:
        """Rank runs by target metric."""
        scored = [(r, r.metrics.get(target_metric)) for r in runs]
        scored = [(r, v) for r, v in scored if v is not None]
        if not scored:
            return None, None
        scored.sort(key=lambda x: x[1], reverse=higher_is_better)
        return scored[0][0], scored[-1][0]

    def _param_importance(
        self, runs: list[RunMetrics], target_metric: str
    ) -> list[tuple[str, float]]:
        return self.hyperparameter_importance(
            [r.run_id for r in runs], target_metric
        )

    def _metric_correlations(
        self, runs: list[RunMetrics], target_metric: str
    ) -> dict[str, float]:
        """Compute Pearson correlation of each metric with target_metric."""
        target = [r.metrics[target_metric] for r in runs if target_metric in r.metrics]
        if len(target) < 3:
            return {}
        result: dict[str, float] = {}
        target_arr = np.array(target, dtype=np.float64)
        for r in runs[0].metrics:
            if r == target_metric:
                continue
            vals = [run.metrics[r] for run in runs if r in run.metrics]
            if len(vals) != len(target):
                continue
            vals_arr = np.array(vals, dtype=np.float64)
            corr = np.corrcoef(vals_arr, target_arr)[0, 1]
            if not np.isnan(corr):
                result[r] = float(corr)
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _spearman_r(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Compute Spearman rank correlation between two 1-D arrays."""
    if len(x) != len(y) or len(x) < 3:
        return float("nan")
    rank_x = _argsort_rank(x)
    rank_y = _argsort_rank(y)
    corr = np.corrcoef(rank_x, rank_y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def _argsort_rank(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return rank array (0 to 1 normalised) via argsort."""
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=np.float64)
    ranks[order] = np.linspace(0, 1, len(arr))
    return ranks