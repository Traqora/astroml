"""Visualization utilities for experiment tracking.

Renders training metric charts, parallel coordinates, learning curves,
and hyperparameter importance plots for the experiment dashboard.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class ChartData:
    """Preprocessed chart data suitable for rendering by any frontend."""

    chart_type: str  # line, bar, parallel, heatmap, scatter
    title: str
    x_label: str
    y_label: str
    series: list[dict[str, Any]] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    annotations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentVisualizer:
    """Build chart-ready data from experiment runs and metrics.

    Produces dictionary payloads that can be serialized to JSON for
    web dashboards or notebooks.
    """

    # ------------------------------------------------------------------
    # Learning curves
    # ------------------------------------------------------------------

    @staticmethod
    def learning_curve(
        metric_history: dict[str, list[float]],
        title: str = "Learning Curve",
        smoothing_window: int = 0,
    ) -> ChartData:
        """Build a learning-curve chart from metric histories.

        Args:
            metric_history: Mapping of metric name to list of values per step.
            title: Chart title.
            smoothing_window: If > 0, apply moving average with this window size.

        Returns:
            ChartData for a multi-series line chart.
        """
        series = []
        for name, values in metric_history.items():
            y_vals: list[float] = list(values)
            if smoothing_window > 0 and len(y_vals) > smoothing_window:
                y_vals = _moving_average(y_vals, smoothing_window)
            series.append({
                "name": name,
                "data": [
                    {"x": i, "y": round(v, 6)} for i, v in enumerate(y_vals)
                ],
            })

        return ChartData(
            chart_type="line",
            title=title,
            x_label="Step",
            y_label="Value",
            series=series,
        )

    @staticmethod
    def metric_history_chart(
        runs_histories: dict[str, dict[str, list[float]]],
        metric_name: str,
        title: str | None = None,
    ) -> ChartData:
        """Overlay metric histories from multiple runs.

        Args:
            runs_histories: Mapping of run_name -> {metric: [values]}.
            metric_name: Which metric to plot.
            title: Optional chart title.

        Returns:
            ChartData for a multi-series line chart.
        """
        series = []
        for run_name, history in runs_histories.items():
            vals = history.get(metric_name, [])
            series.append({
                "name": run_name,
                "data": [
                    {"x": i, "y": round(v, 6)} for i, v in enumerate(vals)
                ],
            })

        return ChartData(
            chart_type="line",
            title=title or f"{metric_name} over time",
            x_label="Step",
            y_label=metric_name,
            series=series,
        )

    # ------------------------------------------------------------------
    # Parallel coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def parallel_coordinates(
        dimensions: list[str],
        data: list[dict[str, Any]],
        title: str = "Parallel Coordinates",
    ) -> ChartData:
        """Build parallel-coordinates chart data.

        Args:
            dimensions: Ordered list of axis names.
            data: List of records, each with a value for every dimension
                  plus ``run_name`` (str) and ``run_id`` (str) keys.
            title: Chart title.

        Returns:
            ChartData for a parallel-coordinates plot.
        """
        return ChartData(
            chart_type="parallel",
            title=title,
            x_label="",
            y_label="",
            dimensions=dimensions,
            series=data,
        )

    # ------------------------------------------------------------------
    # Hyperparameter importance
    # ------------------------------------------------------------------

    @staticmethod
    def hyperparameter_importance_bar(
        importance: list[tuple[str, float]],
        title: str = "Hyperparameter Importance",
    ) -> ChartData:
        """Build a horizontal bar chart for hyperparameter importance scores.

        Args:
            importance: Sorted list of (param_name, score) tuples.
            title: Chart title.

        Returns:
            ChartData for a bar chart.
        """
        series = [{
            "name": "importance",
            "data": [
                {"x": round(score, 4), "y": name}
                for name, score in importance
            ],
        }]

        return ChartData(
            chart_type="bar",
            title=title,
            x_label="Importance (|correlation|)",
            y_label="Parameter",
            series=series,
        )

    # ------------------------------------------------------------------
    # Metric correlation heatmap
    # ------------------------------------------------------------------

    @staticmethod
    def metric_correlation_heatmap(
        correlation_matrix: dict[str, dict[str, float]],
        title: str = "Metric Correlations",
    ) -> ChartData:
        """Build a heatmap chart from a correlation matrix.

        Args:
            correlation_matrix: Nested dict of metric_name -> {metric_name: value}.
            title: Chart title.

        Returns:
            ChartData for a heatmap.
        """
        metrics = sorted(correlation_matrix.keys())
        cells = []
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                cells.append({
                    "x": m1,
                    "y": m2,
                    "value": round(correlation_matrix[m1].get(m2, 0), 4),
                })

        series = [{"name": "correlation", "data": cells}]

        return ChartData(
            chart_type="heatmap",
            title=title,
            x_label="Metric",
            y_label="Metric",
            series=series,
        )

    # ------------------------------------------------------------------
    # Metric comparison (bar chart)
    # ------------------------------------------------------------------

    @staticmethod
    def metric_comparison_bars(
        runs: list[dict[str, Any]],
        metric_names: list[str],
        title: str = "Metric Comparison",
    ) -> ChartData:
        """Build grouped bar chart for comparing runs across metrics.

        Args:
            runs: List of {run_name, run_id, metric_name: value} dicts.
            metric_names: Metrics to include.
            title: Chart title.

        Returns:
            ChartData for a grouped bar chart.
        """
        series = []
        for run in runs:
            data_points = [
                {"x": m, "y": round(run.get(m, 0), 6)} for m in metric_names
            ]
            series.append({
                "name": run.get("run_name", run.get("run_id", "unknown")),
                "data": data_points,
            })

        return ChartData(
            chart_type="bar",
            title=title,
            x_label="Metric",
            y_label="Value",
            series=series,
        )

    # ------------------------------------------------------------------
    # Scatter (e.g. param vs metric)
    # ------------------------------------------------------------------

    @staticmethod
    def scatter_plot(
        x_values: list[float],
        y_values: list[float],
        labels: list[str] | None = None,
        x_label: str = "X",
        y_label: str = "Y",
        title: str = "Scatter Plot",
    ) -> ChartData:
        """Build a scatter plot.

        Args:
            x_values: X-axis values.
            y_values: Y-axis values.
            labels: Optional point labels.
            x_label: X-axis label.
            y_label: Y-axis label.
            title: Chart title.

        Returns:
            ChartData for a scatter chart.
        """
        data_points = []
        for i, (xv, yv) in enumerate(zip(x_values, y_values)):
            point: dict[str, Any] = {"x": round(xv, 6), "y": round(yv, 6)}
            if labels and i < len(labels):
                point["label"] = labels[i]
            data_points.append(point)

        return ChartData(
            chart_type="scatter",
            title=title,
            x_label=x_label,
            y_label=y_label,
            series=[{"name": "points", "data": data_points}],
        )

    # ------------------------------------------------------------------
    # Training convergence summary
    # ------------------------------------------------------------------

    @staticmethod
    def convergence_summary(
        runs_histories: dict[str, dict[str, list[float]]],
        metric_name: str = "loss",
        patience: int = 5,
        min_delta: float = 1e-4,
    ) -> dict[str, Any]:
        """Compute convergence statistics for each run.

        Args:
            runs_histories: Mapping of run_name -> {metric: [values]}.
            metric_name: Which metric to analyze for convergence.
            patience: Number of steps with no improvement before convergence.
            min_delta: Minimum change to count as improvement.

        Returns:
            Dict with per-run convergence data: steps_to_converge, final_value,
            best_step, best_value.
        """
        result: dict[str, Any] = {}
        for run_name, history in runs_histories.items():
            values = history.get(metric_name, [])
            if not values:
                result[run_name] = {
                    "steps_to_converge": None,
                    "final_value": None,
                    "best_step": None,
                    "best_value": None,
                }
                continue

            arr = np.array(values, dtype=np.float64)
            best_idx = int(np.argmin(arr))
            best_value = float(arr[best_idx])

            # Simple convergence: last step where improvement > min_delta
            converged_at = len(arr) - 1
            best_so_far = arr[0]
            best_so_far_idx = 0
            for i in range(1, len(arr)):
                if arr[i] < best_so_far - min_delta:
                    best_so_far = arr[i]
                    best_so_far_idx = i
                elif i - best_so_far_idx >= patience:
                    converged_at = best_so_far_idx
                    break

            result[run_name] = {
                "steps_to_converge": int(converged_at),
                "final_value": float(arr[-1]),
                "best_step": int(best_idx),
                "best_value": best_value,
            }

        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _moving_average(values: list[float], window: int) -> list[float]:
    """Apply simple moving average smoothing.

    Args:
        values: Input sequence.
        window: Smoothing window size.

    Returns:
        Smoothed list of same length.
    """
    if window <= 1:
        return list(values)
    arr = np.array(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    return [float(v) for v in smoothed]