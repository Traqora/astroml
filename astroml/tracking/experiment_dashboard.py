"""Experiment dashboard for listing, searching, cloning, tagging, and reporting.

Provides a high-level API over grouped experiment runs backed by MLflow
or an in-memory store for environments without MLflow.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from astroml.tracking.run_comparator import RunComparator, RunMetrics
from astroml.tracking.visualizations import ChartData, ExperimentVisualizer

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Represents a logical experiment grouping one or more runs."""

    experiment_id: str
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    artifact_uri: str | None = None
    runs: list[RunMetrics] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    """Structured report for an experiment.

    Can be exported as JSON, Markdown, or plain text.
    """

    experiment_id: str
    experiment_name: str
    generated_at: str
    num_runs: int
    best_run: dict[str, Any] | None = None
    metric_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    param_importance: list[tuple[str, float]] = field(default_factory=list)
    comparison_charts: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)


class ExperimentDashboard:
    """Central experiment tracking dashboard.

    Supports listing, searching, tagging, cloning, and generating reports
    for experiments and their runs.
    """

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._comparator = RunComparator()
        self._visualizer = ExperimentVisualizer()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Unique experiment name.
            description: Short description.
            tags: Optional list of tags.
            notes: Free-form notes.
            metadata: Arbitrary key-value metadata.

        Returns:
            The created Experiment.

        Raises:
            ValueError: If an experiment with the given name already exists.
        """
        if any(e.name == name for e in self._experiments.values()):
            raise ValueError(f"Experiment '{name}' already exists")

        exp = Experiment(
            experiment_id=uuid.uuid4().hex[:12],
            name=name,
            description=description,
            tags=tags or [],
            notes=notes,
            metadata=metadata or {},
        )
        self._experiments[exp.experiment_id] = exp
        logger.info("Created experiment '%s' (id=%s)", name, exp.experiment_id)
        return exp

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Retrieve an experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Experiment if found, None otherwise.
        """
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        tag_filter: list[str] | None = None,
        search_query: str | None = None,
        sort_by: str = "created_at",
        reverse: bool = False,
    ) -> list[Experiment]:
        """List experiments with optional filtering and search.

        Args:
            tag_filter: Only return experiments with ALL of these tags.
            search_query: Free-text search across name, description, notes.
            sort_by: Attribute to sort by (created_at, updated_at, name).
            reverse: Whether to reverse sort order.

        Returns:
            Filtered and sorted list of experiments.
        """
        results = list(self._experiments.values())

        if tag_filter:
            required = set(tag_filter)
            results = [e for e in results if required.issubset(set(e.tags))]

        if search_query:
            q = search_query.lower()
            results = [
                e
                for e in results
                if q in e.name.lower()
                or q in e.description.lower()
                or q in e.notes.lower()
                or any(q in t.lower() for t in e.tags)
            ]

        results.sort(
            key=lambda e: getattr(e, sort_by, ""),
            reverse=reverse,
        )
        return results

    def update_experiment(
        self,
        experiment_id: str,
        name: str | None = None,
        description: str | None = None,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> Experiment:
        """Update mutable fields of an experiment.

        Args:
            experiment_id: Experiment to update.
            name: New name (if provided).
            description: New description.
            notes: New notes.
            tags: New tag set (replaces existing tags).

        Returns:
            Updated Experiment.

        Raises:
            ValueError: If experiment not found.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        if name is not None:
            exp.name = name
        if description is not None:
            exp.description = description
        if notes is not None:
            exp.notes = notes
        if tags is not None:
            exp.tags = tags
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        return exp

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: Experiment to delete.

        Returns:
            True if deleted, False if not found.
        """
        if experiment_id in self._experiments:
            del self._experiments[experiment_id]
            logger.info("Deleted experiment %s", experiment_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def add_run(self, experiment_id: str, run: RunMetrics) -> None:
        """Add a run to an experiment.

        Args:
            experiment_id: Target experiment.
            run: RunMetrics to attach.

        Raises:
            ValueError: If experiment not found.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        exp.runs.append(run)
        self._comparator.add_run(run)
        exp.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_run(self, experiment_id: str, run_id: str) -> None:
        """Remove a run from an experiment.

        Args:
            experiment_id: Experiment containing the run.
            run_id: Run to remove.

        Raises:
            ValueError: If experiment not found.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        exp.runs = [r for r in exp.runs if r.run_id != run_id]
        self._comparator.remove_run(run_id)
        exp.updated_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def add_tag(self, experiment_id: str, tag: str) -> Experiment:
        """Add a tag to an experiment.

        Args:
            experiment_id: Experiment to tag.
            tag: Tag to add.

        Returns:
            Updated experiment.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        if tag not in exp.tags:
            exp.tags.append(tag)
            exp.updated_at = datetime.now(timezone.utc).isoformat()
        return exp

    def remove_tag(self, experiment_id: str, tag: str) -> Experiment:
        """Remove a tag from an experiment.

        Args:
            experiment_id: Experiment to untag.
            tag: Tag to remove.

        Returns:
            Updated experiment.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")
        if tag in exp.tags:
            exp.tags.remove(tag)
            exp.updated_at = datetime.now(timezone.utc).isoformat()
        return exp

    # ------------------------------------------------------------------
    # Cloning
    # ------------------------------------------------------------------

    def clone_experiment(
        self,
        experiment_id: str,
        new_name: str | None = None,
        copy_runs: bool = False,
    ) -> Experiment:
        """Clone an experiment with optional run copying.

        Args:
            experiment_id: Source experiment.
            new_name: Name for the clone (defaults to '{name} (clone)').
            copy_runs: If True, also copy attached runs.

        Returns:
            The cloned Experiment.

        Raises:
            ValueError: If source experiment not found.
        """
        src = self._experiments.get(experiment_id)
        if src is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        clone = Experiment(
            experiment_id=uuid.uuid4().hex[:12],
            name=new_name or f"{src.name} (clone)",
            description=src.description,
            tags=list(src.tags),
            notes=f"Cloned from {src.name} ({src.experiment_id})",
            metadata=dict(src.metadata),
        )

        if copy_runs:
            clone.runs = [copy.deepcopy(r) for r in src.runs]
            for r in clone.runs:
                self._comparator.add_run(r)

        self._experiments[clone.experiment_id] = clone
        logger.info(
            "Cloned experiment '%s' -> '%s' (id=%s)",
            src.experiment_id,
            clone.experiment_id,
            clone.name,
        )
        return clone

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_runs(
        self,
        run_ids: list[str],
        target_metric: str = "accuracy",
        higher_is_better: bool = True,
    ):
        """Compare a set of runs and return structured results.

        Args:
            run_ids: List of run IDs to compare.
            target_metric: Primary metric for ranking.
            higher_is_better: Whether higher is better.

        Returns:
            ComparisonResult with diffs and importances.
        """
        return self._comparator.compare(run_ids, target_metric, higher_is_better)

    def hyperparameter_importance(
        self,
        run_ids: list[str],
        target_metric: str = "accuracy",
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Get hyperparameter importance scores across runs.

        Args:
            run_ids: Run IDs to analyze.
            target_metric: Target metric.
            top_k: Maximum number of params.

        Returns:
            Sorted (param, score) list.
        """
        return self._comparator.hyperparameter_importance(
            run_ids, target_metric, top_k
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(
        self,
        experiment_id: str,
        target_metric: str = "accuracy",
        include_charts: bool = True,
    ) -> ExperimentReport:
        """Generate a structured experiment report.

        Args:
            experiment_id: Experiment to report on.
            target_metric: Primary metric for analysis.
            include_charts: Whether to include chart data.

        Returns:
            ExperimentReport with metrics, importance, and charts.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment '{experiment_id}' not found")

        run_ids = [r.run_id for r in exp.runs]
        num_runs = len(exp.runs)

        # Determine best run
        best_run_data: dict[str, Any] | None = None
        if exp.runs:
            scored = [
                (r, r.metrics.get(target_metric))
                for r in exp.runs
                if target_metric in r.metrics
            ]
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                best = scored[0][0]
                best_run_data = {
                    "run_id": best.run_id,
                    "run_name": best.run_name,
                    "metrics": best.metrics,
                    "params": best.params,
                    "status": best.status,
                }

        # Metric summary
        metric_summary: dict[str, dict[str, float]] = {}
        for r in exp.runs:
            for k, v in r.metrics.items():
                metric_summary.setdefault(k, {"min": float("inf"), "max": float("-inf"), "mean": 0.0, "count": 0})
                s = metric_summary[k]
                s["min"] = min(s["min"], v)
                s["max"] = max(s["max"], v)
                s["mean"] += v
                s["count"] += 1

        for k in list(metric_summary.keys()):
            if metric_summary[k]["count"] > 0:
                metric_summary[k]["mean"] = round(
                    metric_summary[k]["mean"] / metric_summary[k]["count"], 6
                )
            del metric_summary[k]["count"]

        # Param importance
        param_importance: list[tuple[str, float]] = []
        if run_ids and num_runs >= 2:
            param_importance = self._comparator.hyperparameter_importance(
                run_ids, target_metric
            )

        # Charts
        charts: list[dict[str, Any]] = []
        if include_charts and exp.runs:
            # Learning curves
            for r in exp.runs:
                if r.metric_history:
                    chart = self._visualizer.learning_curve(
                        r.metric_history,
                        title=f"Learning Curve: {r.run_name}",
                    )
                    charts.append(_chart_to_dict(chart))
                    break  # One is enough for the report

            # Metric comparison bars
            if num_runs >= 2:
                common_metrics = set(exp.runs[0].metrics.keys())
                for r in exp.runs[1:]:
                    common_metrics &= set(r.metrics.keys())
                if common_metrics:
                    runs_data = [
                        {
                            "run_name": r.run_name,
                            "run_id": r.run_id,
                            **r.metrics,
                        }
                        for r in exp.runs
                    ]
                    chart = self._visualizer.metric_comparison_bars(
                        runs_data,
                        sorted(common_metrics),
                        title=f"{exp.name}: Metric Comparison",
                    )
                    charts.append(_chart_to_dict(chart))

            # Hyperparameter importance
            if param_importance:
                chart = self._visualizer.hyperparameter_importance_bar(
                    param_importance,
                    title=f"{exp.name}: Hyperparameter Importance",
                )
                charts.append(_chart_to_dict(chart))

        return ExperimentReport(
            experiment_id=exp.experiment_id,
            experiment_name=exp.name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            num_runs=num_runs,
            best_run=best_run_data,
            metric_summary=metric_summary,
            param_importance=param_importance,
            comparison_charts=charts,
            notes=exp.notes,
        )

    def export_report_json(
        self,
        experiment_id: str,
        target_metric: str = "accuracy",
    ) -> dict[str, Any]:
        """Export experiment report as a JSON-serializable dict.

        Args:
            experiment_id: Experiment to export.
            target_metric: Primary metric.

        Returns:
            JSON-serializable dict.
        """
        report = self.generate_report(experiment_id, target_metric)
        return {
            "experiment_id": report.experiment_id,
            "experiment_name": report.experiment_name,
            "generated_at": report.generated_at,
            "num_runs": report.num_runs,
            "best_run": report.best_run,
            "metric_summary": report.metric_summary,
            "param_importance": [
                {"parameter": p, "importance": s}
                for p, s in report.param_importance
            ],
            "charts": report.comparison_charts,
            "notes": report.notes,
        }

    def export_report_markdown(
        self,
        experiment_id: str,
        target_metric: str = "accuracy",
    ) -> str:
        """Export experiment report as Markdown.

        Args:
            experiment_id: Experiment to export.
            target_metric: Primary metric.

        Returns:
            Markdown string.
        """
        report = self.generate_report(experiment_id, target_metric, include_charts=False)
        lines = [
            f"# Experiment Report: {report.experiment_name}",
            "",
            f"- **Experiment ID:** `{report.experiment_id}`",
            f"- **Generated:** {report.generated_at}",
            f"- **Total Runs:** {report.num_runs}",
            "",
        ]

        if report.best_run:
            lines.extend([
                "## Best Run",
                f"- **Name:** {report.best_run['run_name']}",
                f"- **ID:** `{report.best_run['run_id']}`",
                f"- **Status:** {report.best_run['status']}",
                "",
                "### Metrics",
            ])
            for k, v in report.best_run["metrics"].items():
                lines.append(f"- {k}: {v}")
            lines.append("")

        if report.metric_summary:
            lines.append("## Metric Summary")
            lines.append("")
            lines.append("| Metric | Min | Max | Mean |")
            lines.append("|--------|-----|-----|------|")
            for k, s in sorted(report.metric_summary.items()):
                lines.append(f"| {k} | {s['min']} | {s['max']} | {s['mean']} |")
            lines.append("")

        if report.param_importance:
            lines.append("## Hyperparameter Importance")
            lines.append("")
            lines.append("| Parameter | Importance |")
            lines.append("|-----------|------------|")
            for p, s in report.param_importance:
                lines.append(f"| {p} | {s:.4f} |")
            lines.append("")

        if report.notes:
            lines.extend(["## Notes", "", report.notes, ""])

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def dashboard_stats(self) -> dict[str, Any]:
        """Get aggregate statistics for the dashboard summary.

        Returns:
            Dict with total experiments, runs, recent activity, etc.
        """
        total_experiments = len(self._experiments)
        total_runs = sum(len(e.runs) for e in self._experiments.values())

        all_tags: dict[str, int] = {}
        for e in self._experiments.values():
            for t in e.tags:
                all_tags[t] = all_tags.get(t, 0) + 1

        recent = sorted(
            self._experiments.values(),
            key=lambda e: e.updated_at,
            reverse=True,
        )[:5]

        return {
            "total_experiments": total_experiments,
            "total_runs": total_runs,
            "tag_counts": all_tags,
            "recent_experiments": [
                {
                    "id": e.experiment_id,
                    "name": e.name,
                    "runs": len(e.runs),
                    "tags": e.tags,
                    "updated_at": e.updated_at,
                }
                for e in recent
            ],
        }


def _chart_to_dict(chart: ChartData) -> dict[str, Any]:
    """Convert a ChartData to a JSON-compatible dict.

    Args:
        chart: ChartData object.

    Returns:
        Dict representation.
    """
    return {
        "chart_type": chart.chart_type,
        "title": chart.title,
        "x_label": chart.x_label,
        "y_label": chart.y_label,
        "series": chart.series,
        "dimensions": chart.dimensions,
        "annotations": chart.annotations,
        "metadata": chart.metadata,
    }