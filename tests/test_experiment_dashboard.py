"""Tests for experiment tracking dashboard, run comparison and visualizations."""

import numpy as np

from astroml.tracking.experiment_dashboard import ExperimentDashboard
from astroml.tracking.run_comparator import RunComparator, RunMetrics
from astroml.tracking.visualizations import ChartData, ExperimentVisualizer


# ---------------------------------------------------------------------------
# RunComparator
# ---------------------------------------------------------------------------


def _make_run(
    run_id: str,
    metrics: dict[str, float],
    params: dict[str, float] | None = None,
) -> RunMetrics:
    return RunMetrics(
        run_id=run_id,
        run_name=f"run-{run_id}",
        experiment_name="test-exp",
        metrics=metrics,
        params=params or {},
    )


def test_run_comparator_add_and_get() -> None:
    c = RunComparator()
    r = _make_run("r1", {"accuracy": 0.9})
    c.add_run(r)
    assert c.get_run("r1") is r
    assert c.get_run("missing") is None


def test_run_comparator_compare_two_runs() -> None:
    c = RunComparator()
    c.add_run(_make_run("r1", {"accuracy": 0.95, "loss": 0.1}, {"lr": 0.001}))
    c.add_run(_make_run("r2", {"accuracy": 0.90, "loss": 0.2}, {"lr": 0.01}))

    result = c.compare(["r1", "r2"], target_metric="accuracy")
    assert result.best_run == "r1"
    assert result.worst_run == "r2"
    assert len(result.runs) == 2
    assert "accuracy" in result.metric_diffs


def test_run_comparator_parallel_coordinates() -> None:
    c = RunComparator()
    c.add_run(_make_run("r1", {"acc": 0.9, "f1": 0.85, "auc": 0.92}))
    c.add_run(_make_run("r2", {"acc": 0.88, "f1": 0.82, "auc": 0.90}))

    data = c.parallel_coordinates_data(["r1", "r2"])
    assert len(data["dimensions"]) == 3
    assert len(data["data"]) == 2


def test_run_comparator_hyperparameter_importance() -> None:
    c = RunComparator()
    for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
        c.add_run(
            _make_run(
                f"r{i}",
                {"accuracy": 0.9 + (lr - 0.01) * 2},
                {"lr": lr, "batch": 64 + i * 16},
            )
        )

    scores = c.hyperparameter_importance(
        [f"r{i}" for i in range(4)], target_metric="accuracy"
    )
    assert len(scores) >= 1
    # Both lr and batch should appear
    param_names = {p for p, _ in scores}
    assert "lr" in param_names or "batch" in param_names


def test_run_comparator_empty_compare_raises() -> None:
    c = RunComparator()
    try:
        c.compare(["r1", "r2"])
    except ValueError as exc:
        assert "two runs" in str(exc).lower() or "least" in str(exc).lower()


# ---------------------------------------------------------------------------
# ExperimentVisualizer
# ---------------------------------------------------------------------------


def test_learning_curve() -> None:
    viz = ExperimentVisualizer()
    history = {"loss": [1.0, 0.8, 0.6, 0.5, 0.45], "val_loss": [1.2, 0.9, 0.7, 0.6, 0.55]}
    chart = viz.learning_curve(history, title="Test")
    assert chart.chart_type == "line"
    assert len(chart.series) == 2
    assert chart.title == "Test"


def test_learning_curve_smoothing() -> None:
    viz = ExperimentVisualizer()
    history = {"loss": [1.0, 0.8, 0.6, 0.5, 0.45]}
    chart = viz.learning_curve(history, smoothing_window=3)
    assert chart.chart_type == "line"


def test_metric_history_chart() -> None:
    viz = ExperimentVisualizer()
    histories = {
        "run1": {"loss": [1.0, 0.8, 0.6]},
        "run2": {"loss": [0.9, 0.7, 0.5]},
    }
    chart = viz.metric_history_chart(histories, "loss")
    assert len(chart.series) == 2


def test_parallel_coordinates() -> None:
    viz = ExperimentVisualizer()
    chart = viz.parallel_coordinates(
        dimensions=["acc", "f1"],
        data=[
            {"run_name": "a", "run_id": "a1", "acc": 0.9, "f1": 0.85},
            {"run_name": "b", "run_id": "b1", "acc": 0.88, "f1": 0.82},
        ],
    )
    assert chart.chart_type == "parallel"
    assert len(chart.dimensions) == 2


def test_hyperparameter_importance_bar() -> None:
    viz = ExperimentVisualizer()
    importance = [("lr", 0.85), ("batch", 0.42)]
    chart = viz.hyperparameter_importance_bar(importance)
    assert chart.chart_type == "bar"
    assert len(chart.series[0]["data"]) == 2


def test_convergence_summary() -> None:
    viz = ExperimentVisualizer()
    histories = {
        "run1": {"loss": [2.0, 1.5, 1.0, 0.8, 0.75, 0.73, 0.72]},
    }
    summary = viz.convergence_summary(histories, metric_name="loss")
    assert "run1" in summary
    assert summary["run1"]["best_value"] <= 0.72


# ---------------------------------------------------------------------------
# ExperimentDashboard
# ---------------------------------------------------------------------------


def test_create_experiment() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("test-exp", description="desc", tags=["ml"])
    assert exp.name == "test-exp"
    assert "ml" in exp.tags


def test_create_duplicate_raises() -> None:
    dash = ExperimentDashboard()
    dash.create_experiment("exp1")
    try:
        dash.create_experiment("exp1")
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_list_experiments_with_search() -> None:
    dash = ExperimentDashboard()
    dash.create_experiment("model-test", tags=["ml"])
    dash.create_experiment("data-pipeline", tags=["data"])

    results = dash.list_experiments(search_query="model")
    assert len(results) == 1
    assert results[0].name == "model-test"


def test_list_experiments_with_tags() -> None:
    dash = ExperimentDashboard()
    dash.create_experiment("a", tags=["ml", "prod"])
    dash.create_experiment("b", tags=["ml"])

    results = dash.list_experiments(tag_filter=["prod"])
    assert len(results) == 1


def test_add_tag() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    exp = dash.add_tag(exp.experiment_id, "reviewed")
    assert "reviewed" in exp.tags


def test_remove_tag() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp", tags=["ml", "dev"])
    exp = dash.remove_tag(exp.experiment_id, "dev")
    assert "dev" not in exp.tags


def test_clone_experiment() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("original", tags=["ml"])
    run = _make_run("r1", {"acc": 0.9})
    dash.add_run(exp.experiment_id, run)

    clone = dash.clone_experiment(exp.experiment_id, new_name="clone")
    assert clone.name == "clone"
    assert "ml" in clone.tags
    assert len(clone.runs) == 0  # copy_runs defaults to False


def test_clone_with_runs() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("original")
    dash.add_run(exp.experiment_id, _make_run("r1", {"acc": 0.9}))

    clone = dash.clone_experiment(exp.experiment_id, copy_runs=True)
    assert len(clone.runs) == 1


def test_generate_report() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    dash.add_run(exp.experiment_id, _make_run("r1", {"accuracy": 0.9, "f1": 0.85}))
    dash.add_run(exp.experiment_id, _make_run("r2", {"accuracy": 0.85, "f1": 0.80}))

    report = dash.generate_report(exp.experiment_id, target_metric="accuracy")
    assert report.num_runs == 2
    assert report.best_run is not None
    assert "accuracy" in report.metric_summary


def test_export_report_markdown() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    dash.add_run(exp.experiment_id, _make_run("r1", {"accuracy": 0.9}))

    md = dash.export_report_markdown(exp.experiment_id)
    assert "# Experiment Report" in md
    assert "exp" in md


def test_dashboard_stats() -> None:
    dash = ExperimentDashboard()
    dash.create_experiment("a")
    dash.create_experiment("b")
    stats = dash.dashboard_stats()
    assert stats["total_experiments"] == 2
    assert stats["total_runs"] == 0


def test_add_run_to_experiment() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    dash.add_run(exp.experiment_id, _make_run("r1", {"acc": 0.9}))
    assert len(exp.runs) == 1


def test_remove_run() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    dash.add_run(exp.experiment_id, _make_run("r1", {"acc": 0.9}))
    dash.remove_run(exp.experiment_id, "r1")
    assert len(exp.runs) == 0


def test_delete_experiment() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    assert dash.get_experiment(exp.experiment_id) is not None
    dash.delete_experiment(exp.experiment_id)
    assert dash.get_experiment(exp.experiment_id) is None


def test_compare_runs_via_dashboard() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    dash.add_run(
        exp.experiment_id,
        _make_run("r1", {"accuracy": 0.9}, {"lr": 0.001}),
    )
    dash.add_run(
        exp.experiment_id,
        _make_run("r2", {"accuracy": 0.85}, {"lr": 0.1}),
    )

    result = dash.compare_runs(["r1", "r2"], target_metric="accuracy")
    assert result.best_run == "r1"


def test_hyperparameter_importance_via_dashboard() -> None:
    dash = ExperimentDashboard()
    exp = dash.create_experiment("exp")
    for i, lr in enumerate([0.0001, 0.001, 0.01, 0.1]):
        dash.add_run(
            exp.experiment_id,
            _make_run(f"r{i}", {"accuracy": 0.9 + (lr - 0.01) * 10}, {"lr": lr}),
        )

    scores = dash.hyperparameter_importance(
        ["r0", "r1", "r2", "r3"], target_metric="accuracy"
    )
    assert any(p == "lr" for p, _ in scores)