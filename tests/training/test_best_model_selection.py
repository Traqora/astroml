"""Regression tests for best-model selection — issue #897.

The AutoML pipeline must (a) rank candidates by mean CV score, (b) choose the
winner deterministically when scores tie, (c) fit the winner on the full
dataset, and (d) skip — but log — a candidate whose evaluation fails rather
than silently dropping it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from astroml.training.model_selection.automl import AutoMLConfig, AutoMLPipeline
from astroml.training.model_selection.benchmark import BenchmarkResult, ModelBenchmark
from astroml.utils.exceptions import AstroMLError


def _result(name: str, score: float) -> BenchmarkResult:
    return BenchmarkResult(
        model_name=name,
        cv_mean=score,
        cv_std=0.01,
        fit_time=0.0,
        predict_time=0.0,
    )


class _StubModel:
    """Minimal estimator recording whether the pipeline fitted it."""

    def __init__(self) -> None:
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StubModel":
        self.fitted = True
        return self


def test_compare_ranks_by_descending_cv_mean_without_mutating_input() -> None:
    results = [_result("low", 0.6), _result("high", 0.9), _result("mid", 0.7)]

    ranked = ModelBenchmark().compare(results)

    assert [r.model_name for r in ranked] == ["high", "mid", "low"]
    assert [r.model_name for r in results] == ["low", "high", "mid"]


def test_best_returns_the_highest_scoring_candidate() -> None:
    results = [_result("low", 0.6), _result("high", 0.9), _result("mid", 0.7)]

    assert ModelBenchmark().best(results).model_name == "high"


def test_ties_are_broken_deterministically_by_candidate_order() -> None:
    """Stable ranking keeps the first declared candidate on an exact tie."""
    results = [_result("first", 0.8), _result("second", 0.8)]

    assert ModelBenchmark().compare(results)[0].model_name == "first"
    assert ModelBenchmark().best(results).model_name == "first"


def test_search_fits_the_best_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {"weak": _StubModel(), "strong": _StubModel()}
    scores = {"weak": 0.55, "strong": 0.91}

    def fake_run(self, candidates, X, y, cv=5, scoring="accuracy", random_state=42):
        name = next(iter(candidates))
        return [_result(name, scores[name])]

    monkeypatch.setattr(ModelBenchmark, "run", fake_run)

    result = AutoMLPipeline().search(
        np.zeros((10, 2)),
        np.array([0, 1] * 5),
        AutoMLConfig(cv=2),
        models=models,
    )

    assert result.best_model_name == "strong"
    assert result.best_score == pytest.approx(0.91)
    assert result.searched == 2
    assert result.fitted_model is models["strong"]
    assert models["strong"].fitted is True
    assert models["weak"].fitted is False


def test_search_uses_candidate_order_to_break_ties(monkeypatch: pytest.MonkeyPatch) -> None:
    models = {"first": _StubModel(), "second": _StubModel()}

    def fake_run(self, candidates, X, y, cv=5, scoring="accuracy", random_state=42):
        name = next(iter(candidates))
        return [_result(name, 0.8)]

    monkeypatch.setattr(ModelBenchmark, "run", fake_run)

    result = AutoMLPipeline().search(
        np.zeros((10, 2)),
        np.array([0, 1] * 5),
        AutoMLConfig(cv=2),
        models=models,
    )

    assert result.best_model_name == "first"
    assert models["first"].fitted is True
    assert models["second"].fitted is False


def test_failed_candidate_is_skipped_and_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    models = {"broken": _StubModel(), "working": _StubModel()}

    def fake_run(self, candidates, X, y, cv=5, scoring="accuracy", random_state=42):
        name = next(iter(candidates))
        if name == "broken":
            raise AstroMLError("candidate exploded")
        return [_result(name, 0.75)]

    monkeypatch.setattr(ModelBenchmark, "run", fake_run)

    with caplog.at_level(
        logging.WARNING, logger="astroml.training.model_selection.automl"
    ):
        result = AutoMLPipeline().search(
            np.zeros((10, 2)),
            np.array([0, 1] * 5),
            AutoMLConfig(cv=2),
            models=models,
        )

    assert result.best_model_name == "working"
    assert result.searched == 1
    assert models["broken"].fitted is False
    assert any(
        "Skipping candidate model 'broken'" in record.getMessage()
        for record in caplog.records
    )


def test_search_raises_when_every_candidate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(self, candidates, X, y, cv=5, scoring="accuracy", random_state=42):
        raise AstroMLError("always broken")

    monkeypatch.setattr(ModelBenchmark, "run", fake_run)

    with pytest.raises(ValueError, match="All candidate models failed"):
        AutoMLPipeline().search(
            np.zeros((10, 2)),
            np.array([0, 1] * 5),
            AutoMLConfig(cv=2),
            models={"broken": _StubModel()},
        )
