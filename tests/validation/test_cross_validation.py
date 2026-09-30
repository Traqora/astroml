"""Comprehensive tests for cross-validation framework with time-series awareness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

from astroml.training.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    evaluate_model_cv,
)
from astroml.validation.cross_validation import (
    CrossValidationReport,
    CrossValidator,
    cross_validate,
)
from astroml.validation.leakage import LeakageError
from astroml.validation.splitters import (
    GroupKFoldSplitter,
    KFoldSplitter,
    PurgedWalkForwardSplitter,
    SlidingWindowSplitter,
    SplitterConfig,
    StratifiedKFoldSplitter,
    TimeSeriesSplitter,
    get_splitter,
)


@pytest.fixture
def synthetic_classification_data():
    """Generate reproducible classification dataset."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 5
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    timestamps = np.arange(1000, 1000 + n_samples)
    groups = np.repeat(np.arange(10), 10)
    return X, y, timestamps, groups


class TestKFoldSplitters:
    """Tests for standard, stratified, and group k-fold splitters."""

    def test_kfold_split_indices(self, synthetic_classification_data):
        X, y, _, _ = synthetic_classification_data
        splitter = KFoldSplitter(n_splits=5, shuffle=False)
        splits = list(splitter.split(X, y))

        assert len(splits) == 5
        assert splitter.get_n_splits(X) == 5

        for train_idx, test_idx in splits:
            assert len(train_idx) == 80
            assert len(test_idx) == 20
            assert len(set(train_idx).intersection(set(test_idx))) == 0
            assert len(set(train_idx).union(set(test_idx))) == 100

    def test_kfold_shuffle_reproducibility(self, synthetic_classification_data):
        X, y, _, _ = synthetic_classification_data
        s1 = KFoldSplitter(n_splits=3, shuffle=True, random_state=42)
        s2 = KFoldSplitter(n_splits=3, shuffle=True, random_state=42)

        splits1 = list(s1.split(X, y))
        splits2 = list(s2.split(X, y))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            assert np.array_equal(tr1, tr2)
            assert np.array_equal(te1, te2)

    def test_stratified_kfold_proportions(self):
        # 80 negative, 20 positive (imbalanced 4:1)
        y = np.array([0] * 80 + [1] * 20)
        X = np.random.randn(100, 2)
        splitter = StratifiedKFoldSplitter(n_splits=5, shuffle=True, random_state=42)

        for train_idx, test_idx in splitter.split(X, y):
            test_y = y[test_idx]
            # Each test fold of 20 samples should have exactly 4 positives
            assert np.sum(test_y == 1) == 4
            assert np.sum(test_y == 0) == 16

    def test_group_kfold_no_overlap(self, synthetic_classification_data):
        X, y, _, groups = synthetic_classification_data
        splitter = GroupKFoldSplitter(n_splits=5)

        for train_idx, test_idx in splitter.split(X, y, groups=groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups.intersection(test_groups)) == 0


class TestTimeSeriesSplitters:
    """Tests for expanding, sliding, and purged walk-forward time-series splitters."""

    def test_time_series_expanding_window(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        splitter = TimeSeriesSplitter(n_splits=4, gap=2)
        splits = list(splitter.split(X, y, timestamps=timestamps))

        assert len(splits) == 4
        for train_idx, test_idx in splits:
            max_train_ts = np.max(timestamps[train_idx])
            min_test_ts = np.min(timestamps[test_idx])
            assert max_train_ts + 2 <= min_test_ts

    def test_sliding_window_splitter(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        splitter = SlidingWindowSplitter(
            n_splits=3,
            window_size=30,
            test_size=15,
            step_size=15,
            gap=1,
        )
        splits = list(splitter.split(X, y, timestamps=timestamps))

        assert len(splits) == 3
        for train_idx, test_idx in splits:
            assert len(train_idx) == 30
            assert len(test_idx) == 15
            assert np.max(timestamps[train_idx]) < np.min(timestamps[test_idx])

    def test_purged_walk_forward_splitter(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        splitter = PurgedWalkForwardSplitter(
            n_splits=4,
            train_ratio=0.4,
            test_ratio=0.15,
            gap_periods=2,
            embargo_periods=3,
        )
        splits = list(splitter.split(X, y, timestamps=timestamps))

        assert len(splits) >= 1
        for train_idx, test_idx in splits:
            assert np.max(timestamps[train_idx]) + 2 <= np.min(timestamps[test_idx])


class TestSplitterFactory:
    """Tests for SplitterConfig and get_splitter factory."""

    def test_factory_creation(self):
        cfg = SplitterConfig(splitter_type="stratified_kfold", n_splits=4, shuffle=True)
        splitter = get_splitter(cfg)
        assert isinstance(splitter, StratifiedKFoldSplitter)
        assert splitter.n_splits == 4

        ts_splitter = get_splitter("time_series", n_splits=3, gap=5)
        assert isinstance(ts_splitter, TimeSeriesSplitter)
        assert ts_splitter.gap == 5

    def test_factory_invalid_name(self):
        with pytest.raises(Exception):
            get_splitter("invalid_splitter_type")


class TestCrossValidationEngine:
    """Tests for cross_validate and CrossValidator."""

    def test_cross_validate_classification(self, synthetic_classification_data):
        X, y, _, _ = synthetic_classification_data
        clf = LogisticRegression(solver="liblinear")

        report = cross_validate(
            estimator=clf,
            X=X,
            y=y,
            cv=3,
            scoring=["accuracy", "f1", "precision", "recall"],
            return_train_score=True,
            return_estimator=True,
        )

        assert isinstance(report, CrossValidationReport)
        assert report.n_splits == 3
        assert len(report.test_scores) == 3
        assert len(report.train_scores) == 3
        assert len(report.fitted_estimators) == 3

        summary = report.metrics_summary
        assert "accuracy" in summary
        assert 0.0 <= summary["accuracy"]["mean"] <= 1.0
        assert summary["accuracy"]["ci_lower"] <= summary["accuracy"]["ci_upper"]

        df = report.to_dataframe()
        assert len(df) == 3
        assert "accuracy" in df.columns

        summary_text = report.summary()
        assert "Cross-Validation Report" in summary_text
        assert "accuracy" in summary_text

    def test_cross_validate_regression(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 3))
        y = X[:, 0] * 2.0 + X[:, 1] * -1.5 + rng.normal(scale=0.1, size=60)

        reg = Ridge()
        report = cross_validate(
            estimator=reg,
            X=X,
            y=y,
            cv=4,
            scoring=["mse", "r2"],
        )

        summary = report.metrics_summary
        assert "mse" in summary
        assert "r2" in summary
        assert summary["mse"]["mean"] >= 0.0

    def test_leakage_detection_raises(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        clf = LogisticRegression(solver="liblinear")

        # Custom leaking splitter where train contains future timestamps
        class LeakingSplitter(KFoldSplitter):
            def split(self, X, y=None, groups=None, timestamps=None):
                yield np.array([80, 81, 82, 83]), np.array([10, 11, 12, 13])

        leaker = LeakingSplitter(n_splits=2)
        with pytest.raises(LeakageError, match="Temporal leakage detected"):
            cross_validate(
                estimator=clf,
                X=X,
                y=y,
                timestamps=timestamps,
                cv=leaker,
                check_leakage=True,
            )

    def test_cross_validator_class(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        validator = CrossValidator(
            splitter="time_series",
            n_splits=3,
            gap=1,
            scoring="accuracy",
        )
        report = validator.evaluate(
            estimator=LogisticRegression(solver="liblinear"),
            X=X,
            y=y,
            timestamps=timestamps,
        )
        assert report.leakage_verified is True
        assert len(report.test_scores) == 3


class TestModelSelectionSearch:
    """Tests for GridSearchCV and RandomizedSearchCV."""

    def test_grid_search_cv(self, synthetic_classification_data):
        X, y, _, _ = synthetic_classification_data
        clf = LogisticRegression(solver="liblinear")
        param_grid = {"C": [0.1, 1.0, 10.0]}

        search = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            refit=True,
        )
        search.fit(X, y)

        assert "C" in search.best_params_
        assert search.best_score_ > 0.0
        assert search.best_estimator_ is not None

        preds = search.predict(X[:5])
        assert len(preds) == 5

    def test_randomized_search_cv(self, synthetic_classification_data):
        X, y, _, _ = synthetic_classification_data
        clf = LogisticRegression(solver="liblinear")
        param_dist = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}

        search = RandomizedSearchCV(
            estimator=clf,
            param_distributions=param_dist,
            n_iter=3,
            cv=3,
            scoring="accuracy",
            random_state=42,
            refit=True,
        )
        search.fit(X, y)

        assert len(search.cv_results_["params"]) == 3
        assert search.best_score_ > 0.0

    def test_evaluate_model_cv_helper(self, synthetic_classification_data):
        X, y, timestamps, _ = synthetic_classification_data
        clf = RandomForestClassifier(n_estimators=10, random_state=42)

        report = evaluate_model_cv(
            estimator=clf,
            X=X,
            y=y,
            splitter_type="purged_walk_forward",
            n_splits=3,
            timestamps=timestamps,
            scoring="accuracy",
        )
        assert isinstance(report, CrossValidationReport)
        assert report.leakage_verified is True
