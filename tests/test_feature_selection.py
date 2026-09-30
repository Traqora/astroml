"""Tests for feature selection modules: filter, wrapper, embedded, hybrid."""

import numpy as np
from numpy.typing import NDArray

from astroml.preprocessing.feature_selection.embedded import EmbeddedSelector
from astroml.preprocessing.feature_selection.filter import FilterSelector
from astroml.preprocessing.feature_selection.hybrid import (
    FeatureSelectionPipeline,
    HybridSelector,
    PipelineStep,
)
from astroml.preprocessing.feature_selection.wrapper import WrapperSelector


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _make_data(
    n_samples: int = 100, n_features: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    # y is largely determined by features 0, 1, 2
    y = (X[:, 0] * 3 + X[:, 1] * 2 - X[:, 2] * 1.5 + rng.standard_normal(n_samples) * 0.5).astype(np.float64)
    return X, y


def _make_classification_data(
    n_samples: int = 100, n_features: int = 10
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float64)
    p = 1 / (1 + np.exp(-(X[:, 0] * 2 + X[:, 1] * 1.5)))
    y = (rng.random(n_samples) < p).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# FilterSelector
# ---------------------------------------------------------------------------


def test_filter_correlation() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="correlation", k=5)
    fs.fit(X, y)
    result = fs.get_selection_result()
    assert result.num_features_selected == 5
    assert len(result.selected_indices) == 5


def test_filter_mutual_info() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="mutual_info", k=5)
    fs.fit(X, y)
    result = fs.get_selection_result()
    assert result.num_features_selected == 5


def test_filter_variance() -> None:
    X, _ = _make_data()
    fs = FilterSelector(method="variance", k=3)
    fs.fit(X)
    result = fs.get_selection_result()
    assert result.num_features_selected == 3


def test_filter_anova() -> None:
    X, y = _make_classification_data()
    fs = FilterSelector(method="anova", k=5)
    fs.fit(X, y)
    result = fs.get_selection_result()
    assert result.num_features_selected == 5


def test_filter_chi2() -> None:
    X, y = _make_classification_data()
    fs = FilterSelector(method="chi2", k=5)
    fs.fit(X, y)
    result = fs.get_selection_result()
    assert result.num_features_selected == 5


def test_filter_get_support() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="correlation", k=3)
    fs.fit(X, y)
    mask = fs.get_support()
    assert mask.sum() == 3
    assert mask.shape[0] == X.shape[1]


def test_filter_fit_transform() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="correlation", k=4)
    X_sel = fs.fit_transform(X, y)
    assert X_sel.shape[1] == 4
    assert X_sel.shape[0] == X.shape[0]


def test_filter_transform_without_fit_raises() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="correlation")
    try:
        fs.transform(X)
    except RuntimeError:
        pass
    else:
        assert False, "Should raise RuntimeError"


def test_filter_invalid_method_raises() -> None:
    try:
        FilterSelector(method="invalid")
    except ValueError:
        pass


def test_filter_with_threshold() -> None:
    X, y = _make_data()
    fs = FilterSelector(method="variance", threshold=0.5)
    fs.fit(X)
    result = fs.get_selection_result()
    assert result.num_features_selected <= X.shape[1]


def test_filter_with_feature_names() -> None:
    X, y = _make_data(50, 5)
    names = [f"feat_{i}" for i in range(5)]
    fs = FilterSelector(method="correlation", k=3)
    fs.fit(X, y, feature_names=names)
    result = fs.get_selection_result()
    assert result.feature_names is not None
    assert len(result.feature_names) == 3


# ---------------------------------------------------------------------------
# EmbeddedSelector
# ---------------------------------------------------------------------------


def test_embedded_tree() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data()
    es = EmbeddedSelector(method="tree", k=5)
    es.fit(X, y)
    result = es.get_selection_result()
    assert result.num_features_selected == 5


def test_embedded_lasso() -> None:
    try:
        from sklearn.linear_model import Lasso  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data()
    es = EmbeddedSelector(method="lasso", k=5, alpha=0.1)
    es.fit(X, y)
    result = es.get_selection_result()
    assert result.num_features_selected == 5


def test_embedded_elasticnet() -> None:
    try:
        from sklearn.linear_model import ElasticNet  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data()
    es = EmbeddedSelector(method="elasticnet", k=5, alpha=0.1)
    es.fit(X, y)
    result = es.get_selection_result()
    assert result.num_features_selected == 5


def test_embedded_threshold() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 10)
    es = EmbeddedSelector(method="tree", threshold=0.1)
    es.fit(X, y)
    result = es.get_selection_result()
    assert result.num_features_selected >= 0


def test_embedded_get_support() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data()
    es = EmbeddedSelector(method="tree", k=3)
    es.fit(X, y)
    mask = es.get_support()
    assert mask.sum() == 3


def test_embedded_fit_transform() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data()
    es = EmbeddedSelector(method="tree", k=4)
    X_sel = es.fit_transform(X, y)
    assert X_sel.shape[1] == 4


# ---------------------------------------------------------------------------
# WrapperSelector
# ---------------------------------------------------------------------------


def _get_estimator():
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return None

    return RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1)


def test_wrapper_rfe() -> None:
    model = _get_estimator()
    if model is None:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 8)
    ws = WrapperSelector(estimator=model, method="rfe", n_features_to_select=4, step=2)
    ws.fit(X, y)
    result = ws.get_selection_result()
    assert result.num_features_selected > 0


def test_wrapper_forward() -> None:
    model = _get_estimator()
    if model is None:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 6)
    ws = WrapperSelector(
        estimator=model,
        method="forward",
        n_features_to_select=3,
        cv=2,
    )
    ws.fit(X, y)
    result = ws.get_selection_result()
    assert 1 <= result.num_features_selected <= 3


def test_wrapper_backward() -> None:
    model = _get_estimator()
    if model is None:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 6)
    ws = WrapperSelector(
        estimator=model,
        method="backward",
        n_features_to_select=3,
        cv=2,
    )
    ws.fit(X, y)
    result = ws.get_selection_result()
    assert 1 <= result.num_features_selected < 6


def test_wrapper_get_support() -> None:
    model = _get_estimator()
    if model is None:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 6)
    ws = WrapperSelector(estimator=model, method="rfe", n_features_to_select=3)
    ws.fit(X, y)
    mask = ws.get_support()
    assert mask.sum() > 0


def test_wrapper_fit_transform() -> None:
    model = _get_estimator()
    if model is None:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 6)
    ws = WrapperSelector(estimator=model, method="rfe", n_features_to_select=3)
    X_sel = ws.fit_transform(X, y)
    assert X_sel.shape[1] > 0


# ---------------------------------------------------------------------------
# HybridSelector
# ---------------------------------------------------------------------------


def test_hybrid_vote() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=8)),
        ("mi", FilterSelector(method="mutual_info", k=8)),
        ("var", FilterSelector(method="variance", k=8)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="vote", min_votes=2)
    hs.fit(X, y)
    result = hs.get_selection_result()
    assert result.num_features_selected > 0


def test_hybrid_intersection() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=5)),
        ("var", FilterSelector(method="variance", k=5)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="intersection")
    hs.fit(X, y)
    result = hs.get_selection_result()
    assert result.num_features_selected >= 0


def test_hybrid_union() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=3)),
        ("var", FilterSelector(method="variance", k=3)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="union")
    hs.fit(X, y)
    result = hs.get_selection_result()
    assert result.num_features_selected >= 1


def test_hybrid_rank_aggregation() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=8)),
        ("mi", FilterSelector(method="mutual_info", k=8)),
        ("var", FilterSelector(method="variance", k=8)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="rank_aggregation", k=5)
    hs.fit(X, y)
    result = hs.get_selection_result()
    assert result.num_features_selected == 5


def test_hybrid_with_k_limit() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=8)),
        ("var", FilterSelector(method="variance", k=8)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="vote", min_votes=1, k=3)
    hs.fit(X, y)
    result = hs.get_selection_result()
    assert result.num_features_selected <= 3


def test_hybrid_get_support() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=5)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="vote", min_votes=1)
    hs.fit(X, y)
    mask = hs.get_support()
    assert mask.sum() > 0


def test_hybrid_fit_transform() -> None:
    X, y = _make_data()
    selectors = [
        ("corr", FilterSelector(method="correlation", k=5)),
        ("var", FilterSelector(method="variance", k=5)),
    ]
    hs = HybridSelector(selectors=selectors, strategy="vote", min_votes=1, k=4)
    X_sel = hs.fit_transform(X, y)
    assert X_sel.shape[1] <= 4


# ---------------------------------------------------------------------------
# FeatureSelectionPipeline
# ---------------------------------------------------------------------------


def test_pipeline_filter_only() -> None:
    X, y = _make_data(50, 10)
    pipe = FeatureSelectionPipeline([
        ("filter", FilterSelector(method="variance", k=6)),
    ])
    X_sel = pipe.fit_transform(X, y, [f"f{i}" for i in range(10)])
    assert X_sel.shape[1] == 6
    assert len(pipe.get_results()) == 1


def test_pipeline_filter_embedded() -> None:
    try:
        from sklearn.ensemble import RandomForestRegressor  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("sklearn not installed")

    X, y = _make_data(50, 10)
    pipe = FeatureSelectionPipeline([
        ("filter", FilterSelector(method="correlation", k=6)),
        ("embedded", EmbeddedSelector(method="tree", k=3)),
    ])
    X_sel = pipe.fit_transform(X, y)
    assert X_sel.shape[1] == 3


def test_pipeline_summary() -> None:
    X, y = _make_data(30, 6)
    pipe = FeatureSelectionPipeline([
        ("filter", FilterSelector(method="variance", k=4)),
    ])
    pipe.fit(X, y)
    summary = pipe.summary()
    assert "Pipeline" in summary
    assert "filter" in summary