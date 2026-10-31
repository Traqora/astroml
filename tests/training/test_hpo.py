"""Tests for hyperparameter optimization and search space validation."""

import pytest

from astroml.training.hyperparameter_optimization import HPOptimizer, SearchSpace
from astroml.tracking.experiment_tracker import ExperimentTracker


def test_hpo() -> None:
    HPOptimizer().optimize()
    ExperimentTracker().track()


def test_search_space_valid_categorical() -> None:
    """Valid categorical search space."""
    space = SearchSpace.from_dict({
        "optimizer": ["adam", "sgd", "adamw"],
        "model_type": ["gcn", "sage", "gat"],
    })
    assert space.params["optimizer"] == ["adam", "sgd", "adamw"]
    assert space.params["model_type"] == ["gcn", "sage", "gat"]


def test_search_space_valid_range() -> None:
    """Valid range search space."""
    space = SearchSpace.from_dict({
        "learning_rate": (0.001, 0.1),
        "dropout": (0.1, 0.5),
        "batch_size": (32, 256),
    })
    assert space.params["learning_rate"] == (0.001, 0.1)
    assert space.params["dropout"] == (0.1, 0.5)
    assert space.params["batch_size"] == (32, 256)


def test_search_space_valid_mixed() -> None:
    """Valid mixed categorical and range search space."""
    space = SearchSpace.from_dict({
        "learning_rate": (0.001, 0.1),
        "optimizer": ["adam", "sgd"],
        "hidden_dims": [[64, 32], [128, 64]],
    })
    assert space.params["learning_rate"] == (0.001, 0.1)
    assert space.params["optimizer"] == ["adam", "sgd"]
    assert space.params["hidden_dims"] == [[64, 32], [128, 64]]


def test_search_space_empty() -> None:
    """Empty search space should raise error."""
    with pytest.raises(ValueError, match="search space must not be empty"):
        SearchSpace.from_dict({})


def test_search_space_empty_key() -> None:
    """Empty parameter name should raise error."""
    with pytest.raises(ValueError, match="parameter name must be a non-empty string"):
        SearchSpace.from_dict({"": ["adam", "sgd"]})


def test_search_space_whitespace_key() -> None:
    """Whitespace-only parameter name should raise error."""
    with pytest.raises(ValueError, match="parameter name must be a non-empty string"):
        SearchSpace.from_dict({"   ": ["adam", "sgd"]})


def test_search_space_empty_list() -> None:
    """Empty list for categorical should raise error."""
    with pytest.raises(ValueError, match="parameter 'optimizer' list must not be empty"):
        SearchSpace.from_dict({"optimizer": []})


def test_search_space_invalid_tuple_length() -> None:
    """Tuple with wrong length should raise error."""
    with pytest.raises(ValueError, match="tuple must have exactly 2 elements"):
        SearchSpace.from_dict({"learning_rate": (0.001, 0.1, 0.5)})

    with pytest.raises(ValueError, match="tuple must have exactly 2 elements"):
        SearchSpace.from_dict({"learning_rate": (0.001,)})


def test_search_space_non_numeric_range() -> None:
    """Non-numeric range values should raise error."""
    with pytest.raises(ValueError, match="range values must be numeric"):
        SearchSpace.from_dict({"learning_rate": ("0.001", "0.1")})

    with pytest.raises(ValueError, match="range values must be numeric"):
        SearchSpace.from_dict({"learning_rate": (0.001, "0.1")})


def test_search_space_min_ge_max() -> None:
    """Range with min >= max should raise error."""
    with pytest.raises(ValueError, match="min must be less than max"):
        SearchSpace.from_dict({"learning_rate": (0.1, 0.001)})

    with pytest.raises(ValueError, match="min must be less than max"):
        SearchSpace.from_dict({"learning_rate": (0.1, 0.1)})


def test_search_space_invalid_type() -> None:
    """Invalid value type should raise error."""
    with pytest.raises(ValueError, match="must be a list \\(categorical\\) or tuple \\(range\\)"):
        SearchSpace.from_dict({"learning_rate": 0.001})

    with pytest.raises(ValueError, match="must be a list \\(categorical\\) or tuple \\(range\\)"):
        SearchSpace.from_dict({"learning_rate": "0.001"})


def test_hp_optimizer_with_dict() -> None:
    """HPOptimizer should accept raw dict and validate it."""
    optimizer = HPOptimizer({"learning_rate": (0.001, 0.1)})
    assert optimizer.search_space is not None
    assert optimizer.search_space.params["learning_rate"] == (0.001, 0.1)


def test_hp_optimizer_with_search_space() -> None:
    """HPOptimizer should accept validated SearchSpace."""
    space = SearchSpace.from_dict({"learning_rate": (0.001, 0.1)})
    optimizer = HPOptimizer(space)
    assert optimizer.search_space is space


def test_hp_optimizer_without_search_space() -> None:
    """HPOptimizer should work without search space."""
    optimizer = HPOptimizer()
    assert optimizer.search_space is None
