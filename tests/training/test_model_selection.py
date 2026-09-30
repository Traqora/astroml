"""Tests for the model selection toolkit and its API router."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from astroml.api.routers.model_selection import router as model_selection_router
from astroml.training.model_selection.automl import (
    AutoMLConfig,
    AutoMLPipeline,
    AutoMLResult,
)
from astroml.training.model_selection.benchmark import ModelBenchmark
from astroml.training.model_selection.meta_learning import (
    MetaLearningRecommender,
    TaskDescriptor,
)
from astroml.training.model_selection.nas import (
    ArchitectureSpec,
    NeuralArchitectureSearch,
)

app = FastAPI()
app.include_router(model_selection_router)
client = TestClient(app)


@pytest.fixture
def classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Create a small linearly separable classification dataset."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


class TestModelBenchmark:
    def test_run_and_rank(self, classification_data):
        X, y = classification_data
        models = {
            "lr": LogisticRegression(max_iter=500),
            "dt": DecisionTreeClassifier(random_state=42),
        }
        results = ModelBenchmark().run(models, X, y, cv=3)
        assert len(results) == 2
        ranked = ModelBenchmark().compare(results)
        assert ranked[0].cv_mean >= ranked[1].cv_mean
        best = ModelBenchmark().best(results)
        assert best.model_name in models
        assert best.fit_time >= 0
        assert best.predict_time >= 0
        assert "max_iter" in best.params or "random_state" in best.params

    def test_run_empty_models_raises(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="At least one model"):
            ModelBenchmark().run({}, X, y)

    def test_run_invalid_data_raises(self):
        with pytest.raises(ValueError, match="matching lengths"):
            ModelBenchmark().run({"lr": LogisticRegression()}, np.array([[1.0]]), np.array([0, 1]))

    def test_run_not_enough_samples(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="folds"):
            ModelBenchmark().run({"lr": LogisticRegression()}, X[:2], y[:2], cv=5)

    def test_best_empty_raises(self):
        with pytest.raises(ValueError, match="No benchmark results"):
            ModelBenchmark().best([])


class TestAutoML:
    def test_search_default_models(self, classification_data):
        X, y = classification_data
        result = AutoMLPipeline().search(X, y, AutoMLConfig(cv=3, max_models=3))
        assert isinstance(result, AutoMLResult)
        assert result.best_model_name in {"logistic_regression", "decision_tree", "random_forest"}
        assert result.searched <= 3
        assert result.fitted_model is not None
        assert len(result.leaderboard) >= 1

    def test_search_custom_models(self, classification_data):
        X, y = classification_data
        models = {"lr": LogisticRegression(max_iter=300)}
        result = AutoMLPipeline().search(X, y, AutoMLConfig(cv=2), models=models)
        assert result.best_model_name == "lr"
        assert result.best_score > 0.5

    def test_predict(self, classification_data):
        X, y = classification_data
        pipeline = AutoMLPipeline()
        result = pipeline.search(X, y, AutoMLConfig(cv=2, max_models=1))
        predictions = pipeline.predict(result, X[:5])
        assert predictions.shape == (5,)

    def test_predict_without_fit_raises(self):
        result = AutoMLResult(best_model_name="x", best_score=0.0)
        with pytest.raises(ValueError, match="no fitted model"):
            AutoMLPipeline().predict(result, np.array([[1.0]]))

    def test_predict_proba(self, classification_data):
        X, y = classification_data
        pipeline = AutoMLPipeline()
        result = pipeline.search(
            X, y, AutoMLConfig(cv=2, max_models=1), models={"lr": LogisticRegression(max_iter=300)}
        )
        probs = pipeline.predict_proba(result, X[:5])
        assert probs.shape == (5, 2)

    def test_predict_proba_without_fit_raises(self):
        result = AutoMLResult(best_model_name="x", best_score=0.0)
        with pytest.raises(ValueError, match="no fitted model"):
            AutoMLPipeline().predict_proba(result, np.array([[1.0]]))

    def test_unsupported_task_raises(self, classification_data):
        X, y = classification_data
        with pytest.raises(ValueError, match="Unsupported task"):
            AutoMLPipeline().search(X, y, AutoMLConfig(task="regression"))

    def test_invalid_data_raises(self):
        with pytest.raises(ValueError, match="matching lengths"):
            AutoMLPipeline().search(np.array([[1.0]]), np.array([0, 1]), AutoMLConfig())

    def test_time_budget_stops(self, classification_data):
        X, y = classification_data
        try:
            result = AutoMLPipeline().search(
                X, y, AutoMLConfig(cv=2, max_models=3, time_budget=0.0001)
            )
            assert result.searched <= 3
        except ValueError:
            # Zero candidates fit inside the (near-zero) budget.
            pass


class TestNeuralArchitectureSearch:
    def test_sample_candidates(self):
        specs = NeuralArchitectureSearch().sample_candidates(5, rng=np.random.default_rng(0))
        assert len(specs) == 5
        assert all(spec.n_layers == len(spec.units) for spec in specs)
        assert all(spec.activation in {"relu", "tanh", "logistic"} for spec in specs)

    def test_sample_candidates_unique(self):
        specs = NeuralArchitectureSearch().sample_candidates(20, rng=np.random.default_rng(1))
        keys = {(s.n_layers, tuple(s.units), s.activation, s.learning_rate_init) for s in specs}
        assert len(keys) == 20

    def test_sample_invalid_n(self):
        with pytest.raises(ValueError, match="positive"):
            NeuralArchitectureSearch().sample_candidates(0)

    def test_enumerate_small_space(self):
        specs = NeuralArchitectureSearch().enumerate_small_space()
        # (4 + 16 + 64) unit layouts across 3 layer counts * 3 activations
        assert len(specs) == (4 + 16 + 64) * 3

    def test_evaluate(self, classification_data):
        X, y = classification_data
        spec = ArchitectureSpec(n_layers=1, units=[16], activation="relu")
        score = NeuralArchitectureSearch().evaluate(spec, X, y, cv=2)
        assert 0.0 <= score <= 1.0

    def test_evaluate_invalid_data(self):
        spec = ArchitectureSpec(n_layers=1, units=[16], activation="relu")
        with pytest.raises(ValueError, match="matching lengths"):
            NeuralArchitectureSearch().evaluate(spec, np.array([[1.0]]), np.array([0, 1]))

    def test_search(self, classification_data):
        X, y = classification_data
        result = NeuralArchitectureSearch(random_state=7).search(X, y, n_candidates=3, cv=2)
        assert result.evaluated == 3
        assert result.best_score == max(score for _, score in result.candidates)
        assert result.best_architecture.n_layers >= 1

    def test_search_time_budget(self, classification_data):
        X, y = classification_data
        try:
            result = NeuralArchitectureSearch().search(
                X, y, n_candidates=5, cv=2, time_budget=0.0001
            )
            assert result.evaluated <= 5
        except ValueError:
            # Zero candidates fit inside the (near-zero) budget.
            pass

    def test_default_build(self):
        spec = ArchitectureSpec(n_layers=1, units=[16], activation="relu")
        model = NeuralArchitectureSearch()._default_build(spec)
        assert model.hidden_layer_sizes == (16,)
        assert model.activation == "relu"


class TestMetaLearning:
    def test_describe(self, classification_data):
        X, y = classification_data
        descriptor = TaskDescriptor.from_data(X, y)
        assert descriptor.n_samples == 120
        assert descriptor.n_features == 4
        assert descriptor.n_classes == 2
        assert descriptor.imbalance_ratio >= 1.0

    def test_describe_invalid(self):
        with pytest.raises(ValueError, match="matching lengths"):
            TaskDescriptor.from_data(np.array([[1.0]]), np.array([0, 1]))

    def test_recommend_without_experience(self, classification_data):
        X, y = classification_data
        model, score, confidence = MetaLearningRecommender().recommend(X, y)
        assert model == "logistic_regression"
        assert score > 0
        assert confidence == 0.0

    def test_recommend_heuristic_small(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(100, 3))
        y = (X[:, 0] > 0).astype(int)
        model, _, _ = MetaLearningRecommender().recommend(X, y)
        assert model == "logistic_regression"

    def test_add_and_recommend_experience(self, classification_data):
        X, y = classification_data
        recommender = MetaLearningRecommender()
        recommender.add_experience(X, y, "random_forest", 0.95)
        assert len(recommender.experiences) == 1
        model, score, confidence = recommender.recommend(X, y)
        assert model == "random_forest"
        assert score == 0.95
        assert confidence > 0.5

    def test_similarity(self, classification_data):
        X, y = classification_data
        a = TaskDescriptor.from_data(X, y)
        b = TaskDescriptor.from_data(X, y)
        assert MetaLearningRecommender().similarity(a, b) == pytest.approx(1.0)

    def test_similarity_zero_vector(self):
        a = TaskDescriptor(
            n_samples=0,
            n_features=0,
            n_classes=0,
            imbalance_ratio=0.0,
            feature_mean=0.0,
            feature_std=0.0,
            sparsity=0.0,
        )
        assert MetaLearningRecommender().similarity(a, a) == 0.0

    def test_clear(self, classification_data):
        X, y = classification_data
        recommender = MetaLearningRecommender()
        recommender.add_experience(X, y, "random_forest", 0.9)
        recommender.clear()
        assert recommender.experiences == []

    def test_describe_method(self, classification_data):
        X, y = classification_data
        descriptor = MetaLearningRecommender().describe(X, y)
        assert isinstance(descriptor, TaskDescriptor)


class TestModelSelectionAPI:
    def test_automl_success(self):
        payload = {
            "X": [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0], [6.0, 5.0]],
            "y": [0, 0, 1, 1, 1, 1],
            "cv": 2,
            "max_models": 2,
        }
        response = client.post("/api/v1/model-selection/automl", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert "best_model" in data
        assert "leaderboard" in data
        assert len(data["leaderboard"]) >= 1

    def test_automl_invalid_data(self):
        response = client.post(
            "/api/v1/model-selection/automl",
            json={"X": [[1.0, 2.0]], "y": [0, 1]},
        )
        assert response.status_code == 400

    def test_nas_success(self):
        payload = {
            "X": [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0], [5.0, 6.0], [6.0, 5.0]],
            "y": [0, 0, 1, 1, 1, 1],
            "n_candidates": 2,
            "cv": 2,
        }
        response = client.post("/api/v1/model-selection/nas", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert "best_architecture" in data
        assert data["best_architecture"]["n_layers"] >= 1

    def test_recommend_success(self):
        payload = {
            "X": [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]],
            "y": [0, 0, 1, 1],
        }
        response = client.post("/api/v1/model-selection/recommend", json=payload)
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["recommended_model"] == "logistic_regression"
        assert "task_description" in data

    def test_recommend_invalid(self):
        response = client.post(
            "/api/v1/model-selection/recommend",
            json={"X": [[1.0]], "y": [0, 1]},
        )
        assert response.status_code == 400
