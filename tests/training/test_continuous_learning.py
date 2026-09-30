"""Comprehensive tests for continuous learning and incremental retraining."""

import math
import random
import unittest

from astroml.training.continuous_learning import (
    ContinuousLearningPipeline,
    ContinuousLearningPipelineConfig,
    ModelVersionManager,
)
from astroml.training.incremental.adaptive_model import (
    AdaptiveModel,
    AdaptiveModelConfig,
    EWCRegularizer,
    ExperienceReplayBuffer,
)
from astroml.training.incremental.online_learner import (
    OnlineLearnerConfig,
    OnlinePassiveAggressiveClassifier,
    OnlinePassiveAggressiveRegressor,
    OnlineSGDClassifier,
    OnlineSGDRegressor,
)
from astroml.training.incremental.stream_trainer import (
    StreamDataIngestor,
    StreamTrainer,
    StreamTrainerConfig,
)


class TestContinuousLearning(unittest.TestCase):
    """Test suite for continuous and incremental learning capabilities."""

    def setUp(self) -> None:
        random.seed(42)

    def test_online_sgd_classifier(self) -> None:
        """Test Online SGD classifier updates and predictions."""
        config = OnlineLearnerConfig(learning_rate=0.05, loss="log_loss", penalty="l2")
        clf = OnlineSGDClassifier(config)

        # Linearly separable 2D data: x1 + x2 > 0 -> 1, else 0
        X = [[1.0, 1.0], [-1.0, -1.0], [2.0, 1.5], [-2.0, -1.5], [1.5, 0.5], [-1.5, -0.5]]
        y = [1, 0, 1, 0, 1, 0]

        for _ in range(50):
            clf.partial_fit(X, y)

        preds = clf.predict([[2.0, 2.0], [-2.0, -2.0]])
        self.assertEqual(preds, [1, 0])
        probas = clf.predict_proba([[2.0, 2.0]])
        self.assertEqual(len(probas[0]), 2)
        self.assertGreater(probas[0][1], probas[0][0])

    def test_online_sgd_regressor(self) -> None:
        """Test Online SGD regressor convergence."""
        config = OnlineLearnerConfig(learning_rate=0.02, loss="squared_error")
        reg = OnlineSGDRegressor(config)

        # Target: y = 2*x1 + 3*x2 + 1
        X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [-1.0, 0.0]]
        y = [3.0, 4.0, 6.0, 8.0, -1.0]

        for _ in range(200):
            reg.partial_fit(X, y)

        pred = reg.predict([[1.0, 1.0]])[0]
        self.assertAlmostEqual(pred, 6.0, delta=0.5)

    def test_passive_aggressive_classifier(self) -> None:
        """Test Passive-Aggressive classifier."""
        config = OnlineLearnerConfig(c_param=1.0)
        pa = OnlinePassiveAggressiveClassifier(config)

        X = [[1.0, 2.0], [-1.0, -2.0], [2.0, 1.0], [-2.0, -1.0]]
        y = [1, 0, 1, 0]

        for _ in range(20):
            pa.partial_fit(X, y)

        preds = pa.predict([[1.5, 1.5], [-1.5, -1.5]])
        self.assertEqual(preds, [1, 0])

    def test_passive_aggressive_regressor(self) -> None:
        """Test Passive-Aggressive regressor."""
        config = OnlineLearnerConfig(c_param=1.0, epsilon=0.01)
        pa_reg = OnlinePassiveAggressiveRegressor(config)

        X = [[1.0], [2.0], [3.0], [4.0]]
        y = [2.0, 4.0, 6.0, 8.0]

        for _ in range(50):
            pa_reg.partial_fit(X, y)

        pred = pa_reg.predict([[5.0]])[0]
        self.assertAlmostEqual(pred, 10.0, delta=1.0)

    def test_experience_replay_buffer(self) -> None:
        """Test reservoir sampling in experience replay buffer."""
        buffer = ExperienceReplayBuffer(max_size=5, random_state=42)
        X_data = [[float(i)] for i in range(20)]
        y_data = [i for i in range(20)]

        buffer.add(X_data, y_data)
        self.assertEqual(buffer.size(), 5)
        self.assertEqual(buffer.total_seen, 20)

        sample_X, sample_y = buffer.sample(3)
        self.assertEqual(len(sample_X), 3)
        self.assertEqual(len(sample_y), 3)

    def test_ewc_regularization(self) -> None:
        """Test Elastic Weight Consolidation anchor computation and penalties."""
        ewc = EWCRegularizer(ewc_lambda=50.0)
        weights = [1.0, -1.0]
        X = [[1.0, 0.5], [0.5, 1.0]]
        y = [1, 0]

        ewc.update_task_anchors(weights, X, y)
        self.assertEqual(ewc.optimal_weights, [1.0, -1.0])
        self.assertEqual(len(ewc.fisher_information), 2)

        # Penalty when weights deviate
        penalty_grad = ewc.penalty_gradient([2.0, -1.0])
        self.assertGreater(penalty_grad[0], 0.0)
        self.assertEqual(penalty_grad[1], 0.0)

    def test_stream_data_ingestor_and_trainer(self) -> None:
        """Test stream batch ingestion and prequential evaluation."""
        model = AdaptiveModel(AdaptiveModelConfig(replay_buffer_size=100))
        trainer = StreamTrainer(model, StreamTrainerConfig(batch_size=10))

        # Generate synthetic stream
        def make_stream():
            for i in range(50):
                x = [random.uniform(-1, 1), random.uniform(-1, 1)]
                y = 1 if (x[0] + x[1] > 0) else 0
                yield x, y

        res = trainer.train_stream(make_stream())
        self.assertEqual(res["total_samples"], 50)
        self.assertIn("accuracy", res["cumulative_metrics"])
        self.assertGreaterEqual(len(res["metrics_history"]), 5)

    def test_model_versioning_and_rollback(self) -> None:
        """Test model version snapshots and rollback mechanism."""
        model = AdaptiveModel()
        manager = ModelVersionManager()

        # Version 1 (Healthy)
        v1 = manager.create_version(model, metrics={"accuracy": 0.95}, samples_trained=100, is_healthy=True)
        self.assertEqual(v1.version, "v1.0.1")

        # Mutate model state
        model.estimator.weights = [999.0, 999.0]
        v2 = manager.create_version(model, metrics={"accuracy": 0.30}, samples_trained=200, is_healthy=False)
        self.assertEqual(v2.version, "v1.0.2")

        # Rollback should restore v1
        restored = manager.rollback(model)
        self.assertEqual(restored.version, "v1.0.1")
        self.assertNotEqual(model.estimator.weights, [999.0, 999.0])

    def test_continuous_learning_pipeline_auto_rollback(self) -> None:
        """Test automated rollback during continuous learning when degradation occurs."""
        config = ContinuousLearningPipelineConfig(
            snapshot_interval_samples=20,
            degradation_threshold=0.3,
            auto_rollback=True,
        )
        pipeline = ContinuousLearningPipeline(config)

        # Stream of good data then corrupt data
        def stream():
            # 60 good samples
            for _ in range(60):
                x1 = random.uniform(0.1, 1.0)
                x2 = random.uniform(0.1, 1.0)
                yield [x1, x2], 1
            # 40 inverted/corrupt samples
            for _ in range(40):
                x1 = random.uniform(0.1, 1.0)
                x2 = random.uniform(0.1, 1.0)
                yield [x1, x2], 0

        res = pipeline.process_stream(stream())
        self.assertGreaterEqual(res["total_samples"], 100)
        self.assertGreaterEqual(res["total_versions"], 2)


if __name__ == "__main__":
    unittest.main()
