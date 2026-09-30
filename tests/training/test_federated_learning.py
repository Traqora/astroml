"""Comprehensive tests for Federated Learning client, server, aggregators, and SecAgg."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from astroml.api.routers.federated import router as federated_router
from astroml.training.federated.aggregator import (
    AggregatorFactory,
    ClientUpdate,
    FedAvgAggregator,
    FedProxAggregator,
    KrumAggregator,
    MedianAggregator,
    TrimmedMeanAggregator,
)
from astroml.training.federated.client import DPConfig, FederatedClient
from astroml.training.federated.secure_aggregation import (
    MaskedUpdate,
    SecureAggregator,
)
from astroml.training.federated.server import (
    DataVolumeWeightedSelector,
    FederatedServer,
    RandomClientSelector,
    RoundRobinSelector,
)


@pytest.fixture
def sample_linear_data():
    np.random.seed(42)
    # Generate 100 samples with 4 features
    X = np.random.randn(100, 4).astype(np.float32)
    true_w = np.array([[2.0], [-1.0], [0.5], [1.5]], dtype=np.float32)
    logits = np.dot(X, true_w) + 0.1
    y = (logits > 0).astype(np.float32)
    return X, y


class TestFederatedClient:
    def test_client_local_training(self, sample_linear_data):
        X, y = sample_linear_data
        init_w = {
            "weight": np.zeros((4, 1), dtype=np.float32),
            "bias": np.zeros((1,), dtype=np.float32),
        }
        client = FederatedClient(
            client_id="client_1",
            initial_weights=init_w,
            local_data=(X, y),
        )

        assert client.sample_count == 100
        initial_eval = client.evaluate()

        update = client.train_epoch(learning_rate=0.1, epochs=5, batch_size=20)
        assert update.client_id == "client_1"
        assert update.sample_count == 100
        assert not np.array_equal(update.weights["weight"], init_w["weight"])

        post_eval = client.evaluate()
        assert post_eval["accuracy"] >= initial_eval["accuracy"]

    def test_differential_privacy_clipping_and_noise(self, sample_linear_data):
        X, y = sample_linear_data
        init_w = {
            "weight": np.zeros((4, 1), dtype=np.float32),
            "bias": np.zeros((1,), dtype=np.float32),
        }
        dp_cfg = DPConfig(
            enabled=True,
            clip_norm=0.5,
            noise_scale=0.05,
            target_epsilon=2.0,
        )
        client = FederatedClient(
            client_id="client_dp",
            initial_weights=init_w,
            local_data=(X, y),
            dp_config=dp_cfg,
        )

        update = client.train_epoch(learning_rate=0.05, epochs=2)
        eps_spent, delta_spent = client.get_privacy_spent()
        assert eps_spent > 0.0
        assert delta_spent > 0.0


class TestAggregators:
    def test_fedavg(self):
        w1 = {"w": np.array([1.0, 2.0])}
        w2 = {"w": np.array([3.0, 4.0])}
        updates = [
            ClientUpdate("c1", w1, sample_count=100),
            ClientUpdate("c2", w2, sample_count=100),
        ]
        agg = FedAvgAggregator()
        res = agg.aggregate(updates)
        np.testing.assert_allclose(res["w"], np.array([2.0, 3.0]))

    def test_fedprox(self):
        w1 = {"w": np.array([1.0, 2.0])}
        w2 = {"w": np.array([3.0, 4.0])}
        global_w = {"w": np.array([0.0, 0.0])}
        updates = [
            ClientUpdate("c1", w1, sample_count=100),
            ClientUpdate("c2", w2, sample_count=100),
        ]
        agg = FedProxAggregator(mu=0.1, learning_rate=1.0)
        res = agg.aggregate(updates, global_weights=global_w)
        assert res["w"] is not None

    def test_trimmed_mean(self):
        # 5 updates with an extreme outlier
        updates = [
            ClientUpdate("c1", {"w": np.array([10.0])}, sample_count=10),
            ClientUpdate("c2", {"w": np.array([10.2])}, sample_count=10),
            ClientUpdate("c3", {"w": np.array([9.8])}, sample_count=10),
            ClientUpdate("c4", {"w": np.array([10.1])}, sample_count=10),
            ClientUpdate("c_bad", {"w": np.array([1000.0])}, sample_count=10),
        ]
        agg = TrimmedMeanAggregator(beta=0.2)
        res = agg.aggregate(updates)
        assert res["w"][0] < 15.0  # Outlier 1000 is trimmed

    def test_median(self):
        updates = [
            ClientUpdate("c1", {"w": np.array([1.0, 100.0])}, sample_count=10),
            ClientUpdate("c2", {"w": np.array([2.0, 2.0])}, sample_count=10),
            ClientUpdate("c3", {"w": np.array([3.0, 3.0])}, sample_count=10),
        ]
        agg = MedianAggregator()
        res = agg.aggregate(updates)
        np.testing.assert_allclose(res["w"], np.array([2.0, 3.0]))

    def test_krum(self):
        updates = [
            ClientUpdate("c1", {"w": np.array([1.0, 1.0])}, sample_count=10),
            ClientUpdate("c2", {"w": np.array([1.1, 0.9])}, sample_count=10),
            ClientUpdate("c3", {"w": np.array([0.9, 1.1])}, sample_count=10),
            ClientUpdate("c_bad", {"w": np.array([100.0, 100.0])}, sample_count=10),
        ]
        agg = KrumAggregator(num_byzantine=1)
        res = agg.aggregate(updates)
        assert res["w"][0] < 5.0

    def test_factory(self):
        for algo in ["fedavg", "fedprox", "trimmed_mean", "median", "krum"]:
            agg = AggregatorFactory.create(algo)
            assert agg is not None


class TestSecureAggregation:
    def test_mask_cancellation(self):
        secagg = SecureAggregator()
        clients = ["client_a", "client_b", "client_c", "client_d"]
        raw_weights = [
            {"w": np.array([1.0, 2.0, 3.0], dtype=np.float64)},
            {"w": np.array([4.0, 5.0, 6.0], dtype=np.float64)},
            {"w": np.array([7.0, 8.0, 9.0], dtype=np.float64)},
            {"w": np.array([10.0, 11.0, 12.0], dtype=np.float64)},
        ]
        diff = secagg.verify_mask_cancellation(raw_weights, clients, round_id=1)
        assert diff < 1e-6


class TestFederatedServer:
    def test_server_round_orchestration(self, sample_linear_data):
        X, y = sample_linear_data
        init_w = {
            "weight": np.zeros((4, 1), dtype=np.float32),
            "bias": np.zeros((1,), dtype=np.float32),
        }
        server = FederatedServer(
            initial_weights=init_w,
            aggregator=FedAvgAggregator(),
            selection_strategy=RoundRobinSelector(),
        )

        # Register 3 clients with data splits
        client_pool = {}
        for i in range(3):
            cid = f"node_{i}"
            server.register_client(cid, sample_count=30)
            c_data = (X[i * 30 : (i + 1) * 30], y[i * 30 : (i + 1) * 30])
            client_pool[cid] = FederatedClient(cid, initial_weights=init_w, local_data=c_data)

        # Run 3 FL rounds
        for r in range(3):
            round_res = server.run_round(client_pool=client_pool, local_epochs=2)
            assert round_res.round_id == r + 1
            assert round_res.client_count == 3

        eval_res = server.evaluate_global_model((X, y))
        assert "accuracy" in eval_res
        assert len(server.get_training_history()) == 3


class TestFederatedAPI:
    @pytest.fixture
    def client(self):
        app = FastAPI()
        app.include_router(federated_router)
        return TestClient(app)

    def test_full_api_workflow(self, client):
        # 1. Create Session
        res_create = client.post(
            "/api/v1/federated/sessions",
            json={
                "session_id": "test_fl_session",
                "input_dim": 4,
                "algorithm": "fedavg",
            },
        )
        assert res_create.status_code == 200
        assert res_create.json()["session_id"] == "test_fl_session"

        # 2. Register Client
        res_reg = client.post(
            "/api/v1/federated/clients/register",
            json={
                "session_id": "test_fl_session",
                "client_id": "edge_client_1",
                "sample_count": 50,
            },
        )
        assert res_reg.status_code == 200

        # 3. Get Global Model
        res_model = client.get("/api/v1/federated/sessions/test_fl_session/global-model")
        assert res_model.status_code == 200
        assert "weight" in res_model.json()["weights"]

        # 4. Submit Update
        res_up = client.post(
            "/api/v1/federated/sessions/test_fl_session/updates",
            json={
                "client_id": "edge_client_1",
                "round_id": 1,
                "weights": {
                    "weight": [[0.1], [0.2], [0.3], [0.4]],
                    "bias": [0.05],
                },
                "sample_count": 50,
                "loss": 0.25,
            },
        )
        assert res_up.status_code == 200

        # 5. Get Session Info
        res_sess = client.get("/api/v1/federated/sessions/test_fl_session")
        assert res_sess.status_code == 200
        assert res_sess.json()["client_count"] >= 1

        # 6. Evaluate
        res_eval = client.post(
            "/api/v1/federated/sessions/test_fl_session/evaluate",
            json={
                "X": [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.0, 0.0]],
                "y": [1.0, 0.0],
            },
        )
        assert res_eval.status_code == 200
        assert "accuracy" in res_eval.json()["metrics"]
