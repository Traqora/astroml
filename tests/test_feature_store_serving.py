"""Tests for feature store online and offline serving, point-in-time correctness, and API."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from astroml.api.app import app
from astroml.features.feature_registry import FeatureDefinition, FeatureRegistryService, FeatureType
from astroml.features.feature_store import FeatureStore
from astroml.features.offline_store import OfflineFeatureStore
from astroml.features.online_store import InMemoryOnlineStore, OnlineFeatureValue


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def feature_store(temp_dir):
    return FeatureStore(temp_dir)


class TestOnlineStore:
    def test_in_memory_online_store_crud(self):
        store = InMemoryOnlineStore(default_ttl_seconds=100)
        # Write via objects
        val = OnlineFeatureValue(entity_id="user_123", feature_name="score", value=42.5)
        count = store.write_online_features([val])
        assert count == 1

        # Read back
        res = store.get_online_features(["user_123"], ["score", "non_existent"])
        assert res["user_123"]["score"] == 42.5
        assert "non_existent" not in res["user_123"]

        # Write via DataFrame
        df = pd.DataFrame(
            {
                "entity_id": ["user_123", "user_456"],
                "tx_count": [10, 20],
                "avg_amount": [150.0, 300.0],
            }
        )
        count_df = store.write_online_features(df)
        assert count_df == 4

        res2 = store.get_online_features(
            ["user_123", "user_456"], ["tx_count", "avg_amount", "score"]
        )
        assert res2["user_123"]["tx_count"] == 10
        assert res2["user_123"]["score"] == 42.5
        assert res2["user_456"]["avg_amount"] == 300.0

        # Delete
        del_count = store.delete_online_features(["user_123"], ["score"])
        assert del_count == 1
        res3 = store.get_online_features(["user_123"], ["score", "tx_count"])
        assert "score" not in res3["user_123"]
        assert res3["user_123"]["tx_count"] == 10


class TestOfflineStore:
    def test_offline_store_and_point_in_time_join(self, temp_dir):
        offline = OfflineFeatureStore(temp_dir / "offline")

        # Create historical feature updates for account A and B over time
        base_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        feature_df = pd.DataFrame(
            {
                "entity_id": ["acc_A", "acc_A", "acc_B", "acc_B"],
                "timestamp": [
                    base_time - timedelta(hours=3),  # 09:00: acc_A tx_rate = 1.0
                    base_time - timedelta(hours=1),  # 11:00: acc_A tx_rate = 5.0
                    base_time - timedelta(hours=4),  # 08:00: acc_B tx_rate = 2.0
                    base_time + timedelta(hours=2),  # 14:00: acc_B tx_rate = 10.0 (future)
                ],
                "tx_rate": [1.0, 5.0, 2.0, 10.0],
            }
        )

        offline.write_offline_features("tx_rate", feature_df)

        # Entity observation table for training at 10:00 and 13:00
        obs_df = pd.DataFrame(
            {
                "entity_id": ["acc_A", "acc_A", "acc_B"],
                "timestamp": [
                    base_time
                    - timedelta(
                        hours=2
                    ),  # 10:00 (acc_A should get 1.0 from 09:00, not 5.0 from 11:00)
                    base_time + timedelta(hours=1),  # 13:00 (acc_A should get 5.0 from 11:00)
                    base_time,  # 12:00 (acc_B should get 2.0 from 08:00, NOT 10.0 from 14:00)
                ],
                "label": [0, 1, 0],
            }
        )

        joined = offline.get_historical_features(obs_df, ["tx_rate"])
        assert len(joined) == 3
        assert "tx_rate" in joined.columns

        # Verify point-in-time correctness without future leakage
        row_0 = joined[
            (joined["entity_id"] == "acc_A")
            & (joined["timestamp"] == base_time - timedelta(hours=2))
        ].iloc[0]
        assert row_0["tx_rate"] == 1.0

        row_1 = joined[
            (joined["entity_id"] == "acc_A")
            & (joined["timestamp"] == base_time + timedelta(hours=1))
        ].iloc[0]
        assert row_1["tx_rate"] == 5.0

        row_2 = joined[(joined["entity_id"] == "acc_B") & (joined["timestamp"] == base_time)].iloc[
            0
        ]
        assert row_2["tx_rate"] == 2.0

        # Stats
        stats = offline.get_feature_statistics("tx_rate")
        assert stats["count"] == 4
        assert stats["min"] == 1.0
        assert stats["max"] == 10.0


class TestFeatureStoreServingIntegration:
    def test_materialize_and_online_serving(self, feature_store):
        # Write batch data to offline
        now = datetime.now(timezone.utc)
        batch_df = pd.DataFrame(
            {
                "entity_id": ["acc_101", "acc_102"],
                "timestamp": [now, now],
                "risk_score": [0.85, 0.12],
            }
        )
        feature_store.write_offline_features("risk_score", batch_df)

        # Materialize to online store
        written = feature_store.materialize_to_online(["risk_score"])
        assert written == 2

        # Online serving lookup
        online_res = feature_store.get_online_features(["acc_101", "acc_102"], ["risk_score"])
        assert online_res["acc_101"]["risk_score"] == 0.85
        assert online_res["acc_102"]["risk_score"] == 0.12


class TestFeaturesRouter:
    @pytest.fixture
    def client(self):
        from astroml.api.routers import features

        app.include_router(features.router)
        return TestClient(app)

    def test_feature_api_endpoints(self, client):
        # Register feature
        reg_resp = client.post(
            "/api/v1/features/register",
            json={
                "name": "api_test_feature",
                "description": "Feature created via API",
                "feature_type": "numeric",
                "tags": ["test", "api"],
                "owner": "ml_team",
            },
        )
        assert reg_resp.status_code == 200
        assert reg_resp.json()["name"] == "api_test_feature"

        # List features
        list_resp = client.get("/api/v1/features")
        assert list_resp.status_code == 200

        # Online feature endpoint
        online_resp = client.post(
            "/api/v1/features/online",
            json={
                "entity_ids": ["acc_x"],
                "feature_names": ["api_test_feature"],
            },
        )
        assert online_resp.status_code == 200
        assert "features" in online_resp.json()
