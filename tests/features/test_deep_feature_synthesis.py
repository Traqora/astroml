"""Unit tests for automated feature engineering with Featuretools and Deep Feature Synthesis."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from astroml.features.deep_feature_synthesis import (
    DeepFeatureSynthesizer,
    DFSPipeline,
    prune_features,
    rank_feature_importance,
)
from astroml.preprocessing.auto_feature_engineering import (
    EntitySetBuilder,
    build_transaction_entityset,
)


@pytest.fixture
def sample_relational_data():
    """Create sample transaction and account DataFrames."""
    accounts_df = pd.DataFrame(
        {
            "account_id": ["acc_1", "acc_2", "acc_3", "acc_4"],
            "account_type": ["retail", "merchant", "retail", "institution"],
        }
    )

    transactions_df = pd.DataFrame(
        {
            "transaction_id": [f"tx_{i}" for i in range(12)],
            "source_account": [
                "acc_1",
                "acc_1",
                "acc_2",
                "acc_3",
                "acc_2",
                "acc_1",
                "acc_3",
                "acc_4",
                "acc_4",
                "acc_2",
                "acc_1",
                "acc_3",
            ],
            "destination_account": [
                "acc_2",
                "acc_3",
                "acc_1",
                "acc_4",
                "acc_3",
                "acc_4",
                "acc_2",
                "acc_1",
                "acc_2",
                "acc_4",
                "acc_3",
                "acc_1",
            ],
            "amount": [10.0, 50.0, 100.0, 20.0, 35.0, 400.0, 5.0, 120.0, 80.0, 15.0, 60.0, 25.0],
            "fee": [0.01, 0.05, 0.1, 0.02, 0.03, 0.4, 0.01, 0.12, 0.08, 0.01, 0.06, 0.02],
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="h"),
        }
    )

    return accounts_df, transactions_df


class TestEntitySetBuilder:
    """Tests for EntitySetBuilder and build_transaction_entityset."""

    def test_entityset_builder_fluent(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        builder = EntitySetBuilder(id="test_es")
        builder.add_dataframe("accounts", acc_df, index="account_id")
        builder.add_dataframe("transactions", tx_df, index="transaction_id", time_index="timestamp")
        builder.add_relationship("accounts", "account_id", "transactions", "source_account")

        es = builder.build()
        assert es.id == "test_es"
        assert len(es.dataframes) == 2
        assert len(es.relationships) == 1

    def test_build_transaction_entityset(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)

        df_names = [df.ww.name for df in es.dataframes]
        assert "accounts" in df_names
        assert "transactions" in df_names
        assert len(es["accounts"]) == 4
        assert len(es["transactions"]) == 12


class TestDeepFeatureSynthesis:
    """Tests for DeepFeatureSynthesizer."""

    def test_fit_transform_and_feature_count(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)

        synthesizer = DeepFeatureSynthesizer(
            target_dataframe_name="accounts",
            agg_primitives=["mean", "sum", "count", "max", "min"],
            trans_primitives=[],
            max_depth=2,
        )
        fm, defs = synthesizer.fit_transform(es)

        assert len(fm) == 4  # 4 accounts
        assert len(fm.columns) > 1
        assert len(defs) == len(fm.columns)
        assert not fm.isnull().any().any()

    def test_transform_new_data(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        es1 = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)

        synthesizer = DeepFeatureSynthesizer(target_dataframe_name="accounts", max_depth=1)
        fm1, defs = synthesizer.fit_transform(es1)

        # Transform on es1 with fitted defs
        fm2 = synthesizer.transform(es1, defs)
        assert fm2.shape == fm1.shape

    def test_save_and_load_feature_definitions(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)

        synthesizer = DeepFeatureSynthesizer(target_dataframe_name="accounts", max_depth=1)
        _, defs = synthesizer.fit_transform(es)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            synthesizer.save_feature_definitions(temp_path)
            assert os.path.exists(temp_path)

            loaded_defs = DeepFeatureSynthesizer.load_feature_definitions(temp_path)
            assert len(loaded_defs) == len(defs)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestFeaturePruningAndSelection:
    """Tests for prune_features."""

    def test_variance_and_correlation_pruning(self):
        df = pd.DataFrame(
            {
                "constant_feat": [1.0, 1.0, 1.0, 1.0, 1.0],
                "informative_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "informative_collinear": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ],  # Correlated with informative_1
                "informative_2": [10.0, 12.0, 8.0, 15.0, 2.0],
                "mostly_missing": [np.nan, np.nan, np.nan, np.nan, 5.0],
            }
        )

        pruned_df, retained_cols = prune_features(
            feature_matrix=df,
            variance_threshold=0.01,
            correlation_threshold=0.99,
            max_missing_rate=0.4,
        )

        assert "constant_feat" not in retained_cols
        assert "mostly_missing" not in retained_cols
        assert "informative_1" in retained_cols
        assert "informative_2" in retained_cols
        assert len(retained_cols) < len(df.columns)


class TestFeatureImportanceRanking:
    """Tests for rank_feature_importance."""

    def test_ranking_methods(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            {
                "f1": rng.normal(size=50),
                "f2": rng.normal(size=50),
                "f3": rng.normal(size=50),
            }
        )
        y = (X["f1"] * 2.0 + rng.normal(scale=0.1, size=50) > 0).astype(int)

        for method in ["random_forest", "gradient_boosting", "mutual_info", "correlation"]:
            ranking = rank_feature_importance(feature_matrix=X, target=y, method=method, top_k=3)
            assert isinstance(ranking, pd.DataFrame)
            assert len(ranking) == 3
            assert "feature" in ranking.columns
            assert "importance" in ranking.columns
            assert "rank" in ranking.columns
            assert ranking["rank"].iloc[0] == 1


class TestDFSPipeline:
    """Tests for unified DFSPipeline."""

    def test_full_pipeline(self, sample_relational_data):
        acc_df, tx_df = sample_relational_data
        es = build_transaction_entityset(transactions_df=tx_df, accounts_df=acc_df)
        labels = np.array([0, 1, 0, 1])

        pipeline = DFSPipeline(
            target_dataframe_name="accounts",
            max_depth=2,
            variance_threshold=0.01,
            top_k_features=5,
        )
        final_df = pipeline.fit_transform(es, target=labels)

        assert len(final_df) == 4
        assert len(final_df.columns) <= 5
        assert pipeline.importance_ranking_ is not None
