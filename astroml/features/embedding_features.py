"""Embedding feature computers for the Feature Store.

Computes embedding-based features from LLM providers including
transaction descriptions, account behavior, and alert descriptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from astroml.features.feature_engine import BaseFeatureComputer, FeatureDependencyType
from astroml.features.feature_store import FeatureType

logger = logging.getLogger(__name__)


class TransactionEmbeddingComputer(BaseFeatureComputer):
    """Computer for transaction description embeddings."""

    def __init__(self, provider: str = "openai", model: str = "text-embedding-ada-002"):
        super().__init__("transaction_embeddings")
        self.provider = provider
        self.model = model
        self.add_dependency(
            "transaction_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "description", "timestamp"]},
        )

    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.validate_input(data, entity_col, timestamp_col)
        try:
            from astroml.llm.features.compute import compute_embeddings
            texts = data.get("description", data.get("memo", data.get("text", "")))
            embeddings = compute_embeddings(
                texts=texts.tolist() if isinstance(texts, pd.Series) else texts,
                provider=self.provider,
                model=self.model,
            )
            embedding_dim = len(embeddings[0]) if embeddings else 0
            result = pd.DataFrame(
                embeddings,
                index=data[entity_col],
                columns=[f"embedding_{i}" for i in range(embedding_dim)],
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise


class AccountBehaviorEmbeddingComputer(BaseFeatureComputer):
    """Computer for account behavior embeddings."""

    def __init__(self, provider: str = "openai", model: str = "text-embedding-ada-002"):
        super().__init__("account_behavior_embeddings")
        self.provider = provider
        self.model = model
        self.add_dependency(
            "account_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "behavior_summary", "timestamp"]},
        )

    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.validate_input(data, entity_col, timestamp_col)
        try:
            from astroml.llm.features.compute import compute_embeddings
            summaries = data.get("behavior_summary", data.get("profile", ""))
            embeddings = compute_embeddings(
                texts=summaries.tolist() if isinstance(summaries, pd.Series) else summaries,
                provider=self.provider,
                model=self.model,
            )
            embedding_dim = len(embeddings[0]) if embeddings else 0
            result = pd.DataFrame(
                embeddings,
                index=data[entity_col],
                columns=[f"behavior_embedding_{i}" for i in range(embedding_dim)],
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise


class AlertEmbeddingComputer(BaseFeatureComputer):
    """Computer for alert description embeddings."""

    def __init__(self, provider: str = "openai", model: str = "text-embedding-ada-002"):
        super().__init__("alert_embeddings")
        self.provider = provider
        self.model = model
        self.add_dependency(
            "alert_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "alert_description", "timestamp"]},
        )

    def compute(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        self.validate_input(data, entity_col, timestamp_col)
        try:
            from astroml.llm.features.compute import compute_embeddings
            alerts = data.get("alert_description", data.get("description", ""))
            embeddings = compute_embeddings(
                texts=alerts.tolist() if isinstance(alerts, pd.Series) else alerts,
                provider=self.provider,
                model=self.model,
            )
            embedding_dim = len(embeddings[0]) if embeddings else 0
            result = pd.DataFrame(
                embeddings,
                index=data[entity_col],
                columns=[f"alert_embedding_{i}" for i in range(embedding_dim)],
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise
