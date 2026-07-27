"""Integration layer connecting LLM outputs to the Feature Store.

Provides high-level API for registering, computing, and managing
LLM-generated features within the feature store lifecycle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from astroml.features.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureType,
    FeatureRegistry,
    FeatureStatus,
)
from astroml.features.feature_versioning import (
    FeatureVersionManager,
    VersionStatus,
    create_version_manager,
)
from astroml.features.llm_features import (
    LLMFeatureCategory,
    LLMFeatureMeta,
    EmbeddingType,
    ScoreType,
)
from astroml.features.llm_generators import LLMFeatureGenerator, GeneratedLLMFeature
from astroml.cache import cached_feature

logger = logging.getLogger(__name__)


@dataclass
class LLMFeatureConfig:
    """Configuration for LLM feature integration."""
    embedding_provider: str = "openai"
    score_model: str = "gpt-4"
    prompt_version: str = "v1"
    ttl_hours: int = 24
    refresh_interval_minutes: int = 60
    enable_materialized_views: bool = True
    backfill_batch_size: int = 1000
    max_retries: int = 3


class LLMFeatureIntegration:
    """Integrates LLM-generated features with the Feature Store.

    Handles registration, computation, materialized views,
    TTL policies, versioning, and backfill support.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        config: Optional[LLMFeatureConfig] = None,
    ):
        self.store = feature_store
        self.config = config or LLMFeatureConfig()
        self.generator = LLMFeatureGenerator(
            embedding_provider=self.config.embedding_provider,
            score_model=self.config.score_model,
            prompt_version=self.config.prompt_version,
        )
        self.version_manager = create_version_manager()
        self._materialized_views: Dict[str, pd.DataFrame] = {}

    def register_llm_features(self) -> List[str]:
        """Register all LLM feature definitions in the feature store."""
        registered = []

        embedding_features = [
            ("transaction_embeddings", "Transaction description embeddings", FeatureType.VECTOR,
             {"embedding_type": EmbeddingType.TRANSACTION_DESCRIPTION.value}),
            ("account_behavior_embeddings", "Account behavior embeddings", FeatureType.VECTOR,
             {"embedding_type": EmbeddingType.ACCOUNT_BEHAVIOR.value}),
            ("alert_embeddings", "Alert description embeddings", FeatureType.VECTOR,
             {"embedding_type": EmbeddingType.ALERT_DESCRIPTION.value}),
        ]
        score_features = [
            ("fraud_probability", "Fraud probability from LLM", FeatureType.NUMERIC,
             {"score_type": ScoreType.FRAUD_PROBABILITY.value}),
            ("explanation_confidence", "Explanation confidence score", FeatureType.NUMERIC,
             {"score_type": ScoreType.EXPLANATION_CONFIDENCE.value}),
            ("uncertainty_estimates", "Uncertainty estimates from LLM", FeatureType.TIME_SERIES,
             {"score_type": ScoreType.UNCERTAINTY_ESTIMATE.value}),
        ]
        meta_features = [
            ("prompt_version", "Prompt version used for generation", FeatureType.TEXT, {}),
            ("model_used", "Model used for generation", FeatureType.TEXT, {}),
            ("latency_attribution", "Latency attribution per feature", FeatureType.NUMERIC, {}),
        ]

        for name, desc, ftype, extra_meta in embedding_features + score_features + meta_features:
            feature_def = FeatureDefinition(
                name=name,
                description=desc,
                feature_type=ftype,
                tags=["llm", "auto-generated"],
                owner="llm-features",
                status=FeatureStatus.PRODUCTION,
                metadata=extra_meta,
            )
            self.store.register_feature(
                name=name,
                computer=None,
                description=desc,
                feature_type=ftype,
            )
            registered.append(name)
            logger.info(f"Registered LLM feature: {name}")

        self._create_version_snapshots(registered)
        return registered

    def _create_version_snapshots(self, feature_names: List[str]) -> None:
        for name in feature_names:
            self.version_manager.create_version(
                feature_name=name,
                description=f"Initial version for {name}",
                code_hash=hash(name) ^ hash(self.config.prompt_version),
                parameters_hash=hash(self.config.embedding_provider),
                data_hash=0,
                created_by="llm-integration",
            )

    @cached_feature(ttl_seconds=3600)
    def compute_and_store(
        self,
        data: pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
    ) -> Dict[str, GeneratedLLMFeature]:
        """Compute LLM features and store in the feature store."""
        features = self.generator.generate_all(data, entity_col, timestamp_col)
        stored = {}
        for feature in features:
            feature_id = f"{feature.name}_v1"
            self.store.storage.store_feature_values(
                feature_id=feature_id,
                values=feature.values,
                metadata={
                    "category": feature.category.value,
                    "prompt_version": feature.meta.prompt_version,
                    "model": feature.meta.model_name,
                    "generated_at": feature.generated_at.isoformat(),
                },
            )
            if self.config.enable_materialized_views:
                self._materialized_views[feature.name] = feature.values
            stored[feature.name] = feature
        return stored

    def get_feature(
        self,
        name: str,
        entity_ids: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Retrieve a computed LLM feature."""
        if self.config.enable_materialized_views and name in self._materialized_views:
            view = self._materialized_views[name]
            if entity_ids:
                return view[view.index.isin(entity_ids)]
            return view
        feature_id = f"{name}_v1"
        return self.store.storage.get_feature_values(
            feature_id=feature_id,
            entity_ids=entity_ids,
        )

    def refresh_materialized_views(self) -> None:
        """Refresh all materialized views based on TTL policy."""
        now = datetime.utcnow()
        for name in list(self._materialized_views.keys()):
            self._materialized_views.pop(name, None)
        logger.info("Refreshed all materialized views")

    def backfill(
        self,
        historical_data: pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
        batch_size: Optional[int] = None,
    ) -> Dict[str, int]:
        """Backfill LLM features for historical data in batches."""
        batch_size = batch_size or self.config.backfill_batch_size
        total_batches = 0
        feature_counts: Dict[str, int] = {}

        for start in range(0, len(historical_data), batch_size):
            batch = historical_data.iloc[start:start + batch_size]
            features = self.compute_and_store(batch, entity_col, timestamp_col)
            for name, feat in features.items():
                feature_counts[name] = feature_counts.get(name, 0) + len(feat.values)
            total_batches += 1
            logger.info(f"Backfill batch {total_batches} complete for {len(features)} features")

        logger.info(f"Backfill complete: {total_batches} batches, {feature_counts} total values")
        return feature_counts

    def get_meta_features(
        self,
        feature_name: str,
    ) -> Dict[str, Any]:
        """Return meta features for a given LLM feature."""
        return {
            "prompt_version": self.config.prompt_version,
            "model_used": self.config.score_model,
            "latency_attribution": 0.0,
        }
