"""Pipeline integration for LLM feature computation.

Extends the feature engineering pipeline to include LLM-generated
features such as embeddings, scores, and meta features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from astroml.features.feature_store import FeatureStore, FeatureType, FeatureDefinition
from astroml.features.feature_engine import ComputationEngine
from astroml.features.embedding_features import (
    TransactionEmbeddingComputer,
    AccountBehaviorEmbeddingComputer,
    AlertEmbeddingComputer,
)
from astroml.features.scoring_features import (
    FraudProbabilityComputer,
    ExplanationConfidenceComputer,
    UncertaintyEstimatorComputer,
)
from astroml.features.llm_generators import LLMFeatureGenerator
from astroml.cache import cached_feature

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the LLM feature pipeline."""
    enable_embeddings: bool = True
    enable_scores: bool = True
    enable_meta: bool = True
    embedding_provider: str = "openai"
    score_model: str = "gpt-4"
    prompt_version: str = "v1"
    batch_size: int = 1000
    ttl_minutes: int = 60
    parallel_computation: bool = True


class LLMFeaturePipeline:
    """Pipeline for computing and managing LLM features.

    Integrates LLM feature computation as part of the broader
    feature engineering pipeline with caching, batching, and
    materialized views.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        computation_engine: ComputationEngine,
        config: Optional[PipelineConfig] = None,
    ):
        self.store = feature_store
        self.engine = computation_engine
        self.config = config or PipelineConfig()
        self.generator = LLMFeatureGenerator(
            embedding_provider=self.config.embedding_provider,
            score_model=self.config.score_model,
            prompt_version=self.config.prompt_version,
        )
        self._register_pipeline_computers()

    def _register_pipeline_computers(self) -> None:
        if self.config.enable_embeddings:
            self.engine.register_computer(TransactionEmbeddingComputer())
            self.engine.register_computer(AccountBehaviorEmbeddingComputer())
            self.engine.register_computer(AlertEmbeddingComputer())
        if self.config.enable_scores:
            self.engine.register_computer(FraudProbabilityComputer(
                model=self.config.score_model,
                prompt_version=self.config.prompt_version,
            ))
            self.engine.register_computer(ExplanationConfidenceComputer(
                model=self.config.score_model,
                prompt_version=self.config.prompt_version,
            ))
            self.engine.register_computer(UncertaintyEstimatorComputer(
                model=self.config.score_model,
            ))

    @cached_feature(ttl_seconds=3600)
    def run_pipeline(
        self,
        data: pd.DataFrame,
        entity_col: str = "entity_id",
        timestamp_col: str = "timestamp",
    ) -> Dict[str, pd.DataFrame]:
        """Run the full LLM feature pipeline on input data."""
        results = {}

        if self.config.enable_embeddings:
            embedding_types = ["transaction_embeddings", "account_behavior_embeddings", "alert_embeddings"]
            for emb_name in embedding_types:
                computer = self.engine.get_computer(emb_name)
                if computer:
                    task = self.engine.create_task(
                        feature_name=emb_name,
                        data=data,
                        computer_name=emb_name,
                        entity_col=entity_col,
                        timestamp_col=timestamp_col,
                    )
                    self.engine.submit_task(task)

        if self.config.enable_scores:
            score_types = ["fraud_probability", "explanation_confidence", "uncertainty_estimates"]
            for score_name in score_types:
                computer = self.engine.get_computer(score_name)
                if computer:
                    task = self.engine.create_task(
                        feature_name=score_name,
                        data=data,
                        computer_name=score_name,
                        entity_col=entity_col,
                        timestamp_col=timestamp_col,
                    )
                    self.engine.submit_task(task)

        completed = self.engine.run_tasks(parallel=self.config.parallel_computation)
        for task_id, task in completed.items():
            if task.status.value == "completed" and task.result is not None:
                results[task.feature_name] = task.result
                self.store.storage.store_feature_values(
                    feature_id=f"{task.feature_name}_v1",
                    values=task.result,
                )

        return results

    def get_llm_feature_names(self) -> List[str]:
        """List all available LLM feature names in the pipeline."""
        features = []
        if self.config.enable_embeddings:
            features.extend([
                "transaction_embeddings",
                "account_behavior_embeddings",
                "alert_embeddings",
            ])
        if self.config.enable_scores:
            features.extend([
                "fraud_probability",
                "explanation_confidence",
                "uncertainty_estimates",
            ])
        if self.config.enable_meta:
            features.extend([
                "prompt_version",
                "model_used",
                "latency_attribution",
            ])
        return features

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Return a summary of the pipeline configuration and state."""
        return {
            "config": {
                "enable_embeddings": self.config.enable_embeddings,
                "enable_scores": self.config.enable_scores,
                "enable_meta": self.config.enable_meta,
                "embedding_provider": self.config.embedding_provider,
                "score_model": self.config.score_model,
                "prompt_version": self.config.prompt_version,
            },
            "registered_computers": self.engine.list_computers(),
            "llm_features": self.get_llm_feature_names(),
        }
