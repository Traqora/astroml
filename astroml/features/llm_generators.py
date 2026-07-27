"""LLM feature generation orchestration.

Provides high-level generators that orchestrate LLM calls and
produce structured features for the Feature Store.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from astroml.features.llm_features import LLMFeatureMeta, LLMFeatureCategory, EmbeddingType, ScoreType
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

logger = logging.getLogger(__name__)


@dataclass
class GeneratedLLMFeature:
    """Container for a generated LLM feature with metadata."""
    name: str
    category: LLMFeatureCategory
    values: pd.DataFrame
    meta: LLMFeatureMeta
    entity_col: str
    generated_at: datetime = field(default_factory=datetime.utcnow)


class LLMFeatureGenerator:
    """Orchestrates LLM feature generation for multiple feature types."""

    def __init__(
        self,
        embedding_provider: str = "openai",
        score_model: str = "gpt-4",
        prompt_version: str = "v1",
    ):
        self.embedding_provider = embedding_provider
        self.score_model = score_model
        self.prompt_version = prompt_version
        self._computers = self._init_computers()

    def _init_computers(self) -> Dict[str, object]:
        return {
            "transaction_embeddings": TransactionEmbeddingComputer(
                provider=self.embedding_provider,
            ),
            "account_behavior_embeddings": AccountBehaviorEmbeddingComputer(
                provider=self.embedding_provider,
            ),
            "alert_embeddings": AlertEmbeddingComputer(
                provider=self.embedding_provider,
            ),
            "fraud_probability": FraudProbabilityComputer(
                model=self.score_model,
                prompt_version=self.prompt_version,
            ),
            "explanation_confidence": ExplanationConfidenceComputer(
                model=self.score_model,
                prompt_version=self.prompt_version,
            ),
            "uncertainty_estimates": UncertaintyEstimatorComputer(
                model=self.score_model,
            ),
        }

    def generate_embeddings(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        embedding_type: EmbeddingType,
    ) -> GeneratedLLMFeature:
        computer_name = {
            EmbeddingType.TRANSACTION_DESCRIPTION: "transaction_embeddings",
            EmbeddingType.ACCOUNT_BEHAVIOR: "account_behavior_embeddings",
            EmbeddingType.ALERT_DESCRIPTION: "alert_embeddings",
        }.get(embedding_type)

        if not computer_name:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        computer = self._computers[computer_name]
        values = computer.compute(data, entity_col, timestamp_col)
        meta = LLMFeatureMeta(
            prompt_version=self.prompt_version,
            model_name=self.embedding_provider,
            provider=self.embedding_provider,
            latency_ms=0.0,
            tokens_used=0,
            cost_usd=0.0,
        )
        return GeneratedLLMFeature(
            name=computer_name,
            category=LLMFeatureCategory.EMBEDDING,
            values=values,
            meta=meta,
            entity_col=entity_col,
        )

    def generate_scores(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
        score_type: ScoreType,
    ) -> GeneratedLLMFeature:
        computer_name = {
            ScoreType.FRAUD_PROBABILITY: "fraud_probability",
            ScoreType.EXPLANATION_CONFIDENCE: "explanation_confidence",
            ScoreType.UNCERTAINTY_ESTIMATE: "uncertainty_estimates",
        }.get(score_type)

        if not computer_name:
            raise ValueError(f"Unknown score type: {score_type}")

        computer = self._computers[computer_name]
        values = computer.compute(data, entity_col, timestamp_col)
        meta = LLMFeatureMeta(
            prompt_version=self.prompt_version,
            model_name=self.score_model,
            provider="openai",
            latency_ms=0.0,
            tokens_used=0,
            cost_usd=0.0,
        )
        return GeneratedLLMFeature(
            name=computer_name,
            category=LLMFeatureCategory.SCORE,
            values=values,
            meta=meta,
            entity_col=entity_col,
        )

    def generate_all(
        self,
        data: pd.DataFrame,
        entity_col: str,
        timestamp_col: str,
    ) -> List[GeneratedLLMFeature]:
        features = []
        for emb_type in EmbeddingType:
            try:
                features.append(
                    self.generate_embeddings(data, entity_col, timestamp_col, emb_type)
                )
            except Exception as e:
                logger.warning(f"Failed to generate {emb_type.value} embeddings: {e}")
        for score_type in ScoreType:
            try:
                features.append(
                    self.generate_scores(data, entity_col, timestamp_col, score_type)
                )
            except Exception as e:
                logger.warning(f"Failed to generate {score_type.value} scores: {e}")
        return features
