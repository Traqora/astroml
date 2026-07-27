"""LLM score feature computers for the Feature Store.

Computes score-based features from LLM providers including
fraud probability, explanation confidence, and uncertainty estimates.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from astroml.features.feature_engine import BaseFeatureComputer, FeatureDependencyType
from astroml.features.feature_store import FeatureType

logger = logging.getLogger(__name__)


class FraudProbabilityComputer(BaseFeatureComputer):
    """Computer for LLM-based fraud probability scores."""

    def __init__(self, model: str = "gpt-4", prompt_version: str = "v1"):
        super().__init__("fraud_probability")
        self.model = model
        self.prompt_version = prompt_version
        self.add_dependency(
            "transaction_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "amount", "description", "timestamp"]},
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
            from astroml.llm.features.compute import compute_fraud_scores
            scores = compute_fraud_scores(
                data=data,
                entity_col=entity_col,
                model=self.model,
                prompt_version=self.prompt_version,
            )
            result = pd.DataFrame(
                {"fraud_probability": scores},
                index=data[entity_col].values,
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise


class ExplanationConfidenceComputer(BaseFeatureComputer):
    """Computer for LLM explanation confidence scores."""

    def __init__(self, model: str = "gpt-4", prompt_version: str = "v1"):
        super().__init__("explanation_confidence")
        self.model = model
        self.prompt_version = prompt_version
        self.add_dependency(
            "explanation_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "explanation", "timestamp"]},
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
            from astroml.llm.features.compute import compute_confidence_scores
            confidence = compute_confidence_scores(
                data=data,
                entity_col=entity_col,
                model=self.model,
                prompt_version=self.prompt_version,
            )
            result = pd.DataFrame(
                {"explanation_confidence": confidence},
                index=data[entity_col].values,
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise


class UncertaintyEstimatorComputer(BaseFeatureComputer):
    """Computer for LLM uncertainty estimates."""

    def __init__(self, model: str = "gpt-4", num_samples: int = 5):
        super().__init__("uncertainty_estimates")
        self.model = model
        self.num_samples = num_samples
        self.add_dependency(
            "prediction_data",
            FeatureDependencyType.DATA,
            {"columns": ["entity_id", "input_text", "timestamp"]},
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
            from astroml.llm.features.compute import compute_uncertainty
            uncertainty = compute_uncertainty(
                data=data,
                entity_col=entity_col,
                model=self.model,
                num_samples=self.num_samples,
            )
            result = pd.DataFrame(
                uncertainty,
                index=data[entity_col].values,
            )
            return result
        except ImportError as e:
            logger.error(f"Could not import compute module: {e}")
            raise
