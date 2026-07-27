"""LLM Feature Store integration layer.

Provides feature computation, integration, and pipeline support
for LLM-generated features in the AstroML feature store.
"""

from .integration import LLMFeatureIntegration, LLMFeatureConfig
from .compute import (
    compute_embeddings,
    compute_fraud_scores,
    compute_confidence_scores,
    compute_uncertainty,
)

__all__ = [
    "LLMFeatureIntegration",
    "LLMFeatureConfig",
    "compute_embeddings",
    "compute_fraud_scores",
    "compute_confidence_scores",
    "compute_uncertainty",
]
