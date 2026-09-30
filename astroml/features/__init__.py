"""Feature computation and management for AstroML.

This module provides feature engineering capabilities including:
- Graph-based feature computation (centrality, structural importance)
- Temporal feature extraction (frequency, burstiness)
- Feature Store for centralized feature management
- LLM-based feature generation and embeddings
- Feature caching, versioning, and transformation

Key components:
- frequency: Transaction frequency and burstiness metrics
- structural_importance: Graph centrality measures
- node_features: Basic node-level features
- asset_diversity: Asset diversity metrics
- feature_store: Centralized feature storage and retrieval
- llm_features: LLM-generated features and embeddings

Exports:
- FeatureStore: Main feature store interface
- FeatureDefinition: Feature metadata and configuration
- FeatureType: Supported feature data types
- Various feature computation functions

Dependencies:
- pandas: Data manipulation
- networkx: Graph algorithms
- scikit-learn: Feature transformations
"""

from . import (
    embedding_features,
    frequency,
    graph_validation,
    imbalance,
    llm_features,
    llm_generators,
    memo,
    pipeline_structural_importance,
    scoring_features,
    structural_importance,
)
from . import pipeline as llm_pipeline
from .embedding_features import (
    AccountBehaviorEmbeddingComputer,
    AlertEmbeddingComputer,
    TransactionEmbeddingComputer,
)
from .feature_cache import (
    CacheStrategy,
    FeatureCache,
    StorageFormat,
    create_feature_cache,
    create_storage_optimizer,
)
from .feature_engine import (
    BaseFeatureComputer,
    ComputationEngine,
    compute_feature,
    create_computation_engine,
)

# Feature Store components
from .feature_store import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureSet,
    FeatureStatus,
    FeatureStorage,
    FeatureStore,
    FeatureType,
    create_feature_store,
    get_feature_store,
)
from .feature_transformers import (
    FeatureEngineering,
    FeatureTransformer,
    TransformationType,
    apply_log_transform,
    apply_standard_scaling,
    create_feature_transformer,
)
from .feature_versioning import (
    ChangeType,
    FeatureVersionManager,
    VersionStatus,
    compute_feature_hash,
    create_version_manager,
)

# LLM Feature Store components
from .llm_features import (
    EmbeddingType,
    LLMFeatureCategory,
    LLMFeatureDefinition,
    LLMFeatureMeta,
    ScoreType,
)
from .llm_generators import GeneratedLLMFeature, LLMFeatureGenerator
from .pipeline import LLMFeaturePipeline, PipelineConfig
from .scoring_features import (
    ExplanationConfidenceComputer,
    FraudProbabilityComputer,
    UncertaintyEstimatorComputer,
)

__all__ = [
    # Original feature modules
    "imbalance",
    "memo",
    "graph_validation",
    "frequency",
    "structural_importance",
    "pipeline_structural_importance",
    # LLM feature modules
    "llm_features",
    "embedding_features",
    "scoring_features",
    "llm_generators",
    "llm_pipeline",
    "LLMFeatureCategory",
    "EmbeddingType",
    "ScoreType",
    "LLMFeatureDefinition",
    "LLMFeatureMeta",
    "TransactionEmbeddingComputer",
    "AccountBehaviorEmbeddingComputer",
    "AlertEmbeddingComputer",
    "FraudProbabilityComputer",
    "ExplanationConfidenceComputer",
    "UncertaintyEstimatorComputer",
    "LLMFeatureGenerator",
    "GeneratedLLMFeature",
    "LLMFeaturePipeline",
    "PipelineConfig",
    # Feature Store core
    "FeatureStore",
    "FeatureDefinition",
    "FeatureType",
    "FeatureStatus",
    "FeatureSet",
    "FeatureStorage",
    "FeatureRegistry",
    "create_feature_store",
    "get_feature_store",
    # Feature computation
    "ComputationEngine",
    "BaseFeatureComputer",
    "create_computation_engine",
    "compute_feature",
    # Feature transformations
    "FeatureTransformer",
    "TransformationType",
    "FeatureEngineering",
    "create_feature_transformer",
    "apply_standard_scaling",
    "apply_log_transform",
    # Feature caching
    "FeatureCache",
    "CacheStrategy",
    "StorageFormat",
    "create_feature_cache",
    "create_storage_optimizer",
    # Feature versioning
    "FeatureVersionManager",
    "VersionStatus",
    "ChangeType",
    "create_version_manager",
    "compute_feature_hash",
]
