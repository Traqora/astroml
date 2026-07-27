"""Feature modules for AstroML.

Expose feature computation utilities and Feature Store here."""
from . import frequency
from . import imbalance
from . import memo
from . import graph_validation
from . import structural_importance
from . import pipeline_structural_importance
from . import llm_features
from . import embedding_features
from . import scoring_features
from . import llm_generators
from . import pipeline as llm_pipeline

# LLM Feature Store components
from .llm_features import (
    LLMFeatureCategory,
    EmbeddingType,
    ScoreType,
    LLMFeatureDefinition,
    LLMFeatureMeta,
)
from .embedding_features import (
    TransactionEmbeddingComputer,
    AccountBehaviorEmbeddingComputer,
    AlertEmbeddingComputer,
)
from .scoring_features import (
    FraudProbabilityComputer,
    ExplanationConfidenceComputer,
    UncertaintyEstimatorComputer,
)
from .llm_generators import LLMFeatureGenerator, GeneratedLLMFeature
from .pipeline import LLMFeaturePipeline, PipelineConfig

# Feature Store components
from .feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureType,
    FeatureStatus,
    FeatureSet,
    FeatureStorage,
    FeatureRegistry,
    create_feature_store,
    get_feature_store,
)

from .feature_engine import (
    ComputationEngine,
    BaseFeatureComputer,
    create_computation_engine,
    compute_feature,
)

from .feature_transformers import (
    FeatureTransformer,
    TransformationType,
    FeatureEngineering,
    create_feature_transformer,
    apply_standard_scaling,
    apply_log_transform,
)

from .feature_cache import (
    FeatureCache,
    CacheStrategy,
    StorageFormat,
    create_feature_cache,
    create_storage_optimizer,
)

from .feature_versioning import (
    FeatureVersionManager,
    VersionStatus,
    ChangeType,
    create_version_manager,
    compute_feature_hash,
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
