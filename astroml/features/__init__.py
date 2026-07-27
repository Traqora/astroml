"""Feature modules for AstroML.

Expose feature computation utilities and Feature Store here."""
from . import frequency
from . import imbalance
from . import memo
from . import graph_validation
from . import structural_importance
from . import pipeline_structural_importance

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
