"""Automated model selection and architecture search toolkit.

Provides an AutoML pipeline, a neural architecture search framework,
meta-learning based recommendations and a model benchmarking suite.
"""

from astroml.training.model_selection.automl import (
    AutoMLConfig,
    AutoMLPipeline,
    AutoMLResult,
)
from astroml.training.model_selection.benchmark import (
    BenchmarkResult,
    ModelBenchmark,
)
from astroml.training.model_selection.meta_learning import (
    ExperienceRecord,
    MetaLearningRecommender,
    TaskDescriptor,
)
from astroml.training.model_selection.nas import (
    ArchitectureSpec,
    NASResult,
    NeuralArchitectureSearch,
)

__all__ = [
    "AutoMLConfig",
    "AutoMLPipeline",
    "AutoMLResult",
    "BenchmarkResult",
    "ModelBenchmark",
    "ExperienceRecord",
    "MetaLearningRecommender",
    "TaskDescriptor",
    "ArchitectureSpec",
    "NASResult",
    "NeuralArchitectureSearch",
]
