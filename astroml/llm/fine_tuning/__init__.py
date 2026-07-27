"""Fine-tuning pipeline for LLMs.

Provides infrastructure for fine-tuning LLMs on domain-specific data
including data preparation, training orchestration, evaluation,
and model registry.

Supported targets:
- Fraud Explanation Model
- SQL Generation Model
- Transaction Classification
- Support Chatbot
"""

from .pipeline import FineTuningPipeline, FineTuneConfig, FineTuneTarget
from .dataset import FineTuneDataset, DatasetConfig, DataQualityValidator
from .trainer import FineTuneTrainer, TrainerConfig, TrainerType
from .evaluator import FineTuneEvaluator, EvaluationResult
from .registry import FineTuneRegistry, FineTuneModelRecord

__all__ = [
    "FineTuningPipeline",
    "FineTuneConfig",
    "FineTuneTarget",
    "FineTuneDataset",
    "DatasetConfig",
    "DataQualityValidator",
    "FineTuneTrainer",
    "TrainerConfig",
    "TrainerType",
    "FineTuneEvaluator",
    "EvaluationResult",
    "FineTuneRegistry",
    "FineTuneModelRecord",
]
