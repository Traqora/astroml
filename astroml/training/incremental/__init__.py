"""Incremental and continuous learning modules."""

from .adaptive_model import (
    AdaptiveModel,
    AdaptiveModelConfig,
    EWCRegularizer,
    ExperienceReplayBuffer,
)
from .online_learner import (
    OnlineLearnerConfig,
    OnlinePassiveAggressiveClassifier,
    OnlinePassiveAggressiveRegressor,
    OnlineSGDClassifier,
    OnlineSGDRegressor,
)
from .stream_trainer import (
    StreamDataIngestor,
    StreamTrainer,
    StreamTrainerConfig,
)

__all__ = [
    "OnlineLearnerConfig",
    "OnlineSGDClassifier",
    "OnlineSGDRegressor",
    "OnlinePassiveAggressiveClassifier",
    "OnlinePassiveAggressiveRegressor",
    "AdaptiveModelConfig",
    "AdaptiveModel",
    "ExperienceReplayBuffer",
    "EWCRegularizer",
    "StreamTrainerConfig",
    "StreamTrainer",
    "StreamDataIngestor",
]
