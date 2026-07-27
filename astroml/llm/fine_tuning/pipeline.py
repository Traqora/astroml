"""Fine-tuning pipeline orchestration.

Orchestrates the end-to-end fine-tuning workflow: data preparation,
training, evaluation, and model registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset import FineTuneDataset, DatasetConfig
from .trainer import FineTuneTrainer, TrainerConfig, TrainerType
from .evaluator import FineTuneEvaluator, EvaluationResult
from .registry import FineTuneRegistry

logger = logging.getLogger(__name__)


class FineTuneTarget(Enum):
    """Supported fine-tuning targets."""
    FRAUD_EXPLANATION = "fraud_explanation"
    SQL_GENERATION = "sql_generation"
    TRANSACTION_CLASSIFICATION = "transaction_classification"
    SUPPORT_CHATBOT = "support_chatbot"


@dataclass
class FineTuneConfig:
    """Configuration for a fine-tuning run."""
    target: FineTuneTarget
    base_model: str = "gpt-3.5-turbo"
    trainer_type: TrainerType = TrainerType.OPENAI
    dataset_config: Optional[DatasetConfig] = None
    trainer_config: Optional[TrainerConfig] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FineTuningPipeline:
    """End-to-end fine-tuning pipeline.

    Orchestrates data preparation, training, evaluation, and
    model registration for domain-specific LLM fine-tuning.
    """

    def __init__(
        self,
        registry: FineTuneRegistry,
        config: FineTuneConfig,
    ):
        self.registry = registry
        self.config = config
        self.dataset: Optional[FineTuneDataset] = None
        self.trainer: Optional[FineTuneTrainer] = None
        self.evaluator: Optional[FineTuneEvaluator] = None
        self.run_id: str = ""
        self._start_time: Optional[datetime] = None

    def prepare_data(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ) -> FineTuneDataset:
        """Prepare and validate training data."""
        dataset_config = self.config.dataset_config or DatasetConfig(
            name=f"{self.config.target.value}_dataset",
            task_type=self.config.target.value,
        )
        self.dataset = FineTuneDataset(config=dataset_config)
        self.dataset.load_from_dataframe(data, text_column, label_column)
        self.dataset.validate()
        self.dataset.split()
        logger.info(
            f"Data prepared: {len(self.dataset.train)} train, "
            f"{len(self.dataset.val)} val, {len(self.dataset.test)} test"
        )
        return self.dataset

    def setup_trainer(self) -> FineTuneTrainer:
        """Set up the trainer based on configuration."""
        trainer_config = self.config.trainer_config or TrainerConfig(
            model=self.config.base_model,
        )
        self.trainer = FineTuneTrainer(
            config=trainer_config,
            trainer_type=self.config.trainer_type,
        )
        return self.trainer

    def train(self) -> str:
        """Run the fine-tuning training."""
        if not self.dataset or not self.trainer:
            raise RuntimeError("Data and trainer must be prepared before training")
        self._start_time = datetime.utcnow()
        self.run_id = self.trainer.train(
            train_data=self.dataset.train,
            val_data=self.dataset.val,
        )
        duration = (datetime.utcnow() - self._start_time).total_seconds()
        logger.info(f"Training complete: run_id={self.run_id}, duration={duration:.1f}s")

        self.registry.register_model(
            model_id=self.run_id,
            target=self.config.target.value,
            base_model=self.config.base_model,
            trainer_type=self.config.trainer_type.value,
            dataset_name=self.dataset.config.name,
            metrics=self.trainer.training_metrics,
            config=self.config,
        )
        return self.run_id

    def evaluate(self) -> EvaluationResult:
        """Evaluate the fine-tuned model against holdout set."""
        if not self.dataset or not self.trainer:
            raise RuntimeError("Must train before evaluating")
        self.evaluator = FineTuneEvaluator(
            model_id=self.run_id,
            model=self.trainer,
        )
        result = self.evaluator.evaluate(
            test_data=self.dataset.test,
            baseline_model=self.config.base_model,
        )
        self.registry.update_metrics(self.run_id, result.metrics)
        return result

    def run(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ) -> EvaluationResult:
        """Run the full fine-tuning pipeline end-to-end."""
        self.prepare_data(data, text_column, label_column)
        self.setup_trainer()
        self.train()
        result = self.evaluate()
        return result
