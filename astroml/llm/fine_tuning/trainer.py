"""Training wrapper for fine-tuning LLMs.

Provides unified interface for OpenAI fine-tuning API,
LoRA/QLoRA for open-source models, and other trainer types.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrainerType(Enum):
    """Supported fine-tuning trainer types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LORA = "lora"
    QLORA = "qlora"


@dataclass
class TrainerConfig:
    """Configuration for the fine-tuning trainer."""
    model: str = "gpt-3.5-turbo"
    learning_rate: float = 1e-5
    num_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    max_seq_length: int = 4096
    use_wandb: bool = False
    wandb_project: str = "astroml-fine-tuning"
    output_dir: str = "./fine_tuned_models"
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


class FineTuneTrainer:
    """Unified trainer for fine-tuning LLMs.

    Supports multiple trainer types:
    - OpenAI: Uses OpenAI fine-tuning API
    - Anthropic: Uses Anthropic fine-tuning API
    - LoRA/QLoRA: Uses PEFT for open-source models
    """

    def __init__(
        self,
        config: TrainerConfig,
        trainer_type: TrainerType = TrainerType.OPENAI,
    ):
        self.config = config
        self.trainer_type = trainer_type
        self.training_metrics: Dict[str, Any] = {}
        self._model_id: str = ""

    def train(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Run fine-tuning training.

        Args:
            train_data: Training records
            val_data: Optional validation records

        Returns:
            Model ID / run ID for the trained model
        """
        if self.trainer_type == TrainerType.OPENAI:
            return self._train_openai(train_data, val_data)
        elif self.trainer_type == TrainerType.ANTHROPIC:
            return self._train_anthropic(train_data, val_data)
        elif self.trainer_type in (TrainerType.LORA, TrainerType.QLORA):
            return self._train_lora(train_data, val_data, self.trainer_type)
        else:
            raise ValueError(f"Unsupported trainer type: {self.trainer_type}")

    def _train_openai(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]],
    ) -> str:
        """Fine-tune using OpenAI API."""
        try:
            import openai
            from .dataset import FineTuneDataset
        except ImportError:
            logger.error("openai package not installed")
            raise

        dataset = FineTuneDataset.__new__(FineTuneDataset)
        dataset.config = None
        formatted_train = dataset.format_for_openai(train_data)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for record in formatted_train:
                f.write(json.dumps(record) + "\n")
            train_path = f.name

        try:
            client = openai.OpenAI()
            with open(train_path, "rb") as f:
                response = client.files.create(file=f, purpose="fine-tune")

            job = client.fine_tuning.jobs.create(
                training_file=response.id,
                model=self.config.model,
                hyperparameters={
                    "n_epochs": self.config.num_epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate_multiplier": self.config.learning_rate,
                },
            )
            self._model_id = job.id
            self.training_metrics = {
                "job_id": job.id,
                "model": self.config.model,
                "status": job.status,
                "created_at": datetime.utcnow().isoformat(),
            }
            return job.id
        finally:
            os.unlink(train_path)

    def _train_anthropic(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]],
    ) -> str:
        """Fine-tune using Anthropic API."""
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed")
            raise

        client = anthropic.Anthropic()
        formatted = [
            {"role": "user", "content": r["input"]}
            for r in train_data
        ]
        response = client.beta.fine_tuning.jobs.create(
            model=self.config.model,
            training_data=formatted,
        )
        self._model_id = response.id
        self.training_metrics = {
            "job_id": response.id,
            "status": "created",
            "created_at": datetime.utcnow().isoformat(),
        }
        return response.id

    def _train_lora(
        self,
        train_data: List[Dict[str, str]],
        val_data: Optional[List[Dict[str, str]]],
        trainer_type: TrainerType,
    ) -> str:
        """Fine-tune using LoRA/QLoRA for open-source models."""
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
                Trainer,
            )
            from peft import (
                LoraConfig,
                get_peft_model,
                prepare_model_for_kbit_training,
            )
            from datasets import Dataset
        except ImportError:
            logger.error(
                "transformers/torch/datasets/peft packages not installed for LoRA"
            )
            raise

        texts = [
            f"{r['input']}\n{r['output']}"
            for r in train_data
        ]
        hf_dataset = Dataset.from_dict({"text": texts})

        use_4bit = trainer_type == TrainerType.QLORA
        if use_4bit:
            quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"}
        else:
            quantization_config = None

        model_name = self.config.model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
        )

        if use_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = hf_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=os.path.join(self.config.output_dir, "logs"),
            report_to="wandb" if self.config.use_wandb else "none",
            seed=self.config.seed,
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
        )

        trainer.train()
        output_path = os.path.join(
            self.config.output_dir,
            f"lora_{self.config.model.replace('/', '_')}_{int(time.time())}",
        )
        trainer.save_model(output_path)
        tokenizer.save_pretrained(output_path)

        self._model_id = output_path
        self.training_metrics = {
            "output_path": output_path,
            "model": self.config.model,
            "trainer_type": trainer_type.value,
            "num_epochs": self.config.num_epochs,
            "lora_rank": self.config.lora_rank,
        }
        return output_path

    def get_model_id(self) -> str:
        """Get the trained model identifier."""
        return self._model_id
