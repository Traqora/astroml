"""Dataset preparation and formatting for fine-tuning.

Handles data loading, validation, splitting, and formatting
for various fine-tuning targets and trainer types.
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for fine-tuning dataset preparation."""
    name: str
    task_type: str
    min_examples: int = 500
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_length: int = 4096
    format_template: str = "{input} -> {output}"
    seed: int = 42


class DataQualityValidator:
    """Validates dataset quality for fine-tuning."""

    def __init__(self, config: DatasetConfig):
        self.config = config

    def validate(self, data: pd.DataFrame) -> List[str]:
        """Validate data quality and return list of issues found."""
        issues = []

        if len(data) < self.config.min_examples:
            issues.append(
                f"Only {len(data)} examples, minimum required is {self.config.min_examples}"
            )

        required_columns = {"input", "output"}
        missing = required_columns - set(data.columns)
        if missing:
            issues.append(f"Missing required columns: {missing}")

        if "input" in data.columns:
            duplicates = data["input"].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate inputs")

            empty_inputs = data["input"].isna().sum()
            if empty_inputs > 0:
                issues.append(f"Found {empty_inputs} empty inputs")

            null_inputs = data["input"].isnull().sum()
            if null_inputs > 0:
                issues.append(f"Found {null_inputs} null inputs")

        if "output" in data.columns:
            empty_outputs = data["output"].isna().sum()
            if empty_outputs > 0:
                issues.append(f"Found {empty_outputs} empty outputs")

        return issues


class FineTuneDataset:
    """Prepared dataset for fine-tuning.

    Handles loading, validation, splitting, and formatting
    of training data for various fine-tuning targets.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.train: List[Dict[str, str]] = []
        self.val: List[Dict[str, str]] = []
        self.test: List[Dict[str, str]] = []
        self._version: str = ""
        self._metadata: Dict[str, Any] = {}

    def load_from_dataframe(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: Optional[str] = None,
    ) -> None:
        """Load dataset from a pandas DataFrame."""
        records = []
        for _, row in data.iterrows():
            record = {"input": str(row.get(text_column, ""))}
            if label_column and label_column in data.columns:
                record["output"] = str(row[label_column])
            else:
                record["output"] = str(row.get("output", row.get("label", "")))
            records.append(record)
        self._raw_records = records
        self._version = self._compute_version(records)
        self._metadata = {
            "total_records": len(records),
            "columns": list(data.columns),
            "loaded_at": datetime.utcnow().isoformat(),
        }

    def load_from_jsonl(self, path: str) -> None:
        """Load dataset from a JSONL file."""
        records = []
        with open(path) as f:
            for line in f:
                records.append(json.loads(line.strip()))
        self._raw_records = records
        self._version = self._compute_version(records)
        self._metadata = {
            "total_records": len(records),
            "source": path,
            "loaded_at": datetime.utcnow().isoformat(),
        }

    def validate(self) -> List[str]:
        """Validate dataset quality."""
        validator = DataQualityValidator(self.config)
        data = pd.DataFrame(self._raw_records)
        issues = validator.validate(data)
        if issues:
            for issue in issues:
                logger.warning(f"Dataset quality issue: {issue}")
        else:
            logger.info("Dataset validation passed")
        return issues

    def split(self) -> None:
        """Split dataset into train/val/test sets."""
        from sklearn.model_selection import train_test_split

        records = list(self._raw_records)
        train_val, self.test = train_test_split(
            records,
            test_size=self.config.test_ratio,
            random_state=self.config.seed,
        )
        val_ratio_adj = self.config.val_ratio / (1 - self.config.test_ratio)
        self.train, self.val = train_test_split(
            train_val,
            test_size=val_ratio_adj,
            random_state=self.config.seed,
        )
        self._metadata.update({
            "train_count": len(self.train),
            "val_count": len(self.val),
            "test_count": len(self.test),
            "split_seed": self.config.seed,
        })

    def format_for_openai(self, records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Format records for OpenAI fine-tuning API."""
        formatted = []
        for record in records:
            formatted.append({
                "messages": [
                    {"role": "user", "content": record["input"]},
                    {"role": "assistant", "content": record["output"]},
                ]
            })
        return formatted

    def format_for_lora(
        self,
        records: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Format records for LoRA/QLoRA fine-tuning."""
        formatted = []
        for record in records:
            formatted.append({
                "input": record["input"],
                "output": record["output"],
                "text": self.config.format_template.format(
                    input=record["input"],
                    output=record["output"],
                ),
            })
        return formatted

    def save(self, path: str) -> None:
        """Save the dataset splits to disk."""
        import json
        import os

        os.makedirs(path, exist_ok=True)
        for split_name, split_data in [
            ("train", self.train),
            ("val", self.val),
            ("test", self.test),
        ]:
            filepath = os.path.join(path, f"{split_name}.jsonl")
            with open(filepath, "w") as f:
                for record in split_data:
                    f.write(json.dumps(record) + "\n")

        metadata_path = os.path.join(path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info(f"Dataset saved to {path}")

    @property
    def version(self) -> str:
        """Get dataset version hash."""
        return self._version

    def _compute_version(self, records: List[Dict[str, str]]) -> str:
        content = json.dumps(records, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]
