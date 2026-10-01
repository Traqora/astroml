from astroml.utils.exceptions import AstroMLError
"""Fine-tuned model registry.

Provides versioned storage, lineage tracking, and deployment
management for fine-tuned models.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FineTuneModelRecord:
    """Record for a fine-tuned model in the registry."""

    model_id: str
    target: str
    base_model: str
    trainer_type: str
    dataset_name: str
    status: str = "registered"
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    deployed_at: str | None = None
    version: int = 1
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FineTuneModelRecord:
        return cls(**data)


class FineTuneRegistry:
    """Registry for fine-tuned models.

    Provides storage, versioning, lineage tracking, and
    deployment management for fine-tuned models.
    """

    def __init__(self, storage_path -> Any: str = "./fine_tune_registry"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._records: dict[str, FineTuneModelRecord] = {}
        self._load_records()

    def _load_records(self) -> None:
        """Load existing records from disk."""
        records_file = os.path.join(self.storage_path, "registry.json")
        if os.path.exists(records_file):
            try:
                with open(records_file) as f:
                    data = json.load(f)
                for record_data in data:
                    record = FineTuneModelRecord.from_dict(record_data)
                    self._records[record.model_id] = record
                logger.info(f"Loaded {len(self._records)} registry records")
            except AstroMLError as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_records(self) -> None:
        """Save records to disk."""
        records_file = os.path.join(self.storage_path, "registry.json")
        data = [record.to_dict() for record in self._records.values()]
        with open(records_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_id: str,
        target: str,
        base_model: str,
        trainer_type: str,
        dataset_name: str,
        metrics: dict[str, float] | None = None,
        config: object | None = None,
    ) -> FineTuneModelRecord:
        """Register a fine-tuned model in the registry."""
        record = FineTuneModelRecord(
            model_id=model_id,
            target=target,
            base_model=base_model,
            trainer_type=trainer_type,
            dataset_name=dataset_name,
            metrics=metrics or {},
            config=asdict(config) if config and hasattr(config, "__dataclass_fields__") else {},
            created_at=datetime.utcnow().isoformat(),
        )

        if model_id in self._records:
            existing = self._records[model_id]
            record.version = existing.version + 1

        self._records[model_id] = record
        self._save_records()
        logger.info(f"Registered model {model_id} (v{record.version}) for target '{target}'")
        return record

    def update_metrics(
        self,
        model_id: str,
        metrics: dict[str, float],
    ) -> FineTuneModelRecord | None:
        """Update metrics for a registered model."""
        record = self._records.get(model_id)
        if not record:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        record.metrics = metrics
        record.updated_at = datetime.utcnow().isoformat()
        self._save_records()
        return record

    def deploy_model(self, model_id: str) -> FineTuneModelRecord | None:
        """Mark a model as deployed."""
        record = self._records.get(model_id)
        if not record:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        record.status = "deployed"
        record.deployed_at = datetime.utcnow().isoformat()
        record.updated_at = datetime.utcnow().isoformat()
        self._save_records()
        return record

    def rollback(self, model_id: str, version: int) -> FineTuneModelRecord | None:
        """Rollback to a previous version of a model."""
        record = self._records.get(model_id)
        if not record:
            logger.warning(f"Model {model_id} not found in registry")
            return None
        record.status = "rolled_back"
        record.updated_at = datetime.utcnow().isoformat()
        self._save_records()

        rollback_record = FineTuneModelRecord(
            model_id=f"{model_id}_v{version}",
            target=record.target,
            base_model=record.base_model,
            trainer_type=record.trainer_type,
            dataset_name=record.dataset_name,
            status="deployed",
            version=version,
        )
        self._records[f"{model_id}_v{version}"] = rollback_record
        self._save_records()
        return rollback_record

    def get_model(self, model_id: str) -> FineTuneModelRecord | None:
        """Get a registered model by ID."""
        return self._records.get(model_id)

    def list_models(
        self,
        target: str | None = None,
        status: str | None = None,
    ) -> list[FineTuneModelRecord]:
        """List registered models with optional filtering."""
        records = list(self._records.values())
        if target:
            records = [r for r in records if r.target == target]
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def get_cost_summary(self) -> dict[str, Any]:
        """Return cost tracking summary per fine-tuning run."""
        summary = {
            "total_models": len(self._records),
            "by_target": {},
            "by_status": {},
        }
        for record in self._records.values():
            summary["by_target"][record.target] = summary["by_target"].get(record.target, 0) + 1
            summary["by_status"][record.status] = summary["by_status"].get(record.status, 0) + 1
        return summary
