"""Golden dataset generation and management for model evaluation."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from astroml.db.schema import GoldenDataset, GoldenDatasetEntry
from astroml.db.session import get_session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Status State Machine
# ---------------------------------------------------------------------------

VALID_DATASET_TRANSITIONS = {
    "draft": ["review", "archived"],
    "review": ["approved", "draft", "archived"],
    "approved": ["archived"],
    "archived": [],  # Terminal state
}

VALID_DATASET_STATUSES = set(VALID_DATASET_TRANSITIONS.keys())


class InvalidDatasetStatusError(ValueError):
    """Raised when an invalid dataset status transition is attempted."""

    pass


class GoldenDatasetGenerator:
    """Core class for generating and managing golden datasets.

    Provides dataset creation, entry management, validation,
    and quality assessment capabilities.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize the golden dataset generator.

        Args:
            session: Optional SQLAlchemy session. If not provided, creates a new session.
        """
        self._session = session
        self._owns_session = session is None

    @property
    def session(self) -> Session:
        """Get the SQLAlchemy session, creating one if needed."""
        if self._session is None:
            self._session = get_session()
        return self._session

    def close(self) -> None:
        """Close the session if we own it."""
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "GoldenDatasetGenerator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Dataset CRUD operations
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        name: str,
        dataset_type: str,
        task_type: str,
        version: str = "1.0.0",
        description: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GoldenDataset:
        """Create a new golden dataset.

        Args:
            name: Dataset name
            dataset_type: Type of dataset (classification, regression, etc.)
            task_type: ML task type
            version: Dataset version
            description: Optional description
            source: Optional data source identifier
            metadata: Optional additional metadata

        Returns:
            Created GoldenDataset instance

        Raises:
            ValueError: If dataset with same name/version exists or invalid parameters
        """
        if dataset_type not in (
            "classification",
            "regression",
            "anomaly_detection",
            "clustering",
            "custom",
        ):
            raise ValueError(f"Invalid dataset_type: '{dataset_type}'")

        existing = self.get_dataset_by_name_version(name, version)
        if existing:
            raise ValueError(f"Dataset '{name}' version '{version}' already exists")

        dataset = GoldenDataset(
            name=name,
            description=description,
            dataset_type=dataset_type,
            task_type=task_type,
            version=version,
            source=source,
            metadata=metadata,
        )
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        logger.info(
            "Created golden dataset: %s (id=%d, type=%s, version=%s)",
            name,
            dataset.id,
            dataset_type,
            version,
        )
        return dataset

    def get_dataset(self, dataset_id: int) -> Optional[GoldenDataset]:
        """Get a dataset by ID.

        Args:
            dataset_id: Dataset ID

        Returns:
            GoldenDataset instance or None if not found
        """
        return self.session.get(GoldenDataset, dataset_id)

    def get_dataset_by_name_version(
        self, name: str, version: str
    ) -> Optional[GoldenDataset]:
        """Get a dataset by name and version.

        Args:
            name: Dataset name
            version: Dataset version

        Returns:
            GoldenDataset instance or None if not found
        """
        stmt = select(GoldenDataset).where(
            GoldenDataset.name == name, GoldenDataset.version == version
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        task_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[GoldenDataset]:
        """List datasets with optional filters.

        Args:
            dataset_type: Filter by dataset type
            task_type: Filter by task type
            status: Filter by status

        Returns:
            List of GoldenDataset instances
        """
        stmt = select(GoldenDataset)
        if dataset_type:
            stmt = stmt.where(GoldenDataset.dataset_type == dataset_type)
        if task_type:
            stmt = stmt.where(GoldenDataset.task_type == task_type)
        if status:
            stmt = stmt.where(GoldenDataset.status == status)
        stmt = stmt.order_by(GoldenDataset.created_at.desc())
        return list(self.session.execute(stmt).scalars().all())

    def update_dataset(
        self,
        dataset_id: int,
        description: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[GoldenDataset]:
        """Update a dataset.

        Args:
            dataset_id: Dataset ID
            description: New description
            source: New source identifier
            metadata: New metadata
            quality_score: New quality score (0-1)

        Returns:
            Updated GoldenDataset instance or None if not found

        Raises:
            ValueError: If quality_score is out of range
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        if description is not None:
            dataset.description = description
        if source is not None:
            dataset.source = source
        if metadata is not None:
            dataset.metadata = metadata
        if quality_score is not None:
            if not 0.0 <= quality_score <= 1.0:
                raise ValueError(f"quality_score must be between 0.0 and 1.0, got {quality_score}")
            dataset.quality_score = quality_score

        self.session.commit()
        self.session.refresh(dataset)
        logger.info("Updated golden dataset: %s (id=%d)", dataset.name, dataset_id)
        return dataset

    def delete_dataset(self, dataset_id: int) -> bool:
        """Delete a dataset and all its entries.

        Args:
            dataset_id: Dataset ID

        Returns:
            True if deleted, False if not found
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return False

        self.session.delete(dataset)
        self.session.commit()
        logger.info("Deleted golden dataset: %s (id=%d)", dataset.name, dataset_id)
        return True

    # ------------------------------------------------------------------
    # Dataset lifecycle management
    # ------------------------------------------------------------------

    def submit_for_review(self, dataset_id: int) -> Optional[GoldenDataset]:
        """Submit a dataset for review.

        Args:
            dataset_id: Dataset ID

        Returns:
            Updated GoldenDataset or None if not found

        Raises:
            InvalidDatasetStatusError: If dataset cannot be submitted
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        self._validate_dataset_status_transition(dataset.status, "review")
        dataset.status = "review"

        self.session.commit()
        self.session.refresh(dataset)
        logger.info("Submitted dataset for review: %s (id=%d)", dataset.name, dataset_id)
        return dataset

    def approve_dataset(self, dataset_id: int) -> Optional[GoldenDataset]:
        """Approve a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            Updated GoldenDataset or None if not found

        Raises:
            InvalidDatasetStatusError: If dataset cannot be approved
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        self._validate_dataset_status_transition(dataset.status, "approved")
        dataset.status = "approved"

        self.session.commit()
        self.session.refresh(dataset)
        logger.info("Approved dataset: %s (id=%d)", dataset.name, dataset_id)
        return dataset

    def reject_dataset(self, dataset_id: int) -> Optional[GoldenDataset]:
        """Reject a dataset (return to draft).

        Args:
            dataset_id: Dataset ID

        Returns:
            Updated GoldenDataset or None if not found

        Raises:
            InvalidDatasetStatusError: If dataset cannot be rejected
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        self._validate_dataset_status_transition(dataset.status, "draft")
        dataset.status = "draft"

        self.session.commit()
        self.session.refresh(dataset)
        logger.info("Rejected dataset (returned to draft): %s (id=%d)", dataset.name, dataset_id)
        return dataset

    def archive_dataset(self, dataset_id: int) -> Optional[GoldenDataset]:
        """Archive a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            Updated GoldenDataset or None if not found

        Raises:
            InvalidDatasetStatusError: If dataset cannot be archived
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None

        self._validate_dataset_status_transition(dataset.status, "archived")
        dataset.status = "archived"

        self.session.commit()
        self.session.refresh(dataset)
        logger.info("Archived dataset: %s (id=%d)", dataset.name, dataset_id)
        return dataset

    @staticmethod
    def _validate_dataset_status_transition(from_status: str, to_status: str) -> None:
        """Validate that a dataset status transition is allowed.

        Args:
            from_status: Current status
            to_status: Target status

        Raises:
            InvalidDatasetStatusError: If transition is not allowed
        """
        if to_status not in VALID_DATASET_STATUSES:
            raise InvalidDatasetStatusError(f"Invalid target status: '{to_status}'")

        if from_status == to_status:
            return  # No-op transition is allowed

        allowed_transitions = VALID_DATASET_TRANSITIONS.get(from_status, [])
        if to_status not in allowed_transitions:
            raise InvalidDatasetStatusError(
                f"Cannot transition from '{from_status}' to '{to_status}'. "
                f"Allowed transitions from '{from_status}': {allowed_transitions}"
            )

    # ------------------------------------------------------------------
    # Entry management
    # ------------------------------------------------------------------

    def add_entry(
        self,
        dataset_id: int,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        difficulty: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> GoldenDatasetEntry:
        """Add an entry to a dataset.

        Args:
            dataset_id: Dataset ID
            input_data: Model input features
            output_data: Ground truth labels
            metadata: Optional entry-specific metadata
            difficulty: Optional difficulty score (0-1)
            confidence: Optional label confidence (0-1)

        Returns:
            Created GoldenDatasetEntry instance

        Raises:
            ValueError: If dataset not found or invalid parameters
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        if difficulty is not None and not 0.0 <= difficulty <= 1.0:
            raise ValueError(f"difficulty must be between 0.0 and 1.0, got {difficulty}")

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        entry = GoldenDatasetEntry(
            dataset_id=dataset_id,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
            difficulty=difficulty,
            confidence=confidence,
        )
        self.session.add(entry)
        self.session.flush()

        # Update dataset size
        dataset.size += 1
        self.session.commit()
        self.session.refresh(entry)
        logger.debug(
            "Added entry to dataset: %s (entry_id=%d, dataset_id=%d)",
            dataset.name,
            entry.id,
            dataset_id,
        )
        return entry

    def add_entries_batch(
        self,
        dataset_id: int,
        entries: List[Dict[str, Any]],
    ) -> List[GoldenDatasetEntry]:
        """Add multiple entries to a dataset in a single transaction.

        Args:
            dataset_id: Dataset ID
            entries: List of entry dicts with keys: input_data, output_data, metadata, difficulty, confidence

        Returns:
            List of created GoldenDatasetEntry instances

        Raises:
            ValueError: If dataset not found or invalid parameters
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        created_entries = []
        for entry_data in entries:
            entry = GoldenDatasetEntry(
                dataset_id=dataset_id,
                input_data=entry_data["input_data"],
                output_data=entry_data["output_data"],
                metadata=entry_data.get("metadata"),
                difficulty=entry_data.get("difficulty"),
                confidence=entry_data.get("confidence"),
            )
            self.session.add(entry)
            created_entries.append(entry)

        # Update dataset size
        dataset.size += len(entries)
        self.session.commit()

        for entry in created_entries:
            self.session.refresh(entry)

        logger.info(
            "Added %d entries to dataset: %s (dataset_id=%d)",
            len(entries),
            dataset.name,
            dataset_id,
        )
        return created_entries

    def get_entry(self, entry_id: int) -> Optional[GoldenDatasetEntry]:
        """Get an entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            GoldenDatasetEntry instance or None if not found
        """
        return self.session.get(GoldenDatasetEntry, entry_id)

    def list_entries(
        self,
        dataset_id: int,
        min_difficulty: Optional[float] = None,
        max_difficulty: Optional[float] = None,
        min_confidence: Optional[float] = None,
    ) -> List[GoldenDatasetEntry]:
        """List entries for a dataset with optional filters.

        Args:
            dataset_id: Dataset ID
            min_difficulty: Filter by minimum difficulty
            max_difficulty: Filter by maximum difficulty
            min_confidence: Filter by minimum confidence

        Returns:
            List of GoldenDatasetEntry instances
        """
        stmt = select(GoldenDatasetEntry).where(
            GoldenDatasetEntry.dataset_id == dataset_id
        )
        if min_difficulty is not None:
            stmt = stmt.where(GoldenDatasetEntry.difficulty >= min_difficulty)
        if max_difficulty is not None:
            stmt = stmt.where(GoldenDatasetEntry.difficulty <= max_difficulty)
        if min_confidence is not None:
            stmt = stmt.where(GoldenDatasetEntry.confidence >= min_confidence)
        stmt = stmt.order_by(GoldenDatasetEntry.created_at)
        return list(self.session.execute(stmt).scalars().all())

    def delete_entry(self, entry_id: int) -> bool:
        """Delete an entry.

        Args:
            entry_id: Entry ID

        Returns:
            True if deleted, False if not found
        """
        entry = self.get_entry(entry_id)
        if not entry:
            return False

        dataset_id = entry.dataset_id
        self.session.delete(entry)
        self.session.flush()

        # Update dataset size
        dataset = self.get_dataset(dataset_id)
        if dataset and dataset.size > 0:
            dataset.size -= 1

        self.session.commit()
        logger.debug("Deleted entry (entry_id=%d)", entry_id)
        return True

    # ------------------------------------------------------------------
    # Dataset validation and quality metrics
    # ------------------------------------------------------------------

    def validate_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """Validate a dataset and return quality metrics.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dictionary with validation results and quality metrics
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        entries = self.list_entries(dataset_id)

        validation_results = {
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "total_entries": len(entries),
            "is_valid": True,
            "issues": [],
            "quality_metrics": {},
        }

        # Check if dataset has entries
        if len(entries) == 0:
            validation_results["is_valid"] = False
            validation_results["issues"].append("Dataset has no entries")
            return validation_results

        # Check for missing data
        entries_with_difficulty = [e for e in entries if e.difficulty is not None]
        entries_with_confidence = [e for e in entries if e.confidence is not None]

        if len(entries_with_difficulty) < len(entries):
            validation_results["issues"].append(
                f"Only {len(entries_with_difficulty)}/{len(entries)} entries have difficulty scores"
            )

        if len(entries_with_confidence) < len(entries):
            validation_results["issues"].append(
                f"Only {len(entries_with_confidence)}/{len(entries)} entries have confidence scores"
            )

        # Calculate quality metrics
        if entries_with_difficulty:
            difficulties = [e.difficulty for e in entries_with_difficulty]
            validation_results["quality_metrics"]["difficulty"] = {
                "mean": sum(difficulties) / len(difficulties),
                "min": min(difficulties),
                "max": max(difficulties),
                "count": len(difficulties),
            }

        if entries_with_confidence:
            confidences = [e.confidence for e in entries_with_confidence]
            validation_results["quality_metrics"]["confidence"] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "count": len(confidences),
            }

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(validation_results)
        validation_results["quality_score"] = quality_score

        # Update dataset with quality score
        dataset.quality_score = quality_score
        self.session.commit()

        logger.info(
            "Validated dataset: %s (id=%d, quality_score=%.2f)",
            dataset.name,
            dataset_id,
            quality_score,
        )

        return validation_results

    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results.

        Args:
            validation_results: Validation results dictionary

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize for missing entries
        if validation_results["total_entries"] == 0:
            return 0.0

        # Penalize for missing difficulty/confidence scores
        quality_metrics = validation_results.get("quality_metrics", {})
        total_entries = validation_results["total_entries"]

        if "difficulty" in quality_metrics:
            diff_coverage = quality_metrics["difficulty"]["count"] / total_entries
            score *= (0.5 + 0.5 * diff_coverage)  # Min 0.5 if full coverage

        if "confidence" in quality_metrics:
            conf_coverage = quality_metrics["confidence"]["count"] / total_entries
            score *= (0.5 + 0.5 * conf_coverage)  # Min 0.5 if full coverage

        # Penalize for issues
        num_issues = len(validation_results["issues"])
        score *= max(0.0, 1.0 - (num_issues * 0.1))

        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Dataset export/import
    # ------------------------------------------------------------------

    def export_dataset(self, dataset_id: int) -> Dict[str, Any]:
        """Export a dataset with all entries.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dictionary with dataset metadata and entries
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Dataset with id {dataset_id} not found")

        entries = self.list_entries(dataset_id)

        export_data = {
            "dataset": {
                "name": dataset.name,
                "description": dataset.description,
                "dataset_type": dataset.dataset_type,
                "task_type": dataset.task_type,
                "version": dataset.version,
                "source": dataset.source,
                "quality_score": dataset.quality_score,
                "metadata": dataset.metadata,
            },
            "entries": [
                {
                    "input_data": entry.input_data,
                    "output_data": entry.output_data,
                    "metadata": entry.metadata,
                    "difficulty": entry.difficulty,
                    "confidence": entry.confidence,
                }
                for entry in entries
            ],
        }

        logger.info("Exported dataset: %s (id=%d, entries=%d)", dataset.name, dataset_id, len(entries))
        return export_data

    def import_dataset(
        self,
        export_data: Dict[str, Any],
        new_name: Optional[str] = None,
        new_version: Optional[str] = None,
    ) -> GoldenDataset:
        """Import a dataset from exported data.

        Args:
            export_data: Exported dataset dictionary
            new_name: Optional new name (uses original if not provided)
            new_version: Optional new version (uses original if not provided)

        Returns:
            Created GoldenDataset instance
        """
        dataset_data = export_data["dataset"]
        entries_data = export_data["entries"]

        dataset = self.create_dataset(
            name=new_name or dataset_data["name"],
            dataset_type=dataset_data["dataset_type"],
            task_type=dataset_data["task_type"],
            version=new_version or dataset_data["version"],
            description=dataset_data.get("description"),
            source=dataset_data.get("source"),
            metadata=dataset_data.get("metadata"),
        )

        # Import entries in batch
        self.add_entries_batch(
            dataset.id,
            entries_data,
        )

        # Restore quality score if available
        if dataset_data.get("quality_score"):
            dataset.quality_score = dataset_data["quality_score"]
            self.session.commit()

        logger.info(
            "Imported dataset: %s (id=%d, entries=%d)",
            dataset.name,
            dataset.id,
            len(entries_data),
        )
        return dataset
