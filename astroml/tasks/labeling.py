"""Batch labeling worker for LLM-based data labeling (issue #475)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from astroml.llm.labeling import (
    ConsensusLabeler,
    DataLabeler,
    HumanReviewQueue,
    LabelDefinition,
)
from astroml.llm.labeling.schemas import (
    ALERT_CATEGORIZATION_SCHEMA,
    ENTITY_RESOLUTION_SCHEMA,
    FRAUD_CLASSIFICATION_SCHEMA,
    SENTIMENT_SCHEMA,
)

logger = logging.getLogger(__name__)


class BatchLabelingWorker:
    """Worker for batch labeling of data items."""

    def __init__(self) -> Any:
        """Initialize batch labeling worker."""
        self.labeler = DataLabeler()
        self.consensus_labeler = ConsensusLabeler()
        self.review_queue = HumanReviewQueue()

        # Register predefined schemas
        self.labeler.register_schema(FRAUD_CLASSIFICATION_SCHEMA)
        self.labeler.register_schema(ALERT_CATEGORIZATION_SCHEMA)
        self.labeler.register_schema(ENTITY_RESOLUTION_SCHEMA)
        self.labeler.register_schema(SENTIMENT_SCHEMA)

    def process_batch(
        self,
        items: List[tuple[str, Any]],
        schema_name: str,
        use_consensus: bool = False,
        auto_review_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """Process a batch of items for labeling.

        Args:
            items: List of (item_id, data) tuples
            schema_name: Name of the schema to use
            use_consensus: Whether to use multi-LLM consensus
            auto_review_threshold: Confidence threshold for auto-accept

        Returns:
            Dictionary with batch processing results
        """
        results = []
        low_confidence_items = []

        for item_id, data in items:
            if use_consensus:
                # Use consensus labeling
                definition = self.labeler.schemas[schema_name].label_definitions[0]
                consensus_result = self.consensus_labeler.label_with_consensus(
                    item_id, data, definition
                )
                results.append(consensus_result.to_dict())

                # Add to review queue if needed
                if consensus_result.requires_human_review:
                    self._add_to_review_queue(item_id, data, consensus_result, definition)
                    low_confidence_items.append(item_id)
            else:
                # Use single labeler
                result = self.labeler.label_item(item_id, data, schema_name)
                results.append(result.to_dict())

                # Check for low confidence labels
                for label in result.labels:
                    if label.confidence < auto_review_threshold:
                        definition = self.labeler.schemas[schema_name].get_definition(
                            label.label_name
                        )
                        if definition and definition.requires_human_review:
                            self._add_to_review_queue(item_id, data, result, definition)
                            low_confidence_items.append(item_id)
                            break

        return {
            "total_items": len(items),
            "processed": len(results),
            "low_confidence_count": len(low_confidence_items),
            "low_confidence_items": low_confidence_items,
            "results": results,
            "review_queue_stats": self.review_queue.get_stats(),
        }

    def _add_to_review_queue(
        self,
        item_id: str,
        data: Any,
        labeling_result: Any,
        definition: LabelDefinition,
    ) -> None:
        """Add low-confidence items to review queue.

        Args:
            item_id: Item ID
            data: Item data
            labeling_result: Labeling result
            definition: Label definition
        """
        if hasattr(labeling_result, "labels"):
            labels = labeling_result.labels
        else:
            labels = (
                labeling_result.individual_results[0]["result"]["labels"]
                if labeling_result.individual_results
                else []
            )

        self.review_queue.add_task(
            item_id=item_id,
            data=data,
            suggested_labels=labels,
            definition=definition,
        )

    def get_review_queue_stats(self) -> Dict[str, Any]:
        """Get review queue statistics.

        Returns:
            Dictionary with statistics
        """
        return self.review_queue.get_stats()

    def assign_review_task(self, task_id: str, user_id: str) -> bool:
        """Assign a review task to a user.

        Args:
            task_id: Task ID
            user_id: User ID

        Returns:
            True if assignment successful
        """
        return self.review_queue.assign_task(task_id, user_id)

    def get_user_tasks(self, user_id: str) -> List[Dict[str, Any]]:
        """Get tasks assigned to a user.

        Args:
            user_id: User ID

        Returns:
            List of task dictionaries
        """
        tasks = self.review_queue.get_user_tasks(user_id)
        return [task.to_dict() for task in tasks]


# Global batch labeling worker instance
batch_labeling_worker = BatchLabelingWorker()
