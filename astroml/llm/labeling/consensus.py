from astroml.utils.exceptions import AstroMLError
"""Multi-LLM consensus for LLM-based data labeling (issue #475)."""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .labeler import DataLabeler
from .schemas import Label, LabelDefinition

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of consensus labeling.

    Attributes:
        item_id: ID of the labeled item
        labels: Consensus labels
        individual_results: Results from each LLM
        agreement_score: Agreement score (0-1)
        requires_human_review: Whether human review is needed
    """

    item_id: str
    labels: List[Label]
    individual_results: List[Dict[str, Any]]
    agreement_score: float
    requires_human_review: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "labels": [l.to_dict() for l in self.labels],
            "individual_results": self.individual_results,
            "agreement_score": self.agreement_score,
            "requires_human_review": self.requires_human_review,
        }


class ConsensusLabeler:
    """Multi-LLM consensus labeling for improved accuracy."""

    def __init__(self, min_agreement -> Any: float = 0.7):
        """Initialize consensus labeler.

        Args:
            min_agreement: Minimum agreement score for auto-accept
        """
        self.min_agreement = min_agreement
        self.llm_clients: Dict[str, Any] = {}
        self.labeler = DataLabeler()

    def add_llm_client(self, name: str, client: Any) -> None:
        """Add an LLM client for consensus.

        Args:
            name: Client name
            client: LLM client instance
        """
        self.llm_clients[name] = client
        logger.info(f"Added LLM client: {name}")

    def label_with_consensus(
        self,
        item_id: str,
        data: Any,
        definition: LabelDefinition,
    ) -> ConsensusResult:
        """Label an item using multiple LLMs and reach consensus.

        Args:
            item_id: ID of the item to label
            data: Data to label
            definition: Label definition

        Returns:
            ConsensusResult with consensus labels
        """
        if not self.llm_clients:
            logger.warning("No LLM clients configured, using single labeler")
            # Fall back to single labeler
            result = self.labeler.label_item(item_id, data, definition.name)
            return ConsensusResult(
                item_id=item_id,
                labels=result.labels,
                individual_results=[result.to_dict()],
                agreement_score=1.0,
                requires_human_review=not result.success,
            )

        # Get labels from each LLM
        individual_results = []
        all_labels = []

        for client_name, client in self.llm_clients.items():
            try:
                result = self.labeler.label_item(item_id, data, definition.name, client)
                individual_results.append({
                    "client": client_name,
                    "result": result.to_dict(),
                })
                all_labels.extend(result.labels)
            except AstroMLError as e:
                logger.error(f"Error from client {client_name}: {e}")
                individual_results.append({
                    "client": client_name,
                    "error": str(e),
                })

        # Compute consensus
        consensus_labels = self._compute_consensus(all_labels, definition)
        agreement_score = self._calculate_agreement(all_labels, definition)

        # Determine if human review is needed
        requires_human_review = agreement_score < self.min_agreement

        return ConsensusResult(
            item_id=item_id,
            labels=consensus_labels,
            individual_results=individual_results,
            agreement_score=agreement_score,
            requires_human_review=requires_human_review,
        )

    def _compute_consensus(
        self,
        labels: List[Label],
        definition: LabelDefinition,
    ) -> List[Label]:
        """Compute consensus labels from multiple LLM outputs.

        Args:
            labels: Labels from multiple LLMs
            definition: Label definition

        Returns:
            Consensus labels
        """
        if not labels:
            return []

        # Group by label name
        by_name: Dict[str, List[Label]] = {}
        for label in labels:
            if label.label_name not in by_name:
                by_name[label.label_name] = []
            by_name[label.label_name].append(label)

        for label_name, label_list in by_name.items():
            # Count value occurrences
            values = [l.value for l in label_list]
            value_counts = Counter(values)

            # Get most common value
            most_common_value, count = value_counts.most_common(1)[0]

            # Calculate average confidence
            avg_confidence = sum(l.confidence for l in label_list) / len(label_list)

            # Boost confidence based on agreement
            agreement_ratio = count / len(label_list)
            boosted_confidence = avg_confidence * agreement_ratio

            consensus_labels.append(
                Label(
                    id=f"consensus_{label_name}",
                    label_name=label_name,
                    value=most_common_value,
                    confidence=boosted_confidence,
                    source="consensus",
                    metadata={
                        "vote_count": count,
                        "total_votes": len(label_list),
                        "agreement_ratio": agreement_ratio,
                    },
                )
            )

        return consensus_labels

    def _calculate_agreement(
        self,
        labels: List[Label],
        definition: LabelDefinition,
    ) -> float:
        """Calculate agreement score among LLMs.

        Args:
            labels: Labels from multiple LLMs
            definition: Label definition

        Returns:
            Agreement score (0-1)
        """
        if not labels:
            return 0.0

        # Group by label name
        by_name: Dict[str, List[Label]] = {}
        for label in labels:
            if label.label_name not in by_name:
                by_name[label.label_name] = []
            by_name[label.label_name].append(label)

        # Calculate agreement for each label
        agreements = []
        for label_name, label_list in by_name.items():
            values = [l.value for l in label_list]
            value_counts = Counter(values)
            most_common_count = value_counts.most_common(1)[0][1]
            agreement = most_common_count / len(label_list)
            agreements.append(agreement)

        # Return average agreement
        return sum(agreements) / len(agreements) if agreements else 0.0
