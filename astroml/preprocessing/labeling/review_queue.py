"""Human-in-the-loop review interface (issue #624).

Extends the existing LLM-based review queue with active-learning-driven
prioritization, label quality tracking, and conflict resolution.

Components:
- ReviewQueue: Prioritised review queue with quality metrics
- LabelQualityMetrics: Per-labeler / per-LF quality tracking
- ConflictResolver: Automated resolution of labeling conflicts
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class LabelQualityMetrics:
    """Per-label-source quality statistics.

    Attributes:
        labeler_id: Name / ID of the labeling source (LF, human, LLM).
        total_labels: Total votes cast.
        agreement_with_consensus: Fraction of votes matching final consensus.
        avg_confidence: Mean confidence score.
        review_rejection_rate: Fraction of labels flagged in review.
    """

    labeler_id: str
    total_labels: int = 0
    agreement_with_consensus: float = 0.0
    avg_confidence: float = 0.0
    review_rejection_rate: float = 0.0

    def update(self, total: int, agreed: int, avg_conf: float, rejected: int) -> None:
        """Update running statistics."""
        self.total_labels += total
        if self.total_labels:
            self.agreement_with_consensus = (
                self.agreement_with_consensus * (self.total_labels - total) + agreed
            ) / self.total_labels
            self.avg_confidence = (
                self.avg_confidence * (self.total_labels - total) + avg_conf * total
            ) / self.total_labels
            self.review_rejection_rate = (
                self.review_rejection_rate * (self.total_labels - total) + rejected
            ) / self.total_labels


@dataclass
class ReviewItem:
    """A single item queued for human review.

    Attributes:
        item_id: Unique item identifier.
        sample: The data sample being labeled.
        labels: All candidate labels (from LFs, LLM, etc.).
        conflict_score: 0 = unanimous, 1 = complete disagreement.
        priority: Higher values are reviewed first.
        created_at: When this item entered the queue.
        status: Current review status.
        resolution: If resolved, the agreed-upon label value.
    """

    item_id: str
    sample: Any
    labels: dict[str, Any] = field(default_factory=dict)
    conflict_score: float = 0.0
    priority: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    resolution: Any = None


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------


class ReviewQueue:
    """Prioritised human-in-the-loop review queue.

    Items with higher *priority* (confidence-weighted conflict) are surfaced
    first so reviewers spend time on the most impactful samples.

    Args:
        max_queue_size: Evict lowest-priority items once this size is exceeded.
    """

    def __init__(self, max_queue_size: int = 10_000) -> None:
        self._items: dict[str, ReviewItem] = {}
        self.max_queue_size = max_queue_size
        self._quality: dict[str, LabelQualityMetrics] = {}

    def enqueue(
        self,
        item_id: str,
        sample: Any,
        labels: dict[str, Any],
        conflict_score: float = 0.0,
        priority: float = 0.0,
    ) -> ReviewItem:
        """Add an item to the review queue.

        Args:
            item_id: Unique item identifier.
            sample: The data sample.
            labels: Dict mapping label-source name → label value.
            conflict_score: Pre-computed conflict score (auto-computed if 0).
            priority: Pre-assigned priority (auto-computed if 0).

        Returns:
            The created :class:`ReviewItem`.
        """
        if conflict_score == 0.0 and len(labels) > 1:
            conflict_score = _compute_conflict(list(labels.values()))
        if priority == 0.0:
            priority = conflict_score  # higher conflict = higher priority

        item = ReviewItem(
            item_id=item_id,
            sample=sample,
            labels=labels,
            conflict_score=conflict_score,
            priority=priority,
        )
        self._items[item_id] = item

        if len(self._items) > self.max_queue_size:
            self._evict_lowest()

        logger.debug("Enqueued review item %s (priority=%.2f)", item_id, priority)
        return item

    def dequeue_batch(self, batch_size: int = 10) -> list[ReviewItem]:
        """Return the highest-priority pending items and mark them in-progress.

        Args:
            batch_size: Maximum items to return.

        Returns:
            Sorted list of :class:`ReviewItem` instances.
        """
        pending = [i for i in self._items.values() if i.status == "pending"]
        pending.sort(key=lambda i: i.priority, reverse=True)
        batch = pending[:batch_size]
        for item in batch:
            item.status = "in_review"
        return batch

    def resolve(
        self,
        item_id: str,
        resolution: Any,
        *,
        quality_updates: dict[str, tuple[int, int, float, int]] | None = None,
    ) -> bool:
        """Mark an item as resolved and update quality metrics.

        Args:
            item_id: Item to resolve.
            resolution: Agreed-upon label value.
            quality_updates: Optional per-labeler quality deltas
                ``{labeler_id: (total, agreed, avg_conf, rejected)}``.

        Returns:
            True if the item was found and resolved.
        """
        if item_id not in self._items:
            return False
        item = self._items[item_id]
        item.status = "resolved"
        item.resolution = resolution

        if quality_updates:
            for labeler_id, (total, agreed, avg_conf, rejected) in quality_updates.items():
                if labeler_id not in self._quality:
                    self._quality[labeler_id] = LabelQualityMetrics(labeler_id=labeler_id)
                self._quality[labeler_id].update(total, agreed, avg_conf, rejected)

        logger.info("Resolved review item %s → %s", item_id, resolution)
        return True

    def get_quality_report(self) -> dict[str, Any]:
        """Return a quality report for all tracked label sources.

        Returns:
            Dict mapping labeler_id → quality snapshot.
        """
        return {
            lid: {
                "total_labels": q.total_labels,
                "agreement_with_consensus": q.agreement_with_consensus,
                "avg_confidence": q.avg_confidence,
                "review_rejection_rate": q.review_rejection_rate,
            }
            for lid, q in self._quality.items()
        }

    def _evict_lowest(self) -> None:
        """Remove the lowest-priority pending item."""
        pending = [(k, i) for k, i in self._items.items() if i.status == "pending"]
        if not pending:
            return
        lowest = min(pending, key=lambda x: x[1].priority)
        del self._items[lowest[0]]
        logger.debug("Evicted low-priority review item %s", lowest[0])


# ---------------------------------------------------------------------------
# Conflict resolver
# ---------------------------------------------------------------------------


class ConflictResolver:
    """Automated resolution of labeling conflicts.

    Supports majority-vote and weighted-vote resolution strategies.

    Args:
        strategy: Resolution strategy (``"majority"`` or ``"weighted"``).
    """

    def __init__(self, strategy: str = "majority") -> None:
        self.strategy = strategy

    def resolve(
        self,
        votes: Sequence[Any],
        *,
        weights: Sequence[float] | None = None,
    ) -> tuple[Any, float]:
        """Resolve a set of conflicting votes into a single label.

        Args:
            votes: Label values from different sources.
            weights: Source weights (used by ``"weighted"`` strategy).

        Returns:
            ``(resolved_label, confidence)`` where confidence is the
            proportion of votes agreeing with the resolution.
        """
        if not votes:
            return None, 0.0

        if self.strategy == "weighted" and weights and len(weights) == len(votes):
            return self._weighted_resolve(votes, weights)

        return self._majority_resolve(votes)

    @staticmethod
    def _majority_resolve(votes: Sequence[Any]) -> tuple[Any, float]:
        counter: dict[Any, int] = {}
        for v in votes:
            counter[v] = counter.get(v, 0) + 1
        winner = max(counter, key=counter.get)  # type: ignore[arg-type]
        confidence = counter[winner] / len(votes)
        return winner, confidence

    @staticmethod
    def _weighted_resolve(
        votes: Sequence[Any],
        weights: Sequence[float],
    ) -> tuple[Any, float]:
        total_weight = sum(weights)
        if total_weight == 0:
            return votes[0], 1.0 / len(votes)

        weighted: dict[Any, float] = {}
        for v, w in zip(votes, weights):
            weighted[v] = weighted.get(v, 0.0) + w
        winner = max(weighted, key=weighted.get)  # type: ignore[arg-type]
        confidence = weighted[winner] / total_weight
        return winner, confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_conflict(values: Sequence[Any]) -> float:
    """Compute conflict score: 0 = unanimous, 1 = complete disagreement."""
    if len(values) <= 1:
        return 0.0
    unique = set(values)
    if len(unique) == 1:
        return 0.0
    # Normalize: max disagreement is when every value is different
    return (len(unique) - 1) / (len(values) - 1)