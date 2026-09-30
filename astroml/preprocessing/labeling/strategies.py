"""Labeling strategies and pipeline orchestration (issue #624).

Provides composable labeling strategies that integrate active learning,
weak supervision, and human review into a unified pipeline.

Components:
- LabelingStrategy: Abstract base for labeling strategies
- ActiveWeakStrategy: Combines active learning with weak supervision
- BatchLabelingStrategy: Bulk labeling of entire datasets
- LabelingPipeline: Full orchestration from unlabeled data to final labels
- LabelingDashboard: Analytics aggregator for labeling metrics
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from astroml.preprocessing.labeling.active_learning import (
    ActiveLearner,
    ActiveLearningResult,
    QueryStrategy,
    SampleScore,
)
from astroml.preprocessing.labeling.review_queue import (
    ConflictResolver,
    ReviewItem,
    ReviewQueue,
)
from astroml.preprocessing.labeling.weak_supervision import (
    LabelingFunction,
    WeakSupervisionModel,
    _compute_vote_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------


class LabelingStrategy(ABC):
    """Abstract base for labeling strategies."""

    @abstractmethod
    def label(
        self,
        samples: Sequence[Any],
        *,
        model: Callable[[Sequence[Any]], np.ndarray] | None = None,
    ) -> LabelingResult:
        """Apply the strategy to produce labels for the given samples."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return strategy statistics."""
        ...


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LabelingResult:
    """Result of a labeling strategy run.

    Attributes:
        labels: Final integer labels (n_samples,).
        probabilities: Class probabilities (n_samples x n_classes).
        confidence: Per-sample confidence scores.
        quality_score: Overall quality estimate (0-1).
        stats: Additional run statistics.
    """

    labels: np.ndarray
    probabilities: np.ndarray | None = None
    confidence: np.ndarray | None = None
    quality_score: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Active + Weak strategy
# ---------------------------------------------------------------------------


class ActiveWeakStrategy(LabelingStrategy):
    """Combines active learning (sample selection) with weak supervision (labeling).

    Active learning selects which samples to label; weak supervision
    produces the labels via labeling functions.

    Args:
        lfs: Labeling functions for weak supervision.
        active_strategy: Query strategy for sample selection.
        batch_size: Samples per active-learning round.
    """

    def __init__(
        self,
        lfs: Sequence[LabelingFunction],
        active_strategy: QueryStrategy | None = None,
        batch_size: int = 10,
    ) -> None:
        self.lfs = list(lfs)
        self.ws_model = WeakSupervisionModel(self.lfs)
        self.active_strategy = active_strategy
        self.batch_size = batch_size
        self._fitted = False

    def label(
        self,
        samples: Sequence[Any],
        *,
        model: Callable[[Sequence[Any]], np.ndarray] | None = None,
    ) -> LabelingResult:
        if not self._fitted:
            self.ws_model.fit(samples)

        if self.active_strategy is not None and model is not None:
            learner = ActiveLearner(
                strategy=self.active_strategy,
                pool=samples,
                batch_size=self.batch_size,
            )
            learner.run(model)
            # Use labeled subset for WS training
            if learner.labeled_indices:
                labeled_samples = [samples[i] for i in learner.labeled_indices]
                self.ws_model.fit(labeled_samples)

        probs = self.ws_model.predict_proba(samples)
        labels = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        return LabelingResult(
            labels=labels,
            probabilities=probs,
            confidence=confidence,
            quality_score=float(np.mean(confidence)),
            stats={"n_lfs": len(self.lfs), "n_samples": len(samples)},
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "n_lfs": len(self.lfs),
            "lf_coverages": {lf.name: lf.coverage for lf in self.lfs},
            "lf_accuracies": {lf.name: lf.accuracy for lf in self.lfs},
        }


# ---------------------------------------------------------------------------
# Batch labeling strategy
# ---------------------------------------------------------------------------


class BatchLabelingStrategy(LabelingStrategy):
    """Bulk labeling of entire datasets using weak supervision only.

    Args:
        lfs: Labeling functions.
        majority_only: If True, use simple majority voting (faster).
    """

    def __init__(
        self,
        lfs: Sequence[LabelingFunction],
        majority_only: bool = False,
    ) -> None:
        self.lfs = list(lfs)
        if majority_only:
            from astroml.preprocessing.labeling.weak_supervision import MajorityVoter

            self.ws_model = MajorityVoter(lfs)
        else:
            self.ws_model = WeakSupervisionModel(lfs)

    def label(
        self,
        samples: Sequence[Any],
        *,
        model: Callable[[Sequence[Any]], np.ndarray] | None = None,
    ) -> LabelingResult:
        self.ws_model.fit(samples)
        probs = self.ws_model.predict_proba(samples)
        labels = np.argmax(probs, axis=1)
        confidence = np.max(probs, axis=1)

        return LabelingResult(
            labels=labels,
            probabilities=probs,
            confidence=confidence,
            quality_score=float(np.mean(confidence)),
            stats={"n_lfs": len(self.lfs), "n_samples": len(samples)},
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "n_lfs": len(self.lfs),
            "lf_coverages": {lf.name: lf.coverage for lf in self.lfs},
        }


# ---------------------------------------------------------------------------
# Labeling pipeline orchestrator
# ---------------------------------------------------------------------------


class LabelingPipeline:
    """Full-orchestration pipeline: unlabeled data → final labels.

    Integrates active learning, weak supervision, conflict resolution,
    and human review into a single callable pipeline.

    Args:
        strategy: The :class:`LabelingStrategy` to use.
        review_queue: Optional :class:`ReviewQueue` for human-in-the-loop.
        conflict_resolver: Optional :class:`ConflictResolver`.
        auto_accept_threshold: Auto-accept labels with confidence >= threshold.
    """

    def __init__(
        self,
        strategy: LabelingStrategy,
        review_queue: ReviewQueue | None = None,
        conflict_resolver: ConflictResolver | None = None,
        auto_accept_threshold: float = 0.85,
    ) -> None:
        self.strategy = strategy
        self.review_queue = review_queue or ReviewQueue()
        self.conflict_resolver = conflict_resolver or ConflictResolver(strategy="majority")
        self.auto_accept_threshold = auto_accept_threshold
        self._pipeline_stats: dict[str, Any] = {
            "total_processed": 0,
            "auto_accepted": 0,
            "queued_for_review": 0,
            "conflicts_resolved": 0,
        }

    def run(
        self,
        samples: Sequence[Any],
        *,
        model: Callable[[Sequence[Any]], np.ndarray] | None = None,
    ) -> LabelingResult:
        """Run the full pipeline on a set of samples.

        Args:
            samples: Input samples to label.
            model: Optional model for active-learning based strategies.

        Returns:
            :class:`LabelingResult` with final labels.
        """
        result = self.strategy.label(samples, model=model)
        confidence = result.confidence
        if confidence is None:
            confidence = np.ones(len(samples))

        final_labels = result.labels.copy()
        n = len(samples)

        for i in range(n):
            if confidence[i] >= self.auto_accept_threshold:
                self._pipeline_stats["auto_accepted"] += 1
            else:
                # Queue for human review
                self.review_queue.enqueue(
                    item_id=f"sample_{i}",
                    sample=samples[i],
                    labels={"ws_model": int(final_labels[i])},
                    conflict_score=1.0 - float(confidence[i]),
                    priority=1.0 - float(confidence[i]),
                )
                self._pipeline_stats["queued_for_review"] += 1

        self._pipeline_stats["total_processed"] += n

        return LabelingResult(
            labels=final_labels,
            probabilities=result.probabilities,
            confidence=confidence,
            quality_score=result.quality_score,
            stats={**result.stats, "pipeline_stats": self._pipeline_stats},
        )

    def resolve_reviews(self, batch_size: int = 50) -> int:
        """Process the review queue in batches.

        Args:
            batch_size: Number of items to resolve per call.

        Returns:
            Number of items resolved.
        """
        items = self.review_queue.dequeue_batch(batch_size)
        resolved = 0
        for item in items:
            votes = list(item.labels.values())
            winner, conf = self.conflict_resolver.resolve(votes)
            self.review_queue.resolve(item.item_id, winner)
            self._pipeline_stats["conflicts_resolved"] += 1
            resolved += 1
        return resolved

    def get_dashboard(self) -> dict[str, Any]:
        """Return analytics dashboard data for the labeling pipeline.

        Returns:
            Dict with pipeline stats, quality report, and strategy stats.
        """
        return {
            "pipeline_stats": self._pipeline_stats,
            "quality_report": self.review_queue.get_quality_report(),
            "strategy_stats": self.strategy.get_stats(),
            "auto_accept_rate": (
                self._pipeline_stats["auto_accepted"] / max(self._pipeline_stats["total_processed"], 1)
            ),
        }


# ---------------------------------------------------------------------------
# Dashboard aggregator
# ---------------------------------------------------------------------------


@dataclass
class LabelingDashboard:
    """Analytics dashboard for labeling operations.

    Aggregates metrics from across the labeling pipeline for monitoring
    and reporting.

    Attributes:
        total_samples: Total samples processed.
        auto_accepted: Samples auto-accepted by the pipeline.
        queued_for_review: Samples sent to human review.
        conflicts_resolved: Review items resolved.
        quality_scores: Per-labeler quality metrics.
        created_at: Dashboard generation timestamp.
    """

    total_samples: int = 0
    auto_accepted: int = 0
    queued_for_review: int = 0
    conflicts_resolved: int = 0
    quality_scores: dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "auto_accepted": self.auto_accepted,
            "queued_for_review": self.queued_for_review,
            "conflicts_resolved": self.conflicts_resolved,
            "quality_scores": self.quality_scores,
            "created_at": self.created_at.isoformat(),
        }