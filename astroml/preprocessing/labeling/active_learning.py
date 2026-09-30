"""Automated data labeling and annotation pipeline (issue #624).

Provides active learning query strategies for selecting the most informative
samples for human labeling, reducing annotation costs while maximizing model
improvement.

Components:
- ActiveLearner: Core active learning orchestrator
- UncertaintySampling: Least-confidence and margin-based sampling
- DiversitySampling: Cluster-based and dissimilarity-based selection
- HybridStrategy: Combined uncertainty + diversity approaches
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SampleScore:
    """A sample with its informativeness score.

    Attributes:
        index: Original index / identifier of the sample
        score: Informativeness score (higher = more informative)
        uncertainty: Uncertainty component (0-1)
        diversity: Diversity component (0-1) — zero for pure uncertainty strategies
        metadata: Optional per-sample metadata
    """

    index: int
    score: float
    uncertainty: float = 1.0
    diversity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveLearningResult:
    """Result of a single active-learning query round.

    Attributes:
        round_id: Zero-based round index
        selected_indices: Indices of samples chosen for labeling
        scores: Per-sample scores for the selected batch
        query_time_ms: Wall-clock time for the query step
    """

    round_id: int
    selected_indices: list[int]
    scores: list[SampleScore]
    query_time_ms: float


# ---------------------------------------------------------------------------
# Strategy base
# ---------------------------------------------------------------------------


class QueryStrategy(ABC, Generic[T]):
    """Abstract base for active-learning query strategies.

    Subclasses implement :meth:`score_samples` and may override
    :meth:`select_batch` for custom batch-selection logic.
    """

    @abstractmethod
    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        """Score every sample in the unlabeled pool.

        Args:
            pool: Unlabeled samples.
            model: Optional callable that returns class probabilities (N x C).
            model_probs: Pre-computed probability matrix (alternative to model).

        Returns:
            One :class:`SampleScore` per pool element.
        """
        ...

    def select_batch(
        self,
        scores: list[SampleScore],
        batch_size: int,
    ) -> list[SampleScore]:
        """Select a batch of the highest-scoring samples (default: top-k).

        Args:
            scores: Scored pool items.
            batch_size: Number of samples to select.

        Returns:
            Top *batch_size* :class:`SampleScore` instances sorted by score descending.
        """
        return sorted(scores, key=lambda s: s.score, reverse=True)[:batch_size]


# ---------------------------------------------------------------------------
# Uncertainty strategies
# ---------------------------------------------------------------------------


class UncertaintySampling(QueryStrategy[T]):
    """Least-confidence uncertainty sampling.

    Scores each sample as ``1 - max(probs)`` so the model is most uncertain
    about the samples it selects.
    """

    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        probs = _resolve_probs(pool, model, model_probs)
        if probs is None:
            return _uniform_scores(len(pool))

        max_probs = np.max(probs, axis=1)
        return [
            SampleScore(
                index=i,
                score=1.0 - float(max_probs[i]),
                uncertainty=1.0 - float(max_probs[i]),
            )
            for i in range(len(pool))
        ]


class MarginSampling(QueryStrategy[T]):
    """Margin-based uncertainty sampling.

    Scores each sample as ``1 - (p_best - p_second)`` — the smaller the gap
    between the top two predicted classes, the higher the uncertainty.
    """

    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        probs = _resolve_probs(pool, model, model_probs)
        if probs is None or probs.shape[1] < 2:
            return _uniform_scores(len(pool))

        # Sort each row: (best, second, ...)
        sorted_rows = -np.sort(-probs, axis=1)  # descending
        margins = sorted_rows[:, 0] - sorted_rows[:, 1]
        n = max(len(pool), probs.shape[0])
        return [
            SampleScore(index=i, score=1.0 - float(margins[i]), uncertainty=1.0 - float(margins[i]))
            for i in range(min(n, len(margins)))
        ]


class EntropySampling(QueryStrategy[T]):
    """Entropy-based uncertainty sampling.

    Scores samples by the entropy of the predicted class distribution
    (higher entropy = more uncertainty).
    """

    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        probs = _resolve_probs(pool, model, model_probs)
        if probs is None:
            return _uniform_scores(len(pool))

        # Clip to avoid log(0)
        eps = 1e-10
        probs = np.clip(probs, eps, 1.0 - eps)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        # Normalise by log(num_classes) so scores live in [0, 1]
        log_k = np.log(probs.shape[1])
        normalized = entropy / log_k if log_k > 0 else entropy
        n = max(len(pool), probs.shape[0])
        return [
            SampleScore(index=i, score=float(normalized[i]), uncertainty=float(normalized[i]))
            for i in range(min(n, len(normalized)))
        ]


# ---------------------------------------------------------------------------
# Diversity strategies
# ---------------------------------------------------------------------------


class DiversitySampling(QueryStrategy[T]):
    """Cluster-based diversity sampling.

    Selects samples that are dissimilar to already-labeled instances,
    ensuring broad coverage of the feature space.

    Args:
        feature_extractor: Callable that maps a sample to a fixed-length
            feature vector.
        labeled_indices: Indices of previously labeled samples.
        diversity_weight: Weight assigned to diversity vs. uncertainty in
            hybrid scoring (default 1.0 for pure diversity).
    """

    def __init__(
        self,
        feature_extractor: Callable[[T], np.ndarray],
        labeled_indices: Iterable[int] | None = None,
        diversity_weight: float = 1.0,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.labeled_indices: set[int] = set(labeled_indices or ())
        self.diversity_weight = diversity_weight
        self._labeled_features: np.ndarray | None = None

    def add_labeled(self, indices: Iterable[int], samples: Sequence[T]) -> None:
        """Register newly labeled samples so they influence future diversity.

        Args:
            indices: Indices of the newly labeled samples.
            samples: The full sample pool (needed to extract features).
        """
        self.labeled_indices.update(indices)
        self._labeled_features = None  # invalidate cache

    def _get_labeled_features(self, pool: Sequence[T]) -> np.ndarray:
        """Return feature matrix of labeled samples (M x D)."""
        if self._labeled_features is not None:
            return self._labeled_features
        feats = [self.feature_extractor(pool[i]) for i in self.labeled_indices]
        self._labeled_features = np.stack(feats) if feats else np.empty((0, 1))
        return self._labeled_features

    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        labeled_feats = self._get_labeled_features(pool)
        if labeled_feats.size == 0:
            return _uniform_scores(len(pool))

        pool_feats = np.stack([self.feature_extractor(s) for s in pool])
        # Minimum cosine distance to any labeled sample
        scores: list[SampleScore] = []
        for i, feat in enumerate(pool_feats):
            if i in self.labeled_indices:
                scores.append(SampleScore(index=i, score=0.0, diversity=0.0))
                continue
            similarities = np.dot(labeled_feats, feat) / (
                np.linalg.norm(labeled_feats, axis=1) * np.linalg.norm(feat) + 1e-10
            )
            diversity = 1.0 - float(np.max(similarities))  # 1 = most diverse
            scores.append(
                SampleScore(
                    index=i,
                    score=diversity * self.diversity_weight,
                    diversity=diversity,
                )
            )
        return scores


# ---------------------------------------------------------------------------
# Hybrid strategy
# ---------------------------------------------------------------------------


class HybridStrategy(QueryStrategy[T]):
    """Combines uncertainty and diversity sampling.

    Score = alpha * uncertainty + (1 - alpha) * diversity.

    Args:
        uncertainty_strategy: An uncertainty-based :class:`QueryStrategy`.
        diversity_strategy: A diversity-based :class:`QueryStrategy`.
        alpha: Blend weight for uncertainty (0-1).
    """

    def __init__(
        self,
        uncertainty_strategy: QueryStrategy[T],
        diversity_strategy: QueryStrategy[T],
        alpha: float = 0.5,
    ) -> None:
        self.uncertainty_strategy = uncertainty_strategy
        self.diversity_strategy = diversity_strategy
        self.alpha = alpha

    def score_samples(
        self,
        pool: Sequence[T],
        *,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> list[SampleScore]:
        u_scores = self.uncertainty_strategy.score_samples(pool, model=model, model_probs=model_probs)
        d_scores = self.diversity_strategy.score_samples(pool, model=model, model_probs=model_probs)

        return [
            SampleScore(
                index=u.index,
                score=self.alpha * u.uncertainty + (1.0 - self.alpha) * d.diversity,
                uncertainty=u.uncertainty,
                diversity=d.diversity,
            )
            for u, d in zip(u_scores, d_scores)
        ]


# ---------------------------------------------------------------------------
# Active learner orchestrator
# ---------------------------------------------------------------------------


class ActiveLearner(Generic[T]):
    """Orchestrates active-learning query rounds.

    Args:
        strategy: The query strategy to use for sample selection.
        pool: The full unlabeled sample pool.
        batch_size: Number of samples to select each round.
        max_rounds: Maximum number of query rounds.
    """

    def __init__(
        self,
        strategy: QueryStrategy[T],
        pool: Sequence[T],
        batch_size: int = 10,
        max_rounds: int = 20,
    ) -> None:
        self.strategy = strategy
        self.pool = pool
        self.batch_size = batch_size
        self.max_rounds = max_rounds
        self.labeled_indices: set[int] = set()
        self.history: list[ActiveLearningResult] = []

    def query_round(
        self,
        model: Callable[[Sequence[T]], np.ndarray] | None = None,
        model_probs: np.ndarray | None = None,
    ) -> ActiveLearningResult:
        """Execute one query round.

        Returns:
            :class:`ActiveLearningResult` describing the selected batch.
        """
        import time

        start = time.perf_counter()
        scores = self.strategy.score_samples(self.pool, model=model, model_probs=model_probs)
        unlabeled_scores = [s for s in scores if s.index not in self.labeled_indices]
        selected = self.strategy.select_batch(unlabeled_scores, self.batch_size)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.labeled_indices.update(s.index for s in selected)
        result = ActiveLearningResult(
            round_id=len(self.history),
            selected_indices=[s.index for s in selected],
            scores=selected,
            query_time_ms=elapsed_ms,
        )
        self.history.append(result)
        logger.info(
            "Active-learning round %d: selected %d samples in %.1f ms",
            result.round_id,
            len(selected),
            elapsed_ms,
        )
        return result

    def run(
        self,
        model: Callable[[Sequence[T]], np.ndarray],
        *,
        oracle: Callable[[T], Any] | None = None,
    ) -> list[ActiveLearningResult]:
        """Run multiple rounds until max_rounds or pool exhaustion.

        Args:
            model: A callable mapping pool → probability matrix (N x C).
            oracle: Optional labeling oracle; called for each selected sample.
                Not required for the query loop to function.

        Returns:
            All :class:`ActiveLearningResult` entries.
        """
        for _ in range(self.max_rounds):
            if len(self.labeled_indices) >= len(self.pool):
                break
            result = self.query_round(model=model)
            if oracle is not None:
                for idx in result.selected_indices:
                    oracle(self.pool[idx])
        return self.history

    @property
    def labeled_count(self) -> int:
        """Number of samples labeled so far."""
        return len(self.labeled_indices)

    @property
    def unlabeled_count(self) -> int:
        """Number of unlabeled samples remaining."""
        return len(self.pool) - len(self.labeled_indices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_probs(
    pool: Sequence[T],
    model: Callable[[Sequence[T]], np.ndarray] | None,
    model_probs: np.ndarray | None,
) -> np.ndarray | None:
    """Return probability matrix derived from *model* or *model_probs*."""
    if model_probs is not None:
        return np.asarray(model_probs, dtype=np.float64)
    if model is not None:
        return np.asarray(model(pool), dtype=np.float64)
    return None


def _uniform_scores(n: int) -> list[SampleScore]:
    """Return uniform scores when probabilities are unavailable."""
    return [SampleScore(index=i, score=1.0 / n, uncertainty=1.0 / n) for i in range(n)]