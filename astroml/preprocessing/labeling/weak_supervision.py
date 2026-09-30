from astroml.utils.exceptions import AstroMLError
"""Weak supervision with labeling functions (issue #624).

Provides a framework for programmatic labeling via labeling functions (LFs)
that can be combined to produce probabilistic training labels at scale.

Components:
- LabelingFunction: Single heuristic that votes on a label
- WeakSupervisionModel: Aggregates LF votes into probabilistic labels
- MajorityVoter: Simple majority-vote aggregation
- SnorkelStyleModel: Generative model (Naive Bayes) for LF aggregation
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Label value reserved for "abstain"
ABSTAIN = -1


# ---------------------------------------------------------------------------
# Labeling function
# ---------------------------------------------------------------------------


@dataclass
class LabelingFunction:
    """A single labeling function that votes on data points.

    Attributes:
        name: Human-readable name.
        fn: Callable ``(sample) -> int`` returning a class label or ABSTAIN.
        label_map: Optional mapping from LF output to canonical labels.
        coverage: Fraction of samples the LF votes on (updated during fit).
        accuracy: Estimated accuracy on labeled data (updated during fit).
    """

    name: str
    fn: Callable[[Any], int]
    label_map: dict[int, str] | None = None
    coverage: float = 0.0
    accuracy: float = 0.0

    def apply(self, sample: Any) -> int:
        """Invoke the LF on a single sample. Returns an integer label or ABSTAIN."""
        try:
            return self.fn(sample)
        except AstroMLError:
            logger.debug("LF %s raised on sample — treating as ABSTAIN", self.name, exc_info=True)
            return ABSTAIN


# ---------------------------------------------------------------------------
# Vote matrix helpers
# ---------------------------------------------------------------------------


def _compute_vote_matrix(
    lfs: Sequence[LabelingFunction],
    samples: Sequence[Any],
) -> np.ndarray:
    """Compute an (n_lfs x n_samples) integer vote matrix.

    Rows = labeling functions, columns = samples.  Value is ABSTAIN (-1)
    when the LF does not vote.
    """
    n_lfs = len(lfs)
    n_samples = len(samples)
    votes = np.full((n_lfs, n_samples), ABSTAIN, dtype=np.int32)
    for j, lf in enumerate(lfs):
        for i, sample in enumerate(samples):
            votes[j, i] = lf.apply(sample)
    return votes


# ---------------------------------------------------------------------------
# Weak supervision model
# ---------------------------------------------------------------------------


class WeakSupervisionModel:
    """Orchestrates labeling functions to produce probabilistic labels.

    Args:
        lfs: Sequence of :class:`LabelingFunction` instances.
    """

    def __init__(self, lfs: Sequence[LabelingFunction]) -> None:
        self.lfs = list(lfs)
        self._cardinality: int | None = None

    def fit(
        self,
        samples: Sequence[Any],
        *,
        gold_labels: np.ndarray | None = None,
        cardinality: int | None = None,
    ) -> None:
        """Fit the model by computing LF statistics.

        Args:
            samples: Training samples.
            gold_labels: Optional ground-truth integer labels for LF accuracy estimation.
            cardinality: Number of classes. Inferred from votes if not given.
        """
        votes = _compute_vote_matrix(self.lfs, samples)

        if cardinality is not None:
            self._cardinality = cardinality
        elif gold_labels is not None:
            self._cardinality = int(np.max(gold_labels) + 1)
        else:
            unique = np.unique(votes[votes != ABSTAIN])
            self._cardinality = int(np.max(unique) + 1) if len(unique) else 2

        for j, lf in enumerate(self.lfs):
            voted = votes[j] != ABSTAIN
            lf.coverage = float(np.mean(voted))
            if gold_labels is not None and np.any(voted):
                lf.accuracy = float(np.mean(votes[j][voted] == gold_labels[voted]))
            else:
                lf.accuracy = 1.0  # optimistic prior

        logger.info(
            "Weak supervision fit on %d samples with %d LFs (cardinality=%d)",
            len(samples),
            len(self.lfs),
            self._cardinality,
        )

    def predict_proba(
        self,
        samples: Sequence[Any],
    ) -> np.ndarray:
        """Return (n_samples x cardinality) probabilistic label matrix.

        Uses the Snorkel-style generative model to de-noise LF votes.
        """
        if self._cardinality is None:
            self._cardinality = 2
        votes = _compute_vote_matrix(self.lfs, samples)
        return _snorkel_generative_model(votes, len(self.lfs), self._cardinality)

    def predict(self, samples: Sequence[Any]) -> np.ndarray:
        """Return hard integer labels (n_samples,)."""
        return np.argmax(self.predict_proba(samples), axis=1)


# ---------------------------------------------------------------------------
# Majority voter
# ---------------------------------------------------------------------------


class MajorityVoter(WeakSupervisionModel):
    """Simple majority-vote aggregation of labeling functions."""

    def predict_proba(self, samples: Sequence[Any]) -> np.ndarray:
        if self._cardinality is None:
            self._cardinality = 2
        votes = _compute_vote_matrix(self.lfs, samples)
        n_samples = len(samples)
        probs = np.zeros((n_samples, self._cardinality), dtype=np.float64)
        for i in range(n_samples):
            col = votes[:, i]
            valid = col[col != ABSTAIN]
            if len(valid) == 0:
                probs[i] = np.ones(self._cardinality) / self._cardinality
                continue
            counter = Counter(valid.tolist())
            for k, v in counter.items():
                probs[i, k] = v / len(valid)
        return probs


# ---------------------------------------------------------------------------
# Snorkel-style generative model
# ---------------------------------------------------------------------------


def _snorkel_generative_model(
    votes: np.ndarray,
    n_lfs: int,
    cardinality: int,
) -> np.ndarray:
    """Naive-Bayes-style generative model for LF vote aggregation.

    Models P(y) and P(LF_j = v | y) for each LF independently, then
    estimates P(y | votes) via Bayes rule.

    Args:
        votes: (n_lfs x n_samples) integer votes, ABSTAIN = -1.
        n_lfs: Number of labeling functions.
        cardinality: Number of distinct classes.

    Returns:
        (n_samples x cardinality) probability matrix.
    """
    n_samples = votes.shape[1]
    probs = np.zeros((n_samples, cardinality), dtype=np.float64)

    # Uniform prior P(y)
    prior = np.ones(cardinality) / cardinality

    # Estimate P(LF_j = label | y) using majority-vote pseudo-labels
    # as a simple approximation (no EM here).
    pseudo_labels = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        col = votes[:, i]
        valid = col[col != ABSTAIN]
        if len(valid):
            pseudo_labels[i] = Counter(valid.tolist()).most_common(1)[0][0]
        else:
            pseudo_labels[i] = 0

    # Estimate per-LF conditional distributions
    lf_cond = np.zeros((n_lfs, cardinality, cardinality), dtype=np.float64)
    for j in range(n_lfs):
        for y in range(cardinality):
            mask = pseudo_labels == y
            if not np.any(mask):
                lf_cond[j, y] = np.ones(cardinality) / cardinality
                continue
            for v in range(cardinality):
                lf_cond[j, y, v] = np.mean((votes[j][mask] == v)) if mask.any() else 0.0

    # Compute posterior for each sample
    for i in range(n_samples):
        log_posterior = np.log(prior + 1e-10)
        for j in range(n_lfs):
            vote = votes[j, i]
            if vote == ABSTAIN:
                continue
            for y in range(cardinality):
                cond = lf_cond[j, y, vote]
                log_posterior[y] += np.log(cond + 1e-10)
        # Normalize
        posterior = np.exp(log_posterior - np.max(log_posterior))
        probs[i] = posterior / (posterior.sum() + 1e-10)

    return probs


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_binary_lfs() -> list[LabelingFunction]:
    """Create example binary classification labeling functions.

    Returns:
        List of three simple heuristic LFs.
    """
    return [
        LabelingFunction(
            name="lf_always_positive",
            fn=lambda _: 1,
        ),
        LabelingFunction(
            name="lf_random",
            fn=lambda _: np.random.choice([0, 1]),
        ),
        LabelingFunction(
            name="lf_length",
            fn=lambda x: 1 if (isinstance(x, str) and len(x) > 10) else ABSTAIN,
        ),
    ]