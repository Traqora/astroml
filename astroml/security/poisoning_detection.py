"""Training-data poisoning detection.

Resolves part of #645.

Detects the three poisoning families that matter for a fraud model trained on
partly-adversary-controlled data:

* **Label flipping** — samples whose label disagrees with their neighbourhood.
* **Outlier / gradient-shaping injection** — samples far from every class
  centroid in feature space.
* **Backdoor triggers** — a rare, near-constant feature pattern that appears
  almost exclusively alongside one label.

Every detector returns per-sample suspicion scores plus a boolean mask, so
callers can drop, quarantine or re-label the flagged rows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PoisoningDetector",
    "PoisoningFinding",
    "PoisoningReport",
    "PoisoningType",
]


class PoisoningType(str, Enum):
    """Poisoning families this module can detect."""

    LABEL_FLIP = "label_flip"
    OUTLIER_INJECTION = "outlier_injection"
    BACKDOOR_TRIGGER = "backdoor_trigger"


@dataclass(frozen=True)
class PoisoningFinding:
    """Detection output for one poisoning family."""

    poisoning_type: PoisoningType
    suspicious_indices: tuple[int, ...]
    scores: NDArray[np.float64]
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def suspicious_count(self) -> int:
        """Number of samples flagged by this detector."""
        return len(self.suspicious_indices)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (scores are omitted)."""
        return {
            "poisoning_type": self.poisoning_type.value,
            "suspicious_count": self.suspicious_count,
            "suspicious_indices": list(self.suspicious_indices),
            "threshold": self.threshold,
            "max_score": float(self.scores.max()) if self.scores.size else 0.0,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PoisoningReport:
    """Combined findings across every detector."""

    sample_count: int
    findings: tuple[PoisoningFinding, ...]

    @property
    def suspicious_indices(self) -> tuple[int, ...]:
        """Union of the indices flagged by any detector, ascending."""
        union: set[int] = set()
        for finding in self.findings:
            union.update(finding.suspicious_indices)
        return tuple(sorted(union))

    @property
    def contamination_rate(self) -> float:
        """Share of the training set flagged by at least one detector."""
        if self.sample_count == 0:
            return 0.0
        return len(self.suspicious_indices) / self.sample_count

    @property
    def is_clean(self) -> bool:
        """Whether no detector flagged anything."""
        return not self.suspicious_indices

    def clean_mask(self) -> NDArray[np.bool_]:
        """Return a boolean mask selecting the unflagged samples."""
        mask = np.ones(self.sample_count, dtype=bool)
        for index in self.suspicious_indices:
            mask[index] = False
        return mask

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "sample_count": self.sample_count,
            "suspicious_count": len(self.suspicious_indices),
            "contamination_rate": self.contamination_rate,
            "is_clean": self.is_clean,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class PoisoningDetector:
    """Screens a training set for poisoned samples.

    Parameters
    ----------
    n_neighbors:
        Neighbourhood size used by the label-flip detector.
    label_flip_threshold:
        Share of disagreeing neighbours above which a label is suspicious.
    outlier_z_threshold:
        Robust z-score above which a sample counts as an injected outlier.
    backdoor_correlation_threshold:
        Label-correlation above which a rare feature pattern is called a
        backdoor trigger.
    max_pairwise_samples:
        Cap on samples used for the O(n^2) neighbour search; larger sets are
        subsampled so screening stays tractable.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 5,
        label_flip_threshold: float = 0.8,
        outlier_z_threshold: float = 4.0,
        backdoor_correlation_threshold: float = 0.9,
        backdoor_max_prevalence: float = 0.15,
        max_pairwise_samples: int = 5_000,
        seed: int | None = 0,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if not 0.0 < label_flip_threshold <= 1.0:
            raise ValueError("label_flip_threshold must be within (0, 1]")
        if outlier_z_threshold <= 0:
            raise ValueError("outlier_z_threshold must be positive")
        self.n_neighbors = n_neighbors
        self.label_flip_threshold = label_flip_threshold
        self.outlier_z_threshold = outlier_z_threshold
        self.backdoor_correlation_threshold = backdoor_correlation_threshold
        self.backdoor_max_prevalence = backdoor_max_prevalence
        self.max_pairwise_samples = max_pairwise_samples
        self.seed = seed

    # ── Individual detectors ─────────────────────────────────────────────────

    def detect_label_flips(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> PoisoningFinding:
        """Flag samples whose label disagrees with their nearest neighbours."""
        features = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y)
        n_samples = features.shape[0]
        scores = np.zeros(n_samples)

        if n_samples > self.n_neighbors:
            indices = self._subsample(n_samples)
            reference = features[indices]
            reference_labels = labels[indices]
            k = min(self.n_neighbors, reference.shape[0] - 1)
            for i in range(n_samples):
                distances = np.linalg.norm(reference - features[i], axis=1)
                # Exclude the sample itself when it is in the reference set.
                order = np.argsort(distances)
                neighbours = [j for j in order if not np.isclose(distances[j], 0.0)][:k]
                if not neighbours:
                    continue
                disagreement = (reference_labels[neighbours] != labels[i]).mean()
                scores[i] = float(disagreement)

        suspicious = np.flatnonzero(scores >= self.label_flip_threshold)
        return PoisoningFinding(
            poisoning_type=PoisoningType.LABEL_FLIP,
            suspicious_indices=tuple(int(i) for i in suspicious),
            scores=scores,
            threshold=self.label_flip_threshold,
            details={"n_neighbors": self.n_neighbors},
        )

    def detect_outliers(
        self, x: NDArray[np.float64], y: NDArray[np.int_] | None = None
    ) -> PoisoningFinding:
        """Flag samples far from their class centroid using a robust z-score.

        Uses median and median-absolute-deviation rather than mean and standard
        deviation so that the injected points cannot mask themselves by
        inflating the spread they are measured against.
        """
        features = np.asarray(x, dtype=np.float64)
        labels = None if y is None else np.asarray(y)
        scores = np.zeros(features.shape[0])

        groups = (
            [(None, np.arange(features.shape[0]))]
            if labels is None
            else [(label, np.flatnonzero(labels == label)) for label in np.unique(labels)]
        )
        for _, indices in groups:
            if indices.size < 3:
                continue
            block = features[indices]
            median = np.median(block, axis=0)
            deviation = np.abs(block - median)
            mad = np.median(deviation, axis=0)
            # 1.4826 rescales MAD to a standard-deviation-equivalent for normal data.
            scale = np.where(mad > 1e-12, mad * 1.4826, 1.0)
            scores[indices] = np.max(deviation / scale, axis=1)

        suspicious = np.flatnonzero(scores >= self.outlier_z_threshold)
        return PoisoningFinding(
            poisoning_type=PoisoningType.OUTLIER_INJECTION,
            suspicious_indices=tuple(int(i) for i in suspicious),
            scores=scores,
            threshold=self.outlier_z_threshold,
            details={"grouped_by_label": labels is not None},
        )

    def detect_backdoor_triggers(
        self, x: NDArray[np.float64], y: NDArray[np.int_]
    ) -> PoisoningFinding:
        """Flag rare feature values that co-occur almost exclusively with one label.

        A backdoor trigger is, by construction, an uncommon and near-constant
        pattern present in the poisoned rows and absent elsewhere — so it shows
        up as a rare value with an extreme label correlation.
        """
        features = np.asarray(x, dtype=np.float64)
        labels = np.asarray(y)
        n_samples, n_features = features.shape
        scores = np.zeros(n_samples)
        triggers: list[dict[str, Any]] = []

        for feature_index in range(n_features):
            column = features[:, feature_index]
            counts = Counter(np.round(column, 6).tolist())
            for value, count in counts.items():
                prevalence = count / n_samples
                if count < 2 or prevalence > self.backdoor_max_prevalence:
                    continue
                mask = np.isclose(column, value)
                subset_labels = labels[mask]
                dominant = Counter(subset_labels.tolist()).most_common(1)[0]
                correlation = dominant[1] / count
                baseline = float((labels == dominant[0]).mean())
                if (
                    correlation >= self.backdoor_correlation_threshold
                    and correlation > baseline + 0.2
                ):
                    scores[mask] = np.maximum(scores[mask], correlation)
                    triggers.append(
                        {
                            "feature_index": feature_index,
                            "value": float(value),
                            "prevalence": prevalence,
                            "target_label": int(dominant[0]),
                            "correlation": correlation,
                        }
                    )

        suspicious = np.flatnonzero(scores >= self.backdoor_correlation_threshold)
        return PoisoningFinding(
            poisoning_type=PoisoningType.BACKDOOR_TRIGGER,
            suspicious_indices=tuple(int(i) for i in suspicious),
            scores=scores,
            threshold=self.backdoor_correlation_threshold,
            details={"triggers": triggers},
        )

    # ── Combined screening ───────────────────────────────────────────────────

    def detect(self, x: NDArray[np.float64], y: NDArray[np.int_]) -> PoisoningReport:
        """Run every detector and return a combined :class:`PoisoningReport`."""
        features = np.asarray(x, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("x must be a 2-D array of shape (n_samples, n_features)")
        labels = np.asarray(y)
        if labels.shape[0] != features.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        return PoisoningReport(
            sample_count=features.shape[0],
            findings=(
                self.detect_label_flips(features, labels),
                self.detect_outliers(features, labels),
                self.detect_backdoor_triggers(features, labels),
            ),
        )

    def sanitize(
        self, x: NDArray[np.float64], y: NDArray[np.int_]
    ) -> tuple[NDArray[np.float64], NDArray[np.int_], PoisoningReport]:
        """Return ``x``/``y`` with flagged samples removed, plus the report."""
        report = self.detect(x, y)
        mask = report.clean_mask()
        return np.asarray(x, dtype=np.float64)[mask], np.asarray(y)[mask], report

    def _subsample(self, n_samples: int) -> NDArray[np.int_]:
        """Return indices of the reference set used for neighbour searches."""
        if n_samples <= self.max_pairwise_samples:
            return np.arange(n_samples)
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(n_samples, size=self.max_pairwise_samples, replace=False))
