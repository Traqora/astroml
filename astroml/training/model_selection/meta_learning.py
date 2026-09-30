"""Meta-learning for model recommendations.

Describes datasets by a compact feature vector, stores past
(description -> winning model) experience, and recommends a model for a
new dataset by nearest-neighbor search over previous tasks with a
heuristic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskDescriptor:
    """Compact numerical description of a machine learning task.

    Attributes:
        n_samples: Number of samples.
        n_features: Number of features.
        n_classes: Number of classes (1 for regression-style tasks).
        imbalance_ratio: Ratio of the majority class to the minority class.
        feature_mean: Mean of the feature matrix.
        feature_std: Standard deviation of the feature matrix.
        sparsity: Fraction of zero entries in the feature matrix.
    """

    n_samples: int
    n_features: int
    n_classes: int
    imbalance_ratio: float
    feature_mean: float
    feature_std: float
    sparsity: float

    @classmethod
    def from_data(cls, X: np.ndarray, y: np.ndarray) -> "TaskDescriptor":
        """Build a descriptor from raw data.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            A :class:`TaskDescriptor` describing the dataset.

        Raises:
            ValueError: If the inputs are invalid.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) == 0 or len(X) != len(y):
            raise ValueError("X and y must be non-empty and have matching lengths")
        counts = np.bincount(y.astype(int))
        if len(counts) == 0:
            raise ValueError("y must contain at least one label")
        majority = float(counts.max())
        minority = float(counts[counts > 0].min())
        flat = X.astype(float)
        return cls(
            n_samples=int(len(X)),
            n_features=int(X.shape[1]),
            n_classes=int(len(counts)),
            imbalance_ratio=majority / minority if minority > 0 else 1.0,
            feature_mean=float(flat.mean()) if flat.size else 0.0,
            feature_std=float(flat.std()) if flat.size else 0.0,
            sparsity=float(np.count_nonzero(flat == 0) / flat.size) if flat.size else 0.0,
        )

    def to_vector(self) -> np.ndarray:
        """Convert the descriptor to a numeric vector for similarity.

        Returns:
            A 1D array of scaled descriptor features.
        """
        return np.array(
            [
                np.log1p(self.n_samples),
                np.log1p(self.n_features),
                self.n_classes,
                self.imbalance_ratio,
                self.feature_mean,
                self.feature_std,
                self.sparsity,
            ],
            dtype=float,
        )


@dataclass
class ExperienceRecord:
    """A past model selection experience.

    Attributes:
        descriptor: Description of the past task.
        model_name: The model that performed best on that task.
        score: The score achieved by that model.
    """

    descriptor: TaskDescriptor
    model_name: str
    score: float


class MetaLearningRecommender:
    """Recommend models using similarity to previously solved tasks."""

    def __init__(self) -> None:
        """Initialize an empty experience database."""
        self.experiences: list[ExperienceRecord] = []

    def describe(self, X: np.ndarray, y: np.ndarray) -> TaskDescriptor:
        """Build a task descriptor for a dataset.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            A :class:`TaskDescriptor` describing the dataset.
        """
        return TaskDescriptor.from_data(X, y)

    def add_experience(self, X: np.ndarray, y: np.ndarray, model_name: str, score: float) -> None:
        """Record a solved task and its winning model.

        Args:
            X: Feature matrix of the solved task.
            y: Target labels of the solved task.
            model_name: Name of the winning model.
            score: Score achieved by the winning model.
        """
        descriptor = TaskDescriptor.from_data(X, y)
        self.experiences.append(
            ExperienceRecord(descriptor=descriptor, model_name=model_name, score=score)
        )

    def recommend(self, X: np.ndarray, y: np.ndarray) -> tuple[str, float, float]:
        """Recommend a model for a new dataset.

        Args:
            X: Feature matrix of the new task.
            y: Target labels of the new task.

        Returns:
            A tuple of (model name, expected score, confidence in ``[0, 1]``).

        Raises:
            ValueError: If the data is invalid.
        """
        descriptor = TaskDescriptor.from_data(X, y)
        if not self.experiences:
            return (*self._heuristic(descriptor), 0.0)
        best_record, best_similarity = self._most_similar(descriptor)
        confidence = float(np.clip(best_similarity, 0.0, 1.0))
        return best_record.model_name, best_record.score, confidence

    def similarity(self, a: TaskDescriptor, b: TaskDescriptor) -> float:
        """Compute cosine similarity between two task descriptors.

        Args:
            a: First descriptor.
            b: Second descriptor.

        Returns:
            Cosine similarity in ``[0, 1]`` (clipped at zero).
        """
        vec_a = a.to_vector()
        vec_b = b.to_vector()
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.clip(np.dot(vec_a, vec_b) / (norm_a * norm_b), 0.0, 1.0))

    def clear(self) -> None:
        """Clear the experience database."""
        self.experiences.clear()

    def _most_similar(self, descriptor: TaskDescriptor) -> tuple[ExperienceRecord, float]:
        """Find the most similar past experience.

        Args:
            descriptor: The new task descriptor.

        Returns:
            The most similar record and its similarity score.
        """
        ranked = sorted(
            self.experiences,
            key=lambda record: self.similarity(descriptor, record.descriptor),
            reverse=True,
        )
        best = ranked[0]
        return best, self.similarity(descriptor, best.descriptor)

    def _heuristic(self, descriptor: TaskDescriptor) -> tuple[str, float]:
        """Recommend a model using dataset-size heuristics.

        Args:
            descriptor: The task descriptor.

        Returns:
            A tuple of (model name, expected score).
        """
        if descriptor.n_samples < 500 or descriptor.n_features <= 5:
            return "logistic_regression", 0.7
        if descriptor.n_samples < 5000:
            return "random_forest", 0.8
        return "gradient_boosting", 0.85
