"""Multi-modal fusion layer (issue #631).

Provides fusion mechanisms for combining embeddings from different
modalities (text, image, tabular) into a unified representation.

Components:
- FusionMethod: Enum of available fusion approaches
- FusionLayer: Abstract base for fusion strategies
- ConcatenationFusion: Simple concatenation + MLP projection
- AttentionFusion: Cross-modal attention-based fusion
- GatedFusion: Learned gating of modality contributions
- CrossModalRetriever: Retrieves cross-modal associations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np

from astroml.training.multimodal.encoders import Modality

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FusionMethod(str, Enum):
    """Available multi-modal fusion strategies."""

    CONCAT = "concat"  # Concatenate → MLP
    ATTENTION = "attention"  # Cross-modal attention
    GATED = "gated"  # Learned gating
    SUM = "sum"  # Element-wise sum
    MEAN = "mean"  # Element-wise mean
    MAX = "max"  # Element-wise max


# ---------------------------------------------------------------------------
# Fusion layer base
# ---------------------------------------------------------------------------


class FusionLayer(ABC):
    """Abstract base for fusion strategies.

    Subclasses must implement :meth:`fuse` and :meth:`get_output_dim`.
    """

    @abstractmethod
    def fuse(
        self,
        embeddings: dict[Modality, np.ndarray],
    ) -> np.ndarray:
        """Fuse multi-modal embeddings into a unified representation.

        Args:
            embeddings: Dict mapping modality → (batch_size, embed_dim) array.

        Returns:
            Fused (batch_size, output_dim) array.
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the dimension of the fused output."""
        ...


# ---------------------------------------------------------------------------
# Simple fusions
# ---------------------------------------------------------------------------


class SumFusion(FusionLayer):
    """Element-wise sum of modality embeddings.

    All modalities must have the same embedding dimension.
    """

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        arrays = list(embeddings.values())
        if not arrays:
            raise ValueError("No embeddings to fuse")
        result = np.zeros_like(arrays[0])
        for a in arrays[1:]:
            result = result + a
        return result

    def get_output_dim(self) -> int:
        # Dimension is not fixed until first fuse call; default to 0
        return 0


class MeanFusion(FusionLayer):
    """Element-wise mean of modality embeddings."""

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        arrays = list(embeddings.values())
        if not arrays:
            raise ValueError("No embeddings to fuse")
        stacked = np.stack(arrays, axis=0)
        return np.mean(stacked, axis=0)

    def get_output_dim(self) -> int:
        return 0


class MaxFusion(FusionLayer):
    """Element-wise max pooling of modality embeddings."""

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        arrays = list(embeddings.values())
        if not arrays:
            raise ValueError("No embeddings to fuse")
        stacked = np.stack(arrays, axis=0)
        return np.max(stacked, axis=0)

    def get_output_dim(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Concatenation fusion
# ---------------------------------------------------------------------------


class ConcatenationFusion(FusionLayer):
    """Concatenate modality embeddings and project through an MLP.

    Args:
        output_dim: Final fused embedding dimension.
        hidden_dim: Hidden layer size for the projection MLP.
        activation: Non-linearity (``"relu"`` or ``"tanh"``).
    """

    def __init__(
        self,
        output_dim: int = 512,
        hidden_dim: int = 1024,
        activation: str = "relu",
    ) -> None:
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self._projection_w: np.ndarray | None = None
        self._projection_b: np.ndarray | None = None
        self._hidden_w: np.ndarray | None = None
        self._hidden_b: np.ndarray | None = None

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        arrays = list(embeddings.values())
        if not arrays:
            raise ValueError("No embeddings to fuse")

        batch_size = arrays[0].shape[0]
        concatenated = np.concatenate(arrays, axis=1)  # (B, total_dim)
        input_dim = concatenated.shape[1]

        # Lazy init
        if self._hidden_w is None:
            rng = np.random.default_rng(42)
            limit_h = np.sqrt(6.0 / (input_dim + self.hidden_dim))
            self._hidden_w = rng.uniform(-limit_h, limit_h, (input_dim, self.hidden_dim)).astype(
                np.float32
            )
            self._hidden_b = np.zeros(self.hidden_dim, dtype=np.float32)

            limit_out = np.sqrt(6.0 / (self.hidden_dim + self.output_dim))
            self._projection_w = rng.uniform(
                -limit_out, limit_out, (self.hidden_dim, self.output_dim)
            ).astype(np.float32)
            self._projection_b = np.zeros(self.output_dim, dtype=np.float32)

        # MLP: concat → hidden → output
        assert self._hidden_w is not None
        assert self._projection_w is not None
        x = concatenated @ self._hidden_w + self._hidden_b
        if self.activation == "relu":
            x = np.maximum(0, x)
        elif self.activation == "tanh":
            x = np.tanh(x)
        x = x @ self._projection_w + self._projection_b

        # L2 normalize
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norm + 1e-8)

    def get_output_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Attention fusion
# ---------------------------------------------------------------------------


class AttentionFusion(FusionLayer):
    """Cross-modal attention-based fusion.

    Computes pairwise attention between modality embeddings and aggregates
    the resulting attended representations.

    Args:
        output_dim: Final fused embedding dimension.
        num_heads: Number of attention heads.
        temperature: Softmax temperature for attention weights.
    """

    def __init__(
        self,
        output_dim: int = 512,
        num_heads: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.temperature = temperature

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        arrays = list(embeddings.values())
        if len(arrays) < 2:
            # Single modality → identity
            x = arrays[0] if arrays else np.zeros((1, self.output_dim))
            return _pad_or_project(x, self.output_dim)

        batch_size = arrays[0].shape[0]

        # Compute pairwise attention across modalities
        fused = np.zeros((batch_size, self.output_dim), dtype=np.float32)
        for arr in arrays:
            # Cross-attend each modality with every other
            attended = self._cross_attend(arr, arrays)
            fused = fused + _pad_or_project(attended, self.output_dim)

        # Normalize
        norm = np.linalg.norm(fused, axis=1, keepdims=True)
        return fused / (norm + 1e-8)

    def _cross_attend(
        self,
        query: np.ndarray,
        keys: list[np.ndarray],
    ) -> np.ndarray:
        """Cross-modal attention: Q attends over K matrices."""
        batch_size, dim = query.shape
        result = np.zeros_like(query)
        count = 0
        for key in keys:
            if key.shape[1] != dim:
                continue
            # Scaled dot-product attention
            scores = query @ key.T / (np.sqrt(dim) * self.temperature)  # (B, B)
            weights = _softmax(scores, axis=1)
            result = result + weights @ key
            count += 1
        return result / max(count, 1)

    def get_output_dim(self) -> int:
        return self.output_dim


# ---------------------------------------------------------------------------
# Gated fusion
# ---------------------------------------------------------------------------


class GatedFusion(FusionLayer):
    """Learned gating mechanism for modality contributions.

    Each modality receives a learned scalar gate weight that controls
    its contribution to the final fused representation.

    Args:
        output_dim: Final fused embedding dimension.
        initial_gates: Optional initial gate values per modality.
    """

    def __init__(
        self,
        output_dim: int = 512,
        initial_gates: dict[Modality, float] | None = None,
    ) -> None:
        self.output_dim = output_dim
        self._gates: dict[Modality, float] = initial_gates or {}

    def set_gate(self, modality: Modality, value: float) -> None:
        """Set the gate weight for a modality.

        Args:
            modality: Data modality.
            value: Gate weight (sigmoid range recommended: 0-1).
        """
        self._gates[modality] = value

    def fuse(self, embeddings: dict[Modality, np.ndarray]) -> np.ndarray:
        batch_size = next(iter(embeddings.values())).shape[0]
        fused = np.zeros((batch_size, self.output_dim), dtype=np.float32)

        for mod, emb in embeddings.items():
            gate = self._gates.get(mod, 1.0)
            projected = _pad_or_project(emb, self.output_dim)
            sigmoid_gate = 1.0 / (1.0 + np.exp(-gate))
            fused = fused + sigmoid_gate * projected

        # Learned gate softmax normalization
        if len(self._gates) > 1 and fused.any():
            norm = np.linalg.norm(fused, axis=1, keepdims=True)
            fused = fused / (norm + 1e-8)
        return fused

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_gates(self) -> dict[Modality, float]:
        """Return current gate values."""
        return dict(self._gates)


# ---------------------------------------------------------------------------
# Cross-modal retriever
# ---------------------------------------------------------------------------


class CrossModalRetriever:
    """Retrieves cross-modal associations between embeddings.

    Given a query in one modality, finds the most similar items in
    another modality's embedding space using cosine similarity.

    Args:
        fusion: The fusion layer used for multi-modal projection.
    """

    def __init__(self, fusion: FusionLayer | None = None) -> None:
        self.fusion = fusion
        self._indexes: dict[Modality, np.ndarray] = {}

    def index(self, modality: Modality, embeddings: np.ndarray) -> None:
        """Index embeddings for a modality.

        Args:
            modality: Modality to index.
            embeddings: (N, D) embedding array.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._indexes[modality] = embeddings / (norms + 1e-8)
        logger.info("Indexed %d %s embeddings", embeddings.shape[0], modality.value)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        target_modality: Modality,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-k most similar items from a target modality.

        Args:
            query_embedding: (D,) query vector in source modality space.
            target_modality: Modality to search in.
            top_k: Number of results to return.

        Returns:
            (indices, similarities) tuple.

        Raises:
            ValueError: If target modality not indexed.
        """
        if target_modality not in self._indexes:
            raise ValueError(f"Modality {target_modality.value} not indexed")

        index = self._indexes[target_modality]
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = index @ query_norm  # cosine similarity
        top_indices = np.argsort(-similarities)[:top_k]
        return top_indices, similarities[top_indices]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_fusion(
    method: FusionMethod,
    output_dim: int = 512,
    **kwargs: Any,
) -> FusionLayer:
    """Create a fusion layer by method name.

    Args:
        method: Fusion method identifier.
        output_dim: Desired output dimension.
        **kwargs: Forwarded to the fusion constructor.

    Returns:
        A concrete :class:`FusionLayer` instance.
    """
    registry: dict[FusionMethod, type[FusionLayer]] = {
        FusionMethod.CONCAT: ConcatenationFusion,
        FusionMethod.ATTENTION: AttentionFusion,
        FusionMethod.GATED: GatedFusion,
        FusionMethod.SUM: SumFusion,
        FusionMethod.MEAN: MeanFusion,
        FusionMethod.MAX: MaxFusion,
    }
    cls = registry.get(method)
    if cls is None:
        raise ValueError(f"Unknown fusion method: {method}")
    if method in (FusionMethod.CONCAT, FusionMethod.ATTENTION, FusionMethod.GATED):
        return cls(output_dim=output_dim, **kwargs)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-10)


def _pad_or_project(x: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or project an embedding array to target_dim."""
    src_dim = x.shape[1]
    if src_dim == target_dim:
        return x
    if src_dim < target_dim:
        pad = np.zeros((x.shape[0], target_dim - src_dim), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    # Linear projection: src_dim → target_dim (random init for simplicity)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1 / np.sqrt(src_dim), (src_dim, target_dim)).astype(np.float32)
    return x @ proj