"""Multi-modal training pipeline (issue #631).

Orchestrates end-to-end multi-modal training: encoding, fusion, and
downstream task optimization with support for cross-modal retrieval.

Components:
- MultiModalPipeline: Full training and inference pipeline
- MultiModalConfig: Pydantic-validated multi-modal configuration
- MultiModalDataBatch: Typed batch of multi-modal data
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from astroml.training.multimodal.encoders import (
    EncoderRegistry,
    ImageEncoder,
    ImageEncoderType,
    Modality,
    MultiModalEncoder,
    TabularEncoder,
    TextEncoder,
    TextEncoderType,
)
from astroml.training.multimodal.fusion import (
    CrossModalRetriever,
    FusionLayer,
    FusionMethod,
    create_fusion,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class MultiModalConfig(BaseModel):
    """Configuration for the multi-modal training pipeline.

    Attributes:
        enabled_modalities: Which modalities to include.
        fusion_method: How to combine modality embeddings.
        fusion_output_dim: Dimension of the fused embedding.
        text_encoder_type: Transformer architecture for text.
        image_encoder_type: CNN/ViT architecture for images.
        tabular_input_dim: Number of tabular features.
        tabular_hidden_dims: Hidden layer sizes for tabular encoder.
        cross_modal_retrieval: Enable cross-modal retrieval.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        epochs: Number of training epochs.
    """

    model_config = ConfigDict(extra="forbid")

    enabled_modalities: list[Modality] = Field(
        default=[Modality.TEXT, Modality.IMAGE, Modality.TABULAR],
        description="Modalities to include in training",
    )
    fusion_method: FusionMethod = Field(
        default=FusionMethod.CONCAT,
        description="How to fuse modality embeddings",
    )
    fusion_output_dim: int = Field(
        default=512,
        ge=32,
        description="Final fused embedding dimension",
    )
    text_encoder_type: TextEncoderType = Field(
        default=TextEncoderType.BERT,
        description="Text encoder architecture",
    )
    text_max_length: int = Field(default=512, ge=1, description="Max token length for text")
    image_encoder_type: ImageEncoderType = Field(
        default=ImageEncoderType.RESNET50,
        description="Image encoder architecture",
    )
    image_input_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Input image size (H, W)",
    )
    tabular_input_dim: int = Field(
        default=64,
        ge=1,
        description="Number of tabular input features",
    )
    tabular_hidden_dims: list[int] = Field(
        default=[256, 128],
        description="Tabular encoder hidden layer sizes",
    )
    tabular_embedding_dim: int = Field(
        default=256,
        ge=16,
        description="Tabular encoder output dimension",
    )
    cross_modal_retrieval: bool = Field(
        default=False,
        description="Enable cross-modal retrieval capabilities",
    )
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")
    epochs: int = Field(default=10, ge=1, description="Number of training epochs")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class MultiModalDataBatch:
    """A batch of multi-modal data.

    Attributes:
        text_inputs: List of text strings or tokenized arrays.
        image_inputs: Image array (B, C, H, W) or None.
        tabular_inputs: Tabular array (B, F) or None.
        labels: Optional label array (B,).
    """

    text_inputs: list[str] | None = None
    image_inputs: np.ndarray | None = None
    tabular_inputs: np.ndarray | None = None
    labels: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Multi-modal pipeline
# ---------------------------------------------------------------------------


class MultiModalPipeline:
    """End-to-end multi-modal training and inference pipeline.

    Args:
        config: Pipeline configuration.
    """

    def __init__(self, config: MultiModalConfig | None = None) -> None:
        self.config = config or MultiModalConfig()
        self.encoder_registry = EncoderRegistry()
        self.fusion_layer: FusionLayer | None = None
        self.retriever: CrossModalRetriever | None = None
        self._trained: bool = False
        self._loss_history: list[float] = []

        self._setup()

    def _setup(self) -> None:
        """Set up encoders and fusion layer based on configuration."""
        # Text encoder
        if Modality.TEXT in self.config.enabled_modalities:
            self.encoder_registry.register(
                TextEncoder(
                    encoder_type=self.config.text_encoder_type,
                    max_length=self.config.text_max_length,
                )
            )

        # Image encoder
        if Modality.IMAGE in self.config.enabled_modalities:
            self.encoder_registry.register(
                ImageEncoder(
                    encoder_type=self.config.image_encoder_type,
                    input_size=self.config.image_input_size,
                )
            )

        # Tabular encoder
        if Modality.TABULAR in self.config.enabled_modalities:
            self.encoder_registry.register(
                TabularEncoder(
                    input_dim=self.config.tabular_input_dim,
                    hidden_dims=self.config.tabular_hidden_dims,
                    embedding_dim=self.config.tabular_embedding_dim,
                )
            )

        # Fusion
        self.fusion_layer = create_fusion(
            method=self.config.fusion_method,
            output_dim=self.config.fusion_output_dim,
        )

        # Cross-modal retrieval
        if self.config.cross_modal_retrieval:
            self.retriever = CrossModalRetriever(self.fusion_layer)

    def train(
        self,
        data: list[MultiModalDataBatch],
        *,
        epochs: int | None = None,
    ) -> dict[str, Any]:
        """Train the multi-modal pipeline.

        Args:
            data: List of training batches.
            epochs: Number of epochs (defaults to config).

        Returns:
            Dict with training results.
        """
        n_epochs = epochs or self.config.epochs
        self._loss_history = []

        for epoch in range(n_epochs):
            epoch_losses: list[float] = []
            for batch in data:
                loss = self._train_step(batch)
                epoch_losses.append(loss)
            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            self._loss_history.append(avg_loss)
            if epoch % max(1, n_epochs // 5) == 0 or epoch == n_epochs - 1:
                logger.info("Epoch %d/%d: loss=%.4f", epoch + 1, n_epochs, avg_loss)

        self._trained = True
        return {
            "epochs_completed": n_epochs,
            "final_loss": self._loss_history[-1] if self._loss_history else None,
            "loss_history": self._loss_history,
            "fusion_method": self.config.fusion_method.value,
            "modalities": [m.value for m in self.config.enabled_modalities],
        }

    def _train_step(self, batch: MultiModalDataBatch) -> float:
        """Execute a single training step.

        Returns:
            Scalar loss value for this batch.
        """
        # Encode each modality
        embeddings: dict[Modality, np.ndarray] = {}
        if batch.text_inputs and Modality.TEXT in self.encoder_registry.list_modalities():
            embeddings[Modality.TEXT] = self.encoder_registry.encode(Modality.TEXT, batch.text_inputs)
        if batch.image_inputs is not None and Modality.IMAGE in self.encoder_registry.list_modalities():
            embeddings[Modality.IMAGE] = self.encoder_registry.encode(Modality.IMAGE, batch.image_inputs)
        if batch.tabular_inputs is not None and Modality.TABULAR in self.encoder_registry.list_modalities():
            embeddings[Modality.TABULAR] = self.encoder_registry.encode(
                Modality.TABULAR, batch.tabular_inputs
            )

        if not embeddings:
            return 0.0

        assert self.fusion_layer is not None
        fused = self.fusion_layer.fuse(embeddings)

        # Simple reconstruction loss (if labels available, use supervised loss)
        if batch.labels is not None:
            # Simulated classification loss
            probs = _classify(fused, int(batch.labels.max()) + 1)
            loss = -np.mean(np.log(probs[np.arange(len(batch.labels)), batch.labels.astype(int)] + 1e-10))
        else:
            # Unsupervised: reconstruction loss
            concat = np.concatenate(list(embeddings.values()), axis=1)
            fused_padded = _pad_or_project(fused, concat.shape[1])
            loss = float(np.mean((concat - fused_padded) ** 2))

        return max(0.0, loss)

    def infer(self, batch: MultiModalDataBatch) -> np.ndarray:
        """Run inference on a batch and return fused embeddings.

        Args:
            batch: Multi-modal data batch.

        Returns:
            (batch_size, fusion_output_dim) fused embedding array.
        """
        embeddings: dict[Modality, np.ndarray] = {}
        if batch.text_inputs and Modality.TEXT in self.encoder_registry.list_modalities():
            embeddings[Modality.TEXT] = self.encoder_registry.encode(Modality.TEXT, batch.text_inputs)
        if batch.image_inputs is not None and Modality.IMAGE in self.encoder_registry.list_modalities():
            embeddings[Modality.IMAGE] = self.encoder_registry.encode(Modality.IMAGE, batch.image_inputs)
        if batch.tabular_inputs is not None and Modality.TABULAR in self.encoder_registry.list_modalities():
            embeddings[Modality.TABULAR] = self.encoder_registry.encode(
                Modality.TABULAR, batch.tabular_inputs
            )

        assert self.fusion_layer is not None
        return self.fusion_layer.fuse(embeddings)

    def retrieve_cross_modal(
        self,
        query: np.ndarray,
        target_modality: Modality,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cross-modal retrieval: find nearest neighbors across modalities.

        Args:
            query: Query embedding vector (D,).
            target_modality: Modality to search.
            top_k: Number of results.

        Returns:
            (indices, similarities).

        Raises:
            RuntimeError: If cross-modal retrieval is not enabled.
        """
        if self.retriever is None:
            raise RuntimeError("Cross-modal retrieval not enabled in config")
        return self.retriever.retrieve(query, target_modality, top_k)

    def index_modality(self, modality: Modality, embeddings: np.ndarray) -> None:
        """Index embeddings for cross-modal retrieval.

        Args:
            modality: Modality to index.
            embeddings: (N, D) embedding array.

        Raises:
            RuntimeError: If cross-modal retrieval is not enabled.
        """
        if self.retriever is None:
            raise RuntimeError("Cross-modal retrieval not enabled in config")
        self.retriever.index(modality, embeddings)

    def get_embedding_dims(self) -> dict[Modality, int]:
        """Return per-modality embedding dimensions."""
        return self.encoder_registry.get_embedding_dims()

    def get_fusion_output_dim(self) -> int:
        """Return the fused embedding dimension."""
        return self.fusion_layer.get_output_dim() if self.fusion_layer else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify(embeddings: np.ndarray, num_classes: int) -> np.ndarray:
    """Simulated classifier head (random projection + softmax)."""
    rng = np.random.default_rng(99)
    w = rng.normal(0, 0.1, (embeddings.shape[1], num_classes)).astype(np.float32)
    logits = embeddings @ w
    return _softmax(logits)


def _softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / (e_x.sum(axis=1, keepdims=True) + 1e-10)


def _pad_or_project(x: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or project to target_dim."""
    src_dim = x.shape[1]
    if src_dim == target_dim:
        return x
    if src_dim < target_dim:
        pad = np.zeros((x.shape[0], target_dim - src_dim), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)
    rng = np.random.default_rng(42)
    proj = rng.normal(0, 1 / np.sqrt(src_dim), (src_dim, target_dim)).astype(np.float32)
    return x @ proj