"""Multi-modal encoders for text, image, and tabular data (issue #631).

Provides modality-specific encoders that produce fixed-length embeddings:
- TextEncoder: BERT/RoBERTa-style transformer text encoding
- ImageEncoder: ResNet/ViT-style image encoding
- TabularEncoder: MLP-based tabular feature encoding
- EncoderRegistry: Centralized encoder management and dispatch

All encoders conform to a common interface returning (batch_size, embed_dim) tensors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Modality(str, Enum):
    """Supported data modalities."""

    TEXT = "text"
    IMAGE = "image"
    TABULAR = "tabular"
    AUDIO = "audio"
    VIDEO = "video"


class TextEncoderType(str, Enum):
    """Available text encoder architectures."""

    BERT = "bert"
    ROBERTA = "roberta"
    DISTILBERT = "distilbert"
    CUSTOM = "custom"


class ImageEncoderType(str, Enum):
    """Available image encoder architectures."""

    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    VIT_BASE = "vit_base"
    EFFICIENTNET_B0 = "efficientnet_b0"


# ---------------------------------------------------------------------------
# Base encoder
# ---------------------------------------------------------------------------


class MultiModalEncoder(ABC):
    """Abstract base for all modality-specific encoders.

    Subclasses must implement :meth:`encode` and :meth:`get_embedding_dim`.
    """

    @abstractmethod
    def encode(self, inputs: Any) -> np.ndarray:
        """Encode raw inputs into a (batch_size, embed_dim) embedding matrix.

        Args:
            inputs: Raw inputs in the encoder's native format.

        Returns:
            Float32 array of shape (batch_size, embed_dim).
        """
        ...

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        ...

    @property
    @abstractmethod
    def modality(self) -> Modality:
        """Return the modality this encoder handles."""
        ...

    def preprocess(self, inputs: Any) -> Any:
        """Optional pre-processing hook (default: identity)."""
        return inputs


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------


class TextEncoder(MultiModalEncoder):
    """Transformer-based text encoder (BERT, RoBERTa, DistilBERT).

    Encodes text sequences into fixed-length embedding vectors suitable
    for downstream multi-modal fusion.

    Args:
        encoder_type: Transformer architecture variant.
        max_length: Maximum token length (truncation + padding).
        pooling: Pooling strategy (``"cls"``, ``"mean"``, ``"max"``).
        embedding_dim: Output embedding dimension (auto-configured per type).
    """

    _DIMS: dict[TextEncoderType, int] = {
        TextEncoderType.BERT: 768,
        TextEncoderType.ROBERTA: 768,
        TextEncoderType.DISTILBERT: 768,
        TextEncoderType.CUSTOM: 512,
    }

    def __init__(
        self,
        encoder_type: TextEncoderType = TextEncoderType.BERT,
        max_length: int = 512,
        pooling: str = "cls",
        embedding_dim: int | None = None,
    ) -> None:
        self.encoder_type = encoder_type
        self.max_length = max_length
        self.pooling = pooling
        self._embedding_dim = embedding_dim or self._DIMS.get(encoder_type, 768)

    @property
    def modality(self) -> Modality:
        return Modality.TEXT

    def encode(self, inputs: list[str] | np.ndarray) -> np.ndarray:
        """Encode text batch into embeddings.

        Args:
            inputs: List of strings or pre-tokenized array.

        Returns:
            (batch_size, embed_dim) float32 array.
        """
        batch_size = len(inputs) if isinstance(inputs, list) else inputs.shape[0]

        # In production this would call a HuggingFace model; here we
        # simulate with a deterministic projection based on text features.
        embeddings = np.zeros((batch_size, self._embedding_dim), dtype=np.float32)
        for i, text in enumerate(inputs):
            text_str = text if isinstance(text, str) else str(text)
            # Simple feature extraction: length, avg char value, etc.
            text_len = min(len(text_str), self.max_length)
            seed = hash(text_str) % (2**31)
            rng = np.random.default_rng(seed)
            # Produce a deterministic pseudo-embedding
            vec = rng.normal(0, 1, self._embedding_dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)  # unit normalize
            embeddings[i] = vec

        return embeddings

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Image encoder
# ---------------------------------------------------------------------------


class ImageEncoder(MultiModalEncoder):
    """CNN/ViT-based image encoder (ResNet, ViT, EfficientNet).

    Encodes images into fixed-length embedding vectors.

    Args:
        encoder_type: Image architecture variant.
        input_size: Expected input image size (H, W).
        embedding_dim: Output embedding dimension.
        pretrained: Whether to use pre-trained weights.
    """

    _DIMS: dict[ImageEncoderType, int] = {
        ImageEncoderType.RESNET18: 512,
        ImageEncoderType.RESNET50: 2048,
        ImageEncoderType.VIT_BASE: 768,
        ImageEncoderType.EFFICIENTNET_B0: 1280,
    }

    def __init__(
        self,
        encoder_type: ImageEncoderType = ImageEncoderType.RESNET50,
        input_size: tuple[int, int] = (224, 224),
        embedding_dim: int | None = None,
        pretrained: bool = True,
    ) -> None:
        self.encoder_type = encoder_type
        self.input_size = input_size
        self._embedding_dim = embedding_dim or self._DIMS.get(encoder_type, 2048)
        self.pretrained = pretrained

    @property
    def modality(self) -> Modality:
        return Modality.IMAGE

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode image batch into embeddings.

        Args:
            inputs: (batch_size, C, H, W) or (batch_size, H, W, C) array.

        Returns:
            (batch_size, embed_dim) float32 array.
        """
        if inputs.ndim == 3:
            inputs = inputs[np.newaxis, ...]
        batch_size = inputs.shape[0]

        # Simulate image encoding: use image statistics as pseudo-features
        embeddings = np.zeros((batch_size, self._embedding_dim), dtype=np.float32)
        for i in range(batch_size):
            img = inputs[i].astype(np.float32)
            # Derive a deterministic embedding from image stats
            img_flat = img.ravel()
            mean_val = float(np.mean(img_flat)) if img_flat.size else 0.0
            std_val = float(np.std(img_flat)) if img_flat.size else 0.0
            seed = int(abs(mean_val * 1000 + std_val * 100)) % (2**31)
            rng = np.random.default_rng(seed)
            vec = rng.normal(0, 1, self._embedding_dim).astype(np.float32)
            vec = (vec - vec.mean()) / (vec.std() + 1e-8)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings[i] = vec

        return embeddings

    def get_embedding_dim(self) -> int:
        return self._embedding_dim

    def preprocess(self, inputs: np.ndarray) -> np.ndarray:
        """Resize and normalize image inputs."""
        # Placeholder: in production this would use torchvision transforms
        return inputs


# ---------------------------------------------------------------------------
# Tabular encoder
# ---------------------------------------------------------------------------


class TabularEncoder(MultiModalEncoder):
    """MLP-based tabular data encoder.

    Encodes structured/relational features into fixed-length embeddings.

    Args:
        input_dim: Number of input features.
        hidden_dims: Hidden layer sizes for the MLP.
        embedding_dim: Output embedding dimension.
        dropout: Dropout rate for regularization.
        normalize: Whether to apply input normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        embedding_dim: int = 256,
        dropout: float = 0.1,
        normalize: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self._embedding_dim = embedding_dim
        self.dropout = dropout
        self.normalize = normalize
        self._weights: list[np.ndarray] | None = None

    @property
    def modality(self) -> Modality:
        return Modality.TABULAR

    def _init_weights(self) -> None:
        """Lazily initialize MLP weights."""
        if self._weights is not None:
            return
        rng = np.random.default_rng(42)
        dims = [self.input_dim] + self.hidden_dims + [self._embedding_dim]
        self._weights = []
        for i in range(len(dims) - 1):
            # Xavier initialization
            limit = np.sqrt(6.0 / (dims[i] + dims[i + 1]))
            w = rng.uniform(-limit, limit, (dims[i], dims[i + 1])).astype(np.float32)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            self._weights.append((w, b))

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Encode tabular batch into embeddings.

        Args:
            inputs: (batch_size, input_dim) float32 array.

        Returns:
            (batch_size, embed_dim) float32 array.
        """
        self._init_weights()
        assert self._weights is not None

        x = inputs.astype(np.float32)
        if self.normalize:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True) + 1e-8
            x = (x - mean) / std

        # Forward through MLP
        for idx, (w, b) in enumerate(self._weights):
            x = x @ w + b
            if idx < len(self._weights) - 1:  # hidden layers only
                x = np.maximum(0, x)  # ReLU
                # Dropout during training
                if self.dropout > 0:
                    mask = np.random.binomial(1, 1.0 - self.dropout, size=x.shape).astype(np.float32)
                    x = x * mask / (1.0 - self.dropout)

        # Final L2 normalize
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / (norms + 1e-8)

    def get_embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Encoder registry
# ---------------------------------------------------------------------------


class EncoderRegistry:
    """Centralized registry for multi-modal encoders.

    Manages encoder lookup by modality and provides convenience methods
    for encoding mixed-modal batches.

    Args:
        encoders: Optional initial mapping of Modality → encoder.
    """

    def __init__(
        self,
        encoders: dict[Modality, MultiModalEncoder] | None = None,
    ) -> None:
        self._encoders: dict[Modality, MultiModalEncoder] = encoders or {}

    def register(self, encoder: MultiModalEncoder) -> None:
        """Register an encoder for its modality.

        Args:
            encoder: A concrete :class:`MultiModalEncoder`.
        """
        self._encoders[encoder.modality] = encoder
        logger.info(
            "Registered %s encoder (dim=%d)",
            encoder.modality.value,
            encoder.get_embedding_dim(),
        )

    def get_encoder(self, modality: Modality) -> MultiModalEncoder | None:
        """Return the encoder for a given modality, or None."""
        return self._encoders.get(modality)

    def encode(
        self,
        modality: Modality,
        inputs: Any,
    ) -> np.ndarray:
        """Encode inputs for a specific modality.

        Args:
            modality: Data modality.
            inputs: Raw inputs.

        Returns:
            (batch_size, embed_dim) embedding array.

        Raises:
            ValueError: If no encoder is registered for the modality.
        """
        encoder = self._encoders.get(modality)
        if encoder is None:
            raise ValueError(f"No encoder registered for modality: {modality.value}")
        return encoder.encode(inputs)

    def encode_batch(
        self,
        batch: dict[Modality, Any],
    ) -> dict[Modality, np.ndarray]:
        """Encode a batch of mixed-modal inputs.

        Args:
            batch: Dict mapping modality → raw inputs.

        Returns:
            Dict mapping modality → embeddings.
        """
        return {mod: self.encode(mod, data) for mod, data in batch.items()}

    def list_modalities(self) -> list[Modality]:
        """Return all registered modalities."""
        return list(self._encoders.keys())

    def get_embedding_dims(self) -> dict[Modality, int]:
        """Return the output dimension for each registered modality."""
        return {mod: enc.get_embedding_dim() for mod, enc in self._encoders.items()}