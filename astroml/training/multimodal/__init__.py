"""Multi-modal model support (issue #631).

Provides encoders for text (BERT/RoBERTa), image (ResNet/ViT), and
tabular data, along with fusion strategies and a complete training
pipeline with cross-modal retrieval.

Components:
- Encoders: TextEncoder, ImageEncoder, TabularEncoder, EncoderRegistry
- Fusion: Concatenation, Attention, Gated, and simple pooling fusions
- Pipeline: End-to-end multi-modal training and inference
- CrossModalRetriever: Hero search across modalities
"""

from __future__ import annotations

from .encoders import (
    EncoderRegistry,
    ImageEncoder,
    ImageEncoderType,
    Modality,
    MultiModalEncoder,
    TabularEncoder,
    TextEncoder,
    TextEncoderType,
)
from .fusion import (
    AttentionFusion,
    ConcatenationFusion,
    CrossModalRetriever,
    FusionLayer,
    FusionMethod,
    GatedFusion,
    MaxFusion,
    MeanFusion,
    SumFusion,
    create_fusion,
)
from .pipeline import MultiModalConfig, MultiModalDataBatch, MultiModalPipeline

__all__ = [
    # Encoders
    "EncoderRegistry",
    "ImageEncoder",
    "ImageEncoderType",
    "Modality",
    "MultiModalEncoder",
    "TabularEncoder",
    "TextEncoder",
    "TextEncoderType",
    # Fusion
    "AttentionFusion",
    "ConcatenationFusion",
    "CrossModalRetriever",
    "FusionLayer",
    "FusionMethod",
    "GatedFusion",
    "MaxFusion",
    "MeanFusion",
    "SumFusion",
    "create_fusion",
    # Pipeline
    "MultiModalConfig",
    "MultiModalDataBatch",
    "MultiModalPipeline",
]