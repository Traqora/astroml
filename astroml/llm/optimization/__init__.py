"""
LLM optimization for efficient model deployment on edge devices.

This module provides quantization, distillation, and compression techniques
to enable local execution and reduce API costs.
"""

from .compressor import CompressionConfig, ModelCompressor
from .distiller import DistillationConfig, KnowledgeDistiller
from .quantizer import ModelQuantizer, QuantizationConfig, QuantizationType
from .registry import OptimizedModelRegistry
from .validator import QualityValidator, ValidationMetrics

__all__ = [
    "ModelQuantizer",
    "QuantizationConfig",
    "QuantizationType",
    "KnowledgeDistiller",
    "DistillationConfig",
    "ModelCompressor",
    "CompressionConfig",
    "QualityValidator",
    "ValidationMetrics",
    "OptimizedModelRegistry",
]
