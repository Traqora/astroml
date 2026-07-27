"""
LLM optimization for efficient model deployment on edge devices.

This module provides quantization, distillation, and compression techniques
to enable local execution and reduce API costs.
"""

from .quantizer import ModelQuantizer, QuantizationConfig, QuantizationType
from .distiller import KnowledgeDistiller, DistillationConfig
from .compressor import ModelCompressor, CompressionConfig
from .validator import QualityValidator, ValidationMetrics
from .registry import OptimizedModelRegistry

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
