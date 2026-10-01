from typing import Any, Dict, List, Optional, Union, Callable
"""Model optimization with ONNX Runtime."""

from importlib import import_module

_LAZY = {
    "ONNXConverter": ("astroml.training.optimization.onnx_converter", "ONNXConverter"),
    "ONNXOptimizer": ("astroml.training.optimization.onnx_optimizer", "ONNXOptimizer"),
    "QuantizationConfig": ("astroml.training.optimization.quantization", "QuantizationConfig"),
}

__all__ = [
    "ONNXConverter",
    "ONNXOptimizer",
    "QuantizationConfig",
]


def __getattr__(name -> Any: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
