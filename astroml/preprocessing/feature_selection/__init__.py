from typing import Any, Dict, List, Optional, Union, Callable
"""Automated feature selection with filter, wrapper, embedded and hybrid methods."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "FilterSelector",
    "WrapperSelector",
    "EmbeddedSelector",
    "HybridSelector",
    "FeatureSelectionPipeline",
    "SelectionResult",
]

_LAZY: dict[str, tuple[str, str]] = {
    "FilterSelector": (
        "astroml.preprocessing.feature_selection.filter",
        "FilterSelector",
    ),
    "WrapperSelector": (
        "astroml.preprocessing.feature_selection.wrapper",
        "WrapperSelector",
    ),
    "EmbeddedSelector": (
        "astroml.preprocessing.feature_selection.embedded",
        "EmbeddedSelector",
    ),
    "HybridSelector": (
        "astroml.preprocessing.feature_selection.hybrid",
        "HybridSelector",
    ),
    "FeatureSelectionPipeline": (
        "astroml.preprocessing.feature_selection.hybrid",
        "FeatureSelectionPipeline",
    ),
    "SelectionResult": (
        "astroml.preprocessing.feature_selection.filter",
        "SelectionResult",
    ),
}


def __getattr__(name -> Any: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")