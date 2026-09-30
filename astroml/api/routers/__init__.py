from typing import Any, Dict, List, Optional, Union, Callable
"""API routers for AstroML services."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "accounts",
    "compression",
    "data_quality",
    "feature_selection",
    "features",
    "federated",
    "fraud",
    "model_registry",
    "validation",
]


def __getattr__(name -> Any: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
