"""Validation modules for AstroML.

Expose validation submodules without eagerly importing the entire validation
stack at package import time. This keeps focused unit tests, such as the
deduplication tests, isolated from unrelated optional dependencies and import-
time failures in other validation modules.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "api_validation",
    "calibration",
    "data_quality",
    "dedupe",
    "hashing",
    "integrity",
    "leakage",
    "pipeline",
    "validator",
]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
