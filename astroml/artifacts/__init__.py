"""Artifact storage management for models and results.

``ArtifactStore`` (fsspec/torch based) is imported lazily so that the
dependency-free run-artifact API in :mod:`astroml.artifacts.run_artifacts`
works without torch or fsspec installed.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ArtifactStore",
    "get_artifact_store",
    "set_artifact_store",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import store

        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
