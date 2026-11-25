from __future__ import annotations

import os
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any


ArtifactLoader = Callable[[Path], Any]


class ModelLoader:
    """Load a trusted model artifact once and cache it by name and path."""

    def __init__(self, artifact_loader: ArtifactLoader | None = None) -> None:
        self.models: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self._artifact_loader = artifact_loader or self._load_pickle

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        with path.open("rb") as artifact:
            return pickle.load(artifact)

    def load_model(self, name: str, path: str | os.PathLike[str]) -> Any:
        """Load and cache a model artifact, returning the cached object on repeat calls.

        Artifacts must be trusted because the default pickle loader can execute code.
        Supply ``artifact_loader`` for another model format or safer deserializer.
        """
        if not name or not name.strip():
            raise ValueError("model name must not be empty")
        artifact_path = Path(path).expanduser()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Model artifact must be an existing file: {artifact_path}")
        if not os.access(artifact_path, os.R_OK):
            raise PermissionError(f"Model artifact is not readable: {artifact_path}")

        resolved_path = artifact_path.resolve()
        if name in self.models:
            if self._paths[name] != resolved_path:
                raise ValueError(f"model {name!r} is already loaded from a different path")
            return self.models[name]

        model = self._artifact_loader(resolved_path)
        self.models[name] = model
        self._paths[name] = resolved_path
        return model

    def get_model(self, name: str) -> Any | None:
        return self.models.get(name)
