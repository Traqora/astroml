"""Runtime environment snapshot for run metadata (issue #1217).

A model's numbers depend on the versions of the libraries that produced them: a
different ``torch`` or ``torch-geometric`` release can change initialisation, kernel
behaviour or default arguments. A run that records only its metrics cannot be
reproduced, or even compared fairly, once the environment has moved on.

:func:`collect_environment` returns the interpreter, platform and installed
versions of the packages the v1 training path depends on, as plain JSON-safe
values. Versions are read from installed package metadata
(:mod:`importlib.metadata`), so no package is imported to ask its version: a missing
or broken optional library cannot make recording the environment fail, and the
snapshot works wherever the run did.
"""

from __future__ import annotations

import platform
from importlib import metadata
from typing import Any, Iterable

__all__ = ["TRACKED_PACKAGES", "collect_environment", "package_versions"]

#: Distribution names recorded by default: the training stack and the libraries
#: whose behaviour most directly changes a model's numbers or its persisted form.
TRACKED_PACKAGES: tuple[str, ...] = (
    "astroml",
    "torch",
    "torch-geometric",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "networkx",
    "sqlalchemy",
    "mlflow",
)


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed version of each named distribution.

    Args:
        names: Distribution names, as ``pip`` knows them (``torch-geometric``, not
            ``torch_geometric``).

    Returns:
        A mapping from each name to its version string, or ``None`` when the
        distribution is not installed. The mapping keeps the order of ``names``.
    """
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_environment(packages: Iterable[str] = TRACKED_PACKAGES) -> dict[str, Any]:
    """Snapshot the interpreter, platform and package versions for run metadata.

    Args:
        packages: Distribution names to record; defaults to :data:`TRACKED_PACKAGES`.

    Returns:
        A JSON-serialisable dict with ``python``, ``implementation``, ``platform``
        and ``packages`` (name to version, ``None`` for a package that is not
        installed).
    """
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": package_versions(packages),
    }
