"""Packaged documentation templates.

Resolves part of #646.

Templates are plain :class:`string.Template` documents so rendering has no
third-party dependency.  ``load_template`` reads them from this package
directory (which is installed as package data), and ``available_templates``
lists what is on offer.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

__all__ = ["TEMPLATE_DIR", "available_templates", "load_template"]

#: Directory holding the packaged ``*.tmpl`` files.
TEMPLATE_DIR = Path(__file__).resolve().parent


def available_templates() -> list[str]:
    """Return the names of every packaged template."""
    return sorted(path.name for path in TEMPLATE_DIR.glob("*.tmpl"))


@lru_cache(maxsize=None)
def load_template(name: str) -> str:
    """Return the contents of the named template.

    Raises ``FileNotFoundError`` when the template does not exist, and
    ``ValueError`` for names that try to escape the template directory.
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid template name {name!r}")
    path = TEMPLATE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"template {name!r} not found; available: {available_templates()}")
    return path.read_text(encoding="utf-8")
