"""Test-local import shim for astroml.features.graph.snapshot — issue #949.

`astroml.features.graph.snapshot` imports `astroml.cache`, whose
`__init__.py` eagerly imports `astroml.cache.graph_cache`, which currently
has a genuine pre-existing SyntaxError (a stray em-dash left the module's
leading docstring literal unterminated, so two docstrings ended up
concatenated with no closing quotes in between) — confirmed on a clean
`upstream/main`, unrelated to this issue, and out of scope to fix here.

That SyntaxError currently makes `astroml.features.graph.snapshot`
unimportable through the normal package path (`astroml.features.__init__`
eagerly imports a chain that reaches `astroml.cache`), which also blocks
the pre-existing `tests/test_snapshot.py` from being collected at all on a
clean checkout. This fixture loads `snapshot.py` directly via
`importlib`, substituting only the one decorator it actually needs
(`cached_graph_snapshot`, which lives in `astroml.cache.redis_cache` and
does *not* touch the broken `graph_cache.py`), so the RFC 7807 error
handling added by #949 can be exercised without requiring a fix to the
unrelated cache bug.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[3] / "astroml" / "features" / "graph" / "snapshot.py"
)
_REDIS_CACHE_PATH = Path(__file__).resolve().parents[3] / "astroml" / "cache" / "redis_cache.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load spec for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def graph_snapshot_module():
    """Return the `astroml.features.graph.snapshot` module, importable
    despite the unrelated pre-existing `astroml.cache.graph_cache`
    SyntaxError (see module docstring)."""
    if "astroml.features.graph.snapshot" in sys.modules:
        return sys.modules["astroml.features.graph.snapshot"]

    if "astroml.cache" not in sys.modules:
        stub_cache_pkg = types.ModuleType("astroml.cache")
        stub_cache_pkg.__path__ = [str(_REDIS_CACHE_PATH.parent)]
        sys.modules["astroml.cache"] = stub_cache_pkg

        redis_cache = _load_module("astroml.cache.redis_cache", _REDIS_CACHE_PATH)
        sys.modules["astroml.cache"].cached_graph_snapshot = redis_cache.cached_graph_snapshot

    return _load_module("astroml.features.graph.snapshot", _SNAPSHOT_PATH)
