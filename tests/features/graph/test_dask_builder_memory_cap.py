"""Driver-side memory cap for the Dask graph builder (issue #880).

Passing ``memory_limit`` to a ``LocalCluster`` caps each *worker*, but the
builder used to pull every edge back onto the driver in one ``compute()``,
so a graph that fit across the cluster could still OOM the process that
asked for it. These tests pin down the explicit ceiling:

* :func:`parse_memory_limit` understands the human-readable values accepted
  by :class:`DaskGraphBuilder`;
* :func:`enforce_memory_cap` rejects an edge set that would exceed the cap
  *before* any Dask work happens; and
* :meth:`DaskGraphBuilder.build_graph` consumes one partition at a time so
  the driver's peak footprint stays proportional to a partition.

Dask is an optional dependency and is not installed in the test
environment, so the small slice of the ``dask.dataframe`` surface the
builder uses is faked locally. The module itself is loaded through the
``dask_builder_module`` fixture (see ``conftest.py``) to sidestep the
unrelated, pre-existing package-import breakage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "astroml" / "features" / "graph" / "dask_builder.py"
)


@pytest.fixture(scope="session")
def dask_builder_module():
    """Load `astroml.features.graph.dask_builder` directly from its file.

    ``astroml.features.__init__`` eagerly imports a chain that currently
    trips unrelated, pre-existing SyntaxErrors on a clean checkout, so the
    normal package import path is blocked. The logic under test lives
    entirely in this module; loading it with ``importlib`` keeps the test
    runnable with ``--noconftest`` until the repo-wide syntax damage (see
    the note in the PR) is repaired.
    """
    name = "astroml.features.graph.dask_builder"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"could not load {name} from {_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakePartition:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def compute(self) -> pd.DataFrame:
        return self._frame


class _FakeDaskDataFrame:
    def __init__(self, frame: pd.DataFrame, npartitions: int) -> None:
        self._frame = frame
        self.npartitions = max(1, npartitions)
        self.compute_calls = 0

    def __getitem__(self, columns):  # noqa: D401 - mirror dask's slicing API
        return self

    def to_delayed(self) -> list[_FakePartition]:
        rows = list(self._frame.index)
        chunks = [
            rows[i : i + max(1, len(rows) // self.npartitions)]
            for i in range(0, len(rows), max(1, len(rows) // self.npartitions))
        ]
        return [_FakePartition(self._frame.loc[chunk]) for chunk in chunks]


class _FakeDaskDataFrameModule:
    def __init__(self) -> None:
        self.from_pandas_calls = 0
        self.last_frame: _FakeDaskDataFrame | None = None

    def from_pandas(self, frame: pd.DataFrame, npartitions: int = 1) -> _FakeDaskDataFrame:
        self.from_pandas_calls += 1
        self.last_frame = _FakeDaskDataFrame(frame, npartitions)
        return self.last_frame


def _make_builder(dask_builder_module, monkeypatch, **kwargs):
    monkeypatch.setattr(dask_builder_module, "DASK_AVAILABLE", True)
    builder = dask_builder_module.DaskGraphBuilder(**kwargs)
    builder.client = object()
    return builder


class TestParseMemoryLimit:
    def test_plain_int_is_bytes(self, dask_builder_module):
        assert dask_builder_module.parse_memory_limit(2048) == 2048

    def test_bare_number_string_is_bytes(self, dask_builder_module):
        assert dask_builder_module.parse_memory_limit("2048") == 2048

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1KB", 1024),
            ("512MB", 512 * 1024**2),
            ("512MiB", 512 * 1024**2),
            ("4GB", 4 * 1024**3),
            ("4gb", 4 * 1024**3),
            ("1.5GB", int(1.5 * 1024**3)),
        ],
    )
    def test_human_readable_units(self, dask_builder_module, text, expected):
        assert dask_builder_module.parse_memory_limit(text) == expected

    @pytest.mark.parametrize("text", ["", "abc", "4XB", "-1GB", "1 2GB"])
    def test_rejects_unparseable_strings(self, dask_builder_module, text):
        with pytest.raises(ValueError):
            dask_builder_module.parse_memory_limit(text)

    @pytest.mark.parametrize("value", [-1, -1024])
    def test_rejects_negative_int(self, dask_builder_module, value):
        with pytest.raises(ValueError):
            dask_builder_module.parse_memory_limit(value)

    def test_rejects_non_int_non_str(self, dask_builder_module):
        with pytest.raises(ValueError):
            dask_builder_module.parse_memory_limit(3.5)

    def test_rejects_bool(self, dask_builder_module):
        with pytest.raises(ValueError):
            dask_builder_module.parse_memory_limit(True)


class TestEstimateEdgeMemory:
    def test_zero_edges_is_zero(self, dask_builder_module):
        assert dask_builder_module.estimate_edge_memory(0) == 0

    def test_default_overhead(self, dask_builder_module):
        assert (
            dask_builder_module.estimate_edge_memory(10)
            == 10 * dask_builder_module.DEFAULT_EDGE_BYTES
        )

    def test_attributes_add_cost(self, dask_builder_module):
        expected = 10 * (
            dask_builder_module.DEFAULT_EDGE_BYTES + dask_builder_module.ATTRIBUTE_BYTES * 2
        )
        assert dask_builder_module.estimate_edge_memory(10, n_attrs=2) == expected

    @pytest.mark.parametrize("n_edges, n_attrs", [(-1, 0), (0, -1), (5, -3)])
    def test_rejects_negative_sizes(self, dask_builder_module, n_edges, n_attrs):
        with pytest.raises(ValueError):
            dask_builder_module.estimate_edge_memory(n_edges, n_attrs=n_attrs)


class TestEnforceMemoryCap:
    def test_under_cap_returns_estimate(self, dask_builder_module):
        estimate = dask_builder_module.estimate_edge_memory(10)
        assert dask_builder_module.enforce_memory_cap(10, estimate + 1) == estimate

    def test_exactly_at_cap_is_allowed(self, dask_builder_module):
        estimate = dask_builder_module.estimate_edge_memory(10)
        assert dask_builder_module.enforce_memory_cap(10, estimate) == estimate

    def test_over_cap_raises_memory_error(self, dask_builder_module):
        with pytest.raises(MemoryError):
            dask_builder_module.enforce_memory_cap(10, 1)

    def test_negative_cap_raises_value_error(self, dask_builder_module):
        with pytest.raises(ValueError):
            dask_builder_module.enforce_memory_cap(10, -1)


class TestDaskGraphBuilderCap:
    def test_negative_max_edges_rejected_at_construction(self, dask_builder_module, monkeypatch):
        monkeypatch.setattr(dask_builder_module, "DASK_AVAILABLE", True)
        with pytest.raises(ValueError):
            dask_builder_module.DaskGraphBuilder(max_edges=-1)

    def test_max_edges_blocks_before_any_dask_work(self, dask_builder_module, monkeypatch):
        fake_dd = _FakeDaskDataFrameModule()
        monkeypatch.setattr(dask_builder_module, "dd", fake_dd, raising=False)
        builder = _make_builder(dask_builder_module, monkeypatch, max_edges=2)
        frame = pd.DataFrame({"source": [1, 2, 3], "target": [4, 5, 6]})
        with pytest.raises(MemoryError):
            builder.build_graph(frame)
        assert fake_dd.from_pandas_calls == 0

    def test_memory_cap_blocks_before_any_dask_work(self, dask_builder_module, monkeypatch):
        fake_dd = _FakeDaskDataFrameModule()
        monkeypatch.setattr(dask_builder_module, "dd", fake_dd, raising=False)
        builder = _make_builder(dask_builder_module, monkeypatch, memory_limit=100)
        frame = pd.DataFrame({"source": [1, 2, 3], "target": [4, 5, 6]})
        with pytest.raises(MemoryError):
            builder.build_graph(frame)
        assert fake_dd.from_pandas_calls == 0

    def test_build_streams_partitions_and_builds_graph(self, dask_builder_module, monkeypatch):
        fake_dd = _FakeDaskDataFrameModule()
        monkeypatch.setattr(dask_builder_module, "dd", fake_dd, raising=False)
        builder = _make_builder(dask_builder_module, monkeypatch, memory_limit="4GB")
        frame = pd.DataFrame(
            {
                "source": [1, 2, 3, 4],
                "target": [2, 3, 4, 5],
                "amount": [10, 20, 30, 40],
            }
        )
        graph = builder.build_graph(frame, edge_attrs=["amount"], partition_size=2)
        assert fake_dd.from_pandas_calls == 1
        assert sorted(graph.nodes()) == [1, 2, 3, 4, 5]
        assert graph.number_of_edges() == 4
        assert graph[1][2]["amount"] == 10
        assert fake_dd.last_frame is not None
        assert fake_dd.last_frame.npartitions > 1

    def test_directed_flag_selects_digraph(self, dask_builder_module, monkeypatch):
        import networkx as nx

        fake_dd = _FakeDaskDataFrameModule()
        monkeypatch.setattr(dask_builder_module, "dd", fake_dd, raising=False)
        builder = _make_builder(dask_builder_module, monkeypatch, memory_limit="4GB")
        frame = pd.DataFrame({"source": [1], "target": [2]})
        assert isinstance(builder.build_graph(frame, directed=True), nx.DiGraph)
        assert not isinstance(builder.build_graph(frame, directed=False), nx.DiGraph)
