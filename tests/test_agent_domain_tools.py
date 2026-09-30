"""Tests for the AstroML domain tools exposed to agents."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

import pytest

from astroml.agent.domain_tools import (
    DOMAIN_TOOLS,
    account_features,
    anomaly_score_tool,
    build_default_registry,
    graph_overview,
    top_accounts,
    window_stats,
)
from astroml.agent.tools import ToolError, ToolRegistry


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


def _edges() -> List[Dict[str, Any]]:
    """Small graph with a self loop and two assets."""
    return [
        {"src": "A", "dst": "B", "timestamp": 100, "amount": 10.0, "asset": "XLM"},
        {"src": "B", "dst": "C", "timestamp": 200, "amount": 5.0, "asset": "XLM"},
        {"src": "C", "dst": "A", "timestamp": 300, "amount": 2.5, "asset": "USD"},
        {"src": "D", "dst": "D", "timestamp": 400, "amount": 1.0, "asset": "USD"},
    ]


class TestGraphOverview:
    def test_reports_size_assets_and_density(self):
        overview = graph_overview(_edges())
        assert overview["num_nodes"] == 4
        assert overview["num_edges"] == 4
        assert overview["num_self_loops"] == 1
        assert overview["assets"] == {"XLM": 2, "USD": 2}
        assert overview["density"] == pytest.approx(4 / 12, abs=1e-6)

    def test_top_accounts_are_deterministic(self):
        overview = graph_overview(_edges())
        assert [entry["account"] for entry in overview["top_accounts"]] == [
            "A",
            "B",
            "C",
            "D",
        ]
        assert all(entry["degree"] == 2 for entry in overview["top_accounts"])

    def test_empty_graph(self):
        overview = graph_overview([])
        assert overview["num_nodes"] == 0
        assert overview["num_edges"] == 0
        assert overview["density"] == 0.0
        assert overview["top_accounts"] == []

    def test_single_node_graph_has_zero_density(self):
        overview = graph_overview([{"src": "A", "dst": "A", "timestamp": 1}])
        assert overview["density"] == 0.0

    def test_accepts_edge_objects(self):
        from astroml.features.graph.snapshot import Edge

        overview = graph_overview([Edge(src="A", dst="B", timestamp=1)])
        assert overview["num_nodes"] == 2
        assert overview["num_edges"] == 1

    def test_rejects_malformed_edges(self):
        with pytest.raises(ToolError, match="missing 'src'"):
            graph_overview([{"dst": "B", "timestamp": 1}])
        with pytest.raises(ToolError, match="must be an object"):
            graph_overview(["not-an-edge"])
        with pytest.raises(ToolError, match="non-numeric 'timestamp'"):
            graph_overview([{"src": "A", "dst": "B", "timestamp": "later"}])


class TestWindowStats:
    def test_filters_by_inclusive_bounds(self):
        stats = window_stats(_edges(), 100, 200)
        assert stats["num_edges"] == 2
        assert stats["num_nodes"] == 3
        assert stats["window_seconds"] == 100
        assert stats["accounts"] == ["A", "B", "C"]

    def test_rejects_inverted_bounds(self):
        with pytest.raises(ToolError, match="start_ts"):
            window_stats(_edges(), 200, 100)

    def test_empty_window(self):
        stats = window_stats(_edges(), 1000, 2000)
        assert stats["num_edges"] == 0
        assert stats["num_nodes"] == 0


class TestAccountFeatures:
    def test_computes_features_for_every_account(self):
        pytest.importorskip("pandas")
        features = account_features(_edges(), ref_time=400.0)
        assert set(features) == {"A", "B", "C", "D"}
        assert features["A"]["out_degree"] == 1
        assert features["A"]["in_degree"] == 1

    def test_filters_requested_accounts(self):
        pytest.importorskip("pandas")
        features = account_features(_edges(), ref_time=400.0, accounts=["A"])
        assert set(features) == {"A"}

    def test_unknown_accounts_raise_tool_error(self):
        pytest.importorskip("pandas")
        with pytest.raises(ToolError, match="unknown accounts"):
            account_features(_edges(), ref_time=400.0, accounts=["ZZZ"])


class TestTopAccounts:
    def test_ranks_by_volume(self):
        pytest.importorskip("pandas")
        ranked = top_accounts(_edges(), metric="volume")
        assert [entry["account"] for entry in ranked] == ["B", "A", "C", "D"]
        assert ranked[0]["score"] == pytest.approx(15.0)

    def test_ranks_by_degree(self):
        pytest.importorskip("pandas")
        ranked = top_accounts(_edges(), metric="out_degree", top_k=2)
        assert len(ranked) == 2

    def test_invalid_metric_raises(self):
        with pytest.raises(ToolError, match="unsupported metric"):
            top_accounts(_edges(), metric="nope")

    def test_invalid_top_k_raises(self):
        with pytest.raises(ToolError, match="top_k"):
            top_accounts(_edges(), top_k=0)


class _StubScorer:
    """Minimal stand-in for an ``InductiveAnomalyScorer``."""

    def __init__(self) -> None:
        self.seen = None

    def score_new_accounts(self, edges, account_ids, ref_time):
        self.seen = (edges, list(account_ids), ref_time)
        return {account: float(index) for index, account in enumerate(account_ids)}


class TestAnomalyScoreTool:
    def test_wraps_a_scorer(self):
        scorer = _StubScorer()
        tool_obj = anomaly_score_tool(scorer)
        assert tool_obj.name == "score_accounts"

        result = _run(
            ToolRegistry([tool_obj]).run(
                "score_accounts",
                {"edges": _edges(), "accounts": ["A", "B"], "ref_time": 10.0},
            )
        )
        assert result.ok is True
        assert result.data == {"A": 0.0, "B": 1.0}
        assert scorer.seen[1] == ["A", "B"]

    def test_custom_name_and_description(self):
        tool_obj = anomaly_score_tool(
            _StubScorer(), name="risk_scores", description="Score risk"
        )
        assert tool_obj.name == "risk_scores"
        assert tool_obj.description == "Score risk"

    def test_rejects_objects_without_the_scoring_method(self):
        tool_obj = anomaly_score_tool(object())
        result = _run(
            ToolRegistry([tool_obj]).run(
                "score_accounts", {"edges": [], "accounts": ["A"], "ref_time": 1.0}
            )
        )
        assert result.ok is False
        assert "score_new_accounts" in result.error

    def test_requires_at_least_one_account(self):
        tool_obj = anomaly_score_tool(_StubScorer())
        result = _run(
            ToolRegistry([tool_obj]).run(
                "score_accounts", {"edges": [], "accounts": [], "ref_time": 1.0}
            )
        )
        assert result.ok is False
        assert "accounts must not be empty" in result.error


class TestDefaultRegistry:
    def test_registers_every_domain_tool(self):
        registry = build_default_registry()
        assert registry.names() == [tool_obj.name for tool_obj in DOMAIN_TOOLS]
        assert registry.names() == [
            "graph_overview",
            "window_stats",
            "account_features",
            "top_accounts",
        ]

    def test_include_restricts_the_registry(self):
        assert build_default_registry(include=["graph_overview"]).names() == [
            "graph_overview"
        ]

    def test_unknown_include_raises(self):
        with pytest.raises(ToolError, match="unknown domain tools"):
            build_default_registry(include=["graph_overview", "ghost"])

    def test_executes_through_the_registry(self):
        registry = build_default_registry(include=["graph_overview"])
        result = _run(registry.run("graph_overview", {"edges": _edges()}))
        assert result.ok is True
        assert result.data["num_edges"] == 4

    def test_schemas_require_documented_arguments(self):
        specs = {spec.name: spec for spec in build_default_registry().specs()}
        assert specs["graph_overview"].parameters["required"] == ["edges"]
        assert specs["window_stats"].parameters["required"] == [
            "edges",
            "start_ts",
            "end_ts",
        ]
        assert "top_k" in specs["top_accounts"].parameters["properties"]
