"""Tests for the ``astroml agent`` command line interface."""
from __future__ import annotations

import asyncio
import json

import pytest

from astroml.agent.cli import _bound_graph_tools, _load_edges, main


def _run(coro):
    """Execute a coroutine from a synchronous test."""
    return asyncio.run(coro)


def _edges_payload():
    return [
        {"src": "A", "dst": "B", "timestamp": 100, "amount": 1.0, "asset": "XLM"},
        {"src": "B", "dst": "C", "timestamp": 200, "amount": 2.0, "asset": "USD"},
    ]


def _edges_file(tmp_path):
    path = tmp_path / "edges.json"
    path.write_text(json.dumps(_edges_payload()), encoding="utf-8")
    return path


class TestLoadEdges:
    def test_loads_a_json_list(self, tmp_path):
        assert len(_load_edges(str(_edges_file(tmp_path)))) == 2

    def test_accepts_wrapped_edges_key(self, tmp_path):
        path = tmp_path / "wrapped.json"
        path.write_text(
            json.dumps({"edges": [{"src": "A", "dst": "B", "timestamp": 1}]}),
            encoding="utf-8",
        )
        assert len(_load_edges(str(path))) == 1

    def test_accepts_utf8_bom(self, tmp_path):
        path = tmp_path / "bom.json"
        path.write_text(
            json.dumps([{"src": "A", "dst": "B", "timestamp": 1}]),
            encoding="utf-8-sig",
        )
        assert len(_load_edges(str(path))) == 1

    def test_rejects_payloads_without_edges(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_edges(str(path))


class TestBoundGraphTools:
    def _tools(self, edges):
        return {tool_obj.name: tool_obj for tool_obj in _bound_graph_tools(edges)}

    def test_exposes_dataset_level_tools(self):
        tools = self._tools(_edges_payload())
        assert set(tools) == {
            "graph_overview",
            "window_stats",
            "account_features",
            "top_accounts",
            "asset_breakdown",
            "sample_edges",
        }

    def test_graph_overview_takes_no_arguments(self):
        assert self._tools(_edges_payload())["graph_overview"].parameters["properties"] == {}

    def test_asset_breakdown_counts_assets(self):
        result = _run(self._tools(_edges_payload())["asset_breakdown"].run())
        assert result == {"XLM": 1, "USD": 1}

    def test_sample_edges_limits_output(self):
        edges = _edges_payload() * 3
        result = _run(self._tools(edges)["sample_edges"].run(limit=2))
        assert len(result) == 2


class TestMain:
    def test_echo_provider_answers_the_goal(self, capsys):
        exit_code = main(["--provider", "echo", "--quiet", "summarise the graph"])
        assert exit_code == 0
        assert "summarise the graph" in capsys.readouterr().out

    def test_json_output_contains_the_trace(self, capsys):
        exit_code = main(["--provider", "echo", "--json", "hello"])
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert payload["success"] is True
        assert payload["goal"] == "hello"
        assert "graph_overview" in payload["metadata"]["tools"]

    def test_edges_file_binds_graph_tools(self, capsys, tmp_path):
        exit_code = main(
            [
                "--provider",
                "echo",
                "--edges",
                str(_edges_file(tmp_path)),
                "--json",
                "hi",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert "sample_edges" in payload["metadata"]["tools"]

    def test_tool_allow_list_is_applied(self, capsys):
        exit_code = main(
            [
                "--provider",
                "echo",
                "--tools",
                "window_stats,top_accounts",
                "--json",
                "hi",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0
        assert payload["metadata"]["tools"] == ["window_stats", "top_accounts"]

    def test_plan_flag_runs_the_planner(self, capsys):
        assert main(["--provider", "echo", "--plan", "--quiet", "hi"]) == 0

    def test_scripted_provider_uses_the_environment_script(self, capsys, monkeypatch):
        monkeypatch.setenv("ASTROML_LLM_SCRIPT", "Final Answer: scripted ok")
        exit_code = main(["--provider", "scripted", "--quiet", "hi"])
        assert exit_code == 0
        assert "scripted ok" in capsys.readouterr().out

    def test_missing_goal_is_a_usage_error(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["--provider", "echo"])
        assert excinfo.value.code == 2

    def test_invalid_provider_is_a_usage_error(self):
        with pytest.raises(SystemExit):
            main(["--provider", "bogus", "hi"])

    def test_unreadable_edges_file_reports_an_error(self, capsys, tmp_path):
        exit_code = main(
            [
                "--provider",
                "echo",
                "--edges",
                str(tmp_path / "missing.json"),
                "hi",
            ]
        )
        assert exit_code == 2
        assert "error:" in capsys.readouterr().err
