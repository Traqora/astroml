"""The CI half of flaky-test detection — issue #715.

``.github/workflows/flaky-detection.yml`` runs the suite several times and hands
the resulting report files to ``python -m astroml.testing.flaky``.  Aggregating
them is ordinary Python, so it is tested here rather than inside the workflow's
YAML — including the exit code the workflow keys off.
"""

from __future__ import annotations

import json
import pathlib

from astroml.testing.flaky import (
    EXHAUSTED,
    KNOWN,
    NEW,
    aggregate_markdown,
    aggregate_reports,
    main,
    read_reports,
)


def _flake(
    nodeid: str,
    classification: str = NEW,
    attempts: int = 2,
    worker: str = "master",
    message: str = "AssertionError: boom",
) -> dict:
    return {
        "nodeid": nodeid,
        "classification": classification,
        "attempts": attempts,
        "retries_allowed": 2,
        "phase": "call",
        "message": message,
        "worker": worker,
        "duration_s": 0.5,
    }


def _report(*flakes: dict) -> dict:
    return {
        "worker": "master",
        "retries_allowed": 2,
        "registry": None,
        "flakes": list(flakes),
        "counts": {NEW: 0, KNOWN: 0, EXHAUSTED: 0},
    }


def _write(path: pathlib.Path, *flakes: dict) -> pathlib.Path:
    path.write_text(json.dumps(_report(*flakes)))
    return path


class TestAggregation:
    """Repeated runs are merged into one row per test."""

    def test_a_test_seen_in_several_runs_gets_one_row(self):
        payloads = [
            _report(_flake("tests/test_a.py::test_x")),
            _report(_flake("tests/test_a.py::test_x")),
            _report(_flake("tests/test_a.py::test_y")),
        ]

        merged = aggregate_reports(payloads)

        assert [flake.nodeid for flake in merged].count("tests/test_a.py::test_x") == 1

    def test_the_number_of_runs_is_counted(self):
        payloads = [
            _report(_flake("tests/test_a.py::test_x")),
            _report(_flake("tests/test_a.py::test_x")),
            _report(),
        ]

        merged = aggregate_reports(payloads)

        assert merged[0].runs == 2

    def test_a_test_seen_in_more_runs_comes_first(self):
        payloads = [
            _report(_flake("tests/test_a.py::sometimes")),
            _report(_flake("tests/test_a.py::often")),
            _report(_flake("tests/test_a.py::often")),
        ]

        merged = aggregate_reports(payloads)

        assert [flake.nodeid for flake in merged] == [
            "tests/test_a.py::often",
            "tests/test_a.py::sometimes",
        ]

    def test_a_test_that_failed_every_attempt_in_one_run_dominates(self):
        """It flaked in one run and was red in another; the red is the signal."""
        payloads = [
            _report(_flake("tests/test_a.py::test_x", NEW)),
            _report(_flake("tests/test_a.py::test_x", EXHAUSTED, attempts=3)),
        ]

        merged = aggregate_reports(payloads)

        assert merged[0].classification == EXHAUSTED
        assert merged[0].attempts == 3

    def test_a_new_flake_is_not_downgraded_by_a_known_one(self):
        payloads = [
            _report(_flake("tests/test_a.py::test_x", KNOWN)),
            _report(_flake("tests/test_a.py::test_x", NEW)),
        ]

        assert aggregate_reports(payloads)[0].classification == NEW

    def test_workers_are_recorded(self):
        payloads = [
            _report(_flake("tests/test_a.py::test_x", worker="gw0")),
            _report(_flake("tests/test_a.py::test_x", worker="gw1")),
        ]

        assert aggregate_reports(payloads)[0].workers == ["gw0", "gw1"]

    def test_repeats_with_nothing_to_report_are_still_counted(self):
        merged = aggregate_reports([_report(), _report(), _report()])

        assert merged == []

    def test_missing_report_files_are_skipped(self, tmp_path):
        """A job that died before writing a report must not break the summary."""
        _write(tmp_path / "flaky-report-1.json", _flake("tests/test_a.py::test_x"))

        payloads = read_reports(
            [tmp_path / "flaky-report-1.json", tmp_path / "flaky-report-2.json"]
        )

        assert len(payloads) == 1


class TestAggregateMarkdown:
    """The summary a maintainer reads in the workflow run."""

    def test_it_lists_the_test_and_how_often_it_flaked(self):
        payloads = [
            _report(_flake("tests/test_a.py::test_x")),
            _report(_flake("tests/test_a.py::test_x")),
            _report(),
        ]

        markdown = aggregate_markdown(payloads)

        assert "`tests/test_a.py::test_x`" in markdown
        assert "2/3" in markdown
        assert "1 new flake(s)" in markdown

    def test_it_says_so_when_nothing_flaked(self):
        assert "No test needed a retry" in aggregate_markdown([_report(), _report()])


class TestCommandLine:
    """The exit code is what the workflow fails on."""

    def test_the_summary_is_printed(self, tmp_path, capsys):
        _write(tmp_path / "report.json", _flake("tests/test_a.py::test_x"))

        exit_code = main([str(tmp_path / "report.json")])

        assert exit_code == 0
        assert "tests/test_a.py::test_x" in capsys.readouterr().out

    def test_an_unregistered_flake_exits_non_zero(self, tmp_path, capsys):
        _write(tmp_path / "report.json", _flake("tests/test_a.py::test_x", NEW))

        exit_code = main([str(tmp_path / "report.json"), "--fail-on-new"])

        assert exit_code == 1
        assert "::error::" in capsys.readouterr().out

    def test_a_registered_flake_exits_zero(self, tmp_path):
        _write(tmp_path / "report.json", _flake("tests/test_a.py::test_x", KNOWN))

        assert main([str(tmp_path / "report.json"), "--fail-on-new"]) == 0

    def test_a_clean_run_exits_zero(self, tmp_path):
        _write(tmp_path / "report.json")

        assert main([str(tmp_path / "report.json"), "--fail-on-new"]) == 0
