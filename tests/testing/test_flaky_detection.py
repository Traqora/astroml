"""Flaky-test detection, retry and reporting — issue #715.

The retry itself is exercised end to end by running pytest on generated tests
through :mod:`pytester`, because the interesting failures are about what a
retry leaves behind in a real session: a fixture cached as failed, a class
instance carrying the failed attempt's state, an intermediate attempt reaching
the reporters as a failure of its own.  The reporting helpers are unit tested
directly.

Every generated test counts its own attempts through a file in the temporary
working directory, so the assertions can tell "ran once" from "ran twice"
without trusting the plugin's own bookkeeping.

The generated module is always called ``test_target.py``, so nodeids are stable
and a registry entry in a test can name a real test.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from astroml.testing.flaky import (
    EXHAUSTED,
    KNOWN,
    NEW,
    Flake,
    FlakyRun,
    load_registry,
    matches_registry,
)

#: Written by the generated tests into their temporary working directory.
COUNTER = "attempts.txt"

#: Name of the generated test module, fixed so nodeids are predictable.
TARGET = "test_target"

#: Prefix of every generated module: an attempt counter kept in a file.
_BODY = f"""
import pathlib

COUNTER = pathlib.Path({COUNTER!r})


def _attempts() -> int:
    return int(COUNTER.read_text()) if COUNTER.exists() else 0


def _bump() -> int:
    count = _attempts() + 1
    COUNTER.write_text(str(count))
    return count
"""

#: A test that fails its first attempt and passes on the second — a flake.
_FLAKES_ONCE = """

def test_flakes_once():
    assert _bump() > 1
"""


def _plugin_args(path: pathlib.Path, retries: int, extra: list[str] | None = None) -> list[str]:
    """Arguments that run the generated tests with the flaky plugin loaded."""
    args = [
        "-p",
        "astroml.testing.flaky",
        f"--flaky-retries={retries}",
        f"--flaky-report={path / 'report.json'}",
        f"--flaky-registry={path / 'registry.txt'}",
    ]
    return args + list(extra or [])


def _report(path: pathlib.Path) -> dict:
    return json.loads((path / "report.json").read_text())


def _flake_of(report: dict, nodeid: str) -> dict:
    for flake in report["flakes"]:
        if flake["nodeid"] == nodeid or flake["nodeid"].endswith(f"::{nodeid}"):
            return flake
    raise AssertionError(f"no flake recorded for {nodeid}: {report['flakes']}")


@pytest.fixture()
def flaky_test(pytester):
    """A generated module holding one test that fails its first attempt."""
    pytester.makepyfile(**{TARGET: _BODY + _FLAKES_ONCE})
    return pytester


class TestRetrying:
    """A failing test is retried, and the retry is what decides the outcome."""

    def test_a_flaky_test_passes_after_a_retry(self, flaky_test):
        result = flaky_test.runpytest(*_plugin_args(flaky_test.path, retries=1))

        # One outcome for the test — the attempt that passed — not one per try.
        result.assert_outcomes(passed=1)

    def test_the_retry_actually_reran_the_test(self, flaky_test):
        flaky_test.runpytest(*_plugin_args(flaky_test.path, retries=1))

        assert (flaky_test.path / COUNTER).read_text() == "2"

    def test_a_test_that_fails_every_attempt_still_fails(self, pytester):
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

def test_always_broken():
    _bump()
    assert False, "broken"
"""
            }
        )

        result = pytester.runpytest(*_plugin_args(pytester.path, retries=2))

        # Retrying must not turn a broken test into a green one.
        result.assert_outcomes(failed=1)
        assert (pytester.path / COUNTER).read_text() == "3"

    def test_retrying_is_off_by_default(self, flaky_test):
        """Without the flag the plugin is inert: one attempt, no report file."""
        result = flaky_test.runpytest("-p", "astroml.testing.flaky")

        result.assert_outcomes(failed=1)
        assert (flaky_test.path / COUNTER).read_text() == "1"
        assert not (flaky_test.path / "flaky-report.json").exists()

    def test_the_marker_opts_a_test_out_of_retrying(self, pytester):
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

import pytest


@pytest.mark.no_flaky_retry
def test_opt_out():
    _bump()
    assert False, "not retried"
"""
            }
        )

        pytester.runpytest(*_plugin_args(pytester.path, retries=3))

        assert (pytester.path / COUNTER).read_text() == "1"

    def test_a_failing_fixture_is_retried_too(self, pytester):
        """A retry has to clear the failed attempt's fixture state.

        The fixture raises on the first attempt.  If that failure stayed cached,
        the second attempt would report the first attempt's error instead of
        running the test, and the test would never pass.
        """
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

import pytest


@pytest.fixture()
def once_broken():
    if _bump() == 1:
        raise RuntimeError("first attempt fails in setup")
    return "ready"


def test_uses_the_fixture(once_broken):
    assert once_broken == "ready"
"""
            }
        )

        result = pytester.runpytest(*_plugin_args(pytester.path, retries=1))

        result.assert_outcomes(passed=1)
        assert (pytester.path / COUNTER).read_text() == "2"

    def test_a_class_instance_is_not_reused_across_attempts(self, pytester):
        """pytest caches one instance per item; a retry must not inherit it."""
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

class TestState:
    def test_state_starts_empty(self):
        self.seen = getattr(self, "seen", [])
        assert _bump() > 1
"""
            }
        )

        result = pytester.runpytest(*_plugin_args(pytester.path, retries=1))

        result.assert_outcomes(passed=1)

    def test_a_skipped_test_is_not_retried(self, pytester):
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

import pytest


@pytest.mark.skip(reason="not applicable")
def test_skipped():
    _bump()
"""
            }
        )

        result = pytester.runpytest(*_plugin_args(pytester.path, retries=2))

        result.assert_outcomes(skipped=1)
        assert not (pytester.path / COUNTER).exists()


class TestReporting:
    """Every retry is surfaced, not silently absorbed."""

    def test_a_new_flake_is_recorded_in_the_report(self, flaky_test):
        flaky_test.runpytest(*_plugin_args(flaky_test.path, retries=2))

        report = _report(flaky_test.path)
        flake = _flake_of(report, "test_flakes_once")

        assert flake["classification"] == NEW
        assert flake["attempts"] == 2
        assert flake["retries_allowed"] == 2
        assert flake["phase"] == "call"
        assert report["counts"][NEW] == 1

    def test_the_report_names_the_test_that_failed_every_attempt(self, pytester):
        pytester.makepyfile(
            **{
                TARGET: _BODY
                + """

def test_always_broken():
    _bump()
    assert False, "broken"
"""
            }
        )

        pytester.runpytest(*_plugin_args(pytester.path, retries=1))

        flake = _flake_of(_report(pytester.path), "test_always_broken")
        assert flake["classification"] == EXHAUSTED
        assert flake["attempts"] == 2

    def test_the_terminal_summary_names_the_flake(self, flaky_test):
        result = flaky_test.runpytest(*_plugin_args(flaky_test.path, retries=1))

        stdout = result.stdout.str()
        assert "[flaky] new" in stdout
        assert f"{TARGET}.py::test_flakes_once" in stdout

    def test_a_registered_test_is_reported_as_known(self, flaky_test):
        (flaky_test.path / "registry.txt").write_text(
            f"# #900 — tracked, timing dependent\n{TARGET}.py::test_flakes_once\n"
        )

        flaky_test.runpytest(*_plugin_args(flaky_test.path, retries=1))

        report = _report(flaky_test.path)
        assert _flake_of(report, "test_flakes_once")["classification"] == KNOWN
        assert report["counts"][NEW] == 0

    def test_an_unregistered_flake_can_fail_the_run(self, flaky_test):
        """The test passed, but it only passed because it was retried."""
        result = flaky_test.runpytest(
            *_plugin_args(flaky_test.path, retries=1, extra=["--flaky-fail-on-new"])
        )

        assert result.ret == pytest.ExitCode.TESTS_FAILED
        result.assert_outcomes(passed=1)

    def test_a_registered_flake_does_not_fail_the_run(self, flaky_test):
        (flaky_test.path / "registry.txt").write_text(f"{TARGET}.py::test_flakes_once\n")

        result = flaky_test.runpytest(
            *_plugin_args(flaky_test.path, retries=1, extra=["--flaky-fail-on-new"])
        )

        assert result.ret == pytest.ExitCode.OK

    def test_a_clean_run_reports_nothing(self, pytester):
        pytester.makepyfile(**{TARGET: _BODY + "\n\ndef test_clean():\n    assert _bump() == 1\n"})

        result = pytester.runpytest(*_plugin_args(pytester.path, retries=1))

        assert (pytester.path / "report.json").exists()
        assert _report(pytester.path)["flakes"] == []
        result.assert_outcomes(passed=1)


class TestRegistryFile:
    """The registry is data, and reading it is tested without a subprocess."""

    def test_a_missing_registry_is_empty_not_an_error(self, tmp_path):
        assert load_registry(tmp_path / "absent.txt") == []
        assert load_registry(None) == []

    def test_comments_and_blanks_are_dropped(self, tmp_path):
        registry = tmp_path / "known_flaky.txt"
        registry.write_text(
            "# a header comment\n"
            "\n"
            "tests/test_stream.py::test_resume  # #1234 — timing dependent\n"
            "tests/test_ingestion.py\n"
        )

        assert load_registry(registry) == [
            "tests/test_stream.py::test_resume",
            "tests/test_ingestion.py",
        ]

    def test_a_pattern_without_a_test_id_covers_the_whole_file(self):
        patterns = ["tests/test_batch_scheduler.py"]

        assert matches_registry("tests/test_batch_scheduler.py::test_flush", patterns)
        assert matches_registry("tests/test_batch_scheduler.py::TestFlush::test_a", patterns)
        assert not matches_registry("tests/test_stream.py::test_cursor", patterns)

    def test_a_full_nodeid_matches_exactly(self):
        patterns = ["tests/test_stream.py::TestCursor::test_resume"]

        assert matches_registry("tests/test_stream.py::TestCursor::test_resume", patterns)
        assert not matches_registry("tests/test_stream.py::TestCursor::test_other", patterns)

    def test_a_glob_matches_a_class(self):
        patterns = ["tests/test_stream.py::TestCursor::*"]

        assert matches_registry("tests/test_stream.py::TestCursor::test_resume", patterns)
        assert not matches_registry("tests/test_stream.py::TestLedger::test_close", patterns)

    def test_an_empty_registry_matches_nothing(self):
        assert not matches_registry("tests/test_stream.py::test_cursor", [])


class TestRunRecord:
    """The record itself — what a report contains and how it is written."""

    def _flake(self, nodeid: str, classification: str, attempts: int = 2) -> Flake:
        return Flake(
            nodeid=nodeid,
            classification=classification,
            attempts=attempts,
            retries_allowed=2,
            phase="call",
            message="AssertionError: boom",
            worker="master",
            duration_s=0.5,
        )

    def test_one_record_per_test(self):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")

        run.record(self._flake("tests/test_a.py::test_x", NEW))
        run.record(self._flake("tests/test_a.py::test_x", KNOWN))

        assert len(run.flakes) == 1
        assert run.flakes[0].classification == KNOWN

    def test_classifications_are_separated(self):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")
        run.record(self._flake("a", NEW))
        run.record(self._flake("b", KNOWN))
        run.record(self._flake("c", EXHAUSTED))

        assert [flake.nodeid for flake in run.new_flakes] == ["a"]
        assert [flake.nodeid for flake in run.known_flakes] == ["b"]
        assert [flake.nodeid for flake in run.failed_tests] == ["c"]

    def test_a_failing_test_is_not_counted_as_a_flake_that_passed(self):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")
        run.record(self._flake("c", EXHAUSTED))

        assert not run.flakes[0].is_passing

    def test_the_report_is_json_serialisable(self, tmp_path):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")
        run.record(self._flake("tests/test_a.py::test_x", NEW))

        written = run.write(tmp_path / "report.json")
        payload = json.loads(written.read_text())

        assert payload["counts"] == {NEW: 1, KNOWN: 0, EXHAUSTED: 0}
        assert payload["flakes"][0]["nodeid"] == "tests/test_a.py::test_x"

    def test_xdist_workers_write_separate_reports(self, tmp_path):
        """Workers cannot share a file without an aggregator, so they don't."""
        run = FlakyRun(retries_allowed=1, registry_path=None, worker="gw0")

        written = run.write(tmp_path / "report.json", worker="gw0")

        assert written.name == "report-gw0.json"
        assert written.exists()

    def test_markdown_summarises_the_run(self):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")
        run.record(self._flake("tests/test_a.py::test_x", NEW))

        markdown = run.to_markdown()

        assert "## Flaky test report" in markdown
        assert "`tests/test_a.py::test_x`" in markdown
        assert "1 new, 0 known, 0 still failing" in markdown

    def test_markdown_says_so_when_nothing_flaked(self):
        run = FlakyRun(retries_allowed=2, registry_path=None, worker="master")

        assert "No test needed a retry." in run.to_markdown()
