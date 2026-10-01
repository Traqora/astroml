"""Flaky-test detection, retry and reporting — issue #715.

A test that passes on the second attempt is a signal, not a success.  Retrying
it keeps the suite usable, but a silent retry throws the signal away: the same
test flakes again next week, nobody sees it, and the suite slowly stops being
trusted.  This plugin retries *and* reports — every test that needed more than
one attempt is named, counted and written to a machine-readable report, so a
flake can be fixed or explicitly registered instead of absorbed.

How it is used
--------------
The suite runs several times on a schedule (``.github/workflows/flaky-detection.yml``)
with retries enabled::

    pytest tests -m "not gpu" --flaky-retries 2 --flaky-report flaky-report.json

Within a run a failing test is retried up to ``--flaky-retries`` times.  The
first attempt that passes is the outcome the suite reports as this test's, and
the test is recorded as a flake.  A test that fails every attempt is recorded
as failing: retrying must not turn a broken test into a green one.

Known and new flakes
--------------------
``tests/known_flaky.txt`` lists tests already known to be flaky (one ``fnmatch``
pattern per line, ``#`` for comments).  A flake in that file is recorded as
``known``; anything else is recorded as ``new``.  With ``--flaky-fail-on-new`` a
new flake fails the run, which is how a regression gets noticed on the branch
that caused it rather than months later — the alternative is a report nobody
reads.  Register a test there only with an issue reference in the comment above
it, so the list stays a list of decisions and not a list of excuses.

Under ``pytest-xdist`` each worker is a separate process with its own view of
what flaked, so each writes its own report (``flaky-report-gw0.json``), and the
repeated runs of the schedule each write their own too.  ``python -m
astroml.testing.flaky <reports...>`` merges any number of them into one
summary, which is what the workflow's reporting job prints.

With the default ``--flaky-retries 0`` the plugin does nothing at all: no
retrying and no report file, so an ordinary ``pytest`` run is unchanged.
"""

from __future__ import annotations

import fnmatch
import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import pytest
from _pytest.reports import TestReport

#: Where known-flaky nodeids are registered, relative to the invocation dir.
DEFAULT_REGISTRY = pathlib.Path("tests/known_flaky.txt")

#: Where the JSON report is written when ``--flaky-report`` is not given.
DEFAULT_REPORT = pathlib.Path("flaky-report.json")

#: Classification of a retried test that is registered as a known flake.
KNOWN = "known"

#: Classification of a retried test that is not registered — the interesting one.
NEW = "new"

#: Classification of a test that failed every attempt it was given.
EXHAUSTED = "failed"

#: Marker that opts a test out of retrying.
_MARKER = "no_flaky_retry"

#: The run being recorded, shared with the hooks through ``config.stash``.
FLAKY_RUN: pytest.StashKey["FlakyRun"] = pytest.StashKey()


def _worker_id() -> str:
    """Identify the process writing a report — ``master`` unless under xdist."""
    return os.environ.get("PYTEST_XDIST_WORKER", "master")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def load_registry(path: pathlib.Path | str | None) -> list[str]:
    """Read the known-flaky registry.

    Args:
        path: Registry file, or None for an empty registry.

    Returns:
        The patterns in the file, in order, with blank lines and ``#``
        comments dropped. A missing file is an empty registry rather than an
        error: having no known flakes is the state worth aiming for.
    """
    if not path:
        return []
    registry = pathlib.Path(path)
    if not registry.exists():
        return []

    patterns: list[str] = []
    for line in registry.read_text().splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            patterns.append(entry)
    return patterns


def matches_registry(nodeid: str, patterns: Iterable[str]) -> bool:
    """Return whether ``nodeid`` is registered as a known-flaky test.

    A pattern without ``::`` matches the whole file it names, so
    ``tests/test_batch_scheduler.py`` covers every test in that module while
    ``tests/test_batch_scheduler.py::TestFlush::*`` narrows it to one class.

    Args:
        nodeid: The test to classify.
        patterns: Patterns from :func:`load_registry`.

    Returns:
        True when the test is a known flake.
    """
    for pattern in patterns:
        if not pattern:
            continue
        if fnmatch.fnmatchcase(nodeid, pattern):
            return True
        if "::" not in pattern and nodeid.split("::", 1)[0] == pattern:
            return True
    return False


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class Flake:
    """One test that needed more than one attempt, or that used them all up."""

    nodeid: str
    classification: str
    attempts: int
    retries_allowed: int
    phase: str
    message: str
    worker: str
    duration_s: float

    @property
    def is_passing(self) -> bool:
        """Whether the test did eventually pass, on a later attempt."""
        return self.classification != EXHAUSTED

    def as_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-serialisable dict."""
        return {
            "nodeid": self.nodeid,
            "classification": self.classification,
            "attempts": self.attempts,
            "retries_allowed": self.retries_allowed,
            "phase": self.phase,
            "message": self.message,
            "worker": self.worker,
            "duration_s": round(self.duration_s, 4),
        }


@dataclass
class FlakyRun:
    """Everything one pytest process observed about flakiness."""

    retries_allowed: int
    registry_path: str | None
    worker: str
    started_at: float = field(default_factory=time.time)
    flakes: list[Flake] = field(default_factory=list)

    def record(self, flake: Flake) -> None:
        """Add a flake, keeping one record per test.

        A test can only reach here once per run, but the record is deduplicated
        anyway so a re-entrant caller cannot double-count a test.
        """
        for index, existing in enumerate(self.flakes):
            if existing.nodeid == flake.nodeid:
                self.flakes[index] = flake
                return
        self.flakes.append(flake)

    @property
    def new_flakes(self) -> list[Flake]:
        """Flakes that are not registered as known, in report order."""
        return [flake for flake in self.flakes if flake.classification == NEW]

    @property
    def known_flakes(self) -> list[Flake]:
        """Flakes that match the registry, in report order."""
        return [flake for flake in self.flakes if flake.classification == KNOWN]

    @property
    def failed_tests(self) -> list[Flake]:
        """Tests that never passed, whatever number of attempts they were given."""
        return [flake for flake in self.flakes if flake.classification == EXHAUSTED]

    def as_dict(self) -> dict[str, Any]:
        """Return the run as a JSON-serialisable dict."""
        return {
            "worker": self.worker,
            "retries_allowed": self.retries_allowed,
            "registry": self.registry_path,
            "started_at": self.started_at,
            "finished_at": time.time(),
            "flakes": [flake.as_dict() for flake in self.flakes],
            "counts": {
                NEW: len(self.new_flakes),
                KNOWN: len(self.known_flakes),
                EXHAUSTED: len(self.failed_tests),
            },
        }

    def write(self, path: pathlib.Path | str, *, worker: str = "") -> pathlib.Path:
        """Write the JSON report, suffixing non-master workers with their id."""
        target = pathlib.Path(path)
        if worker and worker != "master":
            target = target.with_name(f"{target.stem}-{worker}{target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")
        return target

    def to_markdown(self) -> str:
        """Render the run as a markdown section for a CI step summary."""
        lines = [
            "## Flaky test report",
            "",
            f"Worker `{self.worker}`; retries allowed per test: {self.retries_allowed}.",
            "",
        ]
        if not self.flakes:
            lines.append("No test needed a retry.")
            return "\n".join(lines) + "\n"

        lines += [
            f"{len(self.new_flakes)} new, {len(self.known_flakes)} known, "
            f"{len(self.failed_tests)} still failing.",
            "",
            "| test | classification | attempts | phase | worker |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| `{flake.nodeid}` | {flake.classification} | {flake.attempts} "
            f"| {flake.phase} | {flake.worker} |"
            for flake in self.flakes
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Retry
# ---------------------------------------------------------------------------


def _first_failure(reports: Sequence[TestReport]) -> TestReport | None:
    """The first report that failed, or None when the attempt produced none."""
    for report in reports:
        if report.failed:
            return report
    return None


def _describe(report: TestReport, limit: int = 200) -> str:
    """One line describing a failure, for the report — no traceback.

    pytest attaches the exception summary to the report's ``reprcrash``, which
    is the line worth keeping; the traceback above it is what made the failure
    unreadable in the first place.
    """
    reprcrash = getattr(report.longrepr, "reprcrash", None)
    message = getattr(reprcrash, "message", None)
    text = str(message) if message else str(report.longrepr or "")
    stripped = text.strip()
    if not stripped:
        return "(no detail)"
    return stripped.splitlines()[-1][:limit]


def _reset_item_after_failure(item: pytest.Item) -> None:
    """Clear the state a failed attempt leaves on the item and its fixtures.

    Without this the retry inherits the failure instead of re-running: a
    fixture that raised is still cached as failed and the session still
    believes setup failed, so the second attempt would report the first
    attempt's failure — the one thing a retry must not do.
    """
    setup_state = getattr(item.session, "_setupstate", None)
    if setup_state is not None and item in setup_state.stack:
        del setup_state.stack[item]

    fixture_info = getattr(item, "_fixtureinfo", None)
    name2fixturedefs = getattr(fixture_info, "name2fixturedefs", {})
    for fixture_name in name2fixturedefs:
        for fixture_def in name2fixturedefs[fixture_name]:
            cached = getattr(fixture_def, "cached_result", None)
            # cached_result is (result, cache_key, error); only a *failed*
            # fixture needs clearing, or the retry would re-execute every
            # fixture the test uses.
            if cached is not None and cached[2]:
                fixture_def.cached_result = None
                # pytest >= 9 refuses to re-run a fixture that still has
                # finalizers registered from the failed attempt.
                finalizers = getattr(fixture_def, "_finalizers", None)
                if finalizers is not None:
                    finalizers.clear()

    # pytest caches one class instance per item, so a retry would otherwise
    # reuse the instance — and any state stored on ``self`` — of the failure.
    if getattr(item, "_instance", None) is not None:
        del item._instance
        item._obj = None


def _attempt(item: pytest.Item, nextitem: pytest.Item | None) -> list[TestReport]:
    """Run one attempt and return its reports *without* logging them.

    Logging is left to the caller so the intermediate attempts of a retried
    test never reach the reporters as failures of their own.
    """
    from _pytest.runner import runtestprotocol

    return runtestprotocol(item, nextitem=nextitem, log=False)


def run_with_retries(
    item: pytest.Item,
    nextitem: pytest.Item | None,
    *,
    retries: int,
    run: FlakyRun,
    registry: Sequence[str],
) -> list[TestReport]:
    """Run ``item``, retrying it up to ``retries`` times when it fails.

    The deciding attempt is the first one that passes, or the last one allowed;
    its reports are returned so the caller logs one outcome per test rather
    than one per attempt.

    Args:
        item: The collected test.
        nextitem: The following test, per the pytest protocol.
        retries: Extra attempts allowed after the first.
        run: The run being recorded into.
        registry: Known-flaky patterns.

    Returns:
        The reports of the deciding attempt.
    """
    attempts_allowed = retries + 1
    reports: list[TestReport] = []
    first_failure: TestReport | None = None

    for attempt in range(1, attempts_allowed + 1):
        reports = _attempt(item, nextitem)
        failure = _first_failure(reports)

        if failure is None:
            if attempt > 1 and first_failure is not None:
                run.record(
                    Flake(
                        nodeid=item.nodeid,
                        classification=(KNOWN if matches_registry(item.nodeid, registry) else NEW),
                        attempts=attempt,
                        retries_allowed=retries,
                        phase=first_failure.when,
                        message=_describe(first_failure),
                        worker=run.worker,
                        duration_s=sum(report.duration for report in reports),
                    )
                )
            break

        if first_failure is None:
            first_failure = failure

        if attempt == attempts_allowed:
            run.record(
                Flake(
                    nodeid=item.nodeid,
                    classification=EXHAUSTED,
                    attempts=attempt,
                    retries_allowed=retries,
                    phase=failure.when,
                    message=_describe(failure),
                    worker=run.worker,
                    duration_s=failure.duration,
                )
            )
            break

        _reset_item_after_failure(item)

    return reports


# ---------------------------------------------------------------------------
# Aggregation — the CI job that runs the suite repeatedly
# ---------------------------------------------------------------------------


@dataclass
class AggregatedFlake:
    """A flake seen in one or more of the repeated runs.

    One test flaking in three of five runs is a different problem from one
    flaking in one of five, and neither is visible from a single run's report,
    so the number of runs it was seen in is carried alongside it.
    """

    nodeid: str
    classification: str
    runs: int
    attempts: int
    message: str
    workers: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return the aggregate as a JSON-serialisable dict."""
        return {
            "nodeid": self.nodeid,
            "classification": self.classification,
            "runs": self.runs,
            "attempts": self.attempts,
            "message": self.message,
            "workers": self.workers,
        }


def read_reports(paths: Iterable[pathlib.Path | str]) -> list[dict]:
    """Read report files, skipping any that a failed job did not produce."""
    payloads: list[dict] = []
    for path in paths:
        report = pathlib.Path(path)
        if not report.exists():
            continue
        payloads.append(json.loads(report.read_text()))
    return payloads


def aggregate_reports(payloads: Sequence[dict]) -> list[AggregatedFlake]:
    """Merge the reports of repeated runs into one row per flaky test.

    Args:
        payloads: Parsed report files, from :func:`read_reports`.

    Returns:
        One :class:`AggregatedFlake` per test, ordered by classification and
        then by how many runs it flaked in, most first — a test that flaked in
        every run is the one worth looking at.
    """
    merged: dict[str, AggregatedFlake] = {}
    for payload in payloads:
        for flake in payload.get("flakes", []):
            nodeid = flake["nodeid"]
            existing = merged.get(nodeid)
            if existing is None:
                merged[nodeid] = AggregatedFlake(
                    nodeid=nodeid,
                    classification=flake["classification"],
                    runs=1,
                    attempts=flake["attempts"],
                    message=flake["message"],
                    workers=[flake["worker"]],
                )
                continue
            existing.runs += 1
            existing.attempts = max(existing.attempts, flake["attempts"])
            if flake["worker"] not in existing.workers:
                existing.workers.append(flake["worker"])
            # A test that failed every attempt in any run is reported as
            # failing overall, whatever it managed in the others — that run
            # was red on it, and that is the stronger signal.
            if flake["classification"] == EXHAUSTED or existing.classification == EXHAUSTED:
                existing.classification = EXHAUSTED
            elif flake["classification"] == NEW:
                existing.classification = NEW

    order = {NEW: 0, EXHAUSTED: 1, KNOWN: 2}
    return sorted(
        merged.values(), key=lambda flake: (order.get(flake.classification, 3), -flake.runs)
    )


def aggregate_markdown(payloads: Sequence[dict]) -> str:
    """Render repeated runs as a markdown summary for a CI step summary."""
    flakes = aggregate_reports(payloads)
    runs = len(payloads)
    lines = [
        "## Flaky test report",
        "",
        f"{runs} run(s) reported.",
        "",
    ]
    if not flakes:
        lines.append("No test needed a retry in any run.")
        return "\n".join(lines) + "\n"

    new = [flake for flake in flakes if flake.classification == NEW]
    failing = [flake for flake in flakes if flake.classification == EXHAUSTED]
    lines += [
        f"{len(new)} new flake(s) and {len(failing)} test(s) that failed every attempt.",
        "",
        "| test | classification | runs seen | attempts | first failure |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| `{flake.nodeid}` | {flake.classification} | {flake.runs}/{runs} "
        f"| {flake.attempts} | {flake.message} |"
        for flake in flakes
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Aggregate report files and print the markdown summary.

    Used by ``.github/workflows/flaky-detection.yml`` once the repeated runs
    have uploaded their reports; exits non-zero with ``--fail-on-new`` when a
    test that is not in the known-flaky registry flaked, so the schedule is
    heard from instead of quietly going green.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m astroml.testing.flaky",
        description="Aggregate flaky-test reports written by repeated pytest runs.",
    )
    parser.add_argument("reports", nargs="+", help="Report files (missing ones are skipped).")
    parser.add_argument(
        "--fail-on-new",
        action="store_true",
        dest="fail_on_new",
        help="Exit non-zero when a flake is not in the known-flaky registry.",
    )
    args = parser.parse_args(argv)

    payloads = read_reports(args.reports)
    print(aggregate_markdown(payloads))

    flakes = aggregate_reports(payloads)
    if args.fail_on_new and any(flake.classification == NEW for flake in flakes):
        print("::error::flaky tests are not registered in tests/known_flaky.txt")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--flaky-*`` command line options."""
    group = parser.getgroup("flaky", "flaky test detection (issue #715)")
    group.addoption(
        "--flaky-retries",
        action="store",
        type=int,
        default=0,
        dest="flaky_retries",
        help="Extra attempts for a failing test (0, the default, disables retrying).",
    )
    group.addoption(
        "--flaky-report",
        action="store",
        default=str(DEFAULT_REPORT),
        dest="flaky_report",
        help="JSON report path; pass an empty value to write no report.",
    )
    group.addoption(
        "--flaky-registry",
        action="store",
        default=str(DEFAULT_REGISTRY),
        dest="flaky_registry",
        help="File listing tests already known to be flaky.",
    )
    group.addoption(
        "--flaky-fail-on-new",
        action="store_true",
        default=False,
        dest="flaky_fail_on_new",
        help="Exit non-zero when a test that is not registered flaked.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Create the run record and register the opt-out marker."""
    config.addinivalue_line(
        "markers",
        f"{_MARKER}: do not retry this test, even with --flaky-retries (issue #715)",
    )
    registry_path = config.getoption("flaky_registry")
    config.stash[FLAKY_RUN] = FlakyRun(
        retries_allowed=config.getoption("flaky_retries"),
        registry_path=registry_path,
        worker=_worker_id(),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> bool | None:
    """Run the test through the retry loop, or leave it to pytest.

    Returns ``True`` (handled) only when retrying is on and the test has not
    opted out, so a run without the flag is exactly the run it was before this
    plugin existed.
    """
    config = item.config
    retries = config.getoption("flaky_retries")
    if retries <= 0 or item.get_closest_marker(_MARKER) is not None:
        return None

    run = config.stash[FLAKY_RUN]
    registry = load_registry(config.getoption("flaky_registry"))

    # The default protocol emits these around the attempt; replacing the
    # protocol means emitting them here, or reporters report a test with no
    # start and never finalise it.
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    reports = run_with_retries(item, nextitem, retries=retries, run=run, registry=registry)
    for report in reports:
        item.ihook.pytest_runtest_logreport(report=report)
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Print what flaked, so it is visible without opening the JSON report."""
    run = config.stash.get(FLAKY_RUN, None)
    if run is None:
        return
    if not run.flakes:
        if run.retries_allowed > 0:
            terminalreporter.write_line("[flaky] no test needed a retry", bold=True)
        return

    terminalreporter.write_sep("=", "flaky tests")
    for flake in run.flakes:
        terminalreporter.write_line(
            f"[flaky] {flake.classification}: {flake.nodeid} "
            f"(attempt {flake.attempts}/{flake.retries_allowed + 1}, "
            f"first failure in {flake.phase})"
        )
    terminalreporter.write_line(
        f"[flaky] {len(run.new_flakes)} new, {len(run.known_flakes)} known, "
        f"{len(run.failed_tests)} failed every attempt"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the report and, when asked, fail the run over an unregistered flake."""
    config = session.config
    run = config.stash.get(FLAKY_RUN, None)
    if run is None:
        return

    # Only a run that was willing to retry has anything to report; a plain
    # ``pytest`` invocation stays exactly what it was, without leaving a report
    # file behind in the working tree.
    if run.retries_allowed <= 0:
        return

    report_path = config.getoption("flaky_report")
    if report_path:
        written = run.write(report_path, worker=run.worker)
        terminalreporter = config.pluginmanager.get_plugin("terminalreporter")
        if terminalreporter is not None:
            terminalreporter.write_line(f"[flaky] report written to {written}")

    if run.new_flakes and config.getoption("flaky_fail_on_new"):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
