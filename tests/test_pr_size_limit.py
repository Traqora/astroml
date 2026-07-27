"""Tests for the PR size limit CI policy (Issue #557)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astroml.ci.pr_size import (
    COMMENT_MARKER,
    DEFAULT_THRESHOLDS,
    LARGE_REFACTOR_LABEL,
    PullRequestStats,
    SizeThresholds,
    evaluate_pr_size,
    has_large_refactor_label,
    is_exempt,
    main,
    render_comment,
)


def _stats(**overrides: Any) -> PullRequestStats:
    payload: dict[str, Any] = {
        "additions": 10,
        "deletions": 5,
        "changed_files": 3,
        "title": "feat: small change",
        "labels": (),
    }
    payload.update(overrides)
    return PullRequestStats(**payload)


class TestPullRequestStats:
    def test_total_lines_sums_additions_and_deletions(self) -> None:
        assert _stats(additions=120, deletions=30).total_lines == 150

    def test_from_event_parses_webhook_payload(self) -> None:
        stats = PullRequestStats.from_event(
            {
                "pull_request": {
                    "additions": 900,
                    "deletions": 200,
                    "changed_files": 12,
                    "title": "refactor: move modules",
                    "labels": [{"name": "Refactor:Large"}, {"name": "api"}],
                }
            }
        )

        assert stats.total_lines == 1100
        assert stats.changed_files == 12
        assert stats.labels == ("refactor:large", "api")

    def test_from_event_accepts_bare_pull_request_object(self) -> None:
        stats = PullRequestStats.from_event(
            {"additions": 1, "deletions": 2, "changed_files": 1, "title": "x"}
        )
        assert stats.total_lines == 3

    def test_from_event_tolerates_string_labels(self) -> None:
        stats = PullRequestStats.from_event(
            {
                "pull_request": {
                    "additions": 0,
                    "deletions": 0,
                    "changed_files": 0,
                    "labels": ["Refactor:Large", 7, None],
                }
            }
        )
        assert stats.labels == ("refactor:large",)

    def test_from_event_rejects_payload_without_pull_request(self) -> None:
        with pytest.raises(ValueError, match="does not contain a pull_request"):
            PullRequestStats.from_event({"action": "opened"})


class TestExemptions:
    @pytest.mark.parametrize(
        "title",
        ["[large PR] migrate schema", "chore: vendor deps [LARGE PR]"],
    )
    def test_title_exemption_is_case_insensitive(self, title: str) -> None:
        assert is_exempt(title) is True

    def test_title_without_token_is_not_exempt(self) -> None:
        assert is_exempt("feat: a large pull request") is False

    def test_large_refactor_label_detected(self) -> None:
        assert has_large_refactor_label([" Refactor:Large "]) is True

    def test_missing_large_refactor_label(self) -> None:
        assert has_large_refactor_label(["bug", "api"]) is False


class TestThresholds:
    def test_default_file_limit(self) -> None:
        assert DEFAULT_THRESHOLDS.file_limit(large_refactor=False) == 10

    def test_large_refactor_file_limit(self) -> None:
        assert DEFAULT_THRESHOLDS.file_limit(large_refactor=True) == 50


class TestEvaluatePrSize:
    def test_small_pr_passes(self) -> None:
        verdict = evaluate_pr_size(_stats())

        assert verdict.exceeded is False
        assert verdict.should_comment is False
        assert verdict.reasons == ()

    def test_boundary_values_are_allowed(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=1000, deletions=0, changed_files=10)
        )
        assert verdict.exceeded is False

    def test_line_limit_breach_reported(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=900, deletions=200, changed_files=4)
        )

        assert verdict.exceeded is True
        assert verdict.should_comment is True
        assert len(verdict.reasons) == 1
        assert "1100 lines changed" in verdict.reasons[0]

    def test_file_limit_breach_reported(self) -> None:
        verdict = evaluate_pr_size(_stats(changed_files=25))

        assert verdict.exceeded is True
        assert "25 files changed" in verdict.reasons[0]
        assert "10-file limit" in verdict.reasons[0]

    def test_both_limits_breached(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=2000, deletions=500, changed_files=40)
        )
        assert len(verdict.reasons) == 2

    def test_large_refactor_label_raises_file_ceiling(self) -> None:
        verdict = evaluate_pr_size(
            _stats(changed_files=40, labels=(LARGE_REFACTOR_LABEL,))
        )

        assert verdict.large_refactor is True
        assert verdict.exceeded is False

    def test_large_refactor_label_still_capped(self) -> None:
        verdict = evaluate_pr_size(
            _stats(changed_files=60, labels=(LARGE_REFACTOR_LABEL,))
        )

        assert verdict.exceeded is True
        assert "50-file limit for `refactor:large`" in verdict.reasons[0]

    def test_large_refactor_label_does_not_raise_line_ceiling(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=5000, deletions=0, labels=(LARGE_REFACTOR_LABEL,))
        )
        assert verdict.exceeded is True

    def test_title_exemption_suppresses_comment(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=5000, deletions=0, changed_files=99, title="[large PR] x")
        )

        assert verdict.exceeded is True
        assert verdict.exempt is True
        assert verdict.should_comment is False

    def test_custom_thresholds_are_applied(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=50, deletions=0, changed_files=1),
            SizeThresholds(max_lines=10, max_files=1),
        )
        assert verdict.exceeded is True


class TestRenderComment:
    def test_comment_contains_marker_and_metrics(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=900, deletions=300, changed_files=22)
        )
        body = render_comment(verdict)

        assert body.startswith(COMMENT_MARKER)
        assert "| Lines changed | 1200 | 1000 |" in body
        assert "| Files changed | 22 | 10 |" in body
        assert "CI is not blocked" in body
        assert "`refactor:large`" in body
        assert "`[large PR]`" in body

    def test_comment_lists_every_breached_limit(self) -> None:
        verdict = evaluate_pr_size(
            _stats(additions=2000, deletions=0, changed_files=40)
        )
        body = render_comment(verdict)

        for reason in verdict.reasons:
            assert f"- {reason}" in body

    def test_comment_shows_raised_ceiling_for_large_refactor(self) -> None:
        verdict = evaluate_pr_size(
            _stats(changed_files=80, labels=(LARGE_REFACTOR_LABEL,))
        )
        assert "| Files changed | 80 | 50 |" in render_comment(verdict)


class TestMain:
    @staticmethod
    def _write_event(tmp_path: Path, pull_request: dict[str, Any]) -> str:
        event_path = tmp_path / "event.json"
        event_path.write_text(
            json.dumps({"pull_request": pull_request}), encoding="utf-8"
        )
        return str(event_path)

    def test_main_writes_outputs_for_oversized_pr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = self._write_event(
            tmp_path,
            {
                "additions": 2000,
                "deletions": 100,
                "changed_files": 30,
                "title": "feat: big",
                "labels": [],
            },
        )
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))

        assert main([event]) == 0

        written = output.read_text(encoding="utf-8")
        assert "exceeded=true" in written
        assert "should_comment=true" in written
        assert COMMENT_MARKER in written
        assert written.count("PR_SIZE_EOF") == 2

    def test_main_writes_empty_body_when_within_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = self._write_event(
            tmp_path,
            {"additions": 5, "deletions": 5, "changed_files": 2, "title": "fix: typo"},
        )
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))

        assert main([event]) == 0

        written = output.read_text(encoding="utf-8")
        assert "exceeded=false" in written
        assert "should_comment=false" in written
        assert COMMENT_MARKER not in written

    def test_main_reads_event_path_from_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = self._write_event(
            tmp_path,
            {"additions": 1, "deletions": 1, "changed_files": 1, "title": "fix"},
        )
        monkeypatch.setenv("GITHUB_EVENT_PATH", event)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)

        assert main([]) == 0

    def test_main_errors_without_event_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        assert main([]) == 1

    def test_main_is_exempt_and_skips_comment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        event = self._write_event(
            tmp_path,
            {
                "additions": 9000,
                "deletions": 0,
                "changed_files": 200,
                "title": "[large PR] vendor deps",
            },
        )
        output = tmp_path / "output.txt"
        monkeypatch.setenv("GITHUB_OUTPUT", str(output))

        assert main([event]) == 0

        written = output.read_text(encoding="utf-8")
        assert "exempt=true" in written
        assert "should_comment=false" in written
