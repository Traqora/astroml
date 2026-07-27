"""Pull request size evaluation (Issue #557).

Large pull requests are harder to review and carry more merge risk. This
module implements the *soft* enforcement policy used by the
``pr-size-limit`` GitHub Actions workflow:

* a PR over the thresholds gets a warning comment,
* CI is never failed because of size alone,
* ``refactor:large`` raises the file-count ceiling for planned refactors,
* ``[large PR]`` in the title opts out of the check entirely.

The evaluation logic lives here (rather than inline in ``github-script``)
so it is type-checked by ``mypy`` and covered by unit tests. The workflow
invokes it as ``python -m astroml.ci.pr_size``.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, Sequence

#: Marker used to find (and update) the bot's previous size comment.
COMMENT_MARKER: Final[str] = "<!-- astroml:pr-size-limit -->"

#: Title opt-out token. Case-insensitive.
TITLE_EXEMPTION: Final[str] = "[large pr]"

#: Label that raises the file-count ceiling for planned large refactors.
LARGE_REFACTOR_LABEL: Final[str] = "refactor:large"


@dataclass(frozen=True)
class SizeThresholds:
    """Limits a pull request is measured against.

    Attributes:
        max_lines: Maximum combined additions + deletions.
        max_files: Maximum number of changed files.
        max_files_large_refactor: File ceiling when the PR carries the
            ``refactor:large`` label.
    """

    max_lines: int = 1000
    max_files: int = 10
    max_files_large_refactor: int = 50

    def file_limit(self, *, large_refactor: bool) -> int:
        """Return the applicable file ceiling for this PR."""
        return self.max_files_large_refactor if large_refactor else self.max_files


DEFAULT_THRESHOLDS: Final[SizeThresholds] = SizeThresholds()


@dataclass(frozen=True)
class PullRequestStats:
    """Measured size of a pull request.

    Attributes:
        additions: Lines added.
        deletions: Lines removed.
        changed_files: Number of files touched.
        title: Pull request title, used for the ``[large PR]`` opt-out.
        labels: Label names attached to the pull request.
    """

    additions: int
    deletions: int
    changed_files: int
    title: str = ""
    labels: Sequence[str] = field(default_factory=tuple)

    @property
    def total_lines(self) -> int:
        """Combined additions and deletions."""
        return self.additions + self.deletions

    @classmethod
    def from_event(cls, payload: dict[str, Any]) -> "PullRequestStats":
        """Build stats from a GitHub ``pull_request`` event payload.

        Args:
            payload: The full webhook payload, or the ``pull_request``
                object itself.

        Returns:
            Parsed :class:`PullRequestStats`.

        Raises:
            ValueError: If no ``pull_request`` object can be located.
        """
        pull_request = payload.get("pull_request", payload)
        if not isinstance(pull_request, dict) or "changed_files" not in pull_request:
            raise ValueError(
                "Event payload does not contain a pull_request object with "
                "size fields (additions/deletions/changed_files)."
            )

        raw_labels = pull_request.get("labels") or []
        labels = tuple(_label_names(raw_labels))

        return cls(
            additions=int(pull_request.get("additions", 0)),
            deletions=int(pull_request.get("deletions", 0)),
            changed_files=int(pull_request.get("changed_files", 0)),
            title=str(pull_request.get("title", "")),
            labels=labels,
        )


def _label_names(raw_labels: Iterable[Any]) -> list[str]:
    """Normalise GitHub label objects (or plain strings) to lowercase names."""
    names: list[str] = []
    for label in raw_labels:
        if isinstance(label, dict):
            name = label.get("name")
        else:
            name = label
        if isinstance(name, str):
            names.append(name.strip().lower())
    return names


@dataclass(frozen=True)
class SizeVerdict:
    """Outcome of a size evaluation.

    Attributes:
        exceeded: True when at least one applicable limit was exceeded.
        exempt: True when the ``[large PR]`` title opt-out applies.
        large_refactor: True when the ``refactor:large`` label applies.
        reasons: Human-readable descriptions of each breached limit.
        stats: The stats that were evaluated.
        thresholds: The thresholds that were applied.
    """

    exceeded: bool
    exempt: bool
    large_refactor: bool
    reasons: tuple[str, ...]
    stats: PullRequestStats
    thresholds: SizeThresholds

    @property
    def should_comment(self) -> bool:
        """True when a warning comment should be posted."""
        return self.exceeded and not self.exempt


def is_exempt(title: str) -> bool:
    """Return True when the PR title opts out of the size check."""
    return TITLE_EXEMPTION in title.lower()


def has_large_refactor_label(labels: Sequence[str]) -> bool:
    """Return True when the ``refactor:large`` label is present."""
    return LARGE_REFACTOR_LABEL in {label.strip().lower() for label in labels}


def evaluate_pr_size(
    stats: PullRequestStats,
    thresholds: SizeThresholds = DEFAULT_THRESHOLDS,
) -> SizeVerdict:
    """Compare a pull request against the size policy.

    Args:
        stats: Measured pull request size.
        thresholds: Limits to apply. Defaults to :data:`DEFAULT_THRESHOLDS`.

    Returns:
        A :class:`SizeVerdict` describing which limits (if any) were breached.
    """
    exempt = is_exempt(stats.title)
    large_refactor = has_large_refactor_label(stats.labels)
    file_limit = thresholds.file_limit(large_refactor=large_refactor)

    reasons: list[str] = []
    if stats.total_lines > thresholds.max_lines:
        reasons.append(
            f"{stats.total_lines} lines changed "
            f"(+{stats.additions}/-{stats.deletions}) exceeds the "
            f"{thresholds.max_lines}-line limit."
        )
    if stats.changed_files > file_limit:
        suffix = " for `refactor:large` pull requests" if large_refactor else ""
        reasons.append(
            f"{stats.changed_files} files changed exceeds the "
            f"{file_limit}-file limit{suffix}."
        )

    return SizeVerdict(
        exceeded=bool(reasons),
        exempt=exempt,
        large_refactor=large_refactor,
        reasons=tuple(reasons),
        stats=stats,
        thresholds=thresholds,
    )


def render_comment(verdict: SizeVerdict) -> str:
    """Render the warning comment body for an oversized pull request.

    Args:
        verdict: Result of :func:`evaluate_pr_size`.

    Returns:
        Markdown body, prefixed with :data:`COMMENT_MARKER` so the workflow
        can update its previous comment instead of posting duplicates.
    """
    stats = verdict.stats
    file_limit = verdict.thresholds.file_limit(large_refactor=verdict.large_refactor)

    lines = [
        COMMENT_MARKER,
        "### 📏 Pull request size warning",
        "",
        "This pull request exceeds the review-friendly size limits. "
        "**CI is not blocked** — this is guidance, not a gate.",
        "",
        "| Metric | This PR | Limit |",
        "| --- | ---: | ---: |",
        f"| Lines changed | {stats.total_lines} | {verdict.thresholds.max_lines} |",
        f"| Files changed | {stats.changed_files} | {file_limit} |",
        "",
        "**Why it matters**",
        "",
        "Review quality drops sharply with diff size: defects slip through, "
        "review turnaround grows, and reverts become riskier because unrelated "
        "changes are bundled together.",
        "",
        "**What to do**",
        "",
        "- Split the change into a stack of smaller, independently reviewable PRs.",
        "- Land refactors and behaviour changes separately.",
        "- Hide incomplete work behind a feature flag and merge it incrementally.",
        "- Move generated files, lockfiles, or vendored code into their own PR.",
        "",
        "**Legitimately large?**",
        "",
        f"- Add the `{LARGE_REFACTOR_LABEL}` label to raise the file ceiling to "
        f"{verdict.thresholds.max_files_large_refactor} files.",
        "- Add `[large PR]` to the title to skip this check entirely.",
        "",
        "<details><summary>Breached limits</summary>",
        "",
    ]
    lines.extend(f"- {reason}" for reason in verdict.reasons)
    lines.extend(["", "</details>"])
    return "\n".join(lines)


def _write_github_output(verdict: SizeVerdict, path: str) -> None:
    """Append workflow outputs to the ``$GITHUB_OUTPUT`` file."""
    body = render_comment(verdict) if verdict.should_comment else ""
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"exceeded={'true' if verdict.exceeded else 'false'}\n")
        handle.write(f"exempt={'true' if verdict.exempt else 'false'}\n")
        handle.write(
            f"should_comment={'true' if verdict.should_comment else 'false'}\n"
        )
        handle.write("body<<PR_SIZE_EOF\n")
        handle.write(f"{body}\n")
        handle.write("PR_SIZE_EOF\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by the ``pr-size-limit`` workflow.

    Reads the event payload from ``GITHUB_EVENT_PATH`` (or the first CLI
    argument) and writes ``exceeded``, ``exempt``, ``should_comment`` and
    ``body`` to ``$GITHUB_OUTPUT``.

    Args:
        argv: Optional argument vector, defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code. Always ``0`` on a successful evaluation — size
        alone never fails CI.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    event_path = args[0] if args else os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        print("No event payload available (GITHUB_EVENT_PATH unset).", file=sys.stderr)
        return 1

    with open(event_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    verdict = evaluate_pr_size(PullRequestStats.from_event(payload))

    summary = (
        f"lines={verdict.stats.total_lines} "
        f"files={verdict.stats.changed_files} "
        f"exceeded={verdict.exceeded} exempt={verdict.exempt} "
        f"large_refactor={verdict.large_refactor}"
    )
    print(summary)
    for reason in verdict.reasons:
        print(f"  - {reason}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        _write_github_output(verdict, output_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
