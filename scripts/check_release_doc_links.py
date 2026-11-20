#!/usr/bin/env python3
"""Check the documentation links embedded in the release workflow.

The release workflow (``.github/workflows/release.yml``) writes a Markdown
body into the GitHub Release it creates via a ``cat <<'EOF'`` heredoc. That
body is the first thing a user reads after a tag ships, so every documentation
link in it must point at a file that actually exists in the repository -- a
stale link sends someone who just pulled the release to a 404.

The general ``check_doc_links.py`` does not cover this: it scans ``docs/`` plus
``README.md``/``CONTRIBUTING.md``, and the release notes live inside a YAML
heredoc that none of those tools read. This script is the missing check. It
extracts the release-notes heredoc, pulls out the Markdown links, and resolves
each one against the repository root (the base GitHub uses for relative links
in a release body). External URLs are ignored.

Only the first heredoc in the workflow is examined. ``release.yml`` contains
exactly one -- the release notes -- so that is sufficient and keeps the parser
trivial.

Usage::

    python scripts/check_release_doc_links.py                 # default path
    python scripts/check_release_doc_links.py --root /path     # other repo root
    python scripts/check_release_doc_links.py --json           # machine-readable

Exit status is 0 when every link resolves and 1 otherwise, so it drops into
CI (it runs in ``.github/workflows/docs.yml``).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The release workflow this script checks by default.
DEFAULT_WORKFLOW = ".github/workflows/release.yml"

EXTERNAL_SCHEMES = ("http://", "https://", "mailto:", "git:", "ftp://")

#: A heredoc opener: ``<<'EOF'`` / ``<<EOF`` / ``<<-EOF``. The captured group is
#: the delimiter word.
_HEREDOC_OPEN = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")

#: Markdown inline link: ``[label](target)`` with an optional ``"title"``.
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(\s*([^()\s]+)(?:\s+\"[^\"]*\")?\s*\)")

#: Fenced code block delimiters; links inside them are literal text, not links.
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class BrokenLink:
    """One release-notes link that does not resolve."""

    line: int
    target: str
    kind: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"line": self.line, "target": self.target, "kind": self.kind, "detail": self.detail}

    def __str__(self) -> str:
        return f"release notes:{self.line}: {self.target} -- {self.kind}: {self.detail}"


def github_slug(heading: str) -> str:
    """The anchor GitHub generates for a Markdown heading.

    GitHub trims and lowercases first, drops characters that are not word
    characters, spaces or hyphens, then maps each remaining space to a hyphen.
    """
    text = heading.strip().lstrip("#").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def extract_release_notes(workflow_text: str) -> str:
    """Return the body of the first ``cat <<'EOF'`` heredoc in ``workflow_text``.

    The body is returned verbatim (including its YAML indentation) up to but
    not including the closing delimiter line. Per-line indentation is stripped
    by the link extractor, so callers need not normalise it.
    """
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        match = _HEREDOC_OPEN.search(line)
        if not match:
            continue
        delimiter = match.group(2)
        body: list[str] = []
        for body_line in lines[index + 1 :]:
            if body_line.strip() == delimiter:
                return "\n".join(body)
            body.append(body_line)
        raise ValueError(f"unterminated heredoc '{delimiter}' opened at line {index + 1}")
    raise ValueError("no cat <<'EOF' heredoc found in the release workflow")


def iter_markdown_links(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, target)`` for every Markdown link outside fences."""
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _MARKDOWN_LINK.finditer(line):
            yield number, match.group(1)


def _anchor_lines(text: str) -> set[str]:
    """Collect heading anchors a Markdown document can be linked to."""
    anchors: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if line.lstrip().startswith("#"):
            anchors.add(github_slug(line))
    return anchors


def check_link(root: Path, target: str) -> tuple[str, str] | None:
    """Return ``(kind, detail)`` when ``target`` does not resolve, else ``None``.

    Targets resolve against the repository root, matching how GitHub renders
    relative links in a release body.
    """
    if target.startswith(EXTERNAL_SCHEMES):
        return None

    path_part, _, anchor = target.partition("#")
    resolved = (root / path_part).resolve()

    if not resolved.exists():
        rel = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
        return ("missing target", f"no such file: {rel}")

    if not anchor:
        return None

    if resolved.suffix != ".md":
        # Anchors into non-Markdown files cannot be cheaply verified; the path
        # existence check above is the meaningful check.
        return None

    anchors = _anchor_lines(resolved.read_text(encoding="utf-8", errors="replace"))
    if anchor in anchors:
        return None
    return ("missing anchor", f"{resolved.relative_to(root)} has no anchor '{anchor}'")


def check_release_notes(workflow_text: str, root: Path) -> list[BrokenLink]:
    """Return every broken documentation link in the release-notes heredoc."""
    notes = extract_release_notes(workflow_text)
    broken: list[BrokenLink] = []
    for line, target in iter_markdown_links(notes):
        problem = check_link(root, target)
        if problem is not None:
            broken.append(BrokenLink(line, target, problem[0], problem[1]))
    return broken


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=".", help="repository root used to resolve release-notes links"
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help=f"release workflow to check (default: {DEFAULT_WORKFLOW})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    workflow = (root / args.workflow).resolve()
    if not workflow.exists():
        print(f"release workflow not found: {workflow}", file=sys.stderr)
        return 2

    broken = check_release_notes(workflow.read_text(encoding="utf-8"), root)

    if args.json:
        print(json.dumps([link.as_dict() for link in broken], indent=2))
    else:
        if broken and not args.quiet:
            for link in broken:
                print(link)
            print()
        summary = (
            f"checked release notes: {len(broken)} broken doc link(s)"
            if broken
            else "checked release notes: all doc links resolve"
        )
        print(summary)

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
