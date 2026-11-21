#!/usr/bin/env python3
"""Verify that docs/CI_DATABASE_REQUIREMENTS.md matches the real workflows.

The doc tells contributors which CI/CD workflows need PostgreSQL and which do
not. Left to drift, it becomes a lie: someone adds a ``postgres`` service
container to a workflow and forgets the doc, or removes one and leaves a claim
behind. This script is the guardrail.

It scans ``.github/workflows/*.yml`` for the two signals that a job needs
PostgreSQL -- a ``postgres:*`` service container image or a ``postgresql://``
connection string -- and cross-checks them against the two tables in the doc.
A workflow that uses PostgreSQL must be listed under "needs PostgreSQL"; one
listed there must actually use it; one listed under "does NOT need" must not.

Usage::

    python scripts/check_workflow_postgres.py
    python scripts/check_workflow_postgres.py --json

Exit status is 0 when the doc and the workflows agree and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

DEFAULT_DOC = "docs/CI_DATABASE_REQUIREMENTS.md"
WORKFLOW_GLOB = ".github/workflows/*.yml"

#: A ``postgres:*`` service container image.
_POSTGRES_IMAGE = re.compile(r"image:\s*postgres\b")

NEEDS_HEADING = "### Workflows that need PostgreSQL"
NOT_NEEDS_HEADING = "### Workflows that do NOT need PostgreSQL"

#: A link to a workflow file, e.g. ``[ci.yml](../.github/workflows/ci.yml)``.
_WORKFLOW_LINK = re.compile(r"\.github/workflows/([\w.-]+)")


def detect_postgres(workflow_text: str) -> bool:
    """True if ``workflow_text`` starts a postgres service or uses postgresql."""
    return bool(_POSTGRES_IMAGE.search(workflow_text)) or "postgresql://" in workflow_text


def iter_workflow_files(repo_root: Path) -> Iterator[Path]:
    yield from sorted((repo_root / ".github" / "workflows").glob("*.yml"))


def workflows_needing_postgres(repo_root: Path) -> set[str]:
    """Basenames of workflows that need PostgreSQL."""
    return {
        path.name
        for path in iter_workflow_files(repo_root)
        if detect_postgres(path.read_text(encoding="utf-8"))
    }


def _section_lines(doc_text: str, heading: str) -> list[str]:
    """Return the body lines under ``heading`` up to the next heading."""
    lines = doc_text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return []
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## ") or line.startswith("### "):
            break
        body.append(line)
    return body


def _workflows_in_section(doc_text: str, heading: str) -> set[str]:
    return {
        match.group(1)
        for line in _section_lines(doc_text, heading)
        for match in _WORKFLOW_LINK.finditer(line)
    }


def doc_needs_postgres(doc_text: str) -> set[str]:
    return _workflows_in_section(doc_text, NEEDS_HEADING)


def doc_does_not_need_postgres(doc_text: str) -> set[str]:
    return _workflows_in_section(doc_text, NOT_NEEDS_HEADING)


def validate(repo_root: Path, doc_text: str) -> list[str]:
    """Return human-readable mismatch messages (empty when all consistent)."""
    detected = workflows_needing_postgres(repo_root)
    needs = doc_needs_postgres(doc_text)
    not_needs = doc_does_not_need_postgres(doc_text)

    if not needs or not not_needs:
        return ["could not parse both PostgreSQL requirement tables from the doc"]

    mismatches: list[str] = []
    # False negatives: a workflow that needs PostgreSQL must be listed.
    for name in sorted(detected - needs):
        mismatches.append(
            f"{name} uses PostgreSQL but the doc does not list it under 'needs PostgreSQL'"
        )
    # False positives: a workflow listed as needing PostgreSQL must use it.
    for name in sorted(needs - detected):
        mismatches.append(
            f"{name} is listed as needing PostgreSQL but the workflow does not use it"
        )
    # Contradictions: a workflow cannot be listed as not-needing yet use it.
    for name in sorted(not_needs & detected):
        mismatches.append(
            f"{name} is listed as NOT needing PostgreSQL but the workflow uses it"
        )
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument(
        "--doc",
        default=DEFAULT_DOC,
        help=f"doc to check (default: {DEFAULT_DOC})",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    doc_path = (root / args.doc).resolve()
    if not doc_path.exists():
        print(f"doc not found: {doc_path}", file=sys.stderr)
        return 2

    mismatches = validate(root, doc_path.read_text(encoding="utf-8"))

    if args.json:
        print(json.dumps(mismatches, indent=2))
    else:
        for mismatch in mismatches:
            print(mismatch)
        print(
            f"checked {len(list(iter_workflow_files(root)))} workflow(s): "
            f"{len(mismatches)} mismatch(es)"
            if mismatches
            else f"checked {len(list(iter_workflow_files(root)))} workflow(s): doc matches"
        )

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
