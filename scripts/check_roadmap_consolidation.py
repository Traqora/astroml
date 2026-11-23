#!/usr/bin/env python3
"""Verify unsupported feature claims live in docs/ROADMAP.md, not inline.

Before this check existed, "not yet supported" features were scattered as
``## Future Enhancements`` and ``### Planned Features`` sections inside
feature docs, and the README roadmap carried to-dos for capabilities that had
already shipped. Both make the onboarding path inaccurate: a reader following
the quick start cannot tell what works today from what is planned.

This script enforces the consolidation:

- ``docs/DATA_QUALITY_VALIDATION.md`` and ``docs/FRAUD_REGISTRY_CONTRACT.md``
  must no longer carry inline ``Future Enhancements``/``Planned Features``
  headings -- those claims now live in ``docs/ROADMAP.md``.
- ``README.md`` must not list an already-shipped capability
  (feature store, model benchmarking suite, Docker deployment) as an
  unchecked roadmap to-do.
- ``docs/ROADMAP.md`` must exist and contain the consolidated claims.

Usage::

    python scripts/check_roadmap_consolidation.py
    python scripts/check_roadmap_consolidation.py --json

Exit status is 0 when the roadmap is the single source of truth and 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Feature docs that previously carried inline "not yet supported" sections.
CONSOLIDATED_DOCS = ("docs/DATA_QUALITY_VALIDATION.md", "docs/FRAUD_REGISTRY_CONTRACT.md")

#: Headings that signal an inline, unconsolidated future-claim section.
_FUTURE_HEADINGS = re.compile(
    r"^#{2,6}\s+(?:Future Enhancements|Planned Features)\s*$", re.MULTILINE
)

#: Capabilities that have shipped and must not appear as README roadmap to-dos.
STALE_TODOS = ("Feature store", "Model benchmarking suite", "Docker deployment")
# Match only an *unchecked* box (``- [ ]``); a checked ``- [x]`` item is no
# longer a claim that the feature is unsupported.
_STALE_TODO = re.compile(
    r"^\s*-\s*\[ \]\s+(" + "|".join(re.escape(t) for t in STALE_TODOS) + r")\s*$",
    re.MULTILINE,
)

#: Claims that must appear in the consolidated roadmap.
ROADMAP_CLAIMS = (
    "Real-time streaming ingestion",
    "Temporal GNN models",
    "Contrastive learning pipelines",
    "Machine Learning Validation",
    "Event Logging",
)


def find_unconsolidated_future_claims(doc_text: str) -> list[str]:
    """Return inline future-claim headings that should have been moved out."""
    return _FUTURE_HEADINGS.findall(doc_text)


def find_stale_readme_todos(readme_text: str) -> list[str]:
    """Return already-shipped capabilities still listed as roadmap to-dos."""
    return [match.group(1) for match in _STALE_TODO.finditer(readme_text)]


def missing_roadmap_claims(roadmap_text: str) -> list[str]:
    """Return consolidated claims absent from the roadmap."""
    return [claim for claim in ROADMAP_CLAIMS if claim not in roadmap_text]


def validate(repo_root: Path) -> list[str]:
    """Return human-readable mismatch messages (empty when all consistent)."""
    mismatches: list[str] = []

    for rel in CONSOLIDATED_DOCS:
        path = repo_root / rel
        if not path.exists():
            mismatches.append(f"{rel}: file missing")
            continue
        claims = find_unconsolidated_future_claims(path.read_text(encoding="utf-8"))
        if claims:
            mismatches.append(
                f"{rel}: inline future-claim heading(s) not moved to ROADMAP.md: {claims}"
            )

    readme = repo_root / "README.md"
    if not readme.exists():
        mismatches.append("README.md: file missing")
    else:
        stale = find_stale_readme_todos(readme.read_text(encoding="utf-8"))
        if stale:
            mismatches.append(
                f"README.md: already-shipped feature(s) still listed as roadmap to-do(s): {stale}"
            )

    roadmap = repo_root / "docs" / "ROADMAP.md"
    if not roadmap.exists():
        mismatches.append("docs/ROADMAP.md: file missing")
    else:
        missing = missing_roadmap_claims(roadmap.read_text(encoding="utf-8"))
        if missing:
            mismatches.append(
                f"docs/ROADMAP.md: missing consolidated claim(s): {missing}"
            )

    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="repository root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    mismatches = validate(Path(args.root).resolve())

    if args.json:
        print(json.dumps(mismatches, indent=2))
    else:
        for mismatch in mismatches:
            print(mismatch)
        print(
            f"checked roadmap consolidation: {len(mismatches)} issue(s)"
            if mismatches
            else "checked roadmap consolidation: roadmap is the single source of truth"
        )

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
