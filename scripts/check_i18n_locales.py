#!/usr/bin/env python3
"""Validate docs/i18n/languages.yml against the actual locale folders.

Issue #851. docs/i18n/README.md promises reviewers will check that "the code,
folder name, and file name all match languages.yml", but nothing automates it:
a locale can be registered with no folder (a typo, or a folder renamed without
updating the registry), or a folder can exist with no registry entry (added
without the required step), and nothing notices until a reader hits a dead
link. The Spanish (``es``) locale is the only non-source entry today, so a
drifted registry currently means a silently broken (or silently missing)
Spanish README.

This filters languages.yml's locale list down to the ones that are actually
inconsistent with the filesystem, and reports them. Exit status is 0 when the
registry and the folders agree, 1 otherwise.

Usage::

    python scripts/check_i18n_locales.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

I18N_DIR = Path(__file__).resolve().parents[1] / "docs" / "i18n"
LANGUAGES_FILE = I18N_DIR / "languages.yml"
VALID_PROGRESS = frozenset({"stub", "partial", "complete"})


def _load_locales(languages_file: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(languages_file.read_text(encoding="utf-8")) or {}
    return list(data.get("locales", []))


def find_locale_drift(i18n_dir: Path, locales: list[dict[str, Any]]) -> list[str]:
    """Return one human-readable problem description per inconsistent locale.

    A registry entry is inconsistent if its folder is missing, its
    README.<code>.md is missing, or its progress value isn't one of the
    documented stages. A folder under docs/i18n/ with no matching registry
    entry is also reported, the other half of the same drift.
    """
    problems: list[str] = []
    registered_dirs: set[str] = set()

    for locale in locales:
        code = locale.get("code")
        dir_name = locale.get("dir")
        progress = locale.get("progress")

        if not code or not dir_name:
            problems.append(f"locale entry missing 'code' or 'dir': {locale!r}")
            continue

        registered_dirs.add(dir_name)
        locale_dir = i18n_dir / dir_name
        if not locale_dir.is_dir():
            problems.append(f"'{code}' registers dir '{dir_name}' but {locale_dir} does not exist")
            continue

        if progress not in VALID_PROGRESS:
            problems.append(
                f"'{code}' has progress '{progress}', expected one of {sorted(VALID_PROGRESS)}"
            )

        if locale.get("source"):
            continue  # the English source README.md lives at the repo root.

        expected_readme = locale_dir / f"README.{code}.md"
        if not expected_readme.is_file():
            problems.append(f"'{code}' is registered but {expected_readme} does not exist")

    for entry in sorted(i18n_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if entry.name not in registered_dirs:
            problems.append(f"folder '{entry.name}/' exists but is not registered in languages.yml")

    return problems


def main(argv: list[str] | None = None) -> int:
    if not LANGUAGES_FILE.is_file():
        print(f"error: {LANGUAGES_FILE} not found", file=sys.stderr)
        return 1

    locales = _load_locales(LANGUAGES_FILE)
    problems = find_locale_drift(I18N_DIR, locales)

    if problems:
        print(f"docs/i18n locale registry drift ({len(problems)}):", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(f"docs/i18n locale registry is consistent ({len(locales)} locales checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
