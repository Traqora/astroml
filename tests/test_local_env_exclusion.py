"""Local environment files must stay out of version control (issue #1244)."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORED_PATHS = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging.local",
    "service.env",
    ".secrets/api-key.json",
]

ALLOWED_PATHS = [
    ".env.example",
]


def is_ignored(path: str) -> bool:
    """Ask git whether a path would be ignored, without creating any files."""
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", path],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def test_local_environment_files_are_ignored() -> None:
    for path in IGNORED_PATHS:
        assert is_ignored(path), f"{path} is not excluded by .gitignore"


def test_example_environment_file_remains_tracked() -> None:
    for path in ALLOWED_PATHS:
        assert not is_ignored(path), f"{path} must stay available to contributors"
        assert (ROOT / path).is_file(), f"{path} is missing"


def test_gitignore_states_the_explicit_allowlist() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", ".env.*", "*.env", "!.env.example", ".secrets/"):
        assert pattern in content, f"Missing .gitignore pattern {pattern}"
