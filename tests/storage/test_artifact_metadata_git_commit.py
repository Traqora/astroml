"""Regression test for detect_git_commit() error handling (GitHub Issue #840).

detect_git_commit() caught `except AstroMLError`, a domain-specific
exception type that subprocess.run never raises and that wasn't even
imported into this module. Any real failure (git not installed, call
times out) crashed with an unhandled FileNotFoundError/TimeoutExpired
instead of returning None as the docstring promises, exactly when the
function needed to degrade gracefully.
"""

from __future__ import annotations

import subprocess

from astroml.storage.artifact_metadata import detect_git_commit


def test_returns_none_when_git_is_not_installed(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert detect_git_commit() is None


def test_returns_none_when_git_call_times_out(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert detect_git_commit() is None


def test_returns_none_when_not_a_git_checkout(monkeypatch) -> None:
    class FakeResult:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert detect_git_commit() is None


def test_returns_commit_hash_on_success(monkeypatch) -> None:
    class FakeResult:
        returncode = 0
        stdout = "abc123deadbeef\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeResult())
    assert detect_git_commit() == "abc123deadbeef"
