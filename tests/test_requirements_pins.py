"""Regression tests for CPU/CUDA torch pin consistency (GitHub Issue #853).

requirements-cpu.txt previously pinned `torch==2.0.0+cpu` (exact) while
requirements.txt floated `torch>=2.0.0` (GPU, PyPI index) — the two could
resolve to different torch versions even though REQUIREMENTS.md's pin
policy says the lower bound is held in lockstep across files. Fixed to
`torch>=2.0.0` in both; the `+cpu` build is selected by
`--extra-index-url https://download.pytorch.org/whl/cpu`, not by the
version specifier (a local version label like `+cpu` can only be combined
with `==`/`!=`, never `>=`, per PEP 440).
"""

from __future__ import annotations

import pathlib
import re

import pytest
from packaging.requirements import InvalidRequirement, Requirement

_ROOT = pathlib.Path(__file__).parent.parent
_REQUIREMENTS_FILES = [
    _ROOT / "requirements.txt",
    _ROOT / "requirements-cpu.txt",
    _ROOT / "requirements-minimal.txt",
]

_REQ_LINE = re.compile(r"^[A-Za-z0-9_.\-]+\s*[<>=!~].*")


def _iter_requirement_lines(path: pathlib.Path) -> list[str]:
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        lines.append(stripped)
    return lines


def _torch_specifier(path: pathlib.Path) -> str | None:
    for line in _iter_requirement_lines(path):
        if line.split("=")[0].split(">")[0].split("<")[0].strip() == "torch":
            return line
    return None


def test_every_requirement_line_is_valid_pep440() -> None:
    for path in _REQUIREMENTS_FILES:
        for line in _iter_requirement_lines(path):
            if not _REQ_LINE.match(line):
                continue
            try:
                Requirement(line)
            except InvalidRequirement as exc:
                pytest.fail(f"{path.name}: invalid requirement '{line}': {exc}")


def test_torch_lower_bound_matches_between_gpu_and_cpu_requirements() -> None:
    gpu_line = _torch_specifier(_ROOT / "requirements.txt")
    cpu_line = _torch_specifier(_ROOT / "requirements-cpu.txt")

    assert gpu_line is not None, "requirements.txt must pin torch"
    assert cpu_line is not None, "requirements-cpu.txt must pin torch"

    gpu_spec = Requirement(gpu_line).specifier
    cpu_spec = Requirement(cpu_line).specifier

    assert str(gpu_spec) == str(cpu_spec), (
        f"torch version specifier drifted between requirements.txt ({gpu_spec}) "
        f"and requirements-cpu.txt ({cpu_spec}) — REQUIREMENTS.md's pin policy "
        "requires these to be held in lockstep"
    )
