from __future__ import annotations

import os
import pathlib
import subprocess


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "reset-local.sh"


def _run_reset(
    tmp_path: pathlib.Path,
    *args: str,
    input_text: str | None = None,
    down_exit: int = 0,
) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log = tmp_path / "docker.log"
    docker = fake_bin / "docker"
    docker.write_text(
        """#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_DOCKER_LOG}"
if [[ "$*" == "compose down"* ]]; then
    exit "${FAKE_DOCKER_DOWN_EXIT:-0}"
fi
exit 0
""",
        encoding="utf-8",
    )
    docker.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["FAKE_DOCKER_LOG"] = str(log)
    env["FAKE_DOCKER_DOWN_EXIT"] = str(down_exit)
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
    )
    return result, log


def test_reset_preserves_named_volumes(tmp_path: pathlib.Path) -> None:
    result, log = _run_reset(tmp_path)

    assert result.returncode == 0
    commands = log.read_text(encoding="utf-8")
    assert "compose down --remove-orphans\n" in commands
    assert "--volumes" not in commands
    assert "compose up -d --build postgres redis api\n" in commands


def test_reset_cancellation_does_not_remove_data(tmp_path: pathlib.Path) -> None:
    result, log = _run_reset(
        tmp_path,
        "--destroy-user-data",
        input_text="no\n",
    )

    assert result.returncode == 2
    assert "no data was removed" in result.stderr
    commands = log.read_text(encoding="utf-8")
    assert "compose down" not in commands
    assert "compose up" not in commands


def test_reset_failure_does_not_continue_to_restart(tmp_path: pathlib.Path) -> None:
    result, log = _run_reset(tmp_path, down_exit=42)

    assert result.returncode == 42
    commands = log.read_text(encoding="utf-8")
    assert "compose down --remove-orphans\n" in commands
    assert "compose up" not in commands
