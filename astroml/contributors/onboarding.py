"""Contributor onboarding checklist module.

Tracks a new contributor's progress through the onboarding steps:
  1. Fork the repository
  2. Set up the development environment
  3. Run the test suite
  4. Open a first pull request

Progress is stored per GitHub username and persisted to a JSON file.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ONBOARDING_STEPS: List[str] = [
    "fork_repo",
    "setup_dev_environment",
    "run_tests",
    "first_pr",
]

STEP_LABELS: Dict[str, str] = {
    "fork_repo": "Fork the repository",
    "setup_dev_environment": "Set up dev environment",
    "run_tests": "Run the test suite",
    "first_pr": "Open your first pull request",
}

_DEFAULT_STORE = Path(os.environ.get("ONBOARDING_STORE", "data/onboarding_progress.json"))


@dataclass
class OnboardingProgress:
    github_username: str
    completed_steps: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_steps(self) -> int:
        return len(ONBOARDING_STEPS)

    @property
    def progress_pct(self) -> int:
        if not self.total_steps:
            return 0
        return round(len(self.completed_steps) / self.total_steps * 100)

    @property
    def is_complete(self) -> bool:
        return set(ONBOARDING_STEPS).issubset(set(self.completed_steps))

    def remaining_steps(self) -> List[str]:
        done = set(self.completed_steps)
        return [s for s in ONBOARDING_STEPS if s not in done]

    def checklist(self) -> List[Dict[str, object]]:
        done = set(self.completed_steps)
        return [
            {"step": step, "label": STEP_LABELS[step], "completed": step in done}
            for step in ONBOARDING_STEPS
        ]


class OnboardingTracker:
    """Persists and retrieves onboarding progress per contributor."""

    def __init__(self, store_path: Path = _DEFAULT_STORE) -> None:
        self._path = store_path

    def _load(self) -> Dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2))

    def get(self, github_username: str) -> OnboardingProgress:
        data = self._load()
        raw = data.get(github_username)
        if not raw:
            return OnboardingProgress(github_username=github_username)
        return OnboardingProgress(
            github_username=raw["github_username"],
            completed_steps=raw.get("completed_steps", []),
            started_at=raw.get("started_at", datetime.now(timezone.utc).isoformat()),
            last_updated=raw.get("last_updated", datetime.now(timezone.utc).isoformat()),
        )

    def complete_step(self, github_username: str, step: str) -> OnboardingProgress:
        if step not in ONBOARDING_STEPS:
            raise ValueError(f"Unknown onboarding step: {step!r}. Valid steps: {ONBOARDING_STEPS}")

        progress = self.get(github_username)
        if step not in progress.completed_steps:
            progress.completed_steps.append(step)
            progress.last_updated = datetime.now(timezone.utc).isoformat()

        data = self._load()
        data[github_username] = asdict(progress)
        self._save(data)
        return progress

    def reset(self, github_username: str) -> OnboardingProgress:
        data = self._load()
        data.pop(github_username, None)
        self._save(data)
        return OnboardingProgress(github_username=github_username)

    def all_progress(self) -> List[OnboardingProgress]:
        data = self._load()
        return [
            OnboardingProgress(
                github_username=v["github_username"],
                completed_steps=v.get("completed_steps", []),
                started_at=v.get("started_at", ""),
                last_updated=v.get("last_updated", ""),
            )
            for v in data.values()
        ]
