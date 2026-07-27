"""Contributor onboarding checklist API — Issue #281."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas import OnboardingProgressOut, OnboardingStepIn
from astroml.contributors.onboarding import ONBOARDING_STEPS, OnboardingTracker

router = APIRouter(prefix="/api/v1/contributors", tags=["contributors"])

_tracker = OnboardingTracker(store_path=Path("data/onboarding_progress.json"))


@router.get("/onboarding/{github_username}", response_model=OnboardingProgressOut)
def get_onboarding_progress(github_username: str):
    """Return the onboarding checklist and progress for a contributor."""
    progress = _tracker.get(github_username)
    return _to_out(progress)


@router.post("/onboarding/{github_username}/complete", response_model=OnboardingProgressOut)
def complete_step(github_username: str, body: OnboardingStepIn):
    """Mark an onboarding step as completed."""
    try:
        progress = _tracker.complete_step(github_username, body.step)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Validation error") from exc
    return _to_out(progress)


@router.delete("/onboarding/{github_username}", status_code=204)
def reset_onboarding(github_username: str):
    """Reset a contributor's onboarding progress."""
    _tracker.reset(github_username)


@router.get("/onboarding", response_model=list[OnboardingProgressOut])
def list_all_progress():
    """Return onboarding progress for all contributors."""
    return [_to_out(p) for p in _tracker.all_progress()]


def _to_out(progress) -> OnboardingProgressOut:
    return OnboardingProgressOut(
        github_username=progress.github_username,
        checklist=progress.checklist(),
        completed_count=len(progress.completed_steps),
        total_steps=progress.total_steps,
        progress_pct=progress.progress_pct,
        is_complete=progress.is_complete,
        started_at=progress.started_at,
        last_updated=progress.last_updated,
    )
