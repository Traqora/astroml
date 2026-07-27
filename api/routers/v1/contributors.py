"""Contributors Dashboard API (issue #280).

Endpoints
---------
GET /api/v1/contributors              — Top contributors ranked by commits, PRs, issues
GET /api/v1/contributors/activity     — Contribution activity over time
GET /api/v1/contributors/new          — New contributors (first contribution within window)
GET /api/v1/contributors/{username}   — Single contributor profile with badges
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/contributors", tags=["contributors"])

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
REPO_OWNER = os.getenv("GITHUB_REPO_OWNER", "Traqora")
REPO_NAME = os.getenv("GITHUB_REPO_NAME", "astroml")

# Simple in-process cache: {cache_key: (fetched_at, data)}
_cache: dict[str, tuple[datetime, object]] = {}
CACHE_TTL_SECONDS = 300


def _cached(key: str) -> object | None:
    if key not in _cache:
        return None
    fetched_at, data = _cache[key]
    if (datetime.now(timezone.utc) - fetched_at).total_seconds() > CACHE_TTL_SECONDS:
        del _cache[key]
        return None
    return data


def _store(key: str, data: object) -> None:
    _cache[key] = (datetime.now(timezone.utc), data)


def _headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


async def _gh_get(path: str) -> list | dict:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{GITHUB_API}{path}", headers=_headers())
    if resp.status_code == 404:
        raise HTTPException(status_code=404, detail="GitHub resource not found")
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="GitHub API error")
    return resp.json()


# ── Schemas ───────────────────────────────────────────────────────────────────

class ContributorBadge(BaseModel):
    id: str
    label: str
    description: str


class ContributorOut(BaseModel):
    username: str
    avatar_url: str
    profile_url: str
    commits: int
    pull_requests: int
    issues: int
    total_contributions: int
    badges: list[ContributorBadge]


class ActivityPoint(BaseModel):
    date: str        # ISO date YYYY-MM-DD
    commits: int
    pull_requests: int
    issues: int


class ContributorsResponse(BaseModel):
    contributors: list[ContributorOut]
    total: int


class ActivityResponse(BaseModel):
    activity: list[ActivityPoint]
    days: int


def _assign_badges(commits: int, prs: int, issues: int) -> list[ContributorBadge]:
    badges: list[ContributorBadge] = []
    if commits >= 100:
        badges.append(ContributorBadge(id="centurion", label="Centurion", description="100+ commits"))
    elif commits >= 10:
        badges.append(ContributorBadge(id="active", label="Active Contributor", description="10+ commits"))
    if prs >= 10:
        badges.append(ContributorBadge(id="pr_champion", label="PR Champion", description="10+ pull requests merged"))
    if issues >= 5:
        badges.append(ContributorBadge(id="reporter", label="Reporter", description="5+ issues opened"))
    if commits >= 1 and prs >= 1 and issues >= 1:
        badges.append(ContributorBadge(id="all_rounder", label="All-Rounder", description="Commits, PRs, and issues"))
    return badges


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("", response_model=ContributorsResponse)
async def list_contributors(
    limit: int = Query(20, ge=1, le=100),
    sort_by: str = Query("total", regex="^(commits|pull_requests|issues|total)$"),
):
    """Top contributors ranked by selected metric."""
    cache_key = f"contributors:{REPO_OWNER}/{REPO_NAME}:{limit}:{sort_by}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    # Fetch contributor commit stats
    commit_stats: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/contributors?per_page=100"
    )

    # Fetch PRs and issues counts per contributor
    pr_counts: dict[str, int] = {}
    issue_counts: dict[str, int] = {}

    pr_data: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=closed&per_page=100"
    )
    for pr in pr_data:
        login = pr.get("user", {}).get("login", "")
        if pr.get("merged_at") and login:
            pr_counts[login] = pr_counts.get(login, 0) + 1

    issue_data: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/issues?state=all&per_page=100"
    )
    for issue in issue_data:
        if issue.get("pull_request"):
            continue  # skip PRs listed as issues
        login = issue.get("user", {}).get("login", "")
        if login:
            issue_counts[login] = issue_counts.get(login, 0) + 1

    contributors: list[ContributorOut] = []
    for c in commit_stats:
        login = c.get("login", "")
        commits = c.get("contributions", 0)
        prs = pr_counts.get(login, 0)
        issues = issue_counts.get(login, 0)
        contributors.append(ContributorOut(
            username=login,
            avatar_url=c.get("avatar_url", ""),
            profile_url=c.get("html_url", f"https://github.com/{login}"),
            commits=commits,
            pull_requests=prs,
            issues=issues,
            total_contributions=commits + prs + issues,
            badges=_assign_badges(commits, prs, issues),
        ))

    sort_key = {
        "commits": lambda x: x.commits,
        "pull_requests": lambda x: x.pull_requests,
        "issues": lambda x: x.issues,
        "total": lambda x: x.total_contributions,
    }[sort_by]

    contributors.sort(key=sort_key, reverse=True)
    contributors = contributors[:limit]

    result = ContributorsResponse(contributors=contributors, total=len(contributors))
    _store(cache_key, result)
    return result


@router.get("/activity", response_model=ActivityResponse)
async def contributor_activity(
    days: int = Query(30, ge=7, le=365),
    username: Optional[str] = Query(None),
):
    """Contribution activity bucketed by day over the last N days."""
    cache_key = f"activity:{REPO_OWNER}/{REPO_NAME}:{days}:{username}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    path = f"/repos/{REPO_OWNER}/{REPO_NAME}/commits?since={since}&per_page=100"
    if username:
        path += f"&author={username}"

    commits: list[dict] = await _gh_get(path)

    # Bucket by date
    by_date: dict[str, ActivityPoint] = {}
    for c in commits:
        date_str = (c.get("commit", {}).get("author", {}).get("date", "") or "")[:10]
        if not date_str:
            continue
        if date_str not in by_date:
            by_date[date_str] = ActivityPoint(date=date_str, commits=0, pull_requests=0, issues=0)
        by_date[date_str].commits += 1

    activity = sorted(by_date.values(), key=lambda p: p.date)
    result = ActivityResponse(activity=activity, days=days)
    _store(cache_key, result)
    return result


@router.get("/new", response_model=ContributorsResponse)
async def new_contributors(
    days: int = Query(30, ge=1, le=365),
):
    """Contributors whose first commit to this repo is within the last N days."""
    cache_key = f"new_contributors:{REPO_OWNER}/{REPO_NAME}:{days}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    all_stats: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/stats/contributors"
    )

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    new_ones: list[ContributorOut] = []

    for entry in (all_stats or []):
        weeks: list[dict] = entry.get("weeks", [])
        # Find earliest week with a commit
        first_week = next((w for w in weeks if w.get("c", 0) > 0), None)
        if not first_week:
            continue
        first_ts = datetime.fromtimestamp(first_week["w"], tz=timezone.utc)
        if first_ts >= cutoff:
            author = entry.get("author", {})
            login = author.get("login", "")
            commits = entry.get("total", 0)
            new_ones.append(ContributorOut(
                username=login,
                avatar_url=author.get("avatar_url", ""),
                profile_url=author.get("html_url", f"https://github.com/{login}"),
                commits=commits,
                pull_requests=0,
                issues=0,
                total_contributions=commits,
                badges=_assign_badges(commits, 0, 0),
            ))

    result = ContributorsResponse(contributors=new_ones, total=len(new_ones))
    _store(cache_key, result)
    return result


@router.get("/{username}", response_model=ContributorOut)
async def get_contributor(username: str):
    """Single contributor profile with badges."""
    import re
    if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,38}[a-zA-Z0-9])?$', username):
        raise HTTPException(status_code=400, detail="Invalid GitHub username format")
    cache_key = f"contributor:{username}:{REPO_OWNER}/{REPO_NAME}"
    cached = _cached(cache_key)
    if cached is not None:
        return cached

    user_data: dict = await _gh_get(f"/users/{username}")

    # Commit count for this repo
    commits_data: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/commits?author={username}&per_page=100"
    )
    commits = len(commits_data)

    pr_data: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls?state=closed&per_page=100"
    )
    prs = sum(
        1 for p in pr_data
        if p.get("user", {}).get("login") == username and p.get("merged_at")
    )

    issue_data: list[dict] = await _gh_get(
        f"/repos/{REPO_OWNER}/{REPO_NAME}/issues?state=all&creator={username}&per_page=100"
    )
    issues = sum(1 for i in issue_data if not i.get("pull_request"))

    result = ContributorOut(
        username=username,
        avatar_url=user_data.get("avatar_url", ""),
        profile_url=user_data.get("html_url", f"https://github.com/{username}"),
        commits=commits,
        pull_requests=prs,
        issues=issues,
        total_contributions=commits + prs + issues,
        badges=_assign_badges(commits, prs, issues),
    )
    _store(cache_key, result)
    return result
