"""GitHub issue integration for feedback (issue #308).

Opens a GitHub issue for incoming feedback when a token and repo are
configured; otherwise it's a no-op so the feature works without credentials.
"""
from __future__ import annotations

import logging

import httpx

from api.config import settings

logger = logging.getLogger(__name__)


def github_configured() -> bool:
    return bool(settings.github_token and settings.github_repo)


async def create_feedback_issue(
    category: str, message: str, email: str | None = None
) -> str | None:
    """Create a GitHub issue for the feedback. Returns the issue URL or None."""
    if not github_configured():
        logger.info("GitHub not configured; skipping issue creation")
        return None

    title = f"[feedback/{category}] {message[:60].strip()}"
    body_lines = [message, "", f"_Category: {category}_"]
    if email:
        body_lines.append(f"_Reporter: {email}_")
    payload = {
        "title": title,
        "body": "\n".join(body_lines),
        "labels": ["feedback", category],
    }
    url = f"https://api.github.com/repos/{settings.github_repo}/issues"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {settings.github_token}",
                    "Accept": "application/vnd.github+json",
                },
            )
        resp.raise_for_status()
        return resp.json().get("html_url")
    except Exception as exc:  # issue creation is best-effort
        logger.warning("Failed to create GitHub issue: %s", exc)
        return None
