"""Contributor activity notifications API endpoints.

Endpoints:
  GET /api/v1/notifications                     — get user notifications
  GET /api/v1/notifications/unread              — get unread count
  PUT /api/v1/notifications/{id}/read           — mark as read
  PUT /api/v1/notifications/read-all            — mark all as read
  GET /api/v1/notifications/preferences         — get notification preferences
  PUT /api/v1/notifications/preferences         — update preferences
  POST /api/v1/notifications/webhook/github    — GitHub webhook handler
  GET /api/v1/notifications/digest              — generate digest
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import AuthContext, get_current_auth
from api.database import get_db
from api.models.orm import Notification, NotificationPreference
from api.schemas import (
    NotificationOut,
    NotificationListResponse,
    NotificationPreferenceIn,
    NotificationPreferenceOut,
    WebhookEventIn,
    DigestEmailOut,
)
from astroml.contributors.notifications import (
    NotificationService,
    NotificationPreferences,
    GitHubWebhookHandler,
    DigestEmailGenerator,
)

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])


# ─── User Notifications ───────────────────────────────────────────────────

@router.get("", response_model=NotificationListResponse)
async def get_notifications(
    limit: int = Query(20, ge=1, le=100),
    unread_only: bool = False,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get user notifications."""
    q = select(Notification).where(Notification.user_id == auth.user_id)

    if unread_only:
        q = q.where(Notification.is_read.is_(False))

    q = q.order_by(Notification.created_at.desc()).limit(limit)

    result = await db.execute(q)
    notifications = result.scalars().all()

    # Count unread
    unread_result = await db.execute(
        select(func.count()).select_from(Notification).where(
            Notification.user_id == auth.user_id,
            Notification.is_read.is_(False),
        )
    )
    unread_count = unread_result.scalar()

    return NotificationListResponse(
        data=[NotificationOut.from_orm(n) for n in notifications],
        unread_count=unread_count,
    )


@router.get("/unread")
async def get_unread_count(
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get count of unread notifications."""
    result = await db.execute(
        select(func.count()).select_from(Notification).where(
            Notification.user_id == auth.user_id,
            Notification.is_read.is_(False),
        )
    )
    count = result.scalar()
    return {"unread_count": count}


@router.put("/{notification_id}/read")
async def mark_as_read(
    notification_id: int,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Mark notification as read."""
    result = await db.execute(
        select(Notification).where(Notification.id == notification_id)
    )
    notification = result.scalar_one_or_none()

    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    if notification.user_id != auth.user_id:
        raise HTTPException(status_code=403, detail="Not authorized")

    notification.is_read = True
    await db.flush()

    return NotificationOut.from_orm(notification)


@router.put("/read-all")
async def mark_all_as_read(
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Mark all notifications as read."""
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == auth.user_id,
            Notification.is_read.is_(False),
        )
    )
    notifications = result.scalars().all()

    for notif in notifications:
        notif.is_read = True

    await db.flush()

    return {"marked_as_read": len(notifications)}


# ─── Preferences ──────────────────────────────────────────────────────────

@router.get("/preferences", response_model=NotificationPreferenceOut)
async def get_preferences(
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Get notification preferences."""
    result = await db.execute(
        select(NotificationPreference).where(
            NotificationPreference.user_id == auth.user_id
        )
    )
    pref = result.scalar_one_or_none()

    if not pref:
        # Create default preferences
        pref = NotificationPreference(user_id=auth.user_id)
        db.add(pref)
        await db.flush()

    return NotificationPreferenceOut.from_orm(pref)


@router.put("/preferences", response_model=NotificationPreferenceOut)
async def update_preferences(
    body: NotificationPreferenceIn,
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Update notification preferences."""
    result = await db.execute(
        select(NotificationPreference).where(
            NotificationPreference.user_id == auth.user_id
        )
    )
    pref = result.scalar_one_or_none()

    if not pref:
        pref = NotificationPreference(user_id=auth.user_id)
        db.add(pref)

    pref.email_enabled = body.email_enabled
    pref.slack_enabled = body.slack_enabled
    pref.discord_enabled = body.discord_enabled
    pref.pr_comments = body.pr_comments
    pref.pr_mentions = body.pr_mentions
    pref.issue_comments = body.issue_comments
    pref.issue_mentions = body.issue_mentions
    pref.review_requests = body.review_requests
    pref.digest_frequency = body.digest_frequency
    pref.slack_webhook_url = body.slack_webhook_url
    pref.discord_webhook_url = body.discord_webhook_url

    await db.flush()

    return NotificationPreferenceOut.from_orm(pref)


# ─── Webhooks ─────────────────────────────────────────────────────────────

@router.post("/webhook/github", status_code=202)
async def handle_github_webhook(
    body: WebhookEventIn,
    db: AsyncSession = Depends(get_db),
):
    """Handle GitHub webhook events.

    Processes PR comments, issue comments, review requests, merges.
    """
    handler = GitHubWebhookHandler(db)

    if body.event_type == "pr_comment":
        handler.handle_pr_comment(
            pr_number=body.pr_number,
            commenter=body.commenter,
            content=body.content,
            repo=body.repo,
            link=body.link,
        )
    elif body.event_type == "issue_comment":
        handler.handle_issue_comment(
            issue_number=body.issue_number,
            commenter=body.commenter,
            content=body.content,
            repo=body.repo,
            link=body.link,
        )
    elif body.event_type == "review_request":
        handler.handle_review_request(
            pr_number=body.pr_number,
            reviewer_id=body.reviewer_id,
            repo=body.repo,
            link=body.link,
        )
    elif body.event_type == "pr_merged":
        handler.handle_pr_merged(
            pr_number=body.pr_number,
            author_id=body.author_id,
            repo=body.repo,
            link=body.link,
        )
    else:
        raise HTTPException(status_code=400, detail="Unknown event type")

    await db.commit()
    return {"status": "accepted"}


# ─── Digest & Analytics ───────────────────────────────────────────────────

@router.get("/digest", response_model=DigestEmailOut)
async def get_digest(
    auth: AuthContext = Depends(get_current_auth),
    db: AsyncSession = Depends(get_db),
):
    """Generate weekly digest for user."""
    generator = DigestEmailGenerator(db)
    digest = generator.generate_digest(auth.user_id)

    return DigestEmailOut(
        user_id=digest["user_id"],
        period=digest["period"],
        notifications_count=digest["total_count"],
        generated_at=digest["generated_at"],
    )
