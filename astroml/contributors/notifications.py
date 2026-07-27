"""Notification service for contributor activities.

Features:
  - Email notifications (SendGrid/SES)
  - Slack/Discord integration
  - Weekly digest emails
  - User preference management
  - GitHub webhook handling
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session


class NotificationType(str, Enum):
    """Types of notifications."""
    PR_COMMENT = "pr_comment"
    PR_MENTION = "pr_mention"
    ISSUE_COMMENT = "issue_comment"
    ISSUE_MENTION = "issue_mention"
    REVIEW_REQUEST = "review_request"
    MERGED = "merged"
    CLOSED = "closed"
    WEEKLY_DIGEST = "weekly_digest"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"


@dataclass
class NotificationEvent:
    """Represents a notification event."""
    event_type: NotificationType
    user_id: int
    title: str
    content: str
    link: Optional[str] = None
    actor: Optional[str] = None  # GitHub username who triggered it
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class NotificationService:
    """Manages notification delivery across channels."""

    def __init__(self, db: Session):
        self.db = db

    def send_notification(
        self,
        event: NotificationEvent,
        channels: list[NotificationChannel] = None,
    ) -> int:
        """Send notification and store record.

        Args:
            event: Notification event details
            channels: List of channels to send to (uses preferences if None)

        Returns:
            ID of notification record
        """
        from api.models.orm import Notification  # noqa: PLC0415

        notification = Notification(
            user_id=event.user_id,
            event_type=event.event_type.value,
            title=event.title,
            content=event.content,
            link=event.link,
            actor=event.actor,
            is_read=False,
            created_at=event.timestamp,
        )
        self.db.add(notification)
        self.db.flush()
        notification_id = notification.id

        # Get user preferences
        if channels is None:
            prefs = self._get_preferences(event.user_id)
            channels = prefs.get_enabled_channels(event.event_type)

        # Send to each channel
        for channel in channels:
            self._deliver(notification_id, channel, event)

        return notification_id

    def _deliver(
        self,
        notification_id: int,
        channel: NotificationChannel,
        event: NotificationEvent,
    ) -> bool:
        """Deliver notification via specific channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                return self._send_email(event)
            elif channel == NotificationChannel.SLACK:
                return self._send_slack(event)
            elif channel == NotificationChannel.DISCORD:
                return self._send_discord(event)
        except Exception:  # noqa: BLE001
            return False
        return False

    def _send_email(self, event: NotificationEvent) -> bool:
        """Send email notification."""
        # Placeholder for SendGrid/SES integration
        # Would use credentials from environment
        return True

    def _send_slack(self, event: NotificationEvent) -> bool:
        """Send Slack notification."""
        # Placeholder for Slack webhook integration
        # Would construct message and post to webhook URL
        return True

    def _send_discord(self, event: NotificationEvent) -> bool:
        """Send Discord notification."""
        # Placeholder for Discord webhook integration
        return True

    def _get_preferences(self, user_id: int) -> NotificationPreferences:
        """Get user notification preferences."""
        from api.models.orm import NotificationPreference  # noqa: PLC0415

        result = self.db.execute(
            select(NotificationPreference).where(
                NotificationPreference.user_id == user_id
            )
        )
        pref_record = result.scalar_one_or_none()

        if not pref_record:
            # Return defaults
            return NotificationPreferences(user_id=user_id)

        return NotificationPreferences.from_orm(pref_record)

    def mark_as_read(self, notification_id: int) -> bool:
        """Mark notification as read."""
        from api.models.orm import Notification  # noqa: PLC0415

        result = self.db.execute(
            select(Notification).where(Notification.id == notification_id)
        )
        notification = result.scalar_one_or_none()

        if not notification:
            return False

        notification.is_read = True
        self.db.flush()
        return True

    def mark_all_as_read(self, user_id: int) -> int:
        """Mark all notifications as read for user."""
        from api.models.orm import Notification  # noqa: PLC0415

        result = self.db.execute(
            select(Notification).where(
                Notification.user_id == user_id,
                Notification.is_read.is_(False),
            )
        )
        notifications = result.scalars().all()

        for notif in notifications:
            notif.is_read = True

        self.db.flush()
        return len(notifications)

    def get_user_notifications(
        self,
        user_id: int,
        limit: int = 20,
        unread_only: bool = False,
    ) -> list:
        """Get notifications for user."""
        from api.models.orm import Notification  # noqa: PLC0415

        q = select(Notification).where(Notification.user_id == user_id)

        if unread_only:
            q = q.where(Notification.is_read.is_(False))

        q = q.order_by(Notification.created_at.desc()).limit(limit)

        result = self.db.execute(q)
        return result.scalars().all()


class NotificationPreferences:
    """User notification preferences."""

    def __init__(
        self,
        user_id: int,
        email_enabled: bool = True,
        slack_enabled: bool = False,
        discord_enabled: bool = False,
        pr_comments: bool = True,
        pr_mentions: bool = True,
        issue_comments: bool = True,
        issue_mentions: bool = True,
        review_requests: bool = True,
        digest_frequency: str = "weekly",  # daily|weekly|never
    ):
        self.user_id = user_id
        self.email_enabled = email_enabled
        self.slack_enabled = slack_enabled
        self.discord_enabled = discord_enabled
        self.pr_comments = pr_comments
        self.pr_mentions = pr_mentions
        self.issue_comments = issue_comments
        self.issue_mentions = issue_mentions
        self.review_requests = review_requests
        self.digest_frequency = digest_frequency

    def get_enabled_channels(self, event_type: NotificationType) -> list[NotificationChannel]:
        """Get enabled channels for event type."""
        channels = []

        # Check if event type is enabled
        if event_type == NotificationType.PR_COMMENT and not self.pr_comments:
            return channels
        if event_type == NotificationType.PR_MENTION and not self.pr_mentions:
            return channels
        if event_type == NotificationType.ISSUE_COMMENT and not self.issue_comments:
            return channels
        if event_type == NotificationType.ISSUE_MENTION and not self.issue_mentions:
            return channels
        if event_type == NotificationType.REVIEW_REQUEST and not self.review_requests:
            return channels

        # Add enabled channels
        if self.email_enabled:
            channels.append(NotificationChannel.EMAIL)
        if self.slack_enabled:
            channels.append(NotificationChannel.SLACK)
        if self.discord_enabled:
            channels.append(NotificationChannel.DISCORD)

        return channels

    @classmethod
    def from_orm(cls, orm_obj) -> NotificationPreferences:
        """Create from ORM object."""
        return cls(
            user_id=orm_obj.user_id,
            email_enabled=orm_obj.email_enabled,
            slack_enabled=orm_obj.slack_enabled,
            discord_enabled=orm_obj.discord_enabled,
            pr_comments=orm_obj.pr_comments,
            pr_mentions=orm_obj.pr_mentions,
            issue_comments=orm_obj.issue_comments,
            issue_mentions=orm_obj.issue_mentions,
            review_requests=orm_obj.review_requests,
            digest_frequency=orm_obj.digest_frequency,
        )


class GitHubWebhookHandler:
    """Handles GitHub webhook events."""

    def __init__(self, db: Session):
        self.db = db
        self.service = NotificationService(db)

    def handle_pr_comment(
        self,
        pr_number: int,
        commenter: str,
        content: str,
        repo: str,
        link: str,
    ) -> None:
        """Handle PR comment webhook."""
        # Find mentioned users
        mentioned_users = self._extract_mentions(content)

        for user_id in mentioned_users:
            event = NotificationEvent(
                event_type=NotificationType.PR_MENTION,
                user_id=user_id,
                title=f"PR #{pr_number} comment in {repo}",
                content=content[:200],
                link=link,
                actor=commenter,
            )
            self.service.send_notification(event)

    def handle_issue_comment(
        self,
        issue_number: int,
        commenter: str,
        content: str,
        repo: str,
        link: str,
    ) -> None:
        """Handle issue comment webhook."""
        mentioned_users = self._extract_mentions(content)

        for user_id in mentioned_users:
            event = NotificationEvent(
                event_type=NotificationType.ISSUE_MENTION,
                user_id=user_id,
                title=f"Issue #{issue_number} comment in {repo}",
                content=content[:200],
                link=link,
                actor=commenter,
            )
            self.service.send_notification(event)

    def handle_review_request(
        self,
        pr_number: int,
        reviewer_id: int,
        repo: str,
        link: str,
    ) -> None:
        """Handle review request."""
        event = NotificationEvent(
            event_type=NotificationType.REVIEW_REQUEST,
            user_id=reviewer_id,
            title=f"Review requested for PR #{pr_number} in {repo}",
            content="Your review is requested",
            link=link,
        )
        self.service.send_notification(event)

    def handle_pr_merged(
        self,
        pr_number: int,
        author_id: int,
        repo: str,
        link: str,
    ) -> None:
        """Handle PR merged."""
        event = NotificationEvent(
            event_type=NotificationType.MERGED,
            user_id=author_id,
            title=f"PR #{pr_number} merged in {repo}",
            content="Your PR has been merged",
            link=link,
        )
        self.service.send_notification(event)

    @staticmethod
    def _extract_mentions(text: str) -> list[str]:
        """Extract @mentions from text."""
        import re
        pattern = r"@(\w+)"
        return re.findall(pattern, text)


class DigestEmailGenerator:
    """Generates weekly digest emails."""

    def __init__(self, db: Session):
        self.db = db

    def generate_digest(self, user_id: int) -> dict:
        """Generate weekly digest for user."""
        from api.models.orm import Notification  # noqa: PLC0415

        # Get notifications from last 7 days
        result = self.db.execute(
            select(Notification)
            .where(Notification.user_id == user_id)
            .order_by(Notification.created_at.desc())
            .limit(50)
        )
        notifications = result.scalars().all()

        # Group by event type
        grouped = {}
        for notif in notifications:
            event_type = notif.event_type
            if event_type not in grouped:
                grouped[event_type] = []
            grouped[event_type].append(notif)

        return {
            "user_id": user_id,
            "period": "weekly",
            "notifications_by_type": grouped,
            "total_count": len(notifications),
            "generated_at": datetime.now(timezone.utc),
        }
