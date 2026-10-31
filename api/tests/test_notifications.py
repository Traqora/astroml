"""Tests for contributor activity notifications."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.models.orm import Notification, NotificationPreference
from astroml.contributors.notifications import (
    NotificationEvent,
    NotificationService,
    NotificationType,
)


@pytest.fixture()
def seeded_notifications(db_session):
    """Create test notifications."""
    for i in range(5):
        notif = Notification(
            user_id=1,
            event_type="pr_comment",
            title=f"PR Comment {i}",
            content=f"Content {i}",
            link=f"https://github.com/pr/{i}",
            actor="alice",
            is_read=(i < 2),  # First 2 are read
        )
        db_session.add(notif)
    db_session.commit()


def test_get_notifications(client: TestClient, db_session, seeded_notifications):
    """Test retrieving notifications."""
    response = client.get("/api/v1/notifications")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "unread_count" in data


def test_get_unread_count(client: TestClient, db_session, seeded_notifications):
    """Test getting unread count."""
    response = client.get("/api/v1/notifications/unread")
    assert response.status_code == 200
    data = response.json()
    assert data["unread_count"] == 3  # 5 total - 2 read


def test_mark_notification_as_read(client: TestClient, db_session, seeded_notifications):
    """Test marking notification as read."""
    # Get first unread notification
    notif_result = (
        db_session.query(Notification)
        .filter(Notification.user_id == 1, Notification.is_read == False)
        .first()
    )

    response = client.put(f"/api/v1/notifications/{notif_result.id}/read")
    assert response.status_code == 200
    data = response.json()
    assert data["is_read"] is True


def test_mark_all_as_read(client: TestClient, db_session, seeded_notifications):
    """Test marking all as read."""
    response = client.put("/api/v1/notifications/read-all")
    assert response.status_code == 200
    data = response.json()
    assert data["marked_as_read"] == 3


def test_get_preferences(client: TestClient):
    """Test retrieving notification preferences."""
    response = client.get("/api/v1/notifications/preferences")
    assert response.status_code == 200
    data = response.json()
    assert "email_enabled" in data
    assert "slack_enabled" in data
    assert "digest_frequency" in data


def test_update_preferences(client: TestClient):
    """Test updating notification preferences."""
    prefs = {
        "email_enabled": False,
        "slack_enabled": True,
        "digest_frequency": "daily",
        "pr_comments": False,
    }
    response = client.put("/api/v1/notifications/preferences", json=prefs)
    assert response.status_code == 200
    data = response.json()
    assert data["email_enabled"] is False
    assert data["slack_enabled"] is True
    assert data["digest_frequency"] == "daily"


def test_github_webhook_pr_comment(client: TestClient):
    """Test GitHub webhook for PR comment."""
    webhook_data = {
        "event_type": "pr_comment",
        "pr_number": 42,
        "commenter": "alice",
        "content": "Great work @bob!",
        "repo": "astroml",
        "link": "https://github.com/astroml/pull/42",
    }
    response = client.post("/api/v1/notifications/webhook/github", json=webhook_data)
    assert response.status_code == 202


def test_github_webhook_review_request(client: TestClient):
    """Test GitHub webhook for review request."""
    webhook_data = {
        "event_type": "review_request",
        "pr_number": 50,
        "reviewer_id": 2,
        "repo": "astroml",
        "link": "https://github.com/astroml/pull/50",
    }
    response = client.post("/api/v1/notifications/webhook/github", json=webhook_data)
    assert response.status_code == 202


def test_get_digest(client: TestClient, db_session, seeded_notifications):
    """Test digest generation."""
    response = client.get("/api/v1/notifications/digest")
    assert response.status_code == 200
    data = response.json()
    assert data["period"] == "weekly"
    assert "notifications_count" in data


def test_notifications_batching(db_session: Session):
    """Test that notifications are batched efficiently with single flush."""
    service = NotificationService(db_session)

    # Create multiple notification events
    events = [
        NotificationEvent(
            event_type=NotificationType.PR_MENTION,
            user_id=1,
            title="PR mention 1",
            content="Content 1",
            link="https://github.com/pr/1",
            actor="alice",
        ),
        NotificationEvent(
            event_type=NotificationType.PR_MENTION,
            user_id=2,
            title="PR mention 2",
            content="Content 2",
            link="https://github.com/pr/1",
            actor="alice",
        ),
        NotificationEvent(
            event_type=NotificationType.ISSUE_MENTION,
            user_id=3,
            title="Issue mention 1",
            content="Content 3",
            link="https://github.com/issue/1",
            actor="bob",
        ),
    ]

    # Send notifications in batch
    notification_ids = service.send_notifications_batch(events)

    # Verify all notifications were created
    assert len(notification_ids) == 3
    assert all(isinstance(nid, int) for nid in notification_ids)

    # Verify notifications exist in database
    notifications = (
        db_session.query(Notification)
        .filter(Notification.id.in_(notification_ids))
        .all()
    )
    assert len(notifications) == 3

    # Verify notification details
    assert notifications[0].user_id == 1
    assert notifications[0].title == "PR mention 1"
    assert notifications[1].user_id == 2
    assert notifications[1].title == "PR mention 2"
    assert notifications[2].user_id == 3
    assert notifications[2].title == "Issue mention 1"

    db_session.commit()


def test_notifications_batching_with_multiple_mentions(db_session: Session):
    """Test webhook handler with multiple mentions uses batching."""
    from astroml.contributors.notifications import GitHubWebhookHandler

    handler = GitHubWebhookHandler(db_session)

    # Simulate a PR comment with multiple mentions
    webhook_data = {
        "pr_number": 42,
        "commenter": "alice",
        "content": "Great work @bob, @charlie, and @dave!",
        "repo": "astroml",
        "link": "https://github.com/astroml/pull/42",
    }

    # Before handling, count notifications
    initial_count = db_session.query(Notification).count()

    # Handle the webhook (should batch 3 notifications)
    handler.handle_pr_comment(**webhook_data)

    # Verify 3 new notifications were created
    new_count = db_session.query(Notification).count()
    assert new_count == initial_count + 3

    # Verify all notifications have correct data
    notifications = (
        db_session.query(Notification)
        .filter(Notification.event_type == "pr_mention")
        .all()
    )
    assert len(notifications) == 3
    for notif in notifications:
        assert notif.actor == "alice"
        assert "PR #42" in notif.title
        assert notif.link == "https://github.com/astroml/pull/42"

    db_session.commit()

