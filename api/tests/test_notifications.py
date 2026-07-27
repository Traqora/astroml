"""Tests for contributor activity notifications."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.models.orm import Notification, NotificationPreference


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
    notif_result = db_session.query(Notification).filter(
        Notification.user_id == 1, Notification.is_read == False
    ).first()

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
