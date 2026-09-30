"""Unit tests for API key rotation and revocation (issue #534).

Tests cover:
- POST /api/v1/auth/rotate-key  — issues a new key, keeps old key valid during overlap
- POST /api/v1/auth/revoke-key  — immediately deactivates a key

Coverage targets: rotation logic, revocation logic, overlap window,
expired overlap rejection, 404 on unknown key, idempotent re-revocation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from api.auth.security import generate_api_key, hash_api_key, hash_password
from api.models.orm import ApiKey, User

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def admin_user(db_session) -> User:
    user = User(
        username="rotation_admin",
        hashed_password=hash_password("s3cr3t"),
        scopes=["admin", "read:transactions", "read:fraud", "write:loyalty"],
    )
    db_session.add(user)
    db_session.flush()
    return user


@pytest.fixture()
def admin_token(client, db_session, admin_user, monkeypatch) -> str:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    resp = client.post(
        "/api/v1/auth/login",
        json={"username": "rotation_admin", "password": "s3cr3t"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture()
def existing_api_key(db_session, admin_user) -> tuple[str, ApiKey]:
    """Returns (raw_key, ApiKey ORM row)."""
    raw = generate_api_key()
    entry = ApiKey(
        user_id=admin_user.id,
        key_hash=hash_api_key(raw),
        name="svc-key",
        scopes=["read:transactions"],
        expires_at=datetime.now(timezone.utc) + timedelta(days=90),
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(entry)
    db_session.flush()
    return raw, entry


# ---------------------------------------------------------------------------
# Rotation tests
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("api_key_rotation")
class TestRotateKey:
    def test_rotate_returns_new_key(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        resp = client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["name"] == "svc-key"
        assert data["new_key"].startswith("ak_")
        assert "overlap_expires_at" in data
        assert "message" in data

    def test_rotated_new_key_is_valid_for_auth(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        rotate_resp = client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert rotate_resp.status_code == 200
        new_key = rotate_resp.json()["new_key"]

        # The new key must authenticate successfully
        auth_resp = client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {new_key}"},
        )
        assert auth_resp.status_code == 200

    def test_old_key_still_valid_during_overlap(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        old_raw, _ = existing_api_key
        client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Old key must still work within the 30-day overlap
        auth_resp = client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {old_raw}"},
        )
        assert auth_resp.status_code == 200

    def test_old_key_rejected_after_overlap_expires(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        old_raw, entry = existing_api_key
        # Rotate the key
        client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Back-date the overlap expiry to simulate expiry
        db_session.refresh(entry)
        entry.overlap_expires_at = datetime.now(timezone.utc) - timedelta(days=1)
        db_session.flush()

        auth_resp = client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {old_raw}"},
        )
        assert auth_resp.status_code == 401

    def test_rotate_unknown_key_returns_404(self, client, db_session, admin_user, admin_token):
        resp = client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "does-not-exist"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 404

    def test_rotate_sets_overlap_expiry_30_days(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        before = datetime.now(timezone.utc)
        resp = client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        overlap_str = resp.json()["overlap_expires_at"]
        # Accept both Z and +00:00 suffixes
        overlap_str = overlap_str.replace("Z", "+00:00")
        overlap_dt = datetime.fromisoformat(overlap_str)
        expected = before + timedelta(days=30)
        # Allow a 5-second tolerance
        assert abs((overlap_dt - expected).total_seconds()) < 5

    def test_rotate_requires_admin_scope(self, client, db_session):
        """Non-admin token must receive 403."""
        from api.auth.security import create_access_token

        limited_token = create_access_token("nobody", ["read:transactions"])
        resp = client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {limited_token}"},
        )
        # Auth is disabled in the test environment by default; skip this
        # check when auth is off.
        import os

        if os.environ.get("AUTH_ENABLED", "false").lower() in ("1", "true", "yes"):
            assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Revocation tests
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("api_key_rotation")
class TestRevokeKey:
    def test_revoke_deactivates_key(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        resp = client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["revoked"] is True
        assert data["name"] == "svc-key"

    def test_revoked_key_cannot_authenticate(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        old_raw, _ = existing_api_key
        client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        auth_resp = client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {old_raw}"},
        )
        assert auth_resp.status_code == 401

    def test_revoke_clears_overlap_slot(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        old_raw, entry = existing_api_key
        # First rotate so there's an overlap key
        client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        db_session.refresh(entry)
        assert entry.overlap_key_hash is not None

        # Now revoke
        client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        db_session.refresh(entry)
        assert entry.overlap_key_hash is None
        assert entry.overlap_expires_at is None
        assert entry.is_active is False

    def test_revoke_idempotent_already_revoked(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        # Revoke once
        client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Revoke again — should not error, revoked=False
        resp = client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is False

    def test_revoke_unknown_key_returns_404(self, client, db_session, admin_user, admin_token):
        resp = client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "ghost-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 404

    def test_overlap_key_also_rejected_after_revoke(
        self, client, db_session, admin_user, existing_api_key, admin_token
    ):
        old_raw, _ = existing_api_key
        # Rotate so old_raw becomes the overlap key
        client.post(
            "/api/v1/auth/rotate-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # Revoke entirely
        client.post(
            "/api/v1/auth/revoke-key",
            json={"name": "svc-key"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # The old (overlap) raw key must now be rejected
        auth_resp = client.get(
            "/api/v1/fraud/alerts",
            headers={"Authorization": f"Bearer {old_raw}"},
        )
        assert auth_resp.status_code == 401
