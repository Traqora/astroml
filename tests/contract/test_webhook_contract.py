"""Contract tests for the REST API webhook endpoints (issue #713).

Lightweight contract for the webhook-facing surface, tested against the
mounted routes so the public API cannot drift silently as it evolves:

* ``POST /api/v1/notifications/webhook/github`` accepts the four known
  ``event_type`` values with ``202`` + ``{"status": "accepted"}``.
* Unknown ``event_type`` values are rejected with ``400``.
* Malformed payloads are rejected with ``422`` (FastAPI/Pydantic).
* Non-POST methods are rejected with ``405``.
* Rejected events have no database side effects.

The tests mount only the notifications router on a minimal app (SQLite via
aiosqlite) rather than the full ``api.app`` so they stay independent of
unrelated app wiring.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

# The notifications router is self-contained, but importing it through the
# `api.routers` package __init__ would execute every router import (including
# torch-dependent tracking code) before the module under contract is even
# loaded. Stub the intermediate package modules — with a real __path__ so
# unrelated submodule imports keep resolving from disk — so these contract
# tests stay fast and independent of unrelated app wiring.
_ROUTERS_DIR = Path(__file__).resolve().parent.parent.parent / "api" / "routers"
for _stub_name, _stub_path in (
    ("api.routers", _ROUTERS_DIR),
    ("api.routers.v1", _ROUTERS_DIR / "v1"),
):
    _stub = sys.modules.get(_stub_name)
    if _stub is None:
        _stub = types.ModuleType(_stub_name)
        _stub.__path__ = [str(_stub_path)]
        sys.modules[_stub_name] = _stub

import api.models.orm  # noqa: F401,E402 — registers ORM models on Base.metadata
from api.database import get_sync_db
from api.models.orm import Notification
from api.routers.notifications import router as notifications_router
from astroml.db.schema import Base

WEBHOOK_PATH = "/api/v1/notifications/webhook/github"

BASE_PAYLOAD = {
    "repo": "Traqora/astroml",
    "link": "https://github.com/Traqora/astroml/pull/1",
}


def _payload(event_type: str, **overrides):
    data = {"event_type": event_type, **BASE_PAYLOAD}
    data.update(overrides)
    return data


@pytest.fixture()
def client(tmp_path):
    """Minimal app with only the notifications router + ephemeral SQLite."""
    db_file = tmp_path / "webhook_contract.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def _override_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app = FastAPI()
    app.include_router(notifications_router)
    app.dependency_overrides[get_sync_db] = _override_db
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    engine.dispose()


def _notification_count(client: TestClient) -> int:
    """Count Notification rows through the overridden sync session."""
    override = client.app.dependency_overrides[get_sync_db]
    gen = override()
    session: Session = next(gen)
    try:
        return len(session.execute(select(Notification)).scalars().all())
    finally:
        try:
            next(gen)
        except StopIteration:
            pass
        session.close()


class TestWebhookGithubContract:
    """Payload shape + status-code contract for POST /webhook/github."""

    @pytest.mark.parametrize(
        "event_type,extra",
        [
            (
                "pr_comment",
                {
                    "pr_number": 42,
                    "commenter": "alice",
                    "content": "Great work!",
                },
            ),
            (
                "issue_comment",
                {
                    "issue_number": 7,
                    "commenter": "bob",
                    "content": "Needs a test.",
                },
            ),
            (
                "review_request",
                {"pr_number": 50, "reviewer_id": 2},
            ),
            (
                "pr_merged",
                {"pr_number": 51, "author_id": 3},
            ),
        ],
    )
    def test_known_event_types_are_accepted(self, client, event_type, extra):
        response = client.post(WEBHOOK_PATH, json=_payload(event_type, **extra))
        assert response.status_code == 202
        assert response.headers["content-type"].startswith("application/json")
        assert response.json() == {"status": "accepted"}

    def test_unknown_event_type_is_rejected(self, client):
        response = client.post(WEBHOOK_PATH, json=_payload("deploy_succeeded"))
        assert response.status_code == 400
        assert response.json()["detail"] == "Unknown event type"

    def test_unknown_event_has_no_side_effects(self, client):
        before = _notification_count(client)
        response = client.post(WEBHOOK_PATH, json=_payload("deploy_succeeded"))
        assert response.status_code == 400
        assert _notification_count(client) == before

    @pytest.mark.parametrize(
        "payload",
        [
            {},  # empty body
            {"repo": "x/y", "link": "https://example.com"},  # missing event_type
            {"event_type": "pr_comment"},  # missing repo + link
            {"event_type": "pr_comment", "repo": "x/y"},  # missing link
            {"event_type": "pr_comment", "link": "https://example.com"},  # missing repo
            {
                "event_type": "pr_comment",
                "pr_number": "not-an-int",  # wrong type
                "repo": "x/y",
                "link": "https://example.com",
            },
        ],
    )
    def test_malformed_payloads_are_rejected_with_422(self, client, payload):
        response = client.post(WEBHOOK_PATH, json=payload)
        assert response.status_code == 422

    def test_extra_fields_are_ignored(self, client):
        response = client.post(
            WEBHOOK_PATH,
            json=_payload(
                "pr_comment",
                pr_number=1,
                commenter="alice",
                content="hi",
                future_field="must-not-break-the-contract",
            ),
        )
        assert response.status_code == 202
        assert response.json() == {"status": "accepted"}

    @pytest.mark.parametrize("method", ["get", "put", "delete", "patch"])
    def test_only_post_is_allowed(self, client, method):
        # GET/DELETE take no body in this TestClient version; PUT/PATCH do.
        kwargs = (
            {"json": _payload("pr_comment", pr_number=1)}
            if method in ("put", "patch")
            else {}
        )
        response = getattr(client, method)(WEBHOOK_PATH, **kwargs)
        assert response.status_code == 405

    def test_accepted_mention_creates_notification_for_mentioned_user(self, client):
        before = _notification_count(client)
        response = client.post(
            WEBHOOK_PATH,
            json=_payload(
                "pr_comment",
                pr_number=9,
                commenter="alice",
                content="Please review @bob!",
            ),
        )
        assert response.status_code == 202
        # The handler notifies mentioned users; the accepted event must be
        # observable in the store, pinning the side-effect contract.
        assert _notification_count(client) == before + 1
