"""Tests for the contact form & support ticket endpoints — issue #305."""
from __future__ import annotations

import pytest

VALID = {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "subject": "Need help with the API",
    "message": "I have a question about anomaly scoring.",
}


@pytest.mark.xdist_group("api_contact")
class TestContactForm:
    def test_submit_creates_support_ticket(self, client):
        resp = client.post("/api/v1/contact", json=VALID)
        assert resp.status_code == 201
        data = resp.json()
        assert data["ticket"]["reference"].startswith("TKT-")
        assert data["ticket"]["status"] == "open"
        assert "received" in data["message"].lower()

    def test_submit_invalid_email_returns_422(self, client):
        resp = client.post("/api/v1/contact", json={**VALID, "email": "not-an-email"})
        assert resp.status_code == 422

    def test_submit_blank_message_returns_422(self, client):
        resp = client.post("/api/v1/contact", json={**VALID, "message": "   "})
        assert resp.status_code == 422

    def test_submit_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID.items() if k != "subject"}
        assert client.post("/api/v1/contact", json=payload).status_code == 422

    def test_get_ticket_by_reference(self, client):
        ref = client.post("/api/v1/contact", json=VALID).json()["ticket"]["reference"]
        resp = client.get(f"/api/v1/contact/tickets/{ref}")
        assert resp.status_code == 200
        assert resp.json()["reference"] == ref

    def test_get_unknown_ticket_returns_404(self, client):
        assert client.get("/api/v1/contact/tickets/TKT-DOESNOTEXIST").status_code == 404

    def test_recaptcha_failure_returns_400(self, client, monkeypatch):
        # Force verification to fail regardless of config.
        from api.routers import contact as contact_module

        async def _reject(_token):
            return False

        monkeypatch.setattr(contact_module, "verify_recaptcha", _reject)
        assert client.post("/api/v1/contact", json=VALID).status_code == 400

    def test_emails_sent_on_success(self, client, monkeypatch):
        # Capture the best-effort email dispatch without hitting the network.
        from api.routers import contact as contact_module

        sent = {}

        async def _capture(ticket):
            sent["reference"] = ticket.reference

        monkeypatch.setattr(contact_module, "send_contact_emails", _capture)
        resp = client.post("/api/v1/contact", json=VALID)
        assert resp.status_code == 201
        assert sent.get("reference") == resp.json()["ticket"]["reference"]
