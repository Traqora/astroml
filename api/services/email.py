"""Transactional email for the contact form (issue #305).

Sends via SendGrid when ``sendgrid_api_key`` is configured; otherwise logs the
message so the feature works in development/test without external credentials.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from api.config import settings

logger = logging.getLogger(__name__)

SENDGRID_URL = "https://api.sendgrid.com/v3/mail/send"


async def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email. Returns True if dispatched to the provider."""
    api_key = settings.sendgrid_api_key
    if not api_key:
        logger.info(
            "Email not sent (SendGrid not configured) to=%s subject=%s", to, subject
        )
        return False

    payload: dict[str, Any] = {
        "personalizations": [{"to": [{"email": to}]}],
        "from": {"email": settings.contact_email_from},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                SENDGRID_URL,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
        resp.raise_for_status()
        return True
    except Exception as exc:  # delivery failure must not break the request
        logger.warning("Failed to send email to %s: %s", to, exc)
        return False


async def send_contact_emails(ticket: Any) -> None:
    """Best-effort: notify support and auto-reply to the submitter."""
    await send_email(
        to=settings.contact_support_email,
        subject=f"[{ticket.reference}] New contact: {ticket.subject}",
        body=(
            f"New contact form submission ({ticket.reference})\n\n"
            f"From: {ticket.name} <{ticket.email}>\n"
            f"Subject: {ticket.subject}\n\n"
            f"{ticket.message}\n"
        ),
    )
    await send_email(
        to=ticket.email,
        subject=f"We received your message ({ticket.reference})",
        body=(
            f"Hi {ticket.name},\n\n"
            f"Thanks for reaching out. Your support ticket {ticket.reference} has "
            f"been created and our team will get back to you shortly.\n\n"
            f"Subject: {ticket.subject}\n\n"
            f"— The AstroML team\n"
        ),
    )
