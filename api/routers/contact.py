"""Contact form & support ticket router (issue #305).

Accepts contact form submissions, verifies reCAPTCHA (when configured),
persists a support ticket, and sends a support notification plus an auto-reply
confirmation to the submitter.
"""
from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.models.orm import SupportTicket
from api.schemas import ContactFormIn, ContactSubmitResponse, SupportTicketOut
from api.services.email import send_contact_emails
from api.services.recaptcha import verify_recaptcha

router = APIRouter(prefix="/api/v1/contact", tags=["contact"])


def _generate_reference() -> str:
    """Short, human-quotable ticket reference, e.g. ``TKT-1A2B3C4D``."""
    return "TKT-" + secrets.token_hex(4).upper()


@router.post("", response_model=ContactSubmitResponse, status_code=201)
async def submit_contact_form(
    payload: ContactFormIn,
    db: AsyncSession = Depends(get_db),
) -> ContactSubmitResponse:
    """Validate, spam-check, persist a support ticket, and send emails."""
    if not await verify_recaptcha(payload.recaptcha_token):
        raise HTTPException(status_code=400, detail="reCAPTCHA verification failed")

    ticket = SupportTicket(
        reference=_generate_reference(),
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        message=payload.message,
        status="open",
    )
    db.add(ticket)
    await db.commit()
    await db.refresh(ticket)

    # Email is best-effort — a delivery failure must not fail the submission.
    try:
        await send_contact_emails(ticket)
    except Exception:  # pragma: no cover - defensive
        pass

    return ContactSubmitResponse(
        message="Your message has been received. We'll be in touch shortly.",
        ticket=SupportTicketOut.model_validate(ticket),
    )


@router.get("/tickets/{reference}", response_model=SupportTicketOut)
async def get_ticket(
    reference: str,
    db: AsyncSession = Depends(get_db),
) -> SupportTicket:
    """Look up a support ticket by its reference."""
    result = await db.execute(
        select(SupportTicket).where(SupportTicket.reference == reference)
    )
    ticket = result.scalar_one_or_none()
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket
