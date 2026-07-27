"""reCAPTCHA verification for the contact form (issue #305).

Verification is skipped when no secret is configured, so the contact form works
in development/test without a reCAPTCHA key and only enforces spam protection in
environments where ``contact_recaptcha_secret`` is set.
"""
from __future__ import annotations

import logging

import httpx

from api.config import settings

logger = logging.getLogger(__name__)

VERIFY_URL = "https://www.google.com/recaptcha/api/siteverify"


async def verify_recaptcha(token: str | None) -> bool:
    """Return True when the token is valid, or when verification is disabled."""
    secret = settings.contact_recaptcha_secret
    if not secret:
        return True  # verification disabled
    if not token:
        return False

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                VERIFY_URL, data={"secret": secret, "response": token}
            )
        data = resp.json()
    except Exception as exc:  # network/transport error — fail closed
        logger.warning("reCAPTCHA verification request failed: %s", exc)
        return False

    if not data.get("success", False):
        return False

    # reCAPTCHA v3 returns a score in [0, 1]; enforce a minimum when present.
    score = data.get("score")
    if score is not None and score < settings.contact_recaptcha_min_score:
        return False

    return True
