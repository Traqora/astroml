"""Request body size limit middleware (issue #1247).

Rejects requests whose ``Content-Length`` header or streamed body exceeds
``max_body_size`` bytes with HTTP 413 and an RFC 7807 problem detail.
"""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Receive, Scope, Send

from api.schemas.errors import PROBLEM_CONTENT_TYPE, problem_detail


class ContentSizeLimitMiddleware:
    """ASGI middleware that enforces a maximum request body size.

    Parameters
    ----------
    app:
        The wrapped ASGI application.
    max_body_size:
        Maximum allowed request body size in bytes.  Requests that exceed
        this limit are rejected immediately with HTTP 413.
    """

    def __init__(self, app: ASGIApp, max_body_size: int) -> None:
        self.app = app
        self.max_body_size = max_body_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: reject based on Content-Length before reading the body.
        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                content_length = int(content_length_raw)
            except ValueError:
                content_length = 0
            if content_length > self.max_body_size:
                await self._send_413(send)
                return

        # Slow path: stream and count bytes as they arrive.
        total = 0
        buffered: list[bytes] = []
        body_too_large = False

        async def limited_receive() -> dict:
            nonlocal total, body_too_large
            message = await receive()
            if message["type"] == "http.request":
                chunk: bytes = message.get("body", b"")
                total += len(chunk)
                if total > self.max_body_size:
                    body_too_large = True
                    # Return an empty terminal message so the app can still
                    # process the rejection without blocking on more data.
                    return {"type": "http.request", "body": b"", "more_body": False}
                buffered.append(chunk)
            return message

        # We need to intercept send so we can replace it with our 413 if
        # the body turned out to be too large.  However, since we wrap
        # receive lazily, the app may start sending headers before we know
        # the body is over-limit.  We therefore buffer the app's response
        # until we know the body fits, then flush it; if the body was too
        # large we discard the app's response and emit 413 instead.

        response_started = False
        response_messages: list[dict] = []

        async def buffered_send(message: dict) -> None:
            nonlocal response_started
            response_messages.append(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                response_started = True

        await self.app(scope, limited_receive, buffered_send)

        if body_too_large:
            await self._send_413(send)
        else:
            for message in response_messages:
                await send(message)

    async def _send_413(self, send: Send) -> None:
        payload = problem_detail(
            code="REQUEST_TOO_LARGE",
            title="Request entity too large",
            status=413,
            detail="Request body exceeds the maximum allowed size.",
            details=[],
        )
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", PROBLEM_CONTENT_TYPE.encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})
