from astroml.utils.exceptions import AstroMLError
"""Async streaming client for Stellar Horizon transaction events."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import ssl
from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode, urlparse

Transaction = dict[str, Any]
TransactionHandler = Callable[[Transaction], Any]

# How many recently delivered paging tokens are remembered for de-duplication.
DEFAULT_DEDUPE_CAPACITY = 1024


class HorizonStreamError(RuntimeError):
    """Raised when the Horizon stream returns an invalid HTTP response."""


class HorizonStreamingClient:
    """Consume Horizon transaction events over Server-Sent Events (SSE).

    SSE plus automatic reconnection makes delivery **at-least-once**: when a
    connection drops, Horizon replays the tail of the stream it had already
    started sending, so the same transaction can arrive again. A handler that
    submits claims or moves money must therefore be idempotent, or must not see
    the replay at all.

    By default the client drops replays for you: a transaction whose
    ``paging_token`` was already delivered is logged at ``DEBUG`` and counted in
    :attr:`duplicates_skipped` instead of being handed to the handler again. The
    cursor is still advanced, so skipping never rewinds the stream. Pass
    ``dedupe=False`` to opt out and receive raw at-least-once delivery.

    Transactions with no ``paging_token`` cannot be identified and are always
    delivered. The memory used for de-duplication is bounded by
    ``dedupe_capacity`` tokens.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://horizon.stellar.org",
        endpoint: str = "/transactions",
        cursor: str = "now",
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        dedupe: bool = True,
        dedupe_capacity: int = DEFAULT_DEDUPE_CAPACITY,
        logger: logging.Logger | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        if not parsed.hostname:
            raise ValueError("base_url must include a hostname")
        if reconnect_delay <= 0 or max_reconnect_delay <= 0:
            raise ValueError("Reconnect delays must be positive")
        if dedupe_capacity < 1:
            raise ValueError("dedupe_capacity must be >= 1")

        self._base_url = parsed
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._cursor = str(cursor)
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._dedupe = dedupe
        self._dedupe_capacity = dedupe_capacity
        self._logger = logger or logging.getLogger(__name__)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._duplicates_skipped = 0

    @property
    def cursor(self) -> str:
        """Where the stream will resume, or has resumed to.

        Advances as transactions arrive (see :meth:`_handle_payload`), so
        reading it after :meth:`stop` gives the reconnect point.
        """
        return self._cursor

    @property
    def duplicates_skipped(self) -> int:
        """Number of replayed transactions suppressed since construction."""
        return self._duplicates_skipped

    async def start(self, on_transaction: TransactionHandler) -> None:
        """Begin consuming in a background task and return immediately.

        Args:
            on_transaction: Called per transaction with the parsed Horizon
                payload. May be a coroutine function; the awaitable is awaited.

        Raises:
            RuntimeError: If a stream is already running. Call
                :meth:`stop` first rather than starting a second consumer,
                which would advance ``cursor`` from two places at once.
        """
        if self._task and not self._task.done():
            raise RuntimeError("stream already running")
        self._stop_event.clear()
        self._task = asyncio.create_task(self.stream(on_transaction))

    async def stop(self) -> None:
        """Ask the stream to end and wait for its task to finish.

        Returns once consumption has actually stopped, so a caller can be
        sure no handler is still running — which is what makes it safe to
        read :attr:`cursor` immediately afterwards.

        Side effects: closes the open transport, which makes
        :meth:`_consume_stream` return, and clears the tracked task. Never
        raises: a transport that is already dead is the outcome we wanted.
        """
        self._stop_event.set()
        writer = self._writer
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - transport specific
                self._logger.debug("Error closing Horizon stream writer", exc_info=True)

        if self._task is not None:
            task = self._task
            self._task = None
            # Called from inside the stream task itself (a handler that
            # shuts down), awaiting our own task would deadlock forever.
            if task is not asyncio.current_task():
                await task

    async def stream(self, on_transaction: TransactionHandler) -> None:
        """Reconnect loop: keep consuming until :meth:`stop` is called.

        Exponential backoff restarts at ``reconnect_delay`` after a
        connection that ended cleanly and doubles per failed attempt up to
        ``max_reconnect_delay``, so a Horizon outage costs one retry per
        capped interval instead of a hot loop, while a routine server-side
        disconnect reconnects promptly.

        Args:
            on_transaction: Handler passed through to each consumption; may be
                a coroutine function.

        Note:
            Exceptions from ``on_transaction`` propagate out of this method
            and end the loop — a failing handler is a bug to surface, not a
            disconnected socket to retry.
        """
        delay = self._reconnect_delay

        while not self._stop_event.is_set():
            try:
                await self._consume_stream(on_transaction)
                if self._stop_event.is_set():
                    break
                delay = self._reconnect_delay
                self._logger.warning("Horizon stream disconnected. Reconnecting in %.2fs", delay)
            except asyncio.CancelledError:
                raise
            except AstroMLError:
                if self._stop_event.is_set():
                    break
                self._logger.exception("Horizon stream error. Reconnecting in %.2fs", delay)

            if self._stop_event.is_set():
                break

            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_reconnect_delay)

        self._logger.info("Horizon streaming client stopped")

    async def _consume_stream(self, on_transaction: TransactionHandler) -> None:
        ssl_context = ssl.create_default_context() if self._base_url.scheme == "https" else None
        port = self._base_url.port or (443 if self._base_url.scheme == "https" else 80)
        host_header = self._base_url.hostname
        if self._base_url.port is not None:
            host_header = f"{host_header}:{self._base_url.port}"

        reader, writer = await asyncio.open_connection(
            host=self._base_url.hostname,
            port=port,
            ssl=ssl_context,
        )
        self._writer = writer

        try:
            request = (
                f"GET {self._request_path()} HTTP/1.1\r\n"
                f"Host: {host_header}\r\n"
                "Accept: text/event-stream\r\n"
                "Connection: close\r\n\r\n"
            )
            writer.write(request.encode("ascii"))
            await writer.drain()

            status = await reader.readline()
            if not status:
                raise HorizonStreamError("No HTTP response from Horizon")

            parts = status.decode("iso-8859-1").strip().split(" ", 2)
            if len(parts) < 2:
                raise HorizonStreamError(f"Invalid HTTP status line: {status!r}")
            status_code = int(parts[1])
            if status_code != 200:
                raise HorizonStreamError(f"Unexpected HTTP status: {status_code}")

            while True:
                header_line = await reader.readline()
                if header_line in {b"\r\n", b"\n", b""}:
                    break

            self._logger.info("Connected to Horizon stream: %s", self._request_path())

            data_lines = []
            while not self._stop_event.is_set():
                line = await reader.readline()
                if not line:
                    return

                text = line.decode("utf-8").rstrip("\r\n")
                if text == "":
                    if data_lines:
                        payload = "\n".join(data_lines)
                        data_lines = []
                        await self._handle_payload(payload, on_transaction)
                    continue

                if text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - transport specific
                self._logger.debug("Error closing Horizon stream writer", exc_info=True)
            if self._writer is writer:
                self._writer = None

    async def _handle_payload(
        self,
        payload: str,
        on_transaction: TransactionHandler,
    ) -> None:
        try:
            tx = json.loads(payload)
        except json.JSONDecodeError:
            self._logger.warning("Skipping non-JSON stream payload: %r", payload)
            return

        if not isinstance(tx, dict):
            self._logger.warning("Skipping non-object transaction payload: %r", tx)
            return

        # Issue #983 — advance the cursor optimistically, but roll it back if
        # the handler fails so the reconnect resumes from the last
        # successfully handled transaction instead of skipping this one.
        previous_cursor = self._cursor
        paging_token = tx.get("paging_token")
        if paging_token is not None:
            token = str(paging_token)
            if self._dedupe and self._already_delivered(token):
                self._duplicates_skipped += 1
                self._logger.debug("Horizon stream: skipping replayed paging_token %s", token)
                return
            self._advance_cursor(token)
            self._remember(token)

        try:
            result = on_transaction(tx)
            if inspect.isawaitable(result):
                await result
        except BaseException:
            self._cursor = previous_cursor
            self._logger.warning(
                "Transaction handler failed; cursor rolled back",
                extra={"cursor": previous_cursor, "paging_token": paging_token},
            )
            raise

    def _already_delivered(self, token: str) -> bool:
        """Whether ``token`` was delivered within the de-duplication window."""
        return token in self._seen

    def _remember(self, token: str) -> None:
        """Record ``token`` as delivered, evicting the oldest once full."""
        self._seen[token] = None
        self._seen.move_to_end(token)
        while len(self._seen) > self._dedupe_capacity:
            self._seen.popitem(last=False)

    def _advance_cursor(self, paging_token: str) -> None:
        """Move the resume cursor forward to `paging_token`.

        Horizon paging tokens are monotonically increasing per stream (they
        encode ledger/tx/op ordering), and the cursor is used verbatim as
        the resume point on reconnect. A token that does not advance the
        cursor (or, worse, moves it backward) is inconsistent with a single
        well-behaved stream and, if applied blindly, would make a
        reconnect replay transactions already delivered to `on_transaction`
        or, if the two happen to compare unequal as strings while equal in
        value, loop forever on the same page. Both the current and
        candidate token are compared numerically when possible so this
        does not misfire on differing zero-padding; a token that fails to
        parse as an integer is treated as opaque and, conservatively,
        rejected rather than assumed safe to adopt.
        """
        try:
            candidate = int(paging_token)
        except (TypeError, ValueError):
            self._logger.warning("Ignoring non-numeric Horizon paging_token: %r", paging_token)
            return

        try:
            current = int(self._cursor)
        except (TypeError, ValueError):
            current = None

        if current is not None and candidate <= current:
            self._logger.warning(
                "Ignoring non-advancing Horizon paging_token: %r (current cursor %r)",
                paging_token,
                self._cursor,
            )
            return

        self._cursor = paging_token

    def _request_path(self) -> str:
        query = urlencode({"cursor": self._cursor, "stream": "true"})
        return f"{self._endpoint}?{query}"
