"""Async streaming client for Stellar Horizon transaction events."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import ssl
from collections.abc import Callable
from typing import Any
from urllib.parse import urlencode, urlparse

Transaction = dict[str, Any]
TransactionHandler = Callable[[Transaction], Any]


class HorizonStreamError(RuntimeError):
    """Raised when the Horizon stream returns an invalid HTTP response."""


class HorizonStreamingClient:
    """Consume Horizon transaction events over Server-Sent Events (SSE)."""

    def __init__(
        self,
        *,
        base_url: str = "https://horizon.stellar.org",
        endpoint: str = "/transactions",
        cursor: str = "now",
        reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("base_url must use http or https")
        if not parsed.hostname:
            raise ValueError("base_url must include a hostname")
        if reconnect_delay <= 0 or max_reconnect_delay <= 0:
            raise ValueError("Reconnect delays must be positive")

        self._base_url = parsed
        self._endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self._cursor = str(cursor)
        self._reconnect_delay = reconnect_delay
        self._max_reconnect_delay = max_reconnect_delay
        self._logger = logger or logging.getLogger(__name__)
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._writer: asyncio.StreamWriter | None = None

    @property
    def cursor(self) -> str:
        return self._cursor

    async def start(self, on_transaction: TransactionHandler) -> None:
        if self._task and not self._task.done():
            raise RuntimeError("stream already running")
        self._stop_event.clear()
        self._task = asyncio.create_task(self.stream(on_transaction))

    async def stop(self) -> None:
        self._stop_event.set()
        writer = self._writer
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover - transport specific
                pass

        if self._task is not None:
            task = self._task
            self._task = None
            if task is not asyncio.current_task():
                await task

    async def stream(self, on_transaction: TransactionHandler) -> None:
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
            except Exception:
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

            data_lines: list[str] = []
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
                pass
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

        paging_token = tx.get("paging_token")
        if paging_token is not None:
            candidate = str(paging_token)
            if self._should_rotate_baseline(candidate):
                self._cursor = candidate

        result = on_transaction(tx)
        if inspect.isawaitable(result):
            await result

    def _should_rotate_baseline(self, candidate: str) -> bool:
        """Decide whether ``candidate`` may become the new cursor baseline (#939).

        The cursor is the replay/skip baseline, so a stale event (a replayed
        paging token lower than the current baseline) must never rewind it —
        rewinding replays already-normalized transactions downstream. Tokens
        that cannot be ordered (non-numeric) are accepted so custom cursor
        schemes still advance; equal tokens are ignored.
        """
        if candidate == self._cursor:
            return False
        try:
            return int(candidate) > int(self._cursor)
        except ValueError:
            # Either side is non-numeric (e.g. the initial "now" cursor or a
            # custom scheme): accept the server-provided token.
            return True

    def _request_path(self) -> str:
        query = urlencode({"cursor": self._cursor, "stream": "true"})
        return f"{self._endpoint}?{query}"
