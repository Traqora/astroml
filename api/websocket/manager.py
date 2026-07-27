"""WebSocket connection manager (issue #239)."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 30
MAX_MISSED_PONGS = 3
MAX_MESSAGES_PER_SECOND = 10


@dataclass
class _ClientState:
    websocket: WebSocket
    channel: str
    last_pong: float = field(default_factory=time.monotonic)
    missed_pongs: int = 0
    send_timestamps: list[float] = field(default_factory=list)


class ConnectionManager:
    """Manages WebSocket clients with heartbeat and per-connection rate limiting."""

    def __init__(self) -> None:
        self._clients: dict[str, list[_ClientState]] = {
            "transactions": [],
            "alerts": [],
            "llm": [],
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str) -> _ClientState:
        await websocket.accept()
        client = _ClientState(websocket=websocket, channel=channel)
        async with self._lock:
            self._clients.setdefault(channel, []).append(client)
        logger.info("WebSocket client connected to %s (total=%d)", channel, len(self._clients[channel]))
        return client

    async def disconnect(self, client: _ClientState) -> None:
        async with self._lock:
            bucket = self._clients.get(client.channel, [])
            if client in bucket:
                bucket.remove(client)
        logger.info("WebSocket client disconnected from %s", client.channel)

    def _can_send(self, client: _ClientState) -> bool:
        now = time.monotonic()
        client.send_timestamps = [t for t in client.send_timestamps if now - t < 1.0]
        if len(client.send_timestamps) >= MAX_MESSAGES_PER_SECOND:
            return False
        client.send_timestamps.append(now)
        return True

    async def send_json(self, client: _ClientState, payload: dict[str, Any]) -> None:
        if not self._can_send(client):
            return
        try:
            await client.websocket.send_json(payload)
        except Exception:  # noqa: BLE001
            await self.disconnect(client)

    async def broadcast(self, channel: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            clients = list(self._clients.get(channel, []))

        dead: list[_ClientState] = []
        for client in clients:
            if not self._can_send(client):
                continue
            try:
                await client.websocket.send_json(payload)
            except Exception:  # noqa: BLE001
                dead.append(client)

        for client in dead:
            await self.disconnect(client)

    async def heartbeat_loop(self, client: _ClientState) -> None:
        """Send periodic pings; disconnect after missed pongs."""
        try:
            while True:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if time.monotonic() - client.last_pong > HEARTBEAT_INTERVAL_SECONDS:
                    client.missed_pongs += 1
                else:
                    client.missed_pongs = 0

                if client.missed_pongs >= MAX_MISSED_PONGS:
                    await client.websocket.close(code=1000, reason="heartbeat timeout")
                    break

                await self.send_json(client, {"type": "ping"})
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            pass
        finally:
            await self.disconnect(client)

    def record_pong(self, client: _ClientState) -> None:
        client.last_pong = time.monotonic()
        client.missed_pongs = 0


ws_manager = ConnectionManager()
