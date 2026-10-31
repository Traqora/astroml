"""Tests for the WebSocket connection manager's heartbeat/reconnect cycle
(issue #849).

``ConnectionManager`` is pure asyncio with no FastAPI test client
dependency, so these exercise it directly against a stub ``WebSocket``
rather than spinning up a full app/client round trip.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from api.websocket.manager import (
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_MISSED_PONGS,
    ConnectionManager,
)


class _FakeWebSocket:
    """Records every JSON payload sent and simulates `.close()`/errors."""

    def __init__(self, raise_on_send: bool = False) -> None:
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self.sent: list[dict] = []
        self._raise_on_send = raise_on_send

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict) -> None:
        if self._raise_on_send:
            raise ConnectionResetError("peer closed the connection")
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code
        self.close_reason = reason


@pytest.fixture
def manager() -> ConnectionManager:
    return ConnectionManager()


@pytest.mark.asyncio
async def test_connect_accepts_and_registers_the_client(manager: ConnectionManager):
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")

    assert ws.accepted
    assert client in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_disconnect_removes_the_client(manager: ConnectionManager):
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")

    await manager.disconnect(client)

    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_disconnect_is_a_no_op_for_an_already_removed_client(manager: ConnectionManager):
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")
    await manager.disconnect(client)

    # Calling disconnect a second time must not raise (ValueError from
    # list.remove on a missing item, say).
    await manager.disconnect(client)
    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_record_pong_resets_missed_pong_count(manager: ConnectionManager):
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")
    client.missed_pongs = 2

    manager.record_pong(client)

    assert client.missed_pongs == 0


@pytest.mark.asyncio
async def test_heartbeat_loop_sends_a_ping_once_the_interval_elapses(
    manager: ConnectionManager, monkeypatch
):
    """The heartbeat loop's core reconnect-detection cycle: ping on the
    interval, and a timely pong keeps the client connected with no missed
    pongs recorded.
    """
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")

    # Run the heartbeat loop's body without the real 30s sleep.
    sleep_calls = 0

    async def fast_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    # The client ponged recently, so this tick should ping, not disconnect.
    manager.record_pong(client)
    with pytest.raises(asyncio.CancelledError):
        await manager.heartbeat_loop(client)

    assert {"type": "ping"} in ws.sent
    assert not ws.closed
    # heartbeat_loop's finally clause always disconnects on exit, including
    # a clean cancellation -- the caller (the WebSocket route handler) is
    # responsible for not reconnecting a client whose loop already exited.
    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_heartbeat_loop_disconnects_after_max_missed_pongs(
    manager: ConnectionManager, monkeypatch
):
    """The actual reconnect trigger this issue is about: a client that
    never pongs back gets disconnected (forcing it to reconnect) after
    MAX_MISSED_PONGS consecutive missed checks, not immediately on the
    first one.
    """
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")
    # Simulate a pong received long before the loop started, so every tick
    # below counts as missed.
    client.last_pong -= HEARTBEAT_INTERVAL_SECONDS * (MAX_MISSED_PONGS + 1)

    tick_count = 0

    async def fast_sleep(_seconds: float) -> None:
        nonlocal tick_count
        tick_count += 1

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    await manager.heartbeat_loop(client)

    assert ws.closed
    assert ws.close_reason == "heartbeat timeout"
    # Exactly MAX_MISSED_PONGS ticks: the loop exits on the tick count hits
    # the threshold, not one before or one after.
    assert tick_count == MAX_MISSED_PONGS
    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_heartbeat_loop_resets_missed_count_on_an_intervening_pong(
    manager: ConnectionManager, monkeypatch
):
    """A pong received between misses resets the streak, so the client is
    never disconnected purely for an isolated slow response.
    """
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")
    client.last_pong -= HEARTBEAT_INTERVAL_SECONDS * 2  # one tick behind

    tick_count = 0

    async def fast_sleep(_seconds: float) -> None:
        nonlocal tick_count
        tick_count += 1
        if tick_count == 1:
            # The client pongs back right after the first missed tick.
            manager.record_pong(client)
        if tick_count > 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    with pytest.raises(asyncio.CancelledError):
        await manager.heartbeat_loop(client)

    assert not ws.closed
    assert client.missed_pongs == 0


@pytest.mark.asyncio
async def test_heartbeat_loop_disconnects_cleanly_when_cancelled(
    manager: ConnectionManager, monkeypatch
):
    """Cancelling the heartbeat task (e.g. on a clean client-initiated
    disconnect) must still remove the client from the manager, not leave
    it registered indefinitely.
    """
    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")

    async def cancel_immediately(_seconds: float) -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(asyncio, "sleep", cancel_immediately)

    with pytest.raises(asyncio.CancelledError):
        await manager.heartbeat_loop(client)

    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_send_json_disconnects_the_client_on_a_send_failure(manager: ConnectionManager):
    """A send failure (the peer went away without a clean close) must
    remove the client, or every subsequent broadcast/ping would keep
    failing against a dead connection forever.
    """
    ws = _FakeWebSocket(raise_on_send=True)
    client = await manager.connect(ws, "transactions")

    await manager.send_json(client, {"type": "ping"})

    assert client not in manager._clients["transactions"]


@pytest.mark.asyncio
async def test_broadcast_delivers_to_every_client_on_the_channel(manager: ConnectionManager):
    ws1 = _FakeWebSocket()
    ws2 = _FakeWebSocket()
    client1 = await manager.connect(ws1, "alerts")
    client2 = await manager.connect(ws2, "alerts")

    await manager.broadcast("alerts", {"type": "alert", "id": 1})

    assert {"type": "alert", "id": 1} in ws1.sent
    assert {"type": "alert", "id": 1} in ws2.sent
    assert client1 in manager._clients["alerts"]
    assert client2 in manager._clients["alerts"]


@pytest.mark.asyncio
async def test_broadcast_removes_only_the_client_whose_send_failed(manager: ConnectionManager):
    healthy = _FakeWebSocket()
    dead = _FakeWebSocket(raise_on_send=True)
    healthy_client = await manager.connect(healthy, "alerts")
    dead_client = await manager.connect(dead, "alerts")

    await manager.broadcast("alerts", {"type": "alert", "id": 2})

    assert healthy_client in manager._clients["alerts"]
    assert dead_client not in manager._clients["alerts"]


@pytest.mark.asyncio
async def test_rate_limit_drops_messages_beyond_the_per_second_cap(manager: ConnectionManager):
    from api.websocket.manager import MAX_MESSAGES_PER_SECOND

    ws = _FakeWebSocket()
    client = await manager.connect(ws, "transactions")

    for i in range(MAX_MESSAGES_PER_SECOND + 5):
        await manager.send_json(client, {"type": "tick", "i": i})

    assert len(ws.sent) == MAX_MESSAGES_PER_SECOND


@pytest.mark.asyncio
async def test_llm_chat_handler_logs_unhandled_exceptions(caplog):
    """The LLM chat WebSocket handler's broad except previously bound the
    exception as `e` and never used it -- no log call at all. Confirms it
    now logs at ERROR with a traceback (issue #849's "no broad except
    without logging" requirement).

    The handler's own `except Exception` does not re-raise or break out of
    its `while True` loop, so after the first logged failure this raises
    `WebSocketDisconnect` to exit the loop the same way a real dropped
    connection would, rather than looping on the same `RuntimeError`
    forever.
    """
    from fastapi import WebSocketDisconnect

    from api.routers.ws import _handle_llm_chat
    from api.routers.ws import ws_manager as real_ws_manager

    class _RaisingWebSocket(_FakeWebSocket):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def receive_text(self) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            raise WebSocketDisconnect()

    ws = _RaisingWebSocket()
    # A real _ClientState, produced by the same ws_manager singleton
    # _handle_llm_chat itself calls send_json through: a bare stub object
    # has no send_timestamps attribute, which send_json's rate limiter
    # reads right after logging the error, to tell the client "An error
    # occurred".
    client = await real_ws_manager.connect(ws, "llm")

    with caplog.at_level(logging.ERROR, logger="api.routers.ws"):
        await _handle_llm_chat(ws, client)

    assert any(
        "Unhandled error in LLM chat WebSocket handler" in r.message and r.exc_info is not None
        for r in caplog.records
    )
