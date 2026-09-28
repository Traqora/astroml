"""Idempotency tests for the Horizon streaming client (issue #995).

SSE plus automatic reconnection is at-least-once: when a connection drops,
Horizon replays the tail of the page it had already begun sending. Before
issue #995 the client forwarded every replay straight to ``on_transaction``, so
a single Stellar transaction could reach a handler many times per second of
outage. Downstream that is not a harmless retry — ``ClaimService.submit_claim``
keys pending claims by ``claim_reference`` and rebuilds the submission, so a
replay resets ``retry_count`` to 0 and restarts the retry budget for a claim
that is already in flight.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging

import pytest

from astroml.ingestion.horizon_stream import HorizonStreamingClient

_OK_HEADERS = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"


def _frame(**payload) -> bytes:
    """Encode one SSE `data:` frame with its blank-line terminator."""
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


async def _drain_request(reader: asyncio.StreamReader) -> None:
    """Consume the request line and headers of a test server connection."""
    await reader.readline()
    while True:
        line = await reader.readline()
        if line in {b"\r\n", b"\n", b""}:
            return


async def _wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll ``predicate`` until it is truthy or ``timeout`` elapses."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


def _serve_once(
    frames: bytes, on_transaction, **client_kwargs
) -> tuple[list, HorizonStreamingClient]:
    """Push ``frames`` down a single SSE connection and collect the deliveries.

    Args:
        frames: Raw response body, usually several ``_frame`` calls concatenated.
        on_transaction: Handler invoked by the client.
        **client_kwargs: Passed through to ``HorizonStreamingClient``.

    Returns:
        The transactions the handler saw, and the client that delivered them.
    """
    received: list = []

    async def run_test() -> tuple[list, HorizonStreamingClient]:
        async def handler(reader, writer):
            await _drain_request(reader)
            writer.write(_OK_HEADERS + frames)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}", **client_kwargs)

        async def counting_handler(tx):
            received.append(tx)
            result = on_transaction(tx)
            if inspect.isawaitable(result):
                await result

        try:
            await client._consume_stream(counting_handler)
        finally:
            server.close()
            await server.wait_closed()
        return received, client

    return asyncio.run(run_test())


def _serve_with_replays(
    on_transaction, replays: int = 3, **client_kwargs
) -> tuple[list, HorizonStreamingClient]:
    """Drive the reconnect loop against a server that always replays one transaction.

    The test server answers every connection with the same ``paging_token``,
    which is what Horizon does once a connection has dropped mid-page. The run
    ends once the first delivery lands and at least ``replays`` further replays
    have been observed.

    Returns:
        The transactions the handler saw, and the client that delivered them.
    """
    received: list = []

    async def run_test() -> tuple[list, HorizonStreamingClient]:
        async def handler(reader, writer):
            await _drain_request(reader)
            writer.write(_OK_HEADERS + _frame(id="tx-1", paging_token="12345"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.02,
            max_reconnect_delay=0.05,
            **client_kwargs,
        )

        async def counting_handler(tx):
            received.append(tx)
            result = on_transaction(tx)
            if inspect.isawaitable(result):
                await result

        task = asyncio.create_task(client.stream(counting_handler))
        try:
            await _wait_until(lambda: bool(received))
            await _wait_until(lambda: client.duplicates_skipped >= replays)
        finally:
            await client.stop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            server.close()
            await server.wait_closed()
        return received, client

    return asyncio.run(run_test())


# ---------------------------------------------------------------------------
# Replay suppression
# ---------------------------------------------------------------------------


def test_replayed_transaction_reaches_handler_once():
    """The same paging_token sent repeatedly is delivered exactly once."""
    received, client = _serve_once(_frame(id="tx-1", paging_token="12345") * 3, lambda _tx: None)

    assert [tx["id"] for tx in received] == ["tx-1"]
    assert client.duplicates_skipped == 2


def test_reconnect_replay_is_suppressed():
    """Horizon replaying the tail on every reconnect still yields one delivery."""
    received, client = _serve_with_replays(lambda _tx: None, replays=3)

    assert [tx["paging_token"] for tx in received] == ["12345"]
    assert client.duplicates_skipped >= 3


def test_duplicate_suppression_preserves_cursor_progress():
    """Skipping a replay never rewinds the cursor."""
    _received, client = _serve_with_replays(lambda _tx: None, replays=2)

    assert client.cursor == "12345"


def test_dedupe_can_be_disabled():
    """`dedupe=False` restores raw at-least-once delivery."""
    received, client = _serve_once(
        _frame(id="tx-1", paging_token="77") * 3, lambda _tx: None, dedupe=False
    )

    assert len(received) == 3
    assert {tx["paging_token"] for tx in received} == {"77"}
    assert client.duplicates_skipped == 0


def test_distinct_transactions_are_all_delivered():
    """De-duplication keys on paging_token, not on arrival order."""
    frames = _frame(id="tx-1", paging_token="1") + _frame(id="tx-2", paging_token="2")

    received, _client = _serve_once(frames, lambda _tx: None)

    assert [tx["paging_token"] for tx in received] == ["1", "2"]


def test_transaction_without_paging_token_is_always_delivered():
    """A payload with no paging_token cannot be identified, so it is not filtered."""
    body = json.dumps({"id": "tx-nopaging", "type": "payment"}).encode()
    frames = b"data: " + body + b"\n\n" + b"data: " + body + b"\n\n"

    received, _client = _serve_once(frames, lambda _tx: None)

    assert len(received) == 2
    assert all("paging_token" not in tx for tx in received)


def test_seen_window_is_bounded():
    """The de-duplication memory footprint is capped by `dedupe_capacity`."""
    client = HorizonStreamingClient(dedupe_capacity=3)

    for token in ("1", "2", "3", "4", "5"):
        client._remember(token)

    assert len(client._seen) == 3
    assert list(client._seen) == ["3", "4", "5"]


def test_dedupe_capacity_must_be_positive():
    """A non-positive window is rejected at construction, like the other options."""
    with pytest.raises(ValueError, match="dedupe_capacity"):
        HorizonStreamingClient(dedupe_capacity=0)


def test_duplicate_suppression_is_logged(caplog):
    """A suppressed replay leaves a DEBUG breadcrumb for operators."""
    client = HorizonStreamingClient()

    async def run_test():
        with caplog.at_level(logging.DEBUG, logger="astroml.ingestion.horizon_stream"):
            for _ in range(2):
                await client._handle_payload(
                    json.dumps({"id": "tx-1", "paging_token": "5"}), lambda _tx: None
                )

    asyncio.run(run_test())

    assert client.duplicates_skipped == 1
    assert any("replayed paging_token" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Claim-submission idempotency
# ---------------------------------------------------------------------------


def test_replay_does_not_resubmit_a_claim():
    """The end-to-end property: one transaction yields one claim submission.

    ``ClaimService.submit_claim`` keys pending claims by ``claim_reference`` and
    rebuilds the submission, resetting ``retry_count``. Feeding it a replayed
    transaction therefore restarts the retry budget for an in-flight claim.
    """
    submitted: list[str] = []

    def submit_claim(tx) -> str:
        # Stands in for ClaimService.submit_claim: the claim_reference is the
        # transaction id, so a replay would overwrite the in-flight submission.
        reference = tx["id"]
        submitted.append(reference)
        return reference

    _received, client = _serve_with_replays(submit_claim, replays=3)

    assert submitted == ["tx-1"]
    assert client.duplicates_skipped >= 3


def test_distinct_claims_each_reach_the_service():
    """Distinct transactions still produce one submission each."""
    submitted: list[str] = []
    frames = (
        _frame(id="tx-1", paging_token="1")
        + _frame(id="tx-1", paging_token="1")
        + _frame(id="tx-2", paging_token="2")
    )

    received, _client = _serve_once(frames, lambda tx: submitted.append(tx["id"]))

    assert [tx["id"] for tx in received] == ["tx-1", "tx-2"]
    assert submitted == ["tx-1", "tx-2"]


# ---------------------------------------------------------------------------
# Existing guarantees still hold
# ---------------------------------------------------------------------------


def test_malformed_payloads_are_skipped_not_delivered():
    """Non-JSON and non-object payloads never reach the handler."""
    received: list = []
    client = HorizonStreamingClient()

    async def run_test():
        for payload in ("not json at all", json.dumps([1, 2, 3])):
            await client._handle_payload(payload, received.append)

    asyncio.run(run_test())

    assert received == []
