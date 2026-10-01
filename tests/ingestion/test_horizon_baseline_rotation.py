"""Baseline rotation checks for the Horizon streaming cursor (#939).

The ingest cursor is the replay/skip baseline: it tells Horizon where the
stream left off, so a rotated baseline must be *reused* after a reconnect
and must never rewind (replayed transactions would be double-normalized,
while a lost baseline after ``now`` would silently skip live transactions).

These tests pin the baseline-rotation contract on top of the stream tests
in ``tests/test_horizon_stream.py``: rotation on every valid paging token,
reuse of the rotated baseline in the next request after a disconnect,
monotonicity (never rewinds), and rejection of malformed paging tokens.
"""

import asyncio
import json
from urllib.parse import parse_qs, urlparse

from astroml.ingestion import HorizonStreamingClient


def test_rotated_baseline_is_reused_after_reconnect():
    """After a disconnect, the next request must carry the rotated baseline.

    Rotation happens when a paging token arrives; losing it would replay or
    skip transactions depending on direction.
    """

    async def run_test():
        request_cursors = []
        first_tx_event = asyncio.Event()
        connection_count = 0

        async def handler(reader, writer):
            nonlocal connection_count
            connection_count += 1

            request_line = await reader.readline()
            target = request_line.decode("ascii").split(" ")[1]
            query = parse_qs(urlparse(target).query)
            request_cursors.append(query.get("cursor", [""])[0])

            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            if connection_count == 1:
                # Rotate the baseline once, then close to force a reconnect.
                payload = {"id": "tx-1", "paging_token": "777"}
                response = (
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Type: text/event-stream\r\n"
                    "Connection: close\r\n\r\n"
                    f"data: {json.dumps(payload)}\n\n"
                )
                writer.write(response.encode("utf-8"))
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            first_tx_event.set()
            writer.write(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.05,
            max_reconnect_delay=0.1,
        )

        async def on_transaction(_tx):
            return None

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(first_tx_event.wait(), timeout=2.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert request_cursors[0] == "now"
        assert request_cursors[1] == "777"
        assert client.cursor == "777"

    asyncio.run(run_test())


def test_baseline_never_rewinds_on_rotation():
    """A paging token lower than the current baseline must not rotate it."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            # Rotate forward, then send a stale token from a replayed event.
            body = (
                f"data: {json.dumps({'id': 'tx-1', 'paging_token': '500'})}\n\n"
                f"data: {json.dumps({'id': 'tx-0', 'paging_token': '499'})}\n\n"
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n" + body.encode("utf-8")
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.05,
            max_reconnect_delay=0.1,
        )
        cursors_seen = []

        async def on_transaction(tx):
            cursors_seen.append(client.cursor)

        try:
            await client.start(on_transaction)
            await asyncio.sleep(0.3)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert client.cursor == "500", "stale token must not rewind the baseline"

    asyncio.run(run_test())


def test_baseline_rotates_on_every_valid_paging_token():
    """Rotation is per-event: each valid token immediately becomes the baseline."""

    async def run_test():
        seen = []

        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            events = "".join(
                f"data: {json.dumps({'id': f'tx-{i}', 'paging_token': str(100 + i)})}\n\n"
                for i in range(3)
            )
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n" + events.encode("utf-8")
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            seen.append(client.cursor)

        try:
            await client._consume_stream(on_transaction)
        finally:
            server.close()
            await server.wait_closed()

        assert seen == ["100", "101", "102"]

    asyncio.run(run_test())


def test_baseline_ignores_malformed_paging_tokens():
    """Events without a usable paging token must not rotate the baseline."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            payload = {"id": "tx-no-token", "paging_token": None}
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n" + f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            cursor="seed",
        )

        async def on_transaction(_tx):
            return None

        try:
            await client._consume_stream(on_transaction)
        finally:
            server.close()
            await server.wait_closed()

        assert client.cursor == "seed"

    asyncio.run(run_test())
