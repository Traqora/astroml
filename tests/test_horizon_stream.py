import asyncio
import json
import logging
from urllib.parse import parse_qs, urlparse

import pytest

from astroml.ingestion import (
    HorizonRateLimitedError,
    HorizonStreamError,
    HorizonStreamingClient,
)


def test_package_exports_horizon_streaming_client():
    """Guard against `astroml.ingestion`'s __init__ dropping these exports again.

    `HorizonStreamingClient`/`HorizonStreamError` are defined in
    `astroml.ingestion.horizon_stream` but must also be importable from the
    `astroml.ingestion` package root, since callers (including this test
    module) import them that way (issue #976).
    """
    import astroml.ingestion as ingestion_pkg
    from astroml.ingestion.horizon_stream import (
        HorizonStreamError as DirectHorizonStreamError,
    )
    from astroml.ingestion.horizon_stream import (
        HorizonStreamingClient as DirectHorizonStreamingClient,
    )

    assert ingestion_pkg.HorizonStreamingClient is DirectHorizonStreamingClient
    assert ingestion_pkg.HorizonStreamError is DirectHorizonStreamError
    assert "HorizonStreamingClient" in ingestion_pkg.__all__
    assert "HorizonStreamError" in ingestion_pkg.__all__


def test_horizon_stream_ingests_transactions():
    async def run_test():
        received = []
        received_event = asyncio.Event()

        async def handler(reader, writer):
            request_line = await reader.readline()
            assert request_line.startswith(b"GET /transactions?")
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            payload = {"id": "tx-1", "paging_token": "12345"}
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

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.5,
            max_reconnect_delay=1.0,
        )

        async def on_transaction(tx):
            received.append(tx)
            received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=1.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert received == [{"id": "tx-1", "paging_token": "12345"}]
        assert client.cursor == "12345"

    asyncio.run(run_test())


def test_horizon_stream_reconnects_automatically():
    async def run_test():
        received_ids = []
        request_cursors = []
        received_event = asyncio.Event()
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

            payload = {"id": f"tx-{connection_count}", "paging_token": str(connection_count)}
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

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.05,
            max_reconnect_delay=0.1,
        )

        async def on_transaction(tx):
            received_ids.append(tx["id"])
            if len(received_ids) >= 2:
                received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=2.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert connection_count >= 2
        assert received_ids[:2] == ["tx-1", "tx-2"]
        assert request_cursors[0] == "now"
        assert request_cursors[1] == "1"

    asyncio.run(run_test())


def test_horizon_stream_graceful_shutdown():
    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()

            try:
                while True:
                    writer.write(b": keepalive\n\n")
                    await writer.drain()
                    await asyncio.sleep(0.05)
            except Exception:
                pass
            finally:
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
            await asyncio.sleep(0.2)
            await asyncio.wait_for(client.stop(), timeout=1.0)
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert client._task is None

    asyncio.run(run_test())


def test_horizon_stream_raises_on_non_200_status():
    """`_consume_stream` surfaces a `HorizonStreamError` for non-200 responses."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            writer.write(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            return None

        try:
            with pytest.raises(HorizonStreamError, match="503"):
                await client._consume_stream(on_transaction)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())


def test_cursor_advances_on_an_increasing_paging_token():
    client = HorizonStreamingClient(cursor="1000")
    client._advance_cursor("1001")
    assert client.cursor == "1001"


def test_cursor_ignores_a_non_advancing_paging_token(caplog):
    client = HorizonStreamingClient(cursor="1000")
    with caplog.at_level(logging.WARNING):
        client._advance_cursor("1000")
    assert client.cursor == "1000"
    assert "non-advancing" in caplog.text


def test_cursor_ignores_a_regressing_paging_token(caplog):
    # A malicious or misbehaving upstream sending an earlier paging_token
    # must not move the resume cursor backward: reconnecting on it would
    # replay transactions already delivered to on_transaction.
    client = HorizonStreamingClient(cursor="5000")
    with caplog.at_level(logging.WARNING):
        client._advance_cursor("4999")
    assert client.cursor == "5000"
    assert "non-advancing" in caplog.text


def test_cursor_ignores_a_non_numeric_paging_token(caplog):
    client = HorizonStreamingClient(cursor="1000")
    with caplog.at_level(logging.WARNING):
        client._advance_cursor("not-a-number")
    assert client.cursor == "1000"
    assert "non-numeric" in caplog.text


def test_cursor_accepts_the_first_numeric_token_from_the_default_now_cursor():
    # The default "now" cursor is not itself numeric, so the very first
    # real paging_token received must be adopted unconditionally rather
    # than being rejected as non-advancing against an unparsable baseline.
    client = HorizonStreamingClient()
    assert client.cursor == "now"
    client._advance_cursor("12345")
    assert client.cursor == "12345"


def test_cursor_comparison_is_numeric_not_lexicographic():
    # "9" > "10" lexicographically but not numerically; the cursor must
    # compare as integers so equal-value tokens with different padding, or
    # a genuine increase past a power of ten, are not misjudged.
    client = HorizonStreamingClient(cursor="9")
    client._advance_cursor("10")
    assert client.cursor == "10"


def test_consume_stream_raises_rate_limited_error_on_429():
    """A 429 response raises `HorizonRateLimitedError`, not the generic error."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            writer.write(b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            return None

        try:
            with pytest.raises(HorizonRateLimitedError):
                await client._consume_stream(on_transaction)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())


def test_consume_stream_parses_retry_after_on_429():
    """`Retry-After` (delta-seconds form) is captured on the raised error."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            writer.write(
                b"HTTP/1.1 429 Too Many Requests\r\n"
                b"Retry-After: 7\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            return None

        try:
            with pytest.raises(HorizonRateLimitedError) as exc_info:
                await client._consume_stream(on_transaction)
            assert exc_info.value.retry_after == 7.0
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())


def test_consume_stream_ignores_unparseable_retry_after():
    """An HTTP-date or garbage `Retry-After` value is treated as absent."""

    async def run_test():
        async def handler(reader, writer):
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break
            writer.write(
                b"HTTP/1.1 429 Too Many Requests\r\n"
                b"Retry-After: Wed, 21 Oct 2026 07:28:00 GMT\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(base_url=f"http://127.0.0.1:{port}")

        async def on_transaction(_tx):
            return None

        try:
            with pytest.raises(HorizonRateLimitedError) as exc_info:
                await client._consume_stream(on_transaction)
            assert exc_info.value.retry_after is None
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run_test())


def test_stream_retries_after_429_with_jittered_backoff():
    """`stream()` catches the 429 and retries, rather than crashing the loop.

    Also covers the pre-existing bug this fix touches: `HorizonStreamError`
    is a plain `RuntimeError`, not an `AstroMLError`, so `stream()`'s old
    `except AstroMLError` clause never actually caught it and any non-200
    response (429 or otherwise) crashed the reconnect loop outright.
    """

    async def run_test():
        received_ids = []
        received_event = asyncio.Event()
        connection_count = 0

        async def handler(reader, writer):
            nonlocal connection_count
            connection_count += 1
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            if connection_count == 1:
                writer.write(b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\n\r\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            payload = {"id": "tx-1", "paging_token": "1"}
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

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            rate_limit_backoff=0.05,
            max_rate_limit_backoff=0.1,
        )

        async def on_transaction(tx):
            received_ids.append(tx["id"])
            received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=2.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert connection_count == 2
        assert received_ids == ["tx-1"]

    asyncio.run(run_test())


def test_stream_jittered_wait_is_bounded_by_current_backoff(monkeypatch):
    """The jittered 429 wait is drawn from `[0, rate_limit_backoff]`."""

    async def run_test():
        captured_delays = []

        async def fake_sleep(seconds):
            captured_delays.append(seconds)
            if len(captured_delays) >= 2:
                client._stop_event.set()

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        async def always_429(_on_transaction):
            raise HorizonRateLimitedError()

        client = HorizonStreamingClient(
            base_url="http://127.0.0.1:1",
            rate_limit_backoff=1.0,
            max_rate_limit_backoff=4.0,
        )
        client._consume_stream = always_429

        async def on_transaction(_tx):
            return None

        await client.stream(on_transaction)

        assert len(captured_delays) >= 2
        assert 0.0 <= captured_delays[0] <= 1.0
        # Backoff ceiling doubles after the first 429, so the second
        # jittered wait is drawn from a wider range.
        assert 0.0 <= captured_delays[1] <= 2.0

    asyncio.run(run_test())


def test_stream_resets_rate_limit_backoff_after_clean_reconnect():
    """A clean reconnect after a 429 resets the backoff ceiling, not just the
    normal reconnect delay — a later, independent 429 should not inherit an
    escalated ceiling from an earlier, unrelated rate-limit episode."""

    async def run_test():
        connection_count = 0
        received_ids = []
        received_event = asyncio.Event()

        async def handler(reader, writer):
            nonlocal connection_count
            connection_count += 1
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in {b"\r\n", b"\n", b""}:
                    break

            if connection_count == 1:
                writer.write(b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\n\r\n")
                await writer.drain()
                writer.close()
                await writer.wait_closed()
                return

            payload = {"id": f"tx-{connection_count}", "paging_token": str(connection_count)}
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

        server = await asyncio.start_server(handler, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = HorizonStreamingClient(
            base_url=f"http://127.0.0.1:{port}",
            reconnect_delay=0.02,
            max_reconnect_delay=0.05,
            rate_limit_backoff=0.02,
            max_rate_limit_backoff=0.05,
        )

        async def on_transaction(tx):
            received_ids.append(tx["id"])
            if len(received_ids) >= 2:
                received_event.set()

        try:
            await client.start(on_transaction)
            await asyncio.wait_for(received_event.wait(), timeout=2.0)
            await client.stop()
        finally:
            await client.stop()
            server.close()
            await server.wait_closed()

        assert received_ids == ["tx-2", "tx-3"]

    asyncio.run(run_test())
