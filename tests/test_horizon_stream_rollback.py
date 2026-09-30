"""Cursor rollback regression tests for HorizonStreamingClient (issue #983)."""

import asyncio
import json

import pytest

from astroml.ingestion.horizon_stream import HorizonStreamingClient


def _payload(token: str) -> str:
    return json.dumps({"paging_token": token, "hash": f"h{token}"})


def test_cursor_advances_on_success():
    client = HorizonStreamingClient(cursor="10")
    asyncio.run(client._handle_payload(_payload("11"), lambda tx: None))
    assert client.cursor == "11"


def test_cursor_rolled_back_when_sync_handler_fails():
    client = HorizonStreamingClient(cursor="10")

    def fail(tx):
        raise RuntimeError("handler boom")

    with pytest.raises(RuntimeError, match="handler boom"):
        asyncio.run(client._handle_payload(_payload("11"), fail))
    assert client.cursor == "10"
    assert "cursor=10" in client._request_path()


def test_cursor_rolled_back_when_async_handler_fails():
    client = HorizonStreamingClient(cursor="10")

    async def fail(tx):
        raise ValueError("async boom")

    with pytest.raises(ValueError):
        asyncio.run(client._handle_payload(_payload("11"), fail))
    assert client.cursor == "10"


def test_rollback_only_undoes_failed_transaction():
    client = HorizonStreamingClient(cursor="10")
    seen = []

    def handler(tx):
        if tx["paging_token"] == "12":
            raise RuntimeError("fail on 12")
        seen.append(tx["paging_token"])

    asyncio.run(client._handle_payload(_payload("11"), handler))
    with pytest.raises(RuntimeError):
        asyncio.run(client._handle_payload(_payload("12"), handler))
    assert seen == ["11"]
    assert client.cursor == "11"


def test_invalid_payload_leaves_cursor_untouched():
    client = HorizonStreamingClient(cursor="10")
    asyncio.run(client._handle_payload("not json", lambda tx: None))
    assert client.cursor == "10"
