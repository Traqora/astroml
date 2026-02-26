"""Horizon Streaming Client for real-time Stellar data ingestion.

Connects to a Stellar Horizon server via Server-Sent Events (SSE) and
persists ledger, transaction, and operation data to PostgreSQL.

Usage::

    python -m astroml.ingestion.stream [--cursor CURSOR] [--endpoint /transactions]
"""
from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import signal
import sys
from datetime import timedelta
from typing import Optional

import aiohttp
from aiohttp_sse_client import client as sse_client

from astroml.db.schema import Ledger, Transaction
from astroml.db.session import get_session
from astroml.ingestion.config import StreamConfig
from astroml.ingestion.parsers import parse_ledger, parse_operation, parse_transaction

logger = logging.getLogger("astroml.ingestion.stream")

CURSOR_FILE = pathlib.Path("config/.stream_cursor")


class HorizonStreamClient:
    """Async streaming client for Stellar Horizon SSE endpoints.

    Supports async context manager protocol, automatic reconnection with
    exponential backoff, cursor tracking for resume-on-restart, graceful
    shutdown via SIGINT/SIGTERM, and structured logging.

    Args:
        config: Streaming configuration. Uses defaults if not provided.
    """

    def __init__(self, config: Optional[StreamConfig] = None) -> None:
        self._config = config or StreamConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._last_cursor: Optional[str] = self._config.cursor or self._load_cursor()
        self._retry_count = 0

    # -- Async context manager ------------------------------------------------

    async def __aenter__(self) -> HorizonStreamClient:
        self._session = aiohttp.ClientSession()
        self._running = True
        self._install_signal_handlers()
        logger.info(
            "HorizonStreamClient initialized | horizon=%s endpoint=%s cursor=%s",
            self._config.horizon_url,
            self._config.stream_endpoint,
            self._last_cursor or "now",
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        if self._session:
            await self._session.close()
        logger.info(
            "HorizonStreamClient shut down | last_cursor=%s", self._last_cursor
        )

    # -- Signal handling ------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Register SIGINT and SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        logger.info("Received signal %s, initiating graceful shutdown...", sig.name)
        self._running = False

    # -- Cursor persistence ---------------------------------------------------

    @staticmethod
    def _load_cursor() -> Optional[str]:
        """Load cursor from file if it exists."""
        if CURSOR_FILE.exists():
            text = CURSOR_FILE.read_text().strip()
            return text if text else None
        return None

    @staticmethod
    def _save_cursor(cursor: str) -> None:
        """Persist cursor to file for resume-on-restart."""
        CURSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
        CURSOR_FILE.write_text(cursor)

    # -- Stream URL -----------------------------------------------------------

    def _build_stream_url(self) -> str:
        """Build the full streaming URL with cursor and order parameters."""
        base = f"{self._config.horizon_url}{self._config.stream_endpoint}"
        cursor = self._last_cursor or "now"
        return f"{base}?order=asc&cursor={cursor}"

    # -- Core streaming loop --------------------------------------------------

    async def run(self) -> None:
        """Main streaming loop with automatic reconnection.

        Connects to the Horizon SSE endpoint and processes events.
        On disconnection, reconnects with exponential backoff.
        Exits when ``self._running`` is set to False.
        """
        while self._running:
            try:
                await self._stream()
            except (
                aiohttp.ClientError,
                ConnectionError,
                asyncio.TimeoutError,
            ) as exc:
                if not self._running:
                    break
                await self._handle_reconnect(exc)
            except Exception:
                logger.exception("Unexpected error in stream loop")
                if not self._running:
                    break
                await self._handle_reconnect(None)

        logger.info("Stream loop exited | last_cursor=%s", self._last_cursor)

    async def _stream(self) -> None:
        """Connect to SSE endpoint and process events until disconnection."""
        url = self._build_stream_url()
        logger.info("Connecting to %s", url)

        async with sse_client.EventSource(
            url,
            session=self._session,
            reconnection_time=timedelta(
                seconds=self._config.reconnect_base_seconds
            ),
        ) as event_source:
            self._retry_count = 0
            logger.info("Connected to Horizon stream")

            async for event in event_source:
                if not self._running:
                    break
                if event.data:
                    await self._process_event(event)

    async def _handle_reconnect(self, exc: Optional[Exception]) -> None:
        """Wait with exponential backoff before reconnecting."""
        self._retry_count += 1
        max_retries = self._config.max_retries
        if max_retries > 0 and self._retry_count > max_retries:
            logger.error("Max retries (%d) exceeded, stopping", max_retries)
            self._running = False
            return

        delay = min(
            self._config.reconnect_base_seconds * (2 ** (self._retry_count - 1)),
            self._config.reconnect_max_seconds,
        )
        logger.warning(
            "Connection lost (attempt %d): %s. Reconnecting in %.1fs...",
            self._retry_count,
            exc,
            delay,
        )
        await asyncio.sleep(delay)

    # -- Event processing -----------------------------------------------------

    async def _process_event(self, event) -> None:
        """Parse an SSE event and persist it to the database."""
        try:
            data = json.loads(event.data)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed event: %s", event.data[:200])
            return

        paging_token = data.get("paging_token")
        endpoint = self._config.stream_endpoint

        try:
            if endpoint == "/ledgers":
                await self._persist_ledger(data)
            elif endpoint == "/transactions":
                await self._persist_transaction(data)
            elif endpoint == "/operations":
                await self._persist_operation(data)
            else:
                logger.warning("Unsupported endpoint: %s", endpoint)
                return
        except Exception:
            logger.exception(
                "Failed to persist event (paging_token=%s)", paging_token
            )
            return

        # Update cursor only after successful persistence
        if paging_token:
            self._last_cursor = paging_token
            self._save_cursor(paging_token)
            logger.debug("Cursor updated to %s", paging_token)

    # -- Persistence helpers --------------------------------------------------

    async def _persist_transaction(self, data: dict) -> None:
        """Persist a transaction and a minimal parent ledger stub."""
        tx = parse_transaction(data)
        logger.info(
            "Processing transaction %s (ledger %d)",
            tx.hash[:12],
            tx.ledger_sequence,
        )
        await asyncio.to_thread(self._db_write_transaction, tx)

    @staticmethod
    def _db_write_transaction(tx: Transaction) -> None:
        """Synchronous DB write for a transaction (runs in thread executor)."""
        session = get_session()
        try:
            existing_ledger = session.get(Ledger, tx.ledger_sequence)
            if existing_ledger is None:
                ledger = Ledger(
                    sequence=tx.ledger_sequence,
                    hash="",
                    closed_at=tx.created_at,
                )
                session.merge(ledger)
            session.merge(tx)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def _persist_ledger(self, data: dict) -> None:
        """Persist a ledger."""
        ledger = parse_ledger(data)
        logger.info("Processing ledger %d", ledger.sequence)
        await asyncio.to_thread(self._db_write_model, ledger)

    async def _persist_operation(self, data: dict) -> None:
        """Persist an operation."""
        op = parse_operation(data)
        logger.info("Processing operation %d (type=%s)", op.id, op.type)
        await asyncio.to_thread(self._db_write_model, op)

    @staticmethod
    def _db_write_model(model) -> None:
        """Synchronous DB write for any model (runs in thread executor)."""
        session = get_session()
        try:
            session.merge(model)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # -- Cursor access --------------------------------------------------------

    @property
    def last_cursor(self) -> Optional[str]:
        """The paging_token of the last successfully processed event."""
        return self._last_cursor


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Configure structured logging for the streaming process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )


def _parse_cli_args() -> StreamConfig:
    """Parse command-line arguments into a StreamConfig."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream Stellar blockchain data from Horizon into PostgreSQL.",
    )
    parser.add_argument(
        "--horizon-url",
        default=None,
        help="Horizon server URL (default: testnet, or ASTROML_HORIZON_URL env var)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Streaming endpoint path (default: /transactions)",
    )
    parser.add_argument(
        "--cursor",
        default=None,
        help="Starting cursor/paging_token. Use 'now' for live-only.",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.horizon_url:
        kwargs["horizon_url"] = args.horizon_url
    if args.endpoint:
        kwargs["stream_endpoint"] = args.endpoint
    if args.cursor:
        kwargs["cursor"] = args.cursor

    return StreamConfig(**kwargs)


async def _main() -> None:
    """Async entry point."""
    config = _parse_cli_args()
    async with HorizonStreamClient(config) as client:
        await client.run()
    logger.info("Final cursor: %s", client.last_cursor)


if __name__ == "__main__":
    _configure_logging()
    asyncio.run(_main())
