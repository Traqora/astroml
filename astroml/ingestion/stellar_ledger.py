from astroml.utils.exceptions import AstroMLError
"""Module for downloading historical Stellar ledger data."""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
from typing import Any

import aiohttp

from astroml.ingestion.config import StreamConfig

logger = logging.getLogger("astroml.ingestion.stellar_ledger")


class StellarLedgerDownloader:
    """Downloader for historical Stellar ledger data.

    Supports downloading a range of ledgers, saving them as JSON or XDR,
    and handles pagination and retries.
    """

    def __init__(self, config: StreamConfig | None = None) -> None:
        self._config = config or StreamConfig()
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> StellarLedgerDownloader:
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, _exc_val, _exc_tb) -> None:
        if self._session:
            await self._session.close()

    async def _fetch_with_retry(self, url: str) -> dict[str, Any]:
        """Fetch a URL with exponential backoff retry logic."""
        retry_count = 0
        while True:
            try:
                async with self._session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        logger.warning("Rate limit hit, retrying...")
                    elif response.status >= 500:
                        logger.warning("Server error %d, retrying...", response.status)
                    else:
                        response.raise_for_status()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning("Network error: %s, retrying...", e)

            retry_count += 1
            if self._config.max_retries > 0 and retry_count > self._config.max_retries:
                raise Exception(f"Max retries exceeded for {url}")

            delay = min(
                self._config.reconnect_base_seconds * (2 ** (retry_count - 1)),
                self._config.reconnect_max_seconds,
            )
            await asyncio.sleep(delay)

    async def download_range(
        self,
        start_ledger: int,
        end_ledger: int,
        output_dir: str = "data/ledgers",
        format: str = "json",
        partition_size: int | None = None,
    ) -> None:
        """Download a range of ledgers and save them to disk.

        Args:
            start_ledger: Starting ledger sequence (inclusive).
            end_ledger: Ending ledger sequence (inclusive).
            output_dir: Directory to save the ledger data.
            format: Output format ("json" or "xdr"). Currently only "json" is fully supported via Horizon.
            partition_size: Optional number of ledgers per partition directory.
        """
        if format not in ("json", "xdr"):
            raise ValueError(f"Unsupported format: {format}")

        path = pathlib.Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        current_ledger = start_ledger
        cursor = str(start_ledger - 1)

        while current_ledger <= end_ledger:
            url = f"{self._config.horizon_url}/ledgers?cursor={cursor}&limit=200&order=asc"
            logger.info("Fetching ledgers from cursor %s", cursor)

            data = await self._fetch_with_retry(url)
            records = data.get("_embedded", {}).get("records", [])

            if not records:
                logger.info("No more ledgers found.")
                break

            for record in records:
                seq = record["sequence"]
                if seq > end_ledger:
                    break

                if partition_size and partition_size > 0:
                    partition_dir = path / str((seq // partition_size) * partition_size)
                    partition_dir.mkdir(parents=True, exist_ok=True)
                    out_path = partition_dir
                else:
                    out_path = path

                if format == "json":
                    file_path = out_path / f"ledger_{seq}.json"
                    file_path.write_text(json.dumps(record, indent=2))
                elif format == "xdr":
                    # Horizon provides header_xdr and other XDR fields in the JSON response
                    # For a pure XDR download, we'd typically use ledger archives,
                    # but here we save what Horizon provides.
                    file_path = out_path / f"ledger_{seq}.xdr"
                    file_path.write_text(record.get("header_xdr", ""))

                cursor = record["paging_token"]
                current_ledger = seq + 1

            if current_ledger > end_ledger:
                break

        logger.info(
            "Download complete. Ledgers %d to %d (or last available) saved to %s",
            start_ledger,
            min(current_ledger - 1, end_ledger),
            output_dir,
        )


async def main() -> None:
    """Command line entry point for the downloader.

    Parses ``--start``/``--end``/``--output``/``--format``, applies the central
    logging configuration, and runs the download to completion. Intended to be
    invoked via :func:`asyncio.run` from ``__main__``.
    """
    import argparse  # noqa: E402
    import sys  # noqa: E402

    parser = argparse.ArgumentParser(description="Stellar Ledger Downloader")
    parser.add_argument("--start", type=int, required=True, help="Start ledger sequence")
    parser.add_argument("--end", type=int, required=True, help="End ledger sequence")
    parser.add_argument("--output", default="data/ledgers", help="Output directory")
    parser.add_argument("--format", choices=["json", "xdr"], default="json", help="Output format")
    parser.add_argument("--partition-size", type=int, default=None, help="Number of ledgers per partition directory")

    args = parser.parse_args()

    # Issue #195 — central logging config (level + text/json format)
    # via ASTROML_LOG_LEVEL / ASTROML_LOG_FORMAT env vars.
    from astroml.utils.logging import configure_logging

    configure_logging()

    async with StellarLedgerDownloader() as downloader:
        try:
            await downloader.download_range(
                args.start, args.end, args.output, args.format, args.partition_size
            )
        except Exception as e:
            logger.error("Download failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
