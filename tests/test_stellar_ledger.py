import json
from unittest.mock import AsyncMock, patch

import pathlib
import pytest

from astroml.ingestion.config import StreamConfig
from astroml.ingestion.stellar_ledger import StellarLedgerDownloader, ledger_partition_dir


@pytest.fixture
def mock_config():
    return StreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        reconnect_base_seconds=0.1,
        reconnect_max_seconds=0.2,
        max_retries=2,
    )


@pytest.mark.asyncio
async def test_download_range_success(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"

    mock_response_data = {
        "_embedded": {
            "records": [
                {"sequence": 100, "paging_token": "100_0", "header_xdr": "AAAAA..."},
                {"sequence": 101, "paging_token": "101_0", "header_xdr": "BBBBB..."},
            ]
        }
    }

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response_data
        mock_get.return_value.__aenter__.return_value = mock_resp

        async with downloader:
            await downloader.download_range(100, 101, output_dir=str(output_dir))

    bucket = output_dir / "ledger_bucket_00000000"
    assert (bucket / "ledger_100.json").exists()
    assert (bucket / "ledger_101.json").exists()

    ledger_100 = json.loads((bucket / "ledger_100.json").read_text())
    assert ledger_100["sequence"] == 100


@pytest.mark.asyncio
async def test_download_range_flat_layout(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"

    mock_response_data = {
        "_embedded": {
            "records": [
                {"sequence": 100, "paging_token": "100_0", "header_xdr": "AAAAA..."},
            ]
        }
    }

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json.return_value = mock_response_data
        mock_get.return_value.__aenter__.return_value = mock_resp

        async with downloader:
            await downloader.download_range(
                100, 100, output_dir=str(output_dir), partition_size=None
            )

    assert (output_dir / "ledger_100.json").exists()


@pytest.mark.asyncio
async def test_download_range_pagination(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"

    mock_response_1 = {"_embedded": {"records": [{"sequence": 100, "paging_token": "100_0"}]}}
    mock_response_2 = {"_embedded": {"records": [{"sequence": 101, "paging_token": "101_0"}]}}

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp_1 = AsyncMock()
        mock_resp_1.status = 200
        mock_resp_1.json.return_value = mock_response_1

        mock_resp_2 = AsyncMock()
        mock_resp_2.status = 200
        mock_resp_2.json.return_value = mock_response_2

        mock_get.return_value.__aenter__.side_effect = [mock_resp_1, mock_resp_2]

        async with downloader:
            await downloader.download_range(100, 101, output_dir=str(output_dir))

    bucket = output_dir / "ledger_bucket_00000000"
    assert (bucket / "ledger_100.json").exists()
    assert (bucket / "ledger_101.json").exists()


@pytest.mark.asyncio
async def test_download_range_retry(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"

    mock_response_data = {"_embedded": {"records": [{"sequence": 100, "paging_token": "100_0"}]}}

    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_resp_fail = AsyncMock()
        mock_resp_fail.status = 429

        mock_resp_success = AsyncMock()
        mock_resp_success.status = 200
        mock_resp_success.json.return_value = mock_response_data

        mock_get.return_value.__aenter__.side_effect = [mock_resp_fail, mock_resp_success]

        async with downloader:
            await downloader.download_range(100, 100, output_dir=str(output_dir))

    assert (output_dir / "ledger_bucket_00000000" / "ledger_100.json").exists()


@pytest.mark.asyncio
async def test_download_range_invalid_format(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    with pytest.raises(ValueError, match="Unsupported format"):
        async with downloader:
            await downloader.download_range(100, 101, format="invalid")


def test_ledger_partition_dir():
    assert ledger_partition_dir(0) == pathlib.Path("ledger_bucket_00000000")
    assert ledger_partition_dir(9999) == pathlib.Path("ledger_bucket_00000000")
    assert ledger_partition_dir(10000) == pathlib.Path("ledger_bucket_00010000")
    assert ledger_partition_dir(12345, partition_size=1000) == pathlib.Path("ledger_bucket_00012000")
