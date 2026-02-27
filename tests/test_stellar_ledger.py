import json
import pytest
import aiohttp
from unittest.mock import AsyncMock, patch
from astroml.ingestion.stellar_ledger import StellarLedgerDownloader
from astroml.ingestion.config import StreamConfig

@pytest.fixture
def mock_config():
    return StreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        reconnect_base_seconds=0.1,
        reconnect_max_seconds=0.2,
        max_retries=2
    )

@pytest.mark.asyncio
async def test_download_range_success(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"
    
    mock_response_data = {
        "_embedded": {
            "records": [
                {"sequence": 100, "paging_token": "100_0", "header_xdr": "AAAAA..."},
                {"sequence": 101, "paging_token": "101_0", "header_xdr": "BBBBB..."}
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
            
    assert (output_dir / "ledger_100.json").exists()
    assert (output_dir / "ledger_101.json").exists()
    
    ledger_100 = json.loads((output_dir / "ledger_100.json").read_text())
    assert ledger_100["sequence"] == 100

@pytest.mark.asyncio
async def test_download_range_pagination(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"
    
    # Page 1: Ledger 100
    mock_response_1 = {
        "_embedded": {
            "records": [{"sequence": 100, "paging_token": "100_0"}]
        }
    }
    # Page 2: Ledger 101
    mock_response_2 = {
        "_embedded": {
            "records": [{"sequence": 101, "paging_token": "101_0"}]
        }
    }
    
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
            
    assert (output_dir / "ledger_100.json").exists()
    assert (output_dir / "ledger_101.json").exists()

@pytest.mark.asyncio
async def test_download_range_retry(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    output_dir = tmp_path / "ledgers"
    
    mock_response_data = {
        "_embedded": {
            "records": [{"sequence": 100, "paging_token": "100_0"}]
        }
    }
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        # Fail once with 429, then succeed
        mock_resp_fail = AsyncMock()
        mock_resp_fail.status = 429
        
        mock_resp_success = AsyncMock()
        mock_resp_success.status = 200
        mock_resp_success.json.return_value = mock_response_data
        
        mock_get.return_value.__aenter__.side_effect = [mock_resp_fail, mock_resp_success]
        
        async with downloader:
            await downloader.download_range(100, 100, output_dir=str(output_dir))
            
    assert (output_dir / "ledger_100.json").exists()

@pytest.mark.asyncio
async def test_download_range_invalid_format(mock_config, tmp_path):
    downloader = StellarLedgerDownloader(config=mock_config)
    with pytest.raises(ValueError, match="Unsupported format"):
        async with downloader:
            await downloader.download_range(100, 101, format="invalid")
