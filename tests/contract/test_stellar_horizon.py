import pytest
import requests

# We assume Horizon runs locally for tests or we use the testnet.
HORIZON_URL = "https://horizon-testnet.stellar.org"

def test_stellar_horizon_root_endpoint():
    """Verify the root endpoint of the Stellar Horizon API is reachable and follows expected schema."""
    response = requests.get(HORIZON_URL)
    assert response.status_code == 200, "Stellar Horizon API root endpoint should return 200"
    
    data = response.json()
    # Contract checks: the root should contain links and horizon version
    assert "horizon_version" in data, "Contract broken: 'horizon_version' missing"
    assert "core_version" in data, "Contract broken: 'core_version' missing"
    assert "_links" in data, "Contract broken: '_links' missing"
    
def test_stellar_horizon_fee_stats():
    """Verify the fee stats endpoint to ensure our fee estimation contract holds."""
    response = requests.get(f"{HORIZON_URL}/fee_stats")
    assert response.status_code == 200, "Stellar Horizon API fee_stats endpoint should return 200"
    
    data = response.json()
    assert "last_ledger" in data, "Contract broken: 'last_ledger' missing in fee_stats"
    assert "last_ledger_base_fee" in data, "Contract broken: 'last_ledger_base_fee' missing"
    assert "fee_charged" in data, "Contract broken: 'fee_charged' missing"
    assert "max_fee" in data, "Contract broken: 'max_fee' missing"
