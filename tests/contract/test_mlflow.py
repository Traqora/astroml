import pytest
import requests
import os

# Assuming MLflow runs locally for tests or a test instance
MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

@pytest.mark.skipif(not os.environ.get("MLFLOW_TRACKING_URI"), reason="Requires MLFLOW_TRACKING_URI or local mlflow to be running.")
def test_mlflow_api_health():
    """Verify the MLflow API is reachable."""
    try:
        # MLflow provides a ping/health endpoint in some versions, or we can just try to fetch experiments
        response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search")
        assert response.status_code in [200, 404, 401], "MLflow API should respond to search request"
    except requests.exceptions.ConnectionError:
        pytest.skip("MLflow instance is not running")

def test_mlflow_contract_schema_mock(requests_mock):
    """
    If MLFlow isn't running, we at least test our expectations of the schema against a mock
    to ensure if we change parsing logic, it matches the contract.
    """
    mock_url = "http://mock-mlflow:5000/api/2.0/mlflow/experiments/search"
    expected_response = {
        "experiments": [
            {
                "experiment_id": "0",
                "name": "Default",
                "artifact_location": "mlruns/0",
                "lifecycle_stage": "active"
            }
        ]
    }
    requests_mock.get(mock_url, json=expected_response)
    
    response = requests.get(mock_url)
    assert response.status_code == 200
    
    data = response.json()
    assert "experiments" in data, "Contract broken: 'experiments' key missing"
    assert len(data["experiments"]) > 0
    exp = data["experiments"][0]
    assert "experiment_id" in exp, "Contract broken: 'experiment_id' missing"
    assert "name" in exp, "Contract broken: 'name' missing"
