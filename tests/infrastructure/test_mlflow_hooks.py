import pytest
import mlflow
import torch
import torch.nn as nn
from astroml.tracking.mlflow_tracker import MLflowTracker

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        
    def forward(self, x):
        return self.linear(x)

def test_mlflow_tracker_integration(tmp_path):
    """Test that training registers correctly to MLflow (params, metrics, artifact)
    and that a re-run creates a new run without corrupting the previous one (Issue #710).
    """
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    experiment_name = "test_experiment"
    
    # First run
    tracker1 = MLflowTracker(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name="run_1"
    )
    
    tracker1.log_params({"learning_rate": 0.01, "epochs": 5})
    tracker1.log_metrics({"loss": 0.5}, step=1)
    
    model = DummyModel()
    tracker1.log_model_artifact(model, "model_artifact")
    tracker1.end()
    
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None
    
    runs = mlflow.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.learning_rate"] == "0.01"
    assert runs.iloc[0]["metrics.loss"] == 0.5
    
    # Second run (re-run)
    tracker2 = MLflowTracker(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name="run_2"
    )
    
    tracker2.log_params({"learning_rate": 0.02})
    tracker2.log_metrics({"loss": 0.25}, step=1)
    tracker2.end()
    
    runs = mlflow.search_runs([experiment.experiment_id], order_by=["start_time ASC"])
    assert len(runs) == 2
    assert runs.iloc[0]["params.learning_rate"] == "0.01"
    assert runs.iloc[1]["params.learning_rate"] == "0.02"

