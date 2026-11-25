import pickle

import pytest

from astroml.serving.grpc_server import GrpcServer
from astroml.serving.model_loader import ModelLoader
from astroml.serving.tf_serving import TFServingClient


def test_grpc_server():
    server = GrpcServer()
    server.start(50051)
    assert server.is_running is True
    server.stop()
    assert server.is_running is False

def test_model_loader_loads_artifact_once(tmp_path):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(pickle.dumps({"weights": [1, 2, 3]}))
    calls = []

    def load(path):
        calls.append(path)
        with path.open("rb") as handle:
            return pickle.load(handle)

    loader = ModelLoader(artifact_loader=load)

    first = loader.load_model("model1", artifact)
    second = loader.load_model("model1", artifact)

    assert first == {"weights": [1, 2, 3]}
    assert second is first
    assert loader.get_model("model1") is first
    assert calls == [artifact.resolve()]


def test_model_loader_rejects_missing_or_conflicting_artifact_paths(tmp_path):
    loader = ModelLoader(artifact_loader=lambda path: object())
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"artifact")

    with pytest.raises(FileNotFoundError, match="existing file"):
        loader.load_model("missing", tmp_path / "missing.pkl")

    loader.load_model("model1", artifact)
    other_artifact = tmp_path / "other.pkl"
    other_artifact.write_bytes(b"other")
    with pytest.raises(ValueError, match="different path"):
        loader.load_model("model1", other_artifact)

def test_tf_serving():
    client = TFServingClient("localhost", 8500)
    preds = client.predict("model1", [1.0, 2.0])
    assert preds == [3.0]
