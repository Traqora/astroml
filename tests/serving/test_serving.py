from astroml.serving.grpc_server import GrpcServer
from astroml.serving.model_loader import ModelLoader
from astroml.serving.tf_serving import TFServingClient


def test_grpc_server():
    server = GrpcServer()
    server.start(50051)
    assert server.is_running is True
    server.stop()
    assert server.is_running is False

def test_model_loader():
    loader = ModelLoader()
    loader.load_model("model1", "/tmp/model1")
    assert loader.get_model("model1")["loaded"] is True

def test_tf_serving():
    client = TFServingClient("localhost", 8500)
    preds = client.predict("model1", [1.0, 2.0])
    assert preds == [3.0]
