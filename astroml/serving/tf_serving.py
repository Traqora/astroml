from typing import List


class TFServingClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        
    def predict(self, model_name: str, inputs: List[float]) -> List[float]:
        # Dummy prediction
        return [sum(inputs)]
