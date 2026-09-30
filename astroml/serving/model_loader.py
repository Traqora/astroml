from typing import Any, Optional


class ModelLoader:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        
    def load_model(self, name: str, path: str) -> None:
        self.models[name] = {"path": path, "loaded": True}
        
    def get_model(self, name: str) -> Optional[Any]:
        return self.models.get(name)
