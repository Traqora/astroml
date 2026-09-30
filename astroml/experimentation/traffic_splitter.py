import random
from typing import Optional

class TrafficSplitter:
    def __init__(self, default_split -> Any: float = 0.5):
        self.default_split = default_split
        
    def split_random(self, user_id: Optional[str] = None) -> str:
        if random.random() < self.default_split:
            return "treatment"
        return "control"
        
    def split_cookie(self, cookie_id: str) -> str:
        if hash(cookie_id) % 100 < (self.default_split * 100):
            return "treatment"
        return "control"
