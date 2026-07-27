import time
from typing import Dict, List, Any

class EvictionPolicy:
    def __init__(self, max_size: int = 100, strategy: str = "LRU"):
        self.max_size = max_size
        self.strategy = strategy.upper() # "LRU" or "LFU"
        # Key -> last_accessed or access_count
        self.meta: Dict[str, float] = {}

    def record_access(self, key: str):
        if self.strategy == "LRU":
            self.meta[key] = time.time()
        elif self.strategy == "LFU":
            self.meta[key] = self.meta.get(key, 0.0) + 1.0

    def get_eviction_target(self, current_keys: List[str]) -> Optional[str]:
        if len(current_keys) < self.max_size:
            return None
            
        # Find key with minimum value in metadata
        target_key = None
        min_val = float("inf")
        
        for k in current_keys:
            val = self.meta.get(k, 0.0)
            if val < min_val:
                min_val = val
                target_key = k
                
        return target_key

    def evict_key(self, key: str):
        if key in self.meta:
            del self.meta[key]
