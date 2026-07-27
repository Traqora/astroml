import time
from typing import List

class AdaptiveBuffer:
    def __init__(self, initial_batch_size: int = 1, max_batch_size: int = 10, batch_window_ms: float = 50.0):
        self.batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.batch_window = batch_window_ms / 1000.0
        self.queue: List[str] = []
        self.last_flush = time.time()

    def add(self, token: str) -> List[str]:
        self.queue.append(token)
        now = time.time()
        
        # Flush if batch size exceeded or window closed
        if len(self.queue) >= self.batch_size or (now - self.last_flush) >= self.batch_window:
            return self.flush()
        return []

    def flush(self) -> List[str]:
        if not self.queue:
            return []
        items = self.queue
        self.queue = []
        self.last_flush = time.time()
        return items

    def adjust_backpressure(self, client_rtt_ms: float):
        # Adjust batch size up if client is slow (high RTT) to maximize throughput, or down if fast (low latency)
        if client_rtt_ms > 200:
            self.batch_size = min(self.max_batch_size, self.batch_size + 1)
        elif client_rtt_ms < 50:
            self.batch_size = max(1, self.batch_size - 1)
