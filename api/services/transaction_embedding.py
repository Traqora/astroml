import time
from typing import List, Dict, Any
import numpy as np

class TransactionEmbeddingGenerator:
    def __init__(self, embedding_dim: int = 256, decay_rate: float = 0.05):
        self.embedding_dim = embedding_dim
        self.decay_rate = decay_rate
    
    def serialize_transaction(self, tx: Dict[str, Any]) -> str:
        """
        Serialize a transaction into a consistent string representation for embedding.
        """
        parts = [
            f"type:{tx.get('type', 'unknown')}",
            f"amt:{tx.get('amount', 0)}",
            f"asset:{tx.get('asset', 'native')}",
            f"from:{tx.get('source', '')}",
            f"to:{tx.get('destination', '')}",
            f"ops:{len(tx.get('operations', []))}"
        ]
        return "|".join(parts)
        
    def apply_time_decay(self, embedding: np.ndarray, tx_timestamp: float, current_time: float) -> np.ndarray:
        """
        Implement time-decay on the embedding based on transaction age.
        """
        age_days = (current_time - tx_timestamp) / (24 * 3600)
        if age_days < 0:
            age_days = 0
            
        decay_factor = np.exp(-self.decay_rate * age_days)
        return embedding * decay_factor
        
    def generate_batch_embeddings(self, transactions: List[Dict[str, Any]], current_time: float = None) -> np.ndarray:
        """
        Generate embeddings for transactions to find similar operations with batch processing.
        """
        if current_time is None:
            current_time = time.time()
            
        embeddings = []
        for tx in transactions:
            # 1. Serialize
            serialized = self.serialize_transaction(tx)
            
            # 2. Generate raw embedding (mock representation)
            # In production, this would call an actual embedding model on `serialized`
            np.random.seed(hash(serialized) % (2**32))
            raw_emb = np.random.randn(self.embedding_dim)
            raw_emb = raw_emb / np.linalg.norm(raw_emb)
            
            # 3. Apply time decay
            tx_time = tx.get("timestamp", current_time)
            decayed_emb = self.apply_time_decay(raw_emb, tx_time, current_time)
            
            embeddings.append(decayed_emb)
            
        return np.array(embeddings)
