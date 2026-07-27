import numpy as np
from typing import List

class EmbeddingGenerator:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        self.model_name = model_name
        self.dimension = 3072 if "large" in model_name else 1536

    def generate_embedding(self, text: str) -> List[float]:
        # Hash-deterministic mock embedding generation for testable/consistent vector matching
        # Let's generate a vector of self.dimension based on the character contents of text
        np.random.seed(abs(hash(text)) % (2**32))
        vec = np.random.randn(self.dimension)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.generate_embedding(t) for t in texts]

_embedder = EmbeddingGenerator()

def get_embedder() -> EmbeddingGenerator:
    return _embedder
