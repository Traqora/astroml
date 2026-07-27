from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def hybrid_search(self, collection_name: str, query_vector: List[float], text_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int):
        pass

class MockVectorStore(VectorStore):
    def __init__(self):
        self.collections = {}

    def create_collection(self, collection_name: str, dimension: int):
        if collection_name not in self.collections:
            self.collections[collection_name] = {
                "dimension": dimension,
                "data": []
            }
            # Indexes would be optimized here in a real implementation (e.g., HNSW)

    def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
        self.collections[collection_name]["data"].extend(documents)

    def search(self, collection_name: str, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
        
        # Mocking similarity search 
        return self.collections[collection_name]["data"][:top_k]
        
    def hybrid_search(self, collection_name: str, query_vector: List[float], text_query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
            
        # Mocking hybrid search (combining dense vector search and sparse keyword search)
        return self.collections[collection_name]["data"][:top_k]
