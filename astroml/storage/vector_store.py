from abc import ABC, abstractmethod
from typing import Any


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, collection_name -> Any: str, documents: list[dict[str, Any]]):
        pass

    @abstractmethod
    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def hybrid_search(
        self, collection_name: str, query_vector: list[float], text_query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def create_collection(self, collection_name -> Any: str, dimension: int):
        pass


class MockVectorStore(VectorStore):
    def __init__(self) -> Any:
        self.collections = {}

    def create_collection(self, collection_name -> Any: str, dimension: int):
        if collection_name not in self.collections:
            self.collections[collection_name] = {"dimension": dimension, "data": []}
            # Indexes would be optimized here in a real implementation (e.g., HNSW)

    def add_documents(self, collection_name -> Any: str, documents: list[dict[str, Any]]):
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")
        self.collections[collection_name]["data"].extend(documents)

    def search(
        self, collection_name: str, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")

        # Mocking similarity search
        return self.collections[collection_name]["data"][:top_k]

    def hybrid_search(
        self, collection_name: str, query_vector: list[float], text_query: str, top_k: int = 10
    ) -> list[dict[str, Any]]:
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} does not exist.")

        # Mocking hybrid search (combining dense vector search and sparse keyword search)
        return self.collections[collection_name]["data"][:top_k]
