"""Document retrieval logic for RAG pipeline."""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import time


@dataclass
class RetrievedDocument:
    """Retrieved document with metadata."""

    text: str
    source: str
    relevance_score: float
    rerank_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata or {},
        }


class Retriever:
    """Document retriever for RAG system."""

    def __init__(
        self,
        embeddings_service: Any,
        reranker: Optional[Any] = None,
        top_k: int = 10,
        rerank_to_k: int = 5,
    ):
        """Initialize retriever.

        Args:
            embeddings_service: Service for computing embeddings
            reranker: Optional reranker for reranking results
            top_k: Number of documents to initially retrieve
            rerank_to_k: Number of documents after reranking
        """
        self.embeddings = embeddings_service
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_to_k = rerank_to_k
        self.document_store: List[RetrievedDocument] = []

    def add_documents(self, documents: List[str], sources: List[str], metadata: Optional[List[Dict]] = None) -> None:
        """Add documents to retriever."""
        embeddings, text_ids = self.embeddings.embed_texts_batch(documents, metadata)

        for i, (doc, source) in enumerate(zip(documents, sources)):
            meta = metadata[i] if metadata and i < len(metadata) else {}
            doc_obj = RetrievedDocument(
                text=doc,
                source=source,
                relevance_score=0.0,
                metadata=meta,
            )
            self.document_store.append(doc_obj)

    def retrieve(
        self, query: str, metadata_filter: Optional[Dict] = None
    ) -> List[RetrievedDocument]:
        """Retrieve documents for query.

        Args:
            query: Query text
            metadata_filter: Optional metadata filtering

        Returns:
            List of retrieved documents (reranked if reranker available)
        """
        results = self.embeddings.similarity_search(
            query, top_k=self.top_k, metadata_filter=metadata_filter
        )

        retrieved = []
        for text_id, similarity_score, metadata in results:
            for doc in self.document_store:
                if doc.metadata and doc.metadata.get("text_id") == text_id:
                    doc.relevance_score = similarity_score
                    retrieved.append(doc)
                    break

        if self.reranker and len(retrieved) > 0:
            retrieved = self._rerank_results(query, retrieved)

        return retrieved[: self.rerank_to_k]

    def _rerank_results(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Rerank documents using reranker."""
        if not self.reranker:
            return documents

        rerank_scores = self.reranker.rerank(
            query, [doc.text for doc in documents]
        )

        for doc, score in zip(documents, rerank_scores):
            doc.rerank_score = score

        documents.sort(
            key=lambda d: d.rerank_score if d.rerank_score else d.relevance_score,
            reverse=True,
        )
        return documents

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_documents": len(self.document_store),
            "top_k": self.top_k,
            "rerank_to_k": self.rerank_to_k,
            "has_reranker": self.reranker is not None,
        }


class SimpleReranker:
    """Simple reranker using embedding-based scoring."""

    def __init__(self, embeddings_service: Any):
        """Initialize reranker."""
        self.embeddings = embeddings_service

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Rerank documents."""
        query_emb, _ = self.embeddings.embed_texts_batch([query])
        query_emb = query_emb[0]

        doc_embs, _ = self.embeddings.embed_texts_batch(documents)

        scores = []
        for doc_emb in doc_embs:
            score = self.embeddings._cosine_similarity(query_emb, doc_emb)
            scores.append(score)

        return scores
