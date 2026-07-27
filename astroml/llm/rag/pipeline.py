"""RAG (Retrieval Augmented Generation) pipeline orchestrator."""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from .retriever import Retriever, RetrievedDocument


class RAGPipeline:
    """End-to-end RAG system orchestrator."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: Any,
        system_prompt: Optional[str] = None,
        include_citations: bool = True,
        detect_hallucinations: bool = True,
    ):
        """Initialize RAG pipeline.

        Args:
            retriever: Document retriever
            llm_provider: LLM provider for generation
            system_prompt: Optional system prompt
            include_citations: Whether to include citations in responses
            detect_hallucinations: Whether to detect unsupported claims
        """
        self.retriever = retriever
        self.llm = llm_provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.include_citations = include_citations
        self.detect_hallucinations = detect_hallucinations
        self.query_history: List[Dict[str, Any]] = []

    def query(
        self, query: str, metadata_filter: Optional[Dict] = None, stream: bool = False
    ) -> Tuple[str, List[RetrievedDocument], Dict[str, Any]]:
        """Execute RAG query.

        Args:
            query: User query
            metadata_filter: Optional metadata filtering
            stream: Whether to stream response

        Returns:
            Tuple of (response, retrieved_docs, metadata)
        """
        import time
        start_time = time.time()

        retrieved = self.retriever.retrieve(query, metadata_filter)

        context = self._build_context(query, retrieved)

        if stream:
            response = self.llm.generate_stream(context, system_prompt=self.system_prompt)
        else:
            response = self.llm.generate(context, system_prompt=self.system_prompt)

        if self.include_citations:
            response = self._add_citations(response, retrieved)

        if self.detect_hallucinations:
            hallucination_score = self._detect_hallucinations(response, retrieved)
        else:
            hallucination_score = None

        elapsed = time.time() - start_time

        metadata = {
            "query": query,
            "num_retrieved": len(retrieved),
            "response_time": elapsed,
            "hallucination_score": hallucination_score,
            "timestamp": datetime.now().isoformat(),
        }

        self.query_history.append(metadata)

        return response, retrieved, metadata

    def _build_context(self, query: str, retrieved: List[RetrievedDocument]) -> str:
        """Build context from retrieved documents."""
        context_parts = [f"User Query: {query}\n"]

        if retrieved:
            context_parts.append("Retrieved Context:")
            for i, doc in enumerate(retrieved, 1):
                relevance = doc.relevance_score or 0.0
                context_parts.append(
                    f"\n[Document {i} - Relevance: {relevance:.2f}]"
                )
                context_parts.append(f"Source: {doc.source}")
                context_parts.append(f"Content: {doc.text}\n")
        else:
            context_parts.append("No relevant documents found.")

        return "\n".join(context_parts)

    def _add_citations(self, response: str, retrieved: List[RetrievedDocument]) -> str:
        """Add citations to response."""
        citation_section = "\n\n--- Citations ---\n"

        for i, doc in enumerate(retrieved, 1):
            citation_section += f"[{i}] {doc.source}\n"

        return response + citation_section

    def _detect_hallucinations(
        self, response: str, retrieved: List[RetrievedDocument]
    ) -> float:
        """Detect hallucinations in response.

        Returns:
            Hallucination score (0-1, higher = more hallucinated)
        """
        if not retrieved:
            return 1.0

        source_content = " ".join([doc.text for doc in retrieved])
        source_words = set(source_content.lower().split())

        response_words = set(response.lower().split())
        unsupported_words = response_words - source_words

        hallucination_ratio = len(unsupported_words) / max(1, len(response_words))

        return min(1.0, hallucination_ratio)

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if not self.query_history:
            return {"queries": 0}

        response_times = [q["response_time"] for q in self.query_history]
        hallucination_scores = [
            q["hallucination_score"]
            for q in self.query_history
            if q["hallucination_score"] is not None
        ]

        return {
            "total_queries": len(self.query_history),
            "avg_response_time": sum(response_times) / len(response_times),
            "avg_hallucination_score": (
                sum(hallucination_scores) / len(hallucination_scores)
                if hallucination_scores
                else None
            ),
            "retriever_stats": self.retriever.get_stats(),
        }

    def clear_history(self) -> None:
        """Clear query history."""
        self.query_history = []

    @staticmethod
    def _default_system_prompt() -> str:
        """Get default system prompt."""
        return """You are a helpful assistant that answers questions based on retrieved documents.

Instructions:
- Answer based ONLY on the provided context
- If information is not in the context, say "I don't have enough information"
- Be concise and factual
- Cite sources when relevant"""
