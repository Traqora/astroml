"""Tests for RAG (Retrieval-Augmented Generation) functionality.

Resolves #458: Integration tests for RAG query pipeline using mock providers
and test document fixtures.
"""
from __future__ import annotations

import pytest

from tests.llm.fixtures import RAG_TEST_DOCUMENTS, MOCK_EMBEDDINGS, make_mock_embedding
from tests.llm.utils import assert_valid_rag_response


class TestRAGDocumentFixtures:
    """Tests for RAG test document fixtures."""

    def test_rag_documents_have_required_fields(self, rag_documents):
        required = {"doc_id", "content", "metadata", "embedding"}
        for doc in rag_documents:
            missing = required - doc.keys()
            assert not missing, f"Document {doc.get('doc_id')} missing: {missing}"

    def test_rag_embeddings_are_1536_dim(self, rag_documents):
        for doc in rag_documents:
            assert len(doc["embedding"]) == 1536

    def test_rag_documents_have_non_empty_content(self, rag_documents):
        for doc in rag_documents:
            assert isinstance(doc["content"], str)
            assert len(doc["content"]) > 0

    def test_rag_documents_unique_ids(self, rag_documents):
        ids = [d["doc_id"] for d in rag_documents]
        assert len(ids) == len(set(ids)), "Document IDs must be unique"


class TestMockEmbeddings:
    """Tests for mock embedding generation."""

    def test_embedding_is_correct_dimension(self):
        vec = make_mock_embedding("test text")
        assert len(vec) == 1536

    def test_embedding_is_deterministic(self):
        v1 = make_mock_embedding("same text")
        v2 = make_mock_embedding("same text")
        assert v1 == v2

    def test_different_texts_different_embeddings(self):
        v1 = make_mock_embedding("text A")
        v2 = make_mock_embedding("text B")
        assert v1 != v2

    def test_embedding_values_in_valid_range(self):
        vec = make_mock_embedding("normalised embedding")
        for val in vec:
            assert -1.0 <= val <= 1.0

    def test_mock_embedding_fixture_coverage(self, mock_embeddings):
        """Fixture embeddings should contain at least 3 entries."""
        assert len(mock_embeddings) >= 3


class TestRAGPipelineIntegration:
    """Integration tests for RAG query pipeline with mock provider."""

    @pytest.mark.asyncio
    async def test_rag_query_returns_valid_structure(self, mock_provider):
        """RAG query should produce a valid response dict."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        result = await svc.rag_query(
            query="What is fraud detection?",
            top_k=3,
            user_id="test_user",
        )
        assert_valid_rag_response(result)

    @pytest.mark.asyncio
    async def test_rag_query_returns_requested_top_k(self, mock_provider):
        """top_k parameter controls number of returned documents."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        for top_k in (1, 3, 5):
            result = await svc.rag_query(
                query="test",
                top_k=top_k,
                user_id="test_user",
            )
            assert len(result["documents"]) == top_k

    @pytest.mark.asyncio
    async def test_rag_query_includes_answer(self, mock_provider):
        """RAG response must contain a non-empty answer string."""
        from api.services.llm import LLMService
        svc = LLMService(provider=mock_provider)
        result = await svc.rag_query(
            query="Explain fraud detection",
            top_k=2,
        )
        assert isinstance(result.get("answer"), str)
        assert len(result["answer"]) > 0
