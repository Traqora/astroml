from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class RagSource:
    id: str
    title: str
    url: str
    content: str


@dataclass(frozen=True)
class RagCitation:
    source_id: str
    title: str
    url: str
    snippet: str


RAG_CORPUS: tuple[RagSource, ...] = (
    RagSource(
        id="usage-examples",
        title="AstroML Usage Examples",
        url="/docs/api/usage-examples.md",
        content=(
            "Quick start covers ingestion, graph building, model training, and production integration. "
            "The guide also shows historical backfill, real-time streaming, and advanced patterns."
        ),
    ),
    RagSource(
        id="api-reference",
        title="AstroML API Reference",
        url="/docs/api/reference.md",
        content=(
            "The API reference documents ingestion services, state management, stream ingestion, "
            "HorizonStream, and utility functions with parameter tables and code examples."
        ),
    ),
    RagSource(
        id="faq",
        title="AstroML FAQ Router",
        url="/api/v1/faq",
        content=(
            "The FAQ router supports listing published FAQs, category filtering, full-text search, "
            "and feedback submission for helpfulness tracking."
        ),
    ),
    RagSource(
        id="llm-router",
        title="AstroML LLM Router",
        url="/api/v1/llm",
        content=(
            "The LLM router already provides transaction explanations, query translation, multimodal "
            "context generation, and response validation."
        ),
    ),
)


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


def _score_source(question: str, source: RagSource) -> int:
    question_tokens = _tokenize(question)
    content_tokens = _tokenize(f"{source.title} {source.content}")
    overlap = len(question_tokens & content_tokens)

    bonus = 0
    lowered = question.lower()
    if "faq" in lowered and source.id == "faq":
        bonus += 5
    if any(term in lowered for term in ("example", "quick start", "how do i")) and source.id == "usage-examples":
        bonus += 4
    if any(term in lowered for term in ("reference", "parameter", "schema", "api")) and source.id == "api-reference":
        bonus += 4
    if any(term in lowered for term in ("llm", "rag", "answer", "question")) and source.id == "llm-router":
        bonus += 3

    return overlap * 2 + bonus


def retrieve_sources(question: str, limit: int = 3) -> list[RagSource]:
    ranked = sorted(
        RAG_CORPUS,
        key=lambda source: _score_source(question, source),
        reverse=True,
    )
    selected = [source for source in ranked if _score_source(question, source) > 0]
    if not selected:
        return list(RAG_CORPUS[:limit])
    return selected[:limit]


def build_citations(question: str, sources: Iterable[RagSource]) -> list[RagCitation]:
    citations: list[RagCitation] = []
    question_tokens = _tokenize(question)

    for source in sources:
        content_words = source.content.split()
        if question_tokens:
            matching_index = next(
                (
                    idx
                    for idx, word in enumerate(content_words)
                    if _tokenize(word) & question_tokens
                ),
                0,
            )
        else:
            matching_index = 0

        start = max(0, matching_index - 8)
        end = min(len(content_words), matching_index + 22)
        snippet = " ".join(content_words[start:end]).strip()

        citations.append(
            RagCitation(
                source_id=source.id,
                title=source.title,
                url=source.url,
                snippet=snippet,
            )
        )

    return citations


def build_rag_answer(question: str, citations: list[RagCitation]) -> str:
    if not citations:
        return (
            "I could not find a relevant AstroML source, but the best next step is to consult "
            "the API reference and FAQ router."
        )

    primary = citations[0]
    if primary.source_id == "usage-examples":
        lead = "The docs suggest starting with the quick-start and backfill examples."
    elif primary.source_id == "api-reference":
        lead = "The API reference is the best starting point for implementation details."
    elif primary.source_id == "faq":
        lead = "The FAQ router is the most relevant source for common answers and support flows."
    else:
        lead = "The existing LLM router already exposes several related utilities."

    return (
        f"{lead} Based on the retrieved sources, a practical answer to '{question.strip()}' is to "
        f"follow the documented workflow and verify it against the cited endpoints and guides."
    )

