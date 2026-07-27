"""Document ingestion for RAG system."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import time


class DocumentIngestor:
    """Ingests documents from various sources."""

    def __init__(self, embeddings_service: Any, retriever: Any):
        """Initialize ingestor.

        Args:
            embeddings_service: Service for computing embeddings
            retriever: Retriever to add documents to
        """
        self.embeddings = embeddings_service
        self.retriever = retriever
        self.ingested_count = 0

    def ingest_directory(self, directory: str, pattern: str = "*.md") -> Dict[str, Any]:
        """Ingest all documents from directory.

        Args:
            directory: Directory path
            pattern: File pattern to match

        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        path = Path(directory)

        documents = []
        sources = []
        metadata = []

        for file_path in path.glob(pattern):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                chunks = self.embeddings.chunk_text(content)

                for chunk in chunks:
                    documents.append(chunk)
                    sources.append(str(file_path))
                    metadata.append({
                        "file": file_path.name,
                        "type": "markdown",
                        "ingestion_time": time.time(),
                    })

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        if documents:
            self.retriever.add_documents(documents, sources, metadata)
            self.ingested_count += len(documents)

        elapsed = time.time() - start_time

        return {
            "documents_ingested": len(documents),
            "chunks_created": len(documents),
            "elapsed_seconds": elapsed,
            "total_ingested": self.ingested_count,
        }

    def ingest_text_file(self, file_path: str, source_name: Optional[str] = None) -> Dict[str, Any]:
        """Ingest single text file.

        Args:
            file_path: Path to file
            source_name: Name for source attribution

        Returns:
            Ingestion statistics
        """
        start_time = time.time()
        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = self.embeddings.chunk_text(content)

        source = source_name or str(path)
        sources = [source] * len(chunks)
        metadata = [
            {
                "file": path.name,
                "type": "text",
                "ingestion_time": time.time(),
            }
            for _ in chunks
        ]

        self.retriever.add_documents(chunks, sources, metadata)
        self.ingested_count += len(chunks)

        elapsed = time.time() - start_time

        return {
            "chunks_created": len(chunks),
            "elapsed_seconds": elapsed,
            "total_ingested": self.ingested_count,
        }

    def ingest_texts(
        self, texts: List[str], sources: List[str], metadata: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Ingest list of texts.

        Args:
            texts: List of text documents
            sources: List of source names
            metadata: Optional metadata for each document

        Returns:
            Ingestion statistics
        """
        start_time = time.time()

        all_chunks = []
        all_sources = []
        all_metadata = []

        for i, text in enumerate(texts):
            chunks = self.embeddings.chunk_text(text)

            for chunk in chunks:
                all_chunks.append(chunk)
                all_sources.append(sources[i])

                meta = {"ingestion_time": time.time()}
                if metadata and i < len(metadata):
                    meta.update(metadata[i])
                all_metadata.append(meta)

        if all_chunks:
            self.retriever.add_documents(all_chunks, all_sources, all_metadata)
            self.ingested_count += len(all_chunks)

        elapsed = time.time() - start_time

        return {
            "documents_ingested": len(texts),
            "chunks_created": len(all_chunks),
            "elapsed_seconds": elapsed,
            "total_ingested": self.ingested_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            "total_ingested": self.ingested_count,
        }
