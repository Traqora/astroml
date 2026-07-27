"""Abstract base for embedding providers.

All embedding providers share a common interface so the router and cache
can treat them uniformly regardless of which backend is active.

Trade-off summary
-----------------
+----------------+--------+-------+----------+------------------------------+
| Provider       | Dims   | Cost  | Latency  | Notes                        |
+================+========+=======+==========+==============================+
| OpenAI         | 1536   | $$$   | ~100 ms  | Best quality, API key needed |
| Cohere         | 1024   | $$    | ~120 ms  | Good quality, API key needed |
| HuggingFace    | 768    | $     | ~150 ms  | Inference API or self-hosted |
| Local (TF-IDF) | varies | free  | <1 ms    | Zero deps, fallback only     |
+----------------+--------+-------+----------+------------------------------+

Dimension compatibility
-----------------------
Dense providers return fixed-size vectors; the local TF-IDF encoder returns
a vector whose length grows with vocabulary size.  The ``EmbeddingRouter``
resolves this by:

1. Preferring exact-dimension providers when available.
2. Padding shorter vectors with zeros to match the target dimension.
3. Truncating longer vectors to match the target dimension.
4. Normalising all returned vectors to unit L2 length so cosine similarity
   is equivalent to a dot product.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """Abstract base class for all embedding providers.

    Attributes
    ----------
    name : str
        Human-readable provider name used in logs and metadata.
    output_dim : int
        Fixed output dimension.  Implementations MUST guarantee this length
        for every successful call to :meth:`embed` and :meth:`embed_batch`.
    """

    name: str = "base"
    output_dim: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Return an embedding vector for *text*.

        Parameters
        ----------
        text:
            The input string to embed.

        Returns
        -------
        List of floats with length equal to ``output_dim``.

        Raises
        ------
        EmbeddingError
            If the provider cannot produce an embedding for any reason
            (network error, quota exceeded, invalid input, …).
        """

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Return embedding vectors for a list of texts.

        Default implementations may call :meth:`embed` in a loop.
        Providers that support batch APIs should override this method.

        Returns
        -------
        List of embedding vectors, one per input text, in the same order.
        """

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if the provider can be used right now.

        The default implementation always returns True.  Providers that
        require network access or API keys should override this to do a
        cheap health-check (e.g., verify the env-var is set).
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dim={self.output_dim})"


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider fails to produce a vector."""
