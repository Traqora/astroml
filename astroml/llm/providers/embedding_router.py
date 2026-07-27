"""Embedding router with automatic fallback and dimension normalisation.

The ``EmbeddingRouter`` tries providers in priority order, falling back to
the next one when a provider raises ``EmbeddingError`` or times out.  The
total fallback budget is 500 ms (acceptance criterion).

Dimension compatibility
-----------------------
Different providers return vectors of different lengths.  The router
normalises all outputs to a single ``target_dim`` by:

- Truncating vectors that are longer than ``target_dim``.
- Zero-padding vectors that are shorter than ``target_dim``.
- Re-normalising to unit L2 length after any padding/truncation so that
  cosine similarity is still meaningful.

The ``target_dim`` defaults to the ``output_dim`` of the first (highest-
priority) provider that has a fixed dimension (> 0).  When all providers
have dynamic dimensions (only the local TF-IDF provider), normalisation is
skipped.

Usage example
-------------
.. code-block:: python

    from astroml.llm.providers.embedding_router import EmbeddingRouter, build_default_router

    # Auto-builds OpenAI → Cohere → HuggingFace → Local chain from env vars.
    router = build_default_router()

    vector = router.embed("Detect fraud in Stellar transactions")
    print(len(vector))        # == router.output_dim
    print(router.active_provider.name)   # whichever provider succeeded
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from .embedding_base import EmbeddingProvider, EmbeddingError

logger = logging.getLogger(__name__)

# Hard ceiling on total time spent across all fallback attempts (seconds).
_FALLBACK_BUDGET_S = 0.5


def _normalise(vec: List[float], target_dim: int) -> List[float]:
    """Pad or truncate *vec* to *target_dim* and return unit-normalised list."""
    arr = np.array(vec, dtype=np.float32)
    current = arr.shape[0]

    if current < target_dim:
        arr = np.pad(arr, (0, target_dim - current))
    elif current > target_dim:
        arr = arr[:target_dim]

    norm = float(np.linalg.norm(arr))
    if norm > 1e-9:
        arr /= norm
    return arr.tolist()


class EmbeddingRouter(EmbeddingProvider):
    """Multi-provider embedding router with automatic fallback.

    Parameters
    ----------
    providers:
        Ordered list of providers to try.  The first available provider is
        used; if it fails, the next one is tried, subject to the 500 ms
        fallback budget.
    target_dim:
        Target output dimension for dimension normalisation.  If ``None``
        (default), the router infers it from the first provider with a
        fixed ``output_dim > 0``.  When no fixed-dim provider exists,
        vectors are returned as-is.
    """

    name = "router"

    def __init__(
        self,
        providers: List[EmbeddingProvider],
        target_dim: Optional[int] = None,
    ) -> None:
        if not providers:
            raise ValueError("EmbeddingRouter requires at least one provider")
        self.providers = providers
        self._active: Optional[EmbeddingProvider] = None

        # Determine target_dim from the highest-priority fixed-dim provider.
        if target_dim is not None:
            self._target_dim: Optional[int] = target_dim
        else:
            self._target_dim = next(
                (p.output_dim for p in providers if p.output_dim > 0), None
            )

        self.output_dim = self._target_dim or 0

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def active_provider(self) -> Optional[EmbeddingProvider]:
        """The last provider that successfully produced an embedding."""
        return self._active

    def _try_embed(
        self, provider: EmbeddingProvider, text: str, remaining_s: float
    ) -> Tuple[Optional[List[float]], float]:
        """Attempt a single embed call; return (result, elapsed_s) or (None, elapsed_s)."""
        t0 = time.monotonic()
        try:
            if not provider.is_available():
                return None, 0.0
            vec = provider.embed(text)
            elapsed = time.monotonic() - t0
            return vec, elapsed
        except (EmbeddingError, Exception) as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "EmbeddingRouter: provider %r failed in %.0f ms — %s",
                provider.name,
                elapsed * 1000,
                exc,
            )
            return None, elapsed

    def _normalise_vec(self, vec: List[float]) -> List[float]:
        """Apply dimension normalisation if a target_dim is configured."""
        if self._target_dim is not None and len(vec) != self._target_dim:
            return _normalise(vec, self._target_dim)
        return vec

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    def embed(self, text: str) -> List[float]:
        """Embed *text* using the first available provider, with fallback.

        Falls back to the next provider on failure.  Total time across all
        attempts is capped at 500 ms (acceptance criterion).

        Returns
        -------
        Embedding vector of length ``output_dim`` (or variable length when
        no fixed-dim provider is configured).

        Raises
        ------
        EmbeddingError
            If all providers fail or the fallback budget is exhausted.
        """
        deadline = time.monotonic() + _FALLBACK_BUDGET_S
        last_error: Optional[Exception] = None

        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "EmbeddingRouter: fallback budget exhausted before trying %r",
                    provider.name,
                )
                break

            vec, elapsed = self._try_embed(provider, text, remaining)
            if vec is not None:
                self._active = provider
                logger.debug(
                    "EmbeddingRouter: %r succeeded in %.0f ms",
                    provider.name,
                    elapsed * 1000,
                )
                return self._normalise_vec(vec)

        raise EmbeddingError(
            f"All embedding providers failed within {_FALLBACK_BUDGET_S * 1000:.0f} ms budget"
        )

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using the first available provider, with fallback."""
        deadline = time.monotonic() + _FALLBACK_BUDGET_S
        last_error: Optional[Exception] = None

        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not provider.is_available():
                continue
            t0 = time.monotonic()
            try:
                vecs = provider.embed_batch(texts)
                elapsed = time.monotonic() - t0
                self._active = provider
                logger.debug(
                    "EmbeddingRouter batch: %r succeeded in %.0f ms",
                    provider.name,
                    elapsed * 1000,
                )
                return [self._normalise_vec(v) for v in vecs]
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "EmbeddingRouter batch: provider %r failed in %.0f ms — %s",
                    provider.name,
                    elapsed * 1000,
                    exc,
                )
                last_error = exc

        raise EmbeddingError(
            f"All embedding providers failed within {_FALLBACK_BUDGET_S * 1000:.0f} ms budget"
        )

    def provider_status(self) -> List[dict]:
        """Return availability status for every configured provider."""
        return [
            {
                "name": p.name,
                "output_dim": p.output_dim,
                "available": p.is_available(),
                "active": p is self._active,
            }
            for p in self.providers
        ]


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_default_router(target_dim: Optional[int] = None) -> EmbeddingRouter:
    """Build the standard provider chain from environment variables.

    Priority order: OpenAI → Cohere → HuggingFace → Local

    Any provider whose package is missing or whose API key is absent is
    silently skipped at embed-time (``is_available()`` returns False) and
    the next one in the chain is tried.  The local TF-IDF provider is
    always the final fallback and is always available.

    Parameters
    ----------
    target_dim:
        Override the output dimension.  If ``None``, the router uses the
        first fixed-dim provider's dimension (OpenAI → 1536 if available,
        Cohere → 1024, HuggingFace → 384, else dynamic).
    """
    from .embedding_openai import OpenAIEmbeddingProvider
    from .embedding_cohere import CohereEmbeddingProvider
    from .embedding_huggingface import HuggingFaceEmbeddingProvider
    from .embedding_local import LocalEmbeddingProvider

    providers: List[EmbeddingProvider] = [
        OpenAIEmbeddingProvider(),
        CohereEmbeddingProvider(),
        HuggingFaceEmbeddingProvider(),
        LocalEmbeddingProvider(),
    ]
    return EmbeddingRouter(providers=providers, target_dim=target_dim)
