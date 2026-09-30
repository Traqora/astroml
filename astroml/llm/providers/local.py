from astroml.utils.exceptions import AstroMLError
"""Local model provider (Llama, Mistral) with HuggingFace Hub and mock fallbacks."""

import logging
from collections.abc import Iterator
from typing import Any

from .base import LLMProvider

logger = logging.getLogger(__name__)


class LocalProvider(LLMProvider):
    def __init__(self, api_key -> Any: str = "", model: str = "meta-llama/Llama-2-7b-chat-hf"):
        super().__init__(api_key, model)

    def _generate_raw(self, prompt: str, **kwargs: Any) -> str:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                model=kwargs.pop("model", self.model), token=self.api_key or None
            )
            text = client.text_generation(prompt, **kwargs)
        except AstroMLError as e:
            logger.warning(
                f"Failed to use HuggingFace Hub for local model: {e}. Falling back to mock generation."
            )
            text = f"Mock response from local model {self.model} for prompt: {prompt[:30]}..."

        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(text)
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text

    def get_token_usage(self) -> dict[str, int]:
        return self.last_usage

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(
                model=kwargs.pop("model", self.model), token=self.api_key or None
            )
            for token in client.text_generation(prompt, stream=True, **kwargs):
                yield token
        except AstroMLError:
            yield f"Mock stream from local model {self.model}: {prompt[:30]}"

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=self.api_key or None)
            embedding = client.feature_extraction(text)
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
            return [0.0] * 384
        except AstroMLError:
            return [0.0] * 384

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)
