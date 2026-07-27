"""Label backfill job — auto-label transactions for ML training."""

from typing import Any
from astroml.llm.providers.base import LLMProvider


LABEL_PROMPT = """Classify this transaction into exactly one category: payment, transfer, exchange, fraud, or other.
Transaction: {description}
Amount: {amount}
Category:"""


class LabelJobHandler:
    """Auto-label transactions for ML training."""

    type = "label"
    description = "Auto-label transactions for ML training"

    async def process_item(self, item: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
        prompt = LABEL_PROMPT.format(
            description=item.get("description", ""),
            amount=item.get("amount", "0"),
        )
        label = provider.generate(prompt, max_tokens=20).strip().lower()
        return {"label": label}
