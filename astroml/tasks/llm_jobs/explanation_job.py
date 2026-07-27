"""Explanation backfill job — generate LLM explanations for past fraud alerts."""

from typing import Any
from astroml.llm.providers.base import LLMProvider


EXPLANATION_PROMPT = """Explain this fraud alert in simple terms:
Alert ID: {alert_id}
Transaction: {transaction}
Risk Score: {score}
Reason: {reason}"""


class ExplanationJobHandler:
    """Generate LLM explanations for historical fraud alerts."""

    type = "explanation"
    description = "Generate LLM explanations for past fraud alerts"

    async def process_item(self, item: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
        prompt = EXPLANATION_PROMPT.format(
            alert_id=item.get("alert_id", "unknown"),
            transaction=item.get("transaction", "unknown"),
            score=item.get("score", "N/A"),
            reason=item.get("reason", "N/A"),
        )
        explanation = provider.generate(prompt)
        return {"explanation": explanation}
