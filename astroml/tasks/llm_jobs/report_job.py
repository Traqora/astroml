"""Report backfill job — batch generate monthly reports."""

from typing import Any
from astroml.llm.providers.base import LLMProvider


REPORT_PROMPT = """Generate a monthly summary report based on the following data:
Month: {month}
Total Transactions: {total_tx}
Total Volume: {volume}
Fraud Alerts: {alerts}
Top Categories: {categories}

Report:"""


class ReportJobHandler:
    """Batch generate monthly reports."""

    type = "report"
    description = "Batch generate monthly reports"

    async def process_item(self, item: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
        prompt = REPORT_PROMPT.format(
            month=item.get("month", "unknown"),
            total_tx=item.get("total_tx", 0),
            volume=item.get("volume", "0"),
            alerts=item.get("alerts", 0),
            categories=item.get("categories", "N/A"),
        )
        report = provider.generate(prompt, max_tokens=500)
        return {"report": report}
