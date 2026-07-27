"""LLM context management for blockchain data (issue #360)."""
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BlockchainContextBuilder:
    """Packs blockchain transaction history into an LLM-ready context
    string that fits within a token budget, compressing older days first.
    """

    def __init__(self, token_limit: int = 4000, recent_detailed_days: int = 7, group_size: int = 7):
        self.token_limit = token_limit
        self.recent_detailed_days = recent_detailed_days
        self.group_size = group_size

    def analyze_token_size(self, data: str) -> int:
        """Rough token estimate (~4 characters per token)."""
        return max(1, len(data) // 4)

    def summarize_by_day(self, days: int, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group raw transactions from the last `days` days into daily summaries."""
        cutoff = datetime.now() - timedelta(days=days)
        by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in raw_data:
            ts = datetime.fromisoformat(item.get("timestamp", datetime.now().isoformat()))
            if ts >= cutoff:
                by_day[ts.date().isoformat()].append(item)

        summaries = []
        for day, items in sorted(by_day.items()):
            addresses = {item.get("from_address") for item in items} | {item.get("to_address") for item in items}
            summaries.append({
                "day": day,
                "transaction_count": len(items),
                "total_volume": sum(item.get("amount", 0) for item in items),
                "unique_addresses": len(addresses - {None}),
            })
        return summaries

    def _summary_to_text(self, summary: Dict[str, Any]) -> str:
        return (
            f"{summary['day']}: {summary['transaction_count']} txs, "
            f"volume={summary['total_volume']:.2f}, unique_addresses={summary['unique_addresses']}"
        )

    def compress_data(self, summaries: List[Dict[str, Any]], group_size: int = None) -> str:
        """Aggregate daily summaries into multi-day buckets, summing
        transaction counts/volume exactly so totals don't drift.
        """
        group_size = group_size or self.group_size
        lines = []
        for i in range(0, len(summaries), group_size):
            group = summaries[i:i + group_size]
            start, end = group[0]["day"], group[-1]["day"]
            tx_total = sum(s["transaction_count"] for s in group)
            volume_total = sum(s["total_volume"] for s in group)
            addresses_total = sum(s["unique_addresses"] for s in group)
            lines.append(
                f"{start}..{end} ({len(group)}d): {tx_total} txs, "
                f"volume={volume_total:.2f}, unique_addresses~{addresses_total}"
            )
        return "\n".join(lines)

    def build_context(self, days: int, raw_data: List[Dict[str, Any]]) -> str:
        """Build a token-budgeted context string for the last `days` days
        of blockchain activity, compressing older days if needed.
        """
        summaries = self.summarize_by_day(days, raw_data)

        detailed_text = "\n".join(self._summary_to_text(s) for s in summaries)
        if self.analyze_token_size(detailed_text) <= self.token_limit:
            return detailed_text

        cutoff = max(0, len(summaries) - self.recent_detailed_days)
        older, recent = summaries[:cutoff], summaries[cutoff:]
        recent_text = "\n".join(self._summary_to_text(s) for s in recent)

        if not older:
            return recent_text

        context = self.compress_data(older) + ("\n" + recent_text if recent_text else "")
        if self.analyze_token_size(context) <= self.token_limit:
            return context

        logger.warning("Context still exceeds token_limit=%d after compression; falling back to full aggregation.", self.token_limit)
        return self.compress_data(summaries)
