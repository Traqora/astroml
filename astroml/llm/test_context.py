"""Tests for blockchain LLM context management (issue #360)."""
import unittest
from datetime import datetime, timedelta

from .blockchain_context import BlockchainContextBuilder


def _make_raw_data(num_days: int, txs_per_day: int = 20) -> list:
    now = datetime.now()
    raw_data = []
    for day in range(num_days):
        ts = now - timedelta(days=day)
        for tx in range(txs_per_day):
            raw_data.append({
                "id": f"d{day}-t{tx}",
                "amount": 100.0 + tx,
                "timestamp": ts.isoformat(),
                "from_address": f"0xFrom{day}-{tx % 5}",
                "to_address": f"0xTo{day}-{tx % 5}",
            })
    return raw_data


class BlockchainContextBuilderTests(unittest.TestCase):
    def test_30_days_fits_4k_token_budget(self):
        raw_data = _make_raw_data(30)
        builder = BlockchainContextBuilder(token_limit=4000)

        context = builder.build_context(30, raw_data)

        self.assertLessEqual(builder.analyze_token_size(context), 4000)

    def test_small_input_stays_detailed(self):
        raw_data = _make_raw_data(3)
        builder = BlockchainContextBuilder(token_limit=4000)

        context = builder.build_context(3, raw_data)
        summaries = builder.summarize_by_day(3, raw_data)

        for summary in summaries:
            self.assertIn(summary["day"], context)


class CompressDataTests(unittest.TestCase):
    def test_compression_ratio_at_least_60_percent(self):
        raw_data = _make_raw_data(30)
        builder = BlockchainContextBuilder()
        summaries = builder.summarize_by_day(30, raw_data)
        detailed_text = "\n".join(builder._summary_to_text(s) for s in summaries)

        compressed_text = builder.compress_data(summaries, group_size=7)

        original_tokens = builder.analyze_token_size(detailed_text)
        compressed_tokens = builder.analyze_token_size(compressed_text)
        compression_ratio = 1 - (compressed_tokens / original_tokens)

        self.assertGreaterEqual(compression_ratio, 0.6)

    def test_compression_preserves_totals_within_5_percent(self):
        raw_data = _make_raw_data(30)
        builder = BlockchainContextBuilder()
        summaries = builder.summarize_by_day(30, raw_data)
        original_tx_total = sum(s["transaction_count"] for s in summaries)
        original_volume_total = sum(s["total_volume"] for s in summaries)

        compressed_text = builder.compress_data(summaries, group_size=7)

        compressed_tx_total = sum(
            int(line.split(" txs")[0].split()[-1])
            for line in compressed_text.split("\n")
        )
        compressed_volume_total = sum(
            float(line.split("volume=")[1].split(",")[0])
            for line in compressed_text.split("\n")
        )

        tx_loss = abs(compressed_tx_total - original_tx_total) / original_tx_total
        volume_loss = abs(compressed_volume_total - original_volume_total) / original_volume_total

        self.assertLess(tx_loss, 0.05)
        self.assertLess(volume_loss, 0.05)


if __name__ == "__main__":
    unittest.main()
