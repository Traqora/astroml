- closes #722
- closes #723
- closes #730
- closes #741

### Description
1. **Issue #722**: Added exponential backoff to `_stream_with_retry` in `enhanced_stream.py` to ensure failed Horizon API stream connections or rate limits correctly back off exponentially rather than linearly.
2. **Issue #723**: Extracted a unified helper `extract_asset_string` in `astroml/ingestion/parsers.py` to standardize the parsing of asset types and issuers as `code:issuer` or `XLM`. This helper is now utilized consistently in both `extract_path_payment_hops` and `normalizer.py`.
3. **Issue #730**: Implemented a partitioning strategy for large ledger data sets in `stellar_ledger.py`, allowing grouping of ledger files into directories based on a configurable `--partition-size`. Updated `ledger_reader.py` to recursively search these partition directories via `rglob` for efficient reading.
4. **Issue #741**: Added a `compute_rolling_node_features` function in `astroml/features/node_features.py` that calculates account aggregate features (e.g., received/sent volumes and degrees) filtered by a rolling time window.
- closes #749
- closes #750
- closes #751
- closes #752

- Added readiness/liveness endpoints for training and serving pods
- Added Prometheus metrics for pipeline latency and throughput
- Added alert rule for ingestion success-rate drops
- Implemented structured logging across pipeline components
