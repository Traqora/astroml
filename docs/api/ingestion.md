# Data Ingestion API Documentation

## Overview

The data ingestion module is responsible for fetching, processing, and storing Stellar blockchain data. It provides both batch and streaming capabilities with built-in state management and error handling.

## Core Classes

### IngestionService

The main service for orchestrating data ingestion operations.

#### Class Definition

```python
class IngestionService:
    def __init__(self, state_store: Optional[StateStore] = None) -> None
    def ingest(
        self,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        fetch_fn: Optional[Callable[[int], object]] = None,
        process_fn: Optional[Callable[[int, object], None]] = None,
    ) -> IngestionResult
```

#### Constructor Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `state_store` | `Optional[StateStore]` | No | `None` | State management instance for tracking processed ledgers |

#### Methods

##### ingest()

Ingest ledgers incrementally and idempotently.

**Parameters:**
- `start_ledger` (Optional[int]): Starting ledger ID (inclusive). If None, resumes from last processed ledger + 1 or 0.
- `end_ledger` (Optional[int]): Ending ledger ID (inclusive). If None, processes only the start_ledger if provided.
- `fetch_fn` (Optional[Callable]): Function to fetch data for a ledger ID. Defaults to identity payload.
- `process_fn` (Optional[Callable]): Function to handle processing. Defaults to no-op.

**Returns:** `IngestionResult` object with ingestion statistics.

**Behavior:**
- Skips any ledger already recorded as processed
- Updates state per-ledger for safe retries
- Raises ValueError if end_ledger < start_ledger

**Example:**
```python
from astroml.ingestion import IngestionService

service = IngestionService()

# Ingest historical ledgers
result = service.ingest(
    start_ledger=1000000,
    end_ledger=1100000,
    fetch_fn=lambda ledger_id: fetch_stellar_ledger(ledger_id),
    process_fn=lambda ledger_id, data: store_ledger_data(ledger_id, data)
)

print(f"Attempted: {len(result.attempted)}")
print(f"Processed: {len(result.processed)}")
print(f"Skipped: {len(result.skipped)}")
```

#### ingest_backfill_chunked()

Memory-efficient backfill for very large ledger ranges. Partitions the range into chunks and processes each independently, discarding accumulated data between chunks to keep memory usage bounded.

**Parameters:**
- `start_ledger` (int): First ledger to process (inclusive). If `resume_from_checkpoint=True` and a checkpoint exists, this is overridden by the checkpoint position.
- `end_ledger` (int): Last ledger to process (inclusive).
- `chunk_size` (int): Number of ledgers per memory-bounded batch. Default 10,000.
- `fetch_fn` (Optional[Callable]): Function to fetch data for a ledger ID. Defaults to identity payload.
- `process_fn` (Optional[Callable]): Function to handle processing. Defaults to no-op.
- `batch_size` (int): State-flush cadence inside each chunk. Default 100.
- `resume_from_checkpoint` (bool): If True, resume from the last checkpoint instead of starting from `start_ledger`. Defaults to False.
- `checkpoint_path` (Optional[str]): Optional path to the checkpoint file. If None, uses `.astroml_state/backfill_checkpoint.json`.

**Returns:** Generator yielding one summary dict per chunk with keys: `chunk_start`, `chunk_end`, `processed`, `skipped`, `errors`.

**Behavior:**
- Processes ledgers in chunks to bound memory usage
- Runs garbage collection between chunks
- When `resume_from_checkpoint=True`, saves position after each successful chunk
- On restart with checkpoint, resumes from the last saved position
- Clears checkpoint file on successful completion
- Continues processing even if individual chunks fail (errors are counted)

**Example with checkpointing:**
```python
from astroml.ingestion import IngestionService

service = IngestionService()

# Process a large range with checkpoint support
for chunk_summary in service.ingest_backfill_chunked(
    start_ledger=1000000,
    end_ledger=5000000,
    chunk_size=10000,
    resume_from_checkpoint=True,
    checkpoint_path="backfill_checkpoint.json",
    fetch_fn=lambda ledger_id: fetch_stellar_ledger(ledger_id),
    process_fn=lambda ledger_id, data: store_ledger_data(ledger_id, data)
):
    print(f"Chunk {chunk_summary['chunk_start']}-{chunk_summary['chunk_end']}: "
          f"processed={chunk_summary['processed']}, "
          f"skipped={chunk_summary['skipped']}, "
          f"errors={chunk_summary['errors']}")
```

**Example without checkpointing:**
```python
# Standard chunked backfill (no checkpoint)
for chunk_summary in service.ingest_backfill_chunked(
    start_ledger=1000000,
    end_ledger=5000000,
    chunk_size=10000
):
    print(f"Processed chunk {chunk_summary['chunk_start']}-{chunk_summary['chunk_end']}")
```

#### Failure notifications

`IngestionService` accepts an optional `notifier` — any `(message) -> Any`
callable — which is invoked with a one-line summary whenever an ingestion run
fails. `SlackIntegration.send_webhook` is the intended implementation:

```python
from astroml.chat.slack import SlackConfig, SlackIntegration
from astroml.ingestion import IngestionService

slack = SlackIntegration(SlackConfig(webhook_url=os.environ["SLACK_WEBHOOK_URL"]))
service = IngestionService(notifier=slack.send_webhook)
```

Behavior worth knowing:

| Path | Notified? | Notes |
|------|-----------|-------|
| `ingest()` | Yes | Message names the last attempted ledger and how many succeeded. |
| `ingest_incremental()` | Yes | Delegates to `ingest()`. |
| `ingest_backfill_chunked()` | Yes, once per failed chunk | The chunk exception is caught so the backfill continues, but each failure is still reported (issue #993). |
| No failure | No | A successful run never calls the notifier. |

The notifier's return value is ignored, and an exception raised by the notifier
is logged at `WARNING` rather than propagated — alerting can never mask or
replace the original ingestion error.

### IngestionResult

Container for ingestion operation results.

#### Class Definition

```python
@dataclass
class IngestionResult:
    attempted: List[int]
    processed: List[int]
    skipped: List[int]
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `attempted` | `List[int]` | List of ledger IDs that were attempted |
| `processed` | `List[int]` | List of ledger IDs that were successfully processed |
| `skipped` | `List[int]` | List of ledger IDs that were skipped (already processed) |

### StateStore

Manages ingestion state and tracking of processed ledgers.

#### Class Definition

```python
class StateStore:
    def load(self) -> IngestionState
    def save(self, state: IngestionState) -> None
    def mark_processed(self, ledger_id: int) -> None
    def get_last_processed_ledger(self) -> Optional[int]
```

### BackfillCheckpointManager

Manages checkpoint persistence for chunked backfill operations. Stores the last successfully completed chunk position to enable resuming from that point on restart.

#### Class Definition

```python
class BackfillCheckpointManager:
    def __init__(self, checkpoint_path: Optional[str] = None) -> None
    def save(self, last_ledger: int) -> None
    def load(self) -> Optional[int]
    def clear(self) -> None
```

#### Methods

##### __init__()

Initialize the checkpoint manager.

**Parameters:**
- `checkpoint_path` (Optional[str]): Path to the checkpoint file. If None, uses `.astroml_state/backfill_checkpoint.json`.

##### save()

Save checkpoint position atomically.

**Parameters:**
- `last_ledger` (int): The last ledger successfully processed.

##### load()

Load the last checkpoint position.

**Returns:** Optional[int] - The last ledger processed, or None if no checkpoint exists.

##### clear()

Remove the checkpoint file.

**Example:**
```python
from astroml.ingestion.service import BackfillCheckpointManager

# Use default checkpoint path
checkpoint_mgr = BackfillCheckpointManager()

# Save position after processing a chunk
checkpoint_mgr.save(12345)

# Later, resume from checkpoint
last_position = checkpoint_mgr.load()
if last_position is not None:
    start_ledger = last_position + 1
else:
    start_ledger = 1

# Clear checkpoint on completion
checkpoint_mgr.clear()
```

#### Methods

##### load()

Load the current ingestion state from storage.

**Returns:** `IngestionState` object with current state information.

##### save()

Save the ingestion state to storage.

**Parameters:**
- `state` (IngestionState): State object to save

##### mark_processed()

Mark a specific ledger as processed.

**Parameters:**
- `ledger_id` (int): Ledger ID to mark as processed

##### get_last_processed_ledger()

Get the ID of the last processed ledger.

**Returns:** Optional[int] - Last processed ledger ID or None if none processed.

### IngestionState

Data structure representing ingestion state.

#### Class Definition

```python
@dataclass
class IngestionState:
    last_processed_ledger: Optional[int]
    processed_ledgers: Set[int]
    last_updated: datetime
```

## Enhanced Services

### EnhancedIngestionService

Advanced ingestion service with streaming capabilities and enhanced error handling.

#### Class Definition

```python
class EnhancedIngestionService:
    def __init__(
        self,
        state_store: Optional[StateStore] = None,
        config: Optional[IngestionConfig] = None
    ) -> None
    
    async def stream_ingest(
        self,
        start_ledger: int,
        batch_size: int = 100,
        max_concurrent: int = 10
    ) -> AsyncIterator[IngestionResult]
    
    def backfill(
        self,
        start_ledger: int,
        end_ledger: int,
        batch_size: int = 1000,
        parallel_workers: int = 4
    ) -> IngestionResult
```

#### Methods

##### stream_ingest()

Stream ingestion with async processing and concurrency control.

**Parameters:**
- `start_ledger` (int): Starting ledger ID
- `batch_size` (int): Number of ledgers to process in each batch
- `max_concurrent` (int): Maximum concurrent processing tasks

**Returns:** AsyncIterator[IngestionResult] - Stream of ingestion results

**Example:**
```python
service = EnhancedIngestionService()

async for result in service.stream_ingest(
    start_ledger=1000000,
    batch_size=100,
    max_concurrent=10
):
    print(f"Batch processed: {len(result.processed)}")
```

##### backfill()

Perform bulk historical data ingestion with parallel processing.

**Parameters:**
- `start_ledger` (int): Starting ledger ID
- `end_ledger` (int): Ending ledger ID
- `batch_size` (int): Batch size for processing
- `parallel_workers` (int): Number of parallel workers

**Returns:** IngestionResult - Combined results from all batches

**Example:**
```python
service = EnhancedIngestionService()

result = service.backfill(
    start_ledger=1000000,
    end_ledger=2000000,
    batch_size=1000,
    parallel_workers=4
)

print(f"Total processed: {len(result.processed)}")
```

## Stream Processing

### HorizonStream

Real-time streaming of Stellar Horizon data.

#### Class Definition

```python
class HorizonStream:
    def __init__(
        self,
        horizon_url: str,
        network: str = "testnet",
        cursor: Optional[str] = None
    ) -> None
    
    async def stream_transactions(
        self,
        processors: List[Callable[[Transaction], None]]
    ) -> AsyncIterator[Transaction]
    
    async def stream_ledgers(
        self,
        processors: List[Callable[[Ledger], None]]
    ) -> AsyncIterator[Ledger]
```

#### Methods

##### stream_transactions()

Stream real-time transactions from Stellar Horizon.

**Parameters:**
- `processors` (List[Callable]): List of processing functions for each transaction

**Returns:** AsyncIterator[Transaction] - Stream of transactions

**Example:**
```python
stream = HorizonStream("https://horizon-testnet.stellar.org")

async def process_transaction(tx):
    print(f"Processing transaction: {tx.hash}")

async for transaction in stream.stream_transactions([process_transaction]):
    # Process each transaction
    pass
```

##### stream_ledgers()

Stream real-time ledgers from Stellar Horizon.

**Parameters:**
- `processors` (List[Callable]): List of processing functions for each ledger

**Returns:** AsyncIterator[Ledger] - Stream of ledgers

**Example:**
```python
stream = HorizonStream("https://horizon-testnet.stellar.org")

async def process_ledger(ledger):
    print(f"Processing ledger: {ledger.sequence}")

async for ledger in stream.stream_ledgers([process_ledger]):
    # Process each ledger
    pass
```

### HorizonStreamingClient

SSE client for Horizon transaction events, with automatic reconnection.

#### Delivery guarantees

SSE plus automatic reconnection makes delivery **at-least-once**. When a
connection drops, Horizon replays the tail of the page it had already begun
sending, so the same transaction can arrive again — often many times per second
of outage.

That matters downstream. `ClaimService.submit_claim()` keys pending claims by
`claim_reference` and rebuilds the submission, so a replayed transaction resets
`retry_count` to `0` and restarts the retry budget for a claim that is already
in flight. A handler that moves money has the same problem, one tier worse.

`HorizonStreamingClient` drops replays for you. A transaction whose
`paging_token` was already delivered is logged at `DEBUG` and counted in
`duplicates_skipped` instead of reaching the handler again. The cursor still
advances, so suppressing a replay never rewinds the stream.

De-duplication is keyed on `paging_token` and bounded by `dedupe_capacity`
tokens (default 1024), so memory stays constant on a long-running stream.
Transactions with no `paging_token` cannot be identified and are always
delivered.

#### Constructor Parameters

- `base_url` (str): Horizon base URL, must be `http` or `https`
- `endpoint` (str): SSE path, default `/transactions`
- `cursor` (str): Initial paging token, default `"now"`
- `reconnect_delay` (float): Initial reconnect backoff, default `1.0`
- `max_reconnect_delay` (float): Backoff ceiling, default `30.0`
- `dedupe` (bool): Suppress replayed transactions, default `True`
- `dedupe_capacity` (int): Recently delivered tokens to remember, default `1024`
- `logger` (logging.Logger): Logger for disconnect and replay messages

#### Properties

- `cursor` (str): Current paging token, advanced as transactions are read
- `duplicates_skipped` (int): Replays suppressed since construction

#### Example

```python
from astroml.ingestion.horizon_stream import HorizonStreamingClient

client = HorizonStreamingClient(base_url="https://horizon.stellar.org")

async def submit_claim(tx):
    # Reaches this handler once per transaction, replays excluded.
    claim_service.submit_claim(claim_reference=tx["id"], ...)

await client.start(submit_claim)

print(f"suppressed {client.duplicates_skipped} replays")
```

If your handler is already idempotent (writes keyed on the transaction id into
a table with a unique constraint, for example), opt out and take raw
at-least-once delivery:

```python
client = HorizonStreamingClient(dedupe=False)
```

### EnhancedStream

Enhanced streaming with buffering and error recovery.

#### Class Definition

```python
class EnhancedStream:
    def __init__(
        self,
        config: StreamConfig,
        buffer_size: int = 1000,
        retry_attempts: int = 3
    ) -> None
    
    async def start_streaming(
        self,
        processors: List[Callable],
        error_handler: Optional[Callable] = None
    ) -> None
    
    def get_stream_stats(self) -> StreamStats
```

#### Methods

##### start_streaming()

Start enhanced streaming with buffering and error recovery.

**Parameters:**
- `processors` (List[Callable]): Processing functions
- `error_handler` (Optional[Callable]): Error handling function

**Example:**
```python
config = StreamConfig(
    horizon_url="https://horizon-testnet.stellar.org",
    network="testnet"
)

stream = EnhancedStream(config, buffer_size=1000)

async def handle_error(error, context):
    print(f"Error in streaming: {error}")

await stream.start_streaming(
    processors=[process_transaction, update_graph],
    error_handler=handle_error
)
```

## Data Normalization

### Normalizer

Data normalization and cleaning utilities.

#### Class Definition

```python
class Normalizer:
    def __init__(self, config: NormalizationConfig) -> None
    
    def normalize_transaction(self, tx: RawTransaction) -> NormalizedTransaction
    def normalize_ledger(self, ledger: RawLedger) -> NormalizedLedger
    def normalize_account(self, account: RawAccount) -> NormalizedAccount
```

#### Methods

##### normalize_transaction()

Normalize raw transaction data to standard format.

**Parameters:**
- `tx` (RawTransaction): Raw transaction data from Stellar

**Returns:** NormalizedTransaction - Normalized transaction data

**Example:**
```python
normalizer = Normalizer(NormalizationConfig())

raw_tx = fetch_raw_transaction("abc123...")
normalized_tx = normalizer.normalize_transaction(raw_tx)

print(f"Normalized amount: {normalized_tx.amount}")
print(f"Normalized asset: {normalized_tx.asset}")
```

##### normalize_ledger()

Normalize raw ledger data to standard format.

**Parameters:**
- `ledger` (RawLedger): Raw ledger data from Stellar

**Returns:** NormalizedLedger - Normalized ledger data

##### normalize_account()

Normalize raw account data to standard format.

**Parameters:**
- `account` (RawAccount): Raw account data from Stellar

**Returns:** NormalizedAccount - Normalized account data

## Configuration

### IngestionConfig

Configuration for ingestion services.

#### Class Definition

```python
@dataclass
class IngestionConfig:
    batch_size: int = 1000
    max_retries: int = 3
    timeout: int = 30
    stellar_network: str = "testnet"
    horizon_url: str = "https://horizon-testnet.stellar.org"
    parallel_workers: int = 4
    buffer_size: int = 1000
```

### StreamConfig

Configuration for streaming services.

#### Class Definition

```python
@dataclass
class StreamConfig:
    horizon_url: str
    network: str = "testnet"
    cursor: Optional[str] = None
    reconnect_interval: int = 5
    max_reconnect_attempts: int = 10
```

### NormalizationConfig

Configuration for data normalization.

#### Class Definition

```python
@dataclass
class NormalizationConfig:
    standardize_timestamps: bool = True
    validate_addresses: bool = True
    filter_test_operations: bool = True
    min_amount_threshold: float = 0.0
```

## Error Handling

### Custom Exceptions

#### IngestionError

Base exception for ingestion-related errors.

```python
class IngestionError(Exception):
    """Base exception for ingestion operations."""
    pass
```

#### LedgerNotFoundError

Raised when a requested ledger cannot be found.

```python
class LedgerNotFoundError(IngestionError):
    """Raised when a ledger cannot be found."""
    def __init__(self, ledger_id: int):
        self.ledger_id = ledger_id
        super().__init__(f"Ledger {ledger_id} not found")
```

#### StreamError

Raised when streaming operations fail.

```python
class StreamError(IngestionError):
    """Raised when streaming operations fail."""
    pass
```

#### NormalizationError

Raised when data normalization fails.

```python
class NormalizationError(IngestionError):
    """Raised when data normalization fails."""
    pass
```

### Error Handling Patterns

#### Retry Logic

```python
from astroml.ingestion import IngestionService, IngestionError
import time

def ingest_with_retry(service, start_ledger, end_ledger, max_retries=3):
    for attempt in range(max_retries):
        try:
            return service.ingest(start_ledger, end_ledger)
        except IngestionError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
    return None
```

#### Stream Recovery

```python
async def stream_with_recovery(stream, processors, max_retries=3):
    retry_count = 0
    while retry_count < max_retries:
        try:
            async for result in stream.start_streaming(processors):
                yield result
            break  # Success, exit retry loop
        except StreamError as e:
            retry_count += 1
            if retry_count >= max_retries:
                raise
            await asyncio.sleep(retry_count * 5)  # Wait before retry
```

## Performance Optimization

### Batch Processing

```python
from astroml.ingestion import EnhancedIngestionService

# Optimize for large datasets
service = EnhancedIngestionService(
    config=IngestionConfig(
        batch_size=5000,  # Larger batches
        parallel_workers=8,  # More parallelism
        timeout=60  # Longer timeout
    )
)

result = service.backfill(
    start_ledger=1000000,
    end_ledger=5000000,
    batch_size=5000,
    parallel_workers=8
)
```

### Memory Management

```python
# Process in chunks to manage memory
def process_large_range(start_ledger, end_ledger, chunk_size=100000):
    service = IngestionService()

    for chunk_start in range(start_ledger, end_ledger + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_ledger)

        result = service.ingest(chunk_start, chunk_end)

        # Clear memory after each chunk
        del result

        # Force garbage collection if needed
        import gc
        gc.collect()
```

### Checkpointing for Resumable Backfills

For large backfills that may be interrupted, use checkpointing to resume from the last completed chunk:

```python
from astroml.ingestion import IngestionService

service = IngestionService()

# Resumable backfill - if interrupted, restart from last checkpoint
for chunk_summary in service.ingest_backfill_chunked(
    start_ledger=1000000,
    end_ledger=10000000,
    chunk_size=50000,
    resume_from_checkpoint=True,
    fetch_fn=fetch_ledger,
    process_fn=process_ledger
):
    print(f"Progress: {chunk_summary['chunk_end']} processed")
```

**Benefits:**
- Automatically saves progress after each successful chunk
- Resumes from last checkpoint on restart
- No reprocessing of already-completed chunks
- Checkpoint file is atomically written to avoid corruption
- Checkpoint is cleared on successful completion

**Checkpoint file location:**
- Default: `.astroml_state/backfill_checkpoint.json`
- Customizable via `checkpoint_path` parameter
- Contains: `last_ledger` and `updated_at` timestamp

### Streaming Optimization

```python
# Optimize streaming for high throughput
stream = EnhancedStream(
    config=StreamConfig(
        horizon_url="https://horizon.stellar.org",  # Mainnet for production
        network="mainnet"
    ),
    buffer_size=5000,  # Larger buffer
    retry_attempts=5   # More retries for stability
)

# Use efficient processors
def batch_processor(transactions):
    # Process transactions in batches
    for i in range(0, len(transactions), 100):
        batch = transactions[i:i+100]
        process_transaction_batch(batch)
```

## Testing

### Unit Tests

```python
import pytest
from astroml.ingestion import IngestionService, StateStore

class TestIngestionService:
    def test_ingest_single_ledger(self):
        service = IngestionService()
        
        result = service.ingest(
            start_ledger=1000000,
            end_ledger=1000000,
            fetch_fn=lambda x: {"ledger": x},
            process_fn=lambda x, y: None
        )
        
        assert len(result.attempted) == 1
        assert len(result.processed) == 1
        assert len(result.skipped) == 0
    
    def test_skip_processed_ledger(self):
        state_store = StateStore()
        state_store.mark_processed(1000000)
        
        service = IngestionService(state_store)
        
        result = service.ingest(
            start_ledger=1000000,
            end_ledger=1000000
        )
        
        assert len(result.skipped) == 1
        assert len(result.processed) == 0
```

### Integration Tests

```python
import asyncio
import pytest
from astroml.ingestion import EnhancedIngestionService, HorizonStream

class TestEnhancedIngestion:
    @pytest.mark.asyncio
    async def test_stream_ingestion(self):
        service = EnhancedIngestionService()
        
        results = []
        async for result in service.stream_ingest(
            start_ledger=1000000,
            batch_size=10
        ):
            results.append(result)
            if len(results) >= 3:  # Test with limited results
                break
        
        assert len(results) == 3
        assert all(isinstance(r, IngestionResult) for r in results)
```

## Usage Examples

### Complete Ingestion Pipeline

```python
from astroml.ingestion import EnhancedIngestionService, Normalizer
from astroml.ingestion.config import IngestionConfig, NormalizationConfig

# Configuration
ingestion_config = IngestionConfig(
    batch_size=1000,
    parallel_workers=4,
    stellar_network="testnet"
)

normalization_config = NormalizationConfig(
    standardize_timestamps=True,
    validate_addresses=True
)

# Initialize services
service = EnhancedIngestionService(config=ingestion_config)
normalizer = Normalizer(config=normalization_config)

# Define processing functions
def fetch_ledger(ledger_id):
    """Fetch ledger data from Stellar Horizon."""
    import requests
    response = requests.get(f"https://horizon-testnet.stellar.org/ledgers/{ledger_id}")
    return response.json()

def process_ledger(ledger_id, ledger_data):
    """Process and store ledger data."""
    normalized = normalizer.normalize_ledger(ledger_data)
    # Store in database
    store_normalized_ledger(normalized)

# Perform ingestion
result = service.backfill(
    start_ledger=1000000,
    end_ledger=1100000,
    batch_size=1000,
    parallel_workers=4,
    fetch_fn=fetch_ledger,
    process_fn=process_ledger
)

print(f"Ingestion complete: {result}")
```

### Real-time Streaming

```python
import asyncio
from astroml.ingestion import HorizonStream, EnhancedStream

async def realtime_pipeline():
    """Set up real-time ingestion pipeline."""
    
    # Configure stream
    config = StreamConfig(
        horizon_url="https://horizon-testnet.stellar.org",
        network="testnet"
    )
    
    stream = EnhancedStream(config, buffer_size=1000)
    
    # Define processors
    async def process_transaction(tx):
        """Process individual transaction."""
        # Update graph with new transaction
        await update_transaction_graph(tx)
    
    async def update_anomaly_scores(tx):
        """Update anomaly detection scores."""
        scores = await compute_anomaly_scores(tx)
        await store_anomaly_scores(tx.hash, scores)
    
    async def handle_error(error, context):
        """Handle streaming errors."""
        print(f"Stream error: {error}")
        # Send alert or log to monitoring system
    
    # Start streaming
    await stream.start_streaming(
        processors=[process_transaction, update_anomaly_scores],
        error_handler=handle_error
    )

# Run the pipeline
asyncio.run(realtime_pipeline())
```

---

This comprehensive documentation covers all aspects of the data ingestion module, from basic usage to advanced streaming and performance optimization. The ingestion system is designed to be robust, scalable, and easy to integrate into existing ML pipelines.
