"""Quick start module for AstroML.

Provides a single entry point to wire sample data through the complete
ingestion → graph → train pipeline to produce baseline results.

Usage:
    python -m astroml.quick_start
    # or
    make quickstart
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .benchmarking.config import BenchmarkConfig, ModelConfig, DataConfig, TrainingConfig
from .benchmarking.core import BenchmarkResult, ModelBenchmark
from .db.schema import Ledger, Transaction, Operation, Account, Asset
from .db.session import get_session
from .features.graph.snapshot import Edge, window_snapshot
from .features.graph_validation import validate_graph
from .ingestion.service import IngestionService
from .ingestion.state import StateStore
from .models import LinkPredictor
from .tasks.link_prediction_task import LinkPredictionTask
from .training.temporal_split import temporal_graph_split

logger = logging.getLogger(__name__)


class QuickStartConfig:
    """Configuration for quick start demo."""
    
    # Sample data parameters
    NUM_SAMPLE_LEDGERS = 100
    NUM_ACCOUNTS = 50
    NUM_ASSETS = 5
    TRANSACTIONS_PER_LEDGER = 20
    
    # Training parameters
    TRAIN_EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 0.01
    RANDOM_SEED = 42
    
    # Output
    OUTPUT_DIR = Path("./benchmark_results/quickstart")
    STATE_DIR = Path("./.astroml_state_quickstart")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    logger.info(f"Random seeds set to {seed}")


def generate_sample_ledgers(
    session,
    num_ledgers: int = QuickStartConfig.NUM_SAMPLE_LEDGERS,
    num_accounts: int = QuickStartConfig.NUM_ACCOUNTS,
    num_assets: int = QuickStartConfig.NUM_ASSETS,
    txns_per_ledger: int = QuickStartConfig.TRANSACTIONS_PER_LEDGER,
) -> tuple[List[int], List[str]]:
    """Generate synthetic sample ledgers and transactions.
    
    Returns:
        Tuple of (ledger_sequences, account_ids)
    """
    logger.info(f"Generating {num_ledgers} sample ledgers...")
    
    # Create sample assets
    asset_codes = [f"ASSET{i}" for i in range(num_assets)]
    assets = []
    for code in asset_codes:
        asset = Asset(code=code, issuer="GBRPYHIL2CI3WHZDTOOQFC6EB4RRJC3XNSOLXAUJVLW7IJVUFSZ7ZZXZ")
        session.add(asset)
        assets.append(asset)
    session.commit()
    
    # Create sample accounts
    account_ids = [f"GACCOUNT{i:06d}" for i in range(num_accounts)]
    accounts = []
    for account_id in account_ids:
        account = Account(
            id=account_id,
            balance=1000.0,
            sequence=0,
            flags=0,
            last_modified_ledger=1,
        )
        session.add(account)
        accounts.append(account)
    session.commit()
    
    # Create sample ledgers and transactions
    ledger_sequences = []
    base_time = datetime.utcnow() - timedelta(days=num_ledgers)
    
    for ledger_seq in range(1, num_ledgers + 1):
        ledger = Ledger(
            sequence=ledger_seq,
            hash=f"hash_{ledger_seq:08d}",
            prev_hash=f"hash_{ledger_seq-1:08d}" if ledger_seq > 1 else None,
            closed_at=base_time + timedelta(seconds=ledger_seq * 5),
            successful_transaction_count=txns_per_ledger,
            failed_transaction_count=0,
            operation_count=txns_per_ledger,
        )
        session.add(ledger)
        session.flush()
        ledger_sequences.append(ledger_seq)
        
        # Create transactions for this ledger
        for txn_idx in range(txns_per_ledger):
            src_account = random.choice(account_ids)
            dst_account = random.choice(account_ids)
            
            # Avoid self-loops
            while dst_account == src_account:
                dst_account = random.choice(account_ids)
            
            txn = Transaction(
                hash=f"txn_{ledger_seq}_{txn_idx}",
                ledger_sequence=ledger_seq,
                source_account=src_account,
                created_at=ledger.closed_at,
                fee=100,
                memo=f"sample_txn_{txn_idx}",
            )
            session.add(txn)
            session.flush()
            
            # Create operation (edge)
            asset = random.choice(assets)
            operation = Operation(
                transaction_hash=txn.hash,
                ledger_sequence=ledger_seq,
                type="payment",
                source_account=src_account,
                destination_account=dst_account,
                amount=random.uniform(1, 100),
                asset_code=asset.code,
                asset_issuer=asset.issuer,
                created_at=ledger.closed_at,
            )
            session.add(operation)
    
    session.commit()
    logger.info(f"Generated {len(ledger_sequences)} ledgers with {len(account_ids)} accounts")
    
    return ledger_sequences, account_ids


def build_sample_graph(
    session,
    ledger_sequences: List[int],
    account_ids: List[str],
) -> tuple[List[Edge], dict]:
    """Build a sample transaction graph from generated ledgers.
    
    Returns:
        Tuple of (edges, node_index)
    """
    logger.info("Building sample transaction graph...")
    
    # Query all operations
    operations = session.query(Operation).all()
    
    # Convert to Edge objects
    edges = []
    for op in operations:
        edge = Edge(
            src=op.source_account,
            dst=op.destination_account,
            timestamp=op.created_at.timestamp(),
            asset=op.asset_code,
            amount=float(op.amount),
        )
        edges.append(edge)
    
    # Create node index
    node_index = {account_id: idx for idx, account_id in enumerate(account_ids)}
    
    logger.info(f"Built graph with {len(edges)} edges and {len(node_index)} nodes")
    
    # Validate graph
    try:
        stats = validate_graph(edges, node_index)
        logger.info(f"Graph validation: {stats}")
    except Exception as e:
        logger.warning(f"Graph validation warning: {e}")
    
    return edges, node_index


def train_baseline_model(
    edges: List[Edge],
    node_index: dict,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """Train a baseline link prediction model.
    
    Returns:
        BenchmarkResult with training metrics
    """
    logger.info("Training baseline link prediction model...")
    
    if config is None:
        config = BenchmarkConfig(
            model_name="LinkPredictor",
            model_params={"hidden_dim": 64, "num_layers": 2},
            epochs=QuickStartConfig.TRAIN_EPOCHS,
            batch_size=QuickStartConfig.BATCH_SIZE,
            learning_rate=QuickStartConfig.LEARNING_RATE,
            random_seed=QuickStartConfig.RANDOM_SEED,
        )
    
    # Set seeds for reproducibility
    set_random_seeds(config.random_seed)
    
    # Split edges temporally
    split_result = temporal_graph_split(
        edges,
        train_ratio=0.8,
        time_attr="timestamp",
    )
    
    train_edges = split_result.train_edges
    test_edges = split_result.test_edges
    
    logger.info(f"Split: {len(train_edges)} train edges, {len(test_edges)} test edges")
    
    # Create task and train
    task = LinkPredictionTask(
        context_edges=train_edges,
        future_edges=test_edges,
        node_index=node_index,
        model_params=config.model_params,
        device=config.device,
    )
    
    result = task.train(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
    )
    
    logger.info(f"Training complete. Best metrics: {result.metrics}")
    
    return result


def save_benchmark_config(
    config: BenchmarkConfig,
    result: BenchmarkResult,
    output_dir: Path,
) -> None:
    """Save benchmark configuration and results for reproducibility.
    
    Stores:
    - config.json: Full benchmark configuration with seeds
    - result.json: Benchmark results with metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_dict = asdict(config)
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    logger.info(f"Saved config to {config_path}")
    
    # Save result
    result_dict = asdict(result)
    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    logger.info(f"Saved result to {result_path}")
    
    # Save metadata
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "config_file": str(config_path),
        "result_file": str(result_path),
        "random_seed": config.random_seed,
        "model_name": config.model_name,
        "epochs": config.epochs,
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def run_quickstart() -> int:
    """Run the complete quick start pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    from astroml.utils.logging import configure_logging

    configure_logging()
    
    logger.info("=" * 80)
    logger.info("AstroML Quick Start: Ingestion → Graph → Train Pipeline")
    logger.info("=" * 80)
    
    try:
        # Set random seeds
        set_random_seeds(QuickStartConfig.RANDOM_SEED)
        
        # Step 1: Generate sample data
        logger.info("\n[Step 1/5] Generating sample ledger data...")
        session = get_session()
        ledger_sequences, account_ids = generate_sample_ledgers(
            session,
            num_ledgers=QuickStartConfig.NUM_SAMPLE_LEDGERS,
            num_accounts=QuickStartConfig.NUM_ACCOUNTS,
            num_assets=QuickStartConfig.NUM_ASSETS,
            txns_per_ledger=QuickStartConfig.TRANSACTIONS_PER_LEDGER,
        )
        
        # Step 2: Build graph
        logger.info("\n[Step 2/5] Building transaction graph...")
        edges, node_index = build_sample_graph(session, ledger_sequences, account_ids)
        
        # Step 3: Create benchmark config
        logger.info("\n[Step 3/5] Creating benchmark configuration...")
        config = BenchmarkConfig(
            model_name="LinkPredictor",
            model_params={"hidden_dim": 64, "num_layers": 2},
            epochs=QuickStartConfig.TRAIN_EPOCHS,
            batch_size=QuickStartConfig.BATCH_SIZE,
            learning_rate=QuickStartConfig.LEARNING_RATE,
            random_seed=QuickStartConfig.RANDOM_SEED,
            output_dir=str(QuickStartConfig.OUTPUT_DIR),
        )
        
        # Step 4: Train model
        logger.info("\n[Step 4/5] Training baseline model...")
        result = train_baseline_model(edges, node_index, config)
        
        # Step 5: Save results
        logger.info("\n[Step 5/5] Saving benchmark results...")
        save_benchmark_config(config, result, QuickStartConfig.OUTPUT_DIR)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Quick start completed successfully!")
        logger.info(f"Results saved to: {QuickStartConfig.OUTPUT_DIR}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}", exc_info=True)
        return 1
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(run_quickstart())
