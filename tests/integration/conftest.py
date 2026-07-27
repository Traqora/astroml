"""Shared fixtures for integration tests.

This module provides fixtures for setting up test databases,
sample data, and common test scenarios for integration testing.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from astroml.db.schema import (
    Account,
    Asset,
    Effect,
    GraphAccount,
    GraphEdge,
    Ledger,
    NormalizedTransaction,
    Operation,
    Transaction,
    Base,
)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def test_db_url(tmp_path: Path) -> str:
    """Provide an in-memory SQLite database URL for testing."""
    return f"sqlite:///{tmp_path / 'test.db'}"


@pytest.fixture(scope="function")
def test_engine(test_db_url: str):
    """Create a test database engine."""
    engine = create_engine(test_db_url, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine) -> Session:
    """Create a test database session."""
    factory = sessionmaker(bind=test_engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture(scope="function")
def mock_config(tmp_path: Path):
    """Create a mock configuration file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    config = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "astroml_test",
            "user": "test_user",
            "password": "test_pass",
        },
        "horizon": {
            "url": "https://horizon-testnet.stellar.org",
        },
    }
    
    config_file = config_dir / "database.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    # Change to temp directory for the test
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield tmp_path
    os.chdir(original_cwd)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ledger_data() -> List[Dict[str, Any]]:
    """Sample ledger data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "sequence": 1000,
            "hash": "a" * 64,
            "prev_hash": "b" * 64,
            "closed_at": base_time,
            "successful_transaction_count": 5,
            "failed_transaction_count": 0,
            "operation_count": 10,
            "total_coins": 1000000000.0,
            "fee_pool": 1000000.0,
            "base_fee_in_stroops": 100,
            "protocol_version": 20,
        },
        {
            "sequence": 1001,
            "hash": "c" * 64,
            "prev_hash": "a" * 64,
            "closed_at": base_time + timedelta(seconds=5),
            "successful_transaction_count": 3,
            "failed_transaction_count": 1,
            "operation_count": 8,
            "total_coins": 1000000005.0,
            "fee_pool": 1000005.0,
            "base_fee_in_stroops": 100,
            "protocol_version": 20,
        },
    ]


@pytest.fixture
def sample_transaction_data() -> List[Dict[str, Any]]:
    """Sample transaction data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "hash": "tx1" + "a" * 60,
            "ledger_sequence": 1000,
            "source_account": "G" + "A" * 55,
            "created_at": base_time,
            "fee": 100,
            "operation_count": 2,
            "successful": True,
            "memo_type": "none",
            "memo": None,
        },
        {
            "hash": "tx2" + "b" * 60,
            "ledger_sequence": 1000,
            "source_account": "G" + "B" * 55,
            "created_at": base_time + timedelta(seconds=1),
            "fee": 200,
            "operation_count": 1,
            "successful": True,
            "memo_type": "text",
            "memo": "test",
        },
        {
            "hash": "tx3" + "c" * 60,
            "ledger_sequence": 1001,
            "source_account": "G" + "C" * 55,
            "created_at": base_time + timedelta(seconds=6),
            "fee": 150,
            "operation_count": 3,
            "successful": False,
            "memo_type": "none",
            "memo": None,
        },
    ]


@pytest.fixture
def sample_operation_data() -> List[Dict[str, Any]]:
    """Sample operation data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "transaction_hash": "tx1" + "a" * 60,
            "application_order": 0,
            "type": "payment",
            "source_account": "G" + "A" * 55,
            "destination_account": "G" + "B" * 55,
            "amount": 100.0,
            "asset_code": "XLM",
            "asset_issuer": None,
            "created_at": base_time,
            "details": {"type": "payment"},
        },
        {
            "transaction_hash": "tx1" + "a" * 60,
            "application_order": 1,
            "type": "payment",
            "source_account": "G" + "A" * 55,
            "destination_account": "G" + "C" * 55,
            "amount": 50.0,
            "asset_code": "USDC",
            "asset_issuer": "G" + "D" * 55,
            "created_at": base_time,
            "details": {"type": "payment"},
        },
        {
            "transaction_hash": "tx2" + "b" * 60,
            "application_order": 0,
            "type": "create_account",
            "source_account": "G" + "B" * 55,
            "destination_account": "G" + "E" * 55,
            "amount": None,
            "asset_code": None,
            "asset_issuer": None,
            "created_at": base_time + timedelta(seconds=1),
            "details": {"type": "create_account", "starting_balance": "100.0"},
        },
    ]


@pytest.fixture
def sample_account_data() -> List[Dict[str, Any]]:
    """Sample account data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "account_id": "G" + "A" * 55,
            "balance": 1000.0,
            "sequence": 100,
            "home_domain": "example.com",
            "flags": 0,
            "last_modified_ledger": 1000,
            "created_at": base_time - timedelta(days=30),
            "updated_at": base_time,
        },
        {
            "account_id": "G" + "B" * 55,
            "balance": 500.0,
            "sequence": 50,
            "home_domain": None,
            "flags": 1,
            "last_modified_ledger": 1000,
            "created_at": base_time - timedelta(days=15),
            "updated_at": base_time,
        },
    ]


@pytest.fixture
def sample_asset_data() -> List[Dict[str, Any]]:
    """Sample asset data for testing."""
    return [
        {
            "asset_type": "native",
            "asset_code": "XLM",
            "asset_issuer": None,
            "first_seen_ledger": 1000,
        },
        {
            "asset_type": "credit_alphanum4",
            "asset_code": "USDC",
            "asset_issuer": "G" + "D" * 55,
            "first_seen_ledger": 1000,
        },
    ]


@pytest.fixture
def sample_effect_data() -> List[Dict[str, Any]]:
    """Sample effect data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "account": "G" + "A" * 55,
            "type": "account_debited",
            "amount": -100.0,
            "asset_code": "XLM",
            "asset_issuer": None,
            "destination_account": None,
            "created_at": base_time,
            "details": {"effect_type": "account_debited"},
        },
        {
            "account": "G" + "B" * 55,
            "type": "account_credited",
            "amount": 100.0,
            "asset_code": "XLM",
            "asset_issuer": None,
            "destination_account": None,
            "created_at": base_time,
            "details": {"effect_type": "account_credited"},
        },
    ]


@pytest.fixture
def sample_graph_edges() -> List[Dict[str, Any]]:
    """Sample graph edge data for testing."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    return [
        {
            "edge_type": "transaction",
            "source_account_id": 1,
            "destination_account_id": 2,
            "asset_id": 1,
            "occurred_at": base_time,
            "ledger_sequence": 1000,
            "event_index": 0,
            "transaction_hash": "tx1" + "a" * 60,
            "external_event_id": "evt1",
            "amount": 100.0,
            "status": "completed",
        },
        {
            "edge_type": "payment",
            "source_account_id": 2,
            "destination_account_id": 3,
            "asset_id": 2,
            "occurred_at": base_time + timedelta(seconds=1),
            "ledger_sequence": 1000,
            "event_index": 1,
            "transaction_hash": "tx2" + "b" * 60,
            "external_event_id": "evt2",
            "amount": 50.0,
            "status": "completed",
        },
    ]


# ---------------------------------------------------------------------------
# Populated database fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_test_db(
    test_session: Session,
    sample_ledger_data: List[Dict[str, Any]],
    sample_transaction_data: List[Dict[str, Any]],
    sample_operation_data: List[Dict[str, Any]],
    sample_account_data: List[Dict[str, Any]],
    sample_asset_data: List[Dict[str, Any]],
    sample_effect_data: List[Dict[str, Any]],
) -> Session:
    """Populate test database with sample data."""
    # Add ledgers
    for ledger_data in sample_ledger_data:
        ledger = Ledger(**ledger_data)
        test_session.add(ledger)
    
    # Add assets
    for asset_data in sample_asset_data:
        asset = Asset(**asset_data)
        test_session.add(asset)
    
    test_session.flush()
    
    # Add accounts
    for account_data in sample_account_data:
        account = Account(**account_data)
        test_session.add(account)
    
    # Add transactions
    for tx_data in sample_transaction_data:
        transaction = Transaction(**tx_data)
        test_session.add(transaction)
    
    test_session.flush()
    
    # Add operations
    for op_data in sample_operation_data:
        operation = Operation(**op_data)
        test_session.add(operation)
    
    # Add effects
    for effect_data in sample_effect_data:
        effect = Effect(**effect_data)
        test_session.add(effect)
    
    test_session.commit()
    yield test_session
    test_session.rollback()


@pytest.fixture
def populated_graph_db(
    test_session: Session,
    sample_asset_data: List[Dict[str, Any]],
    sample_graph_edges: List[Dict[str, Any]],
) -> Session:
    """Populate test database with graph data."""
    # Add assets
    for asset_data in sample_asset_data:
        asset = Asset(**asset_data)
        test_session.add(asset)
    
    test_session.flush()
    
    # Add graph accounts
    accounts = [
        GraphAccount(
            id=1,
            account_address="G" + "A" * 55,
            account_type="user",
            first_seen_at=datetime(2024, 1, 1),
            last_seen_at=datetime(2024, 1, 2),
        ),
        GraphAccount(
            id=2,
            account_address="G" + "B" * 55,
            account_type="user",
            first_seen_at=datetime(2024, 1, 1),
            last_seen_at=datetime(2024, 1, 2),
        ),
        GraphAccount(
            id=3,
            account_address="G" + "C" * 55,
            account_type="user",
            first_seen_at=datetime(2024, 1, 1),
            last_seen_at=datetime(2024, 1, 2),
        ),
    ]
    for account in accounts:
        test_session.add(account)
    
    test_session.flush()
    
    # Add graph edges
    for edge_data in sample_graph_edges:
        edge = GraphEdge(**edge_data)
        test_session.add(edge)
    
    test_session.commit()
    yield test_session
    test_session.rollback()


# ---------------------------------------------------------------------------
# Synthetic fraud data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_fraud_patterns() -> Dict[str, Any]:
    """Synthetic fraud pattern configurations for testing."""
    return {
        "sybil_clusters": [
            {
                "cluster_id": "cluster_1",
                "accounts": [f"G{'A' * i}{'B' * (55-i)}" for i in range(5)],
                "coordinator": "G" + "X" * 55,
                "behavior": "circular_transactions",
            }
        ],
        "wash_trading_loops": [
            {
                "loop_id": "loop_1",
                "accounts": [f"G{'C' * i}{'D' * (55-i)}" for i in range(3)],
                "asset": "USDC",
                "frequency": "high",
            }
        ],
    }


@pytest.fixture
def fraud_labels() -> np.ndarray:
    """Sample fraud labels for testing."""
    np.random.seed(42)
    # 10% fraud rate
    labels = np.zeros(1000)
    fraud_indices = np.random.choice(1000, size=100, replace=False)
    labels[fraud_indices] = 1
    return labels


@pytest.fixture
def fraud_scores() -> np.ndarray:
    """Sample fraud scores for testing."""
    np.random.seed(42)
    scores = np.random.beta(2, 5, 1000)
    return scores


# ---------------------------------------------------------------------------
# ML fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_node_features() -> Dict[str, np.ndarray]:
    """Sample node features for ML testing."""
    np.random.seed(42)
    features = {
        f"node_{i}": np.random.randn(16).astype(np.float32)
        for i in range(10)
    }
    return features


@pytest.fixture
def sample_edge_list() -> List[tuple]:
    """Sample edge list for graph testing."""
    edges = [
        ("node_0", "node_1", 1.0, 1000.0),
        ("node_1", "node_2", 0.5, 2000.0),
        ("node_2", "node_3", 2.0, 3000.0),
        ("node_3", "node_4", 1.5, 4000.0),
        ("node_4", "node_0", 0.8, 5000.0),
    ]
    return edges


@pytest.fixture
def sample_training_data() -> tuple:
    """Sample training data for model testing."""
    np.random.seed(42)
    num_samples = 100
    num_features = 16
    
    X = np.random.randn(num_samples, num_features).astype(np.float32)
    y = np.random.randint(0, 2, num_samples)
    
    return X, y


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    return output_dir


# ---------------------------------------------------------------------------
# Mock Horizon API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_horizon_response():
    """Mock Horizon API response data."""
    return {
        "hash": "x" * 64,
        "ledger": 1000,
        "source_account": "G" + "A" * 55,
        "created_at": "2024-01-01T00:00:00Z",
        "fee_charged": 100,
        "operation_count": 2,
        "successful": True,
        "memo_type": "none",
        "paging_token": "12345",
    }
