"""Schema context injection for NL-to-SQL translation."""
from __future__ import annotations

from typing import Dict, Any, List


def get_db_schema_context() -> str:
    """Return database schema information formatted for prompt context injection."""
    return """
Database Schema Tables:
1. ledgers
   - sequence (INTEGER, PK): The ledger sequence number.
   - hash (VARCHAR(64), UNIQUE): Ledger hash.
   - closed_at (DATETIME): Timestamp when ledger closed.
   - operation_count (INTEGER): Number of operations in ledger.
   
2. transactions
   - hash (VARCHAR(64), PK): Transaction hash.
   - ledger_sequence (INTEGER, FK to ledgers.sequence): The sequence number of the ledger this transaction belongs to.
   - source_account (VARCHAR(56)): The account that initiated the transaction.
   - fee_charged (BIGINT): Fee charged in stroops.
   - created_at (DATETIME): Timestamp when transaction was created.

3. operations
   - id (BIGINT, PK): Unique operation identifier.
   - transaction_hash (VARCHAR(64), FK to transactions.hash): Associated transaction.
   - source_account (VARCHAR(56)): Account executing the operation.
   - type (VARCHAR(32)): Type of operation (e.g. 'payment', 'create_account', 'manage_buy_offer').
   - details (JSONB): JSON metadata specific to the operation type (contains amount, asset_code, asset_issuer, etc.).
   
4. accounts
   - account_id (VARCHAR(56), PK): The Stellar public key of the account.
   - balance (NUMERIC): Account balance in XLM.
   - sequence (BIGINT): Account sequence number.
   - risk_score (FLOAT): Account fraud risk score (0.0 to 1.0).
   - updated_at (DATETIME): Last known update time.

5. assets
   - asset_id (INTEGER, PK): Unique asset identifier.
   - code (VARCHAR(12)): Asset code (e.g., 'USDC', 'XLM').
   - issuer (VARCHAR(56)): Issuer account public key.
"""


def get_few_shot_examples() -> List[Dict[str, str]]:
    """Return few-shot NL-to-SQL query mappings for prompt injection."""
    return [
        {
            "nl": "Show me all accounts with balance > 1000 XLM in the last 7 days",
            "sql": "SELECT account_id, balance, updated_at FROM accounts WHERE balance > 1000 AND updated_at >= NOW() - INTERVAL '7 days';"
        },
        {
            "nl": "Top 10 accounts by transaction volume this month",
            "sql": "SELECT source_account, COUNT(hash) AS tx_count FROM transactions WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) GROUP BY source_account ORDER BY tx_count DESC LIMIT 10;"
        },
        {
            "nl": "Show fraud alerts for accounts with risk score > 0.8",
            "sql": "SELECT account_id, risk_score FROM accounts WHERE risk_score > 0.8 ORDER BY risk_score DESC;"
        }
    ]
