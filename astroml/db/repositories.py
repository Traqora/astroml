"""Repository pattern for database access (issue #571)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from astroml.db.models import (
    Account,
    Ledger,
    NormalizedTransaction,
    ProcessedLedger,
    Transaction,
)
from astroml.ingestion.batch import batch_upsert


class LedgerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_range(self, start: int, end: int) -> Sequence[Ledger]:
        stmt = (
            select(Ledger)
            .where(Ledger.sequence >= start)
            .where(Ledger.sequence <= end)
            .order_by(Ledger.sequence)
        )
        return self._session.execute(stmt).scalars().all()

    def get_by_sequence(self, sequence: int) -> Ledger | None:
        stmt = select(Ledger).where(Ledger.sequence == sequence)
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, ledger: Ledger) -> Ledger:
        self._session.add(ledger)
        self._session.flush()
        return ledger

    def get_latest_sequence(self) -> int | None:
        stmt = select(Ledger.sequence).order_by(Ledger.sequence.desc()).limit(1)
        return self._session.execute(stmt).scalar_one_or_none()

    def count(self) -> int:
        from sqlalchemy import func

        return self._session.execute(select(func.count()).select_from(Ledger)).scalar_one()

    def batch_save(self, ledgers: Sequence[Ledger], chunk_size: int = 100) -> int:
        """Save multiple ledgers in chunked batches.

        Args:
            ledgers: Sequence of Ledger instances to save.
            chunk_size: Maximum number of ledgers per commit.

        Returns:
            Total number of ledgers saved.
        """
        return batch_upsert(self._session, ledgers, chunk_size=chunk_size)


class TransactionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_hash(self, hash: str) -> Transaction | None:
        stmt = select(Transaction).where(Transaction.hash == hash)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_ledger_range(self, start: int, end: int) -> Sequence[Transaction]:
        stmt = (
            select(Transaction)
            .where(Transaction.ledger_sequence >= start)
            .where(Transaction.ledger_sequence <= end)
            .order_by(Transaction.created_at)
        )
        return self._session.execute(stmt).scalars().all()

    def get_by_account(self, account_id: str, limit: int = 100) -> Sequence[Transaction]:
        stmt = (
            select(Transaction)
            .where(Transaction.source_account == account_id)
            .order_by(Transaction.created_at.desc())
            .limit(limit)
        )
        return self._session.execute(stmt).scalars().all()

    def save(self, transaction: Transaction) -> Transaction:
        self._session.add(transaction)
        self._session.flush()
        return transaction

    def count_by_ledger(self, ledger_sequence: int) -> int:
        from sqlalchemy import func

        return self._session.execute(
            select(func.count())
            .select_from(Transaction)
            .where(Transaction.ledger_sequence == ledger_sequence)
        ).scalar_one()

    def batch_save(self, transactions: Sequence[Transaction], chunk_size: int = 100) -> int:
        """Save multiple transactions in chunked batches.

        Args:
            transactions: Sequence of Transaction instances to save.
            chunk_size: Maximum number of transactions per commit.

        Returns:
            Total number of transactions saved.
        """
        return batch_upsert(self._session, transactions, chunk_size=chunk_size)


class AccountRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_account_id(self, account_id: str) -> Account | None:
        stmt = select(Account).where(Account.account_id == account_id)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_active_since(self, since: datetime) -> Sequence[Account]:
        stmt = select(Account).where(Account.updated_at >= since).order_by(Account.updated_at)
        return self._session.execute(stmt).scalars().all()

    def save(self, account: Account) -> Account:
        self._session.add(account)
        self._session.flush()
        return account

    def upsert(self, account: Account) -> Account:
        existing = self.get_by_account_id(account.account_id)
        if existing:
            existing.balance = account.balance
            existing.sequence = account.sequence
            existing.home_domain = account.home_domain
            existing.flags = account.flags
            existing.last_modified_ledger = account.last_modified_ledger
            existing.updated_at = account.updated_at
            self._session.flush()
            return existing
        return self.save(account)

    def batch_upsert(self, accounts: Sequence[Account], chunk_size: int = 100) -> int:
        """Upsert multiple accounts in chunked batches.

        Uses the same logic as :meth:`upsert` but commits in chunks
        for better performance during bulk ingestion.

        Args:
            accounts: Sequence of Account instances to upsert.
            chunk_size: Maximum number of accounts per commit.

        Returns:
            Total number of accounts upserted.
        """
        if not accounts:
            return 0

        account_ids = [a.account_id for a in accounts]
        stmt = select(Account).where(Account.account_id.in_(account_ids))
        existing = {a.account_id: a for a in self._session.execute(stmt).scalars().all()}

        to_insert: list[Account] = []
        for account in accounts:
            if account.account_id in existing:
                ex = existing[account.account_id]
                ex.balance = account.balance
                ex.sequence = account.sequence
                ex.home_domain = account.home_domain
                ex.flags = account.flags
                ex.last_modified_ledger = account.last_modified_ledger
                ex.updated_at = account.updated_at
            else:
                to_insert.append(account)

        count = 0
        for i in range(0, len(to_insert), chunk_size):
            chunk = to_insert[i : i + chunk_size]
            for acc in chunk:
                self._session.add(acc)
            self._session.flush()
            self._session.commit()
            count += len(chunk)

        if existing:
            self._session.flush()
            self._session.commit()
            count += len(existing)

        return count


class ProcessedLedgerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_sequence(self, sequence: int) -> ProcessedLedger | None:
        stmt = select(ProcessedLedger).where(ProcessedLedger.ledger_sequence == sequence)
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, processed: ProcessedLedger) -> ProcessedLedger:
        self._session.add(processed)
        self._session.flush()
        return processed

    def is_processed(self, sequence: int) -> bool:
        return self.get_by_sequence(sequence) is not None

    def get_by_status(self, status: str) -> Sequence[ProcessedLedger]:
        stmt = (
            select(ProcessedLedger)
            .where(ProcessedLedger.status == status)
            .order_by(ProcessedLedger.ledger_sequence)
        )
        return self._session.execute(stmt).scalars().all()


class NormalizedTransactionRepository:
    """Idempotent writes for ``normalized_transactions`` (issue #728).

    ``session.merge()`` resolves by primary key, and this table's primary key is
    the surrogate ``id``.  A freshly normalized row has no ``id``, so a retry --
    a replayed Horizon event, a resumed batch after a restart, a page fetched
    twice -- merged as an insert every time and duplicated the activity.  These
    methods key on the natural key the model now carries,
    ``(ledger_sequence, operation_id, hop_index)``, so the same operation
    written twice lands on one row.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_natural_key(
        self,
        ledger_sequence: int,
        operation_id: int,
        hop_index: int = 0,
    ) -> NormalizedTransaction | None:
        stmt = select(NormalizedTransaction).where(
            NormalizedTransaction.ledger_sequence == ledger_sequence,
            NormalizedTransaction.operation_id == operation_id,
            NormalizedTransaction.hop_index == hop_index,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, record: NormalizedTransaction) -> NormalizedTransaction:
        self._session.add(record)
        self._session.flush()
        return record

    def upsert(self, record: NormalizedTransaction) -> NormalizedTransaction:
        """Insert ``record``, or update the row already holding its natural key.

        A record carrying no natural key is inserted as-is: it cannot be
        deduplicated against anything, and silently dropping it would lose
        activity.  Returns the persistent instance (existing or new).
        """
        if not self._has_natural_key(record):
            return self.save(record)

        existing = self.get_by_natural_key(
            record.ledger_sequence, record.operation_id, record.hop_index
        )
        if existing is None:
            return self.save(record)

        self._apply(existing, record)
        self._session.flush()
        return existing

    def batch_upsert(
        self,
        records: Sequence[NormalizedTransaction],
        chunk_size: int = 100,
    ) -> int:
        """Upsert many records, resolving their natural keys in one query.

        Returns the number of rows written (inserted plus updated).
        """
        if not records:
            return 0

        keyed = [r for r in records if self._has_natural_key(r)]
        unkeyed = [r for r in records if not self._has_natural_key(r)]

        existing = self._existing_by_natural_key(keyed)

        updated = 0
        to_insert: list[NormalizedTransaction] = []
        for record in keyed:
            key = (record.ledger_sequence, record.operation_id, record.hop_index)
            found = existing.get(key)
            if found is None:
                to_insert.append(record)
            else:
                self._apply(found, record)
                updated += 1

        inserted = 0
        for i in range(0, len(to_insert), chunk_size):
            chunk = to_insert[i : i + chunk_size]
            for record in chunk:
                self._session.add(record)
            self._session.flush()
            self._session.commit()
            inserted += len(chunk)

        for record in unkeyed:
            self._session.add(record)
            inserted += 1

        if updated or unkeyed:
            self._session.flush()
            self._session.commit()

        return inserted + updated

    def count(self) -> int:
        return self._session.query(NormalizedTransaction).count()

    @staticmethod
    def _has_natural_key(record: NormalizedTransaction) -> bool:
        return record.ledger_sequence is not None and record.operation_id is not None

    def _existing_by_natural_key(
        self,
        records: Sequence[NormalizedTransaction],
    ) -> dict[tuple[int, int, int], NormalizedTransaction]:
        """Fetch every stored row matching one of ``records``' natural keys."""
        if not records:
            return {}

        operation_ids = {r.operation_id for r in records}
        stmt = select(NormalizedTransaction).where(
            NormalizedTransaction.operation_id.in_(operation_ids)
        )
        rows = self._session.execute(stmt).scalars().all()
        return {
            (row.ledger_sequence, row.operation_id, row.hop_index): row
            for row in rows
            if row.ledger_sequence is not None and row.operation_id is not None
        }

    @staticmethod
    def _apply(target: NormalizedTransaction, source: NormalizedTransaction) -> None:
        """Copy the replayable fields of ``source`` onto ``target``.

        The natural key itself is never rewritten: a row is identified by it, so
        an incoming record with the same key is the same activity restated.
        """
        target.transaction_hash = source.transaction_hash
        target.sender = source.sender
        target.receiver = source.receiver
        target.asset = source.asset
        target.amount = source.amount
        target.timestamp = source.timestamp
