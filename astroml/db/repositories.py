"""Repository pattern for database access (issue #571)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from astroml.db.models import Account, Asset, GraphEdge, Ledger, Operation, ProcessedLedger, Transaction


class LedgerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_range(self, start: int, end: int) -> Sequence[Ledger]:
        stmt = select(Ledger).where(Ledger.sequence >= start).where(Ledger.sequence <= end).order_by(Ledger.sequence)
        return self._session.execute(stmt).scalars().all()

    def get_by_sequence(self, sequence: int) -> Optional[Ledger]:
        stmt = select(Ledger).where(Ledger.sequence == sequence)
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, ledger: Ledger) -> Ledger:
        self._session.add(ledger)
        self._session.flush()
        return ledger

    def get_latest_sequence(self) -> Optional[int]:
        stmt = select(Ledger.sequence).order_by(Ledger.sequence.desc()).limit(1)
        return self._session.execute(stmt).scalar_one_or_none()

    def count(self) -> int:
        from sqlalchemy import func
        return self._session.execute(select(func.count()).select_from(Ledger)).scalar_one()

    def delete(self, ledger: Ledger) -> None:
        self._session.delete(ledger)


class TransactionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_hash(self, hash: str) -> Optional[Transaction]:
        stmt = select(Transaction).where(Transaction.hash == hash)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_ledger_range(self, start: int, end: int) -> Sequence[Transaction]:
        stmt = select(Transaction).where(Transaction.ledger_sequence >= start).where(Transaction.ledger_sequence <= end).order_by(Transaction.created_at)
        return self._session.execute(stmt).scalars().all()

    def get_by_account(self, account_id: str, limit: int = 100) -> Sequence[Transaction]:
        stmt = select(Transaction).where(Transaction.source_account == account_id).order_by(Transaction.created_at.desc()).limit(limit)
        return self._session.execute(stmt).scalars().all()

    def save(self, transaction: Transaction) -> Transaction:
        self._session.add(transaction)
        self._session.flush()
        return transaction

    def count_by_ledger(self, ledger_sequence: int) -> int:
        from sqlalchemy import func
        return self._session.execute(select(func.count()).select_from(Transaction).where(Transaction.ledger_sequence == ledger_sequence)).scalar_one()


class AccountRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_account_id(self, account_id: str) -> Optional[Account]:
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


class ProcessedLedgerRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_sequence(self, sequence: int) -> Optional[ProcessedLedger]:
        stmt = select(ProcessedLedger).where(ProcessedLedger.ledger_sequence == sequence)
        return self._session.execute(stmt).scalar_one_or_none()

    def save(self, processed: ProcessedLedger) -> ProcessedLedger:
        self._session.add(processed)
        self._session.flush()
        return processed

    def is_processed(self, sequence: int) -> bool:
        return self.get_by_sequence(sequence) is not None

    def get_by_status(self, status: str) -> Sequence[ProcessedLedger]:
        stmt = select(ProcessedLedger).where(ProcessedLedger.status == status).order_by(ProcessedLedger.ledger_sequence)
        return self._session.execute(stmt).scalars().all()
