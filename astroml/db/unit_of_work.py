"""Unit of Work pattern for transaction management (issue #571)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy.orm import Session

from astroml.db.repositories import AccountRepository, LedgerRepository, ProcessedLedgerRepository, TransactionRepository


class UnitOfWork:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._ledgers: Optional[LedgerRepository] = None
        self._transactions: Optional[TransactionRepository] = None
        self._accounts: Optional[AccountRepository] = None
        self._processed_ledgers: Optional[ProcessedLedgerRepository] = None

    @property
    def ledgers(self) -> LedgerRepository:
        if self._ledgers is None:
            self._ledgers = LedgerRepository(self._session)
        return self._ledgers

    @property
    def transactions(self) -> TransactionRepository:
        if self._transactions is None:
            self._transactions = TransactionRepository(self._session)
        return self._transactions

    @property
    def accounts(self) -> AccountRepository:
        if self._accounts is None:
            self._accounts = AccountRepository(self._session)
        return self._accounts

    @property
    def processed_ledgers(self) -> ProcessedLedgerRepository:
        if self._processed_ledgers is None:
            self._processed_ledgers = ProcessedLedgerRepository(self._session)
        return self._processed_ledgers

    def commit(self) -> None:
        self._session.commit()

    def rollback(self) -> None:
        self._session.rollback()

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> UnitOfWork:
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        if exc_type is not None:
            self.rollback()
        self.close()


@contextmanager
def unit_of_work(session: Session) -> Iterator[UnitOfWork]:
    uow = UnitOfWork(session)
    try:
        yield uow
        uow.commit()
    except Exception:
        uow.rollback()
        raise
    finally:
        uow.close()
