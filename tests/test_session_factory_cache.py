"""Session factory caching regression tests (issue #982)."""

import pytest

from astroml.db import session as session_mod


@pytest.fixture
def fake_engine(monkeypatch):
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")
    calls = {"engine": 0, "factory": 0}
    real_sessionmaker = session_mod.sessionmaker

    def fake_get_engine():
        calls["engine"] += 1
        return engine

    def counting_sessionmaker(*args, **kwargs):
        calls["factory"] += 1
        return real_sessionmaker(*args, **kwargs)

    monkeypatch.setattr(session_mod, "get_engine", fake_get_engine)
    monkeypatch.setattr(session_mod, "sessionmaker", counting_sessionmaker)
    session_mod.get_session_factory.cache_clear()
    yield engine, calls
    session_mod.get_session_factory.cache_clear()


def test_factory_built_once_across_sessions(fake_engine):
    engine, calls = fake_engine
    sessions = [session_mod.get_session() for _ in range(5)]
    try:
        assert calls["factory"] == 1
        assert calls["engine"] == 1
        assert all(s.get_bind() is engine for s in sessions)
    finally:
        for s in sessions:
            s.close()


def test_each_call_returns_distinct_session(fake_engine):
    a, b = session_mod.get_session(), session_mod.get_session()
    try:
        assert a is not b
    finally:
        a.close()
        b.close()


def test_cache_clear_rebuilds_factory(fake_engine):
    _, calls = fake_engine
    session_mod.get_session().close()
    session_mod.get_session_factory.cache_clear()
    session_mod.get_session().close()
    assert calls["factory"] == 2
