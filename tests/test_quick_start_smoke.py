"""Smoke tests for the quick-start orchestration."""

from astroml import quick_start


def test_run_quickstart_returns_failure_when_session_creation_fails(monkeypatch):
    """Session setup failures should return the documented failure exit code."""

    def fail_get_session():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(quick_start, "get_session", fail_get_session)
    monkeypatch.setattr("astroml.utils.logging.configure_logging", lambda: None)

    assert quick_start.run_quickstart() == 1
