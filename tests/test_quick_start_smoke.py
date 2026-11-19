"""Smoke tests for the quick-start orchestration."""

from astroml import quick_start
from astroml import cli
from unittest import mock
import pytest


def test_run_quickstart_returns_failure_when_session_creation_fails(monkeypatch):
    """Session setup failures should return the documented failure exit code."""

    def fail_get_session():
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(quick_start, "get_session", fail_get_session)
    monkeypatch.setattr("astroml.utils.logging.configure_logging", lambda: None)

    assert quick_start.run_quickstart() == 1


def test_run_quickstart_returns_success_for_completed_pipeline(monkeypatch):
    """A completed synthetic pipeline should return the success exit code."""
    fake_session = mock.Mock()
    monkeypatch.setattr(quick_start, "get_session", lambda: fake_session)
    monkeypatch.setattr("astroml.utils.logging.configure_logging", lambda: None)
    monkeypatch.setattr(
        quick_start,
        "generate_sample_ledgers",
        lambda session, **kwargs: ([1], ["GACCOUNT"]),
    )
    monkeypatch.setattr(
        quick_start,
        "build_sample_graph",
        lambda session, ledgers, accounts: ([], {}),
    )
    monkeypatch.setattr(quick_start, "train_baseline_model", lambda edges, nodes, config: mock.Mock())
    monkeypatch.setattr(quick_start, "save_benchmark_config", lambda config, result, output: None)

    assert quick_start.run_quickstart() == 0
    fake_session.close.assert_called_once_with()


def test_quickstart_cli_rejects_zero_ledgers() -> None:
    """Quick-start numeric boundaries should fail clearly instead of corrupting a run."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["quickstart", "--num-ledgers", "0"])
    assert exc.value.code == 2


def test_quickstart_module_applies_custom_parameters(monkeypatch):
    """The module entrypoint should support the documented custom parameters."""
    run = mock.Mock(return_value=0)
    monkeypatch.setattr(quick_start, "run_quickstart", run)

    assert quick_start.main(
        ["--num-ledgers", "200", "--num-accounts", "100", "--epochs", "20", "--seed", "7"]
    ) == 0
    assert quick_start.QuickStartConfig.NUM_SAMPLE_LEDGERS == 200
    assert quick_start.QuickStartConfig.NUM_ACCOUNTS == 100
    assert quick_start.QuickStartConfig.TRAIN_EPOCHS == 20
    assert quick_start.QuickStartConfig.RANDOM_SEED == 7
