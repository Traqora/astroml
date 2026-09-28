"""Mocked Slack-integration tests for the chat and ingestion notifier paths (#993).

``astroml/chat/slack.py`` is the chat side's only outbound integration and the
thing :class:`~astroml.ingestion.service.IngestionService` accepts as its
``notifier``, yet nothing exercised it: the module sat at 39% line coverage and
the chunked-backfill failure path never reached it at all.

Every test here patches ``astroml.chat.slack.requests.post`` so no network call
is ever attempted.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from astroml.chat.slack import SlackConfig, SlackIntegration
from astroml.ingestion.service import IngestionService
from astroml.ingestion.state import StateStore

_POST = "astroml.chat.slack.requests.post"


@pytest.fixture
def store(tmp_path) -> StateStore:
    """A state store rooted in ``tmp_path`` so tests never touch real state."""
    return StateStore(path=str(tmp_path / "state" / "ingest.json"))


@pytest.fixture
def slack() -> SlackIntegration:
    """A webhook-backed Slack integration with a non-routable test URL."""
    return SlackIntegration(SlackConfig(webhook_url="https://hooks.slack.test/abc"))


def _response(status_code: int = 200, **kwargs) -> MagicMock:
    return MagicMock(status_code=status_code, **kwargs)


def _boom_on(bad_ledger: int):
    def process(ledger_id, _payload):
        if ledger_id == bad_ledger:
            raise RuntimeError(f"boom at {bad_ledger}")

    return process


# ---------------------------------------------------------------------------
# SlackConfig / environment
# ---------------------------------------------------------------------------


class TestSlackConfig:
    def test_defaults(self):
        config = SlackConfig()
        assert config.webhook_url is None
        assert config.bot_token is None
        assert config.channel == "#support"

    def test_from_env_reads_all_variables(self, monkeypatch):
        from astroml.chat.slack import SlackIntegration as Integration

        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.test/env")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        monkeypatch.setenv("SLACK_CHANNEL", "#ops")

        config = Integration.create_slack_config_from_env()

        assert config.webhook_url == "https://hooks.slack.test/env"
        assert config.bot_token == "xoxb-test"
        assert config.channel == "#ops"

    def test_from_env_falls_back_to_defaults(self, monkeypatch):
        for name in ("SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN", "SLACK_CHANNEL"):
            monkeypatch.delenv(name, raising=False)

        config = SlackIntegration.create_slack_config_from_env()

        assert config.webhook_url is None
        assert config.bot_token is None
        assert config.channel == "#support"


# ---------------------------------------------------------------------------
# send_webhook
# ---------------------------------------------------------------------------


class TestSendWebhook:
    def test_posts_message_and_returns_true(self, slack):
        with patch(_POST, return_value=_response(200)) as post:
            assert slack.send_webhook("all good") is True

        post.assert_called_once()
        args, kwargs = post.call_args
        assert args[0] == "https://hooks.slack.test/abc"
        assert kwargs["json"] == {"text": "all good"}
        assert kwargs["timeout"] == 10

    def test_missing_webhook_url_skips_http(self, caplog):
        integration = SlackIntegration(SlackConfig())

        with patch(_POST) as post:
            with caplog.at_level(logging.WARNING):
                assert integration.send_webhook("nobody listening") is False

        post.assert_not_called()
        assert any("webhook URL not configured" in r.message for r in caplog.records)

    def test_non_200_returns_false_and_logs(self, slack, caplog):
        with patch(_POST, return_value=_response(500)):
            with caplog.at_level(logging.ERROR):
                assert slack.send_webhook("server on fire") is False

        assert any("500" in r.message for r in caplog.records)

    def test_transport_error_returns_false_and_logs(self, slack, caplog):
        with patch(_POST, side_effect=ConnectionError("dns failure")):
            with caplog.at_level(logging.ERROR):
                assert slack.send_webhook("unreachable") is False

        assert any("dns failure" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Convenience notifiers
# ---------------------------------------------------------------------------


class TestConvenienceNotifiers:
    @pytest.mark.parametrize(
        ("method", "args", "expected"),
        [
            ("notify_new_chat", ("alice", "sess-1"), ["alice", "sess-1"]),
            ("notify_agent_assigned", ("bob", "sess-2"), ["bob", "sess-2"]),
            ("notify_offline_message", ("carol", "carol@example.com"), ["carol"]),
        ],
    )
    def test_delegates_to_webhook_with_details(self, slack, method, args, expected):
        with patch(_POST, return_value=_response(200)) as post:
            assert getattr(slack, method)(*args) is True

        text = post.call_args.kwargs["json"]["text"]
        for fragment in expected:
            assert fragment in text

    def test_offline_message_includes_email(self, slack):
        with patch(_POST, return_value=_response(200)) as post:
            slack.notify_offline_message("carol", "carol@example.com")

        assert "carol@example.com" in post.call_args.kwargs["json"]["text"]

    def test_convenience_notifier_reports_webhook_failure(self, slack):
        with patch(_POST, return_value=_response(404)):
            assert slack.notify_new_chat("alice", "sess-1") is False


# ---------------------------------------------------------------------------
# send_direct_message
# ---------------------------------------------------------------------------


class TestSendDirectMessage:
    @pytest.fixture
    def dm(self) -> SlackIntegration:
        return SlackIntegration(
            SlackConfig(webhook_url="https://hooks.slack.test/abc", bot_token="xoxb-test")
        )

    def test_missing_bot_token_skips_http(self, slack, caplog):
        with patch(_POST) as post:
            with caplog.at_level(logging.WARNING):
                assert slack.send_direct_message("U123", "hi") is False

        post.assert_not_called()
        assert any("bot token not configured" in r.message for r in caplog.records)

    def test_posts_authorized_message(self, dm):
        response = _response(200)
        response.json.return_value = {"ok": True}

        with patch(_POST, return_value=response) as post:
            assert dm.send_direct_message("U123", "hi there") is True

        args, kwargs = post.call_args
        assert args[0] == "https://slack.com/api/chat.postMessage"
        assert kwargs["headers"]["Authorization"] == "Bearer xoxb-test"
        assert kwargs["json"] == {"channel": "U123", "text": "hi there"}
        assert kwargs["timeout"] == 10

    def test_api_level_error_returns_false(self, dm, caplog):
        response = _response(200)
        response.json.return_value = {"ok": False, "error": "channel_not_found"}

        with patch(_POST, return_value=response):
            with caplog.at_level(logging.ERROR):
                assert dm.send_direct_message("U123", "hi") is False

        assert any("channel_not_found" in r.message for r in caplog.records)

    def test_http_error_returns_false(self, dm, caplog):
        with patch(_POST, return_value=_response(429)):
            with caplog.at_level(logging.ERROR):
                assert dm.send_direct_message("U123", "hi") is False

        assert any("429" in r.message for r in caplog.records)

    def test_transport_error_returns_false(self, dm, caplog):
        with patch(_POST, side_effect=TimeoutError("read timed out")):
            with caplog.at_level(logging.ERROR):
                assert dm.send_direct_message("U123", "hi") is False

        assert any("read timed out" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# IngestionService notifier contract
# ---------------------------------------------------------------------------


class TestIngestionNotifierContract:
    def test_ingest_failure_posts_to_slack(self, store, slack):
        service = IngestionService(state_store=store, notifier=slack.send_webhook)

        with patch(_POST, return_value=_response(200)) as post:
            result = service.ingest(start_ledger=1, end_ledger=3, process_fn=_boom_on(2))

        assert result.errors == ["boom at 2"]
        text = post.call_args.kwargs["json"]["text"]
        assert "boom at 2" in text
        assert "1 processed" in text

    def test_failed_webhook_does_not_change_ingest_result(self, store, slack):
        """A webhook that returns False must not alter the IngestionResult."""
        with patch(_POST, return_value=_response(500)):
            with_notifier = IngestionService(state_store=store, notifier=slack.send_webhook).ingest(
                start_ledger=1, end_ledger=3, process_fn=_boom_on(1)
            )

        without_notifier = IngestionService(state_store=store).ingest(
            start_ledger=1, end_ledger=3, process_fn=_boom_on(1)
        )

        assert with_notifier.errors == ["boom at 1"]
        assert with_notifier.attempted == without_notifier.attempted
        assert with_notifier.processed == without_notifier.processed
        assert with_notifier.skipped == without_notifier.skipped

    def test_backfill_chunk_failure_posts_to_slack(self, store, slack):
        """Regression (#993): a failed backfill chunk must reach the notifier.

        ``ingest_backfill_chunked`` catches the per-chunk exception and keeps
        going, so before this the notifier was never called: a backfill where
        every chunk failed produced only ``errors: 1`` summaries and no alert.
        """
        service = IngestionService(state_store=store, notifier=slack.send_webhook)

        with patch(_POST, return_value=_response(200)) as post:
            summaries = list(
                service.ingest_backfill_chunked(
                    start_ledger=1,
                    end_ledger=4,
                    chunk_size=2,
                    process_fn=_boom_on(1),
                )
            )

        assert [s["errors"] for s in summaries] == [1, 0]
        assert post.call_count == 1
        assert "boom at 1" in post.call_args.kwargs["json"]["text"]

    def test_successful_backfill_does_not_notify(self, store, slack):
        service = IngestionService(state_store=store, notifier=slack.send_webhook)

        with patch(_POST) as post:
            summaries = list(service.ingest_backfill_chunked(1, 4, chunk_size=2))

        assert all(s["errors"] == 0 for s in summaries)
        post.assert_not_called()

    def test_notifier_error_never_breaks_a_backfill_chunk(self, store, caplog):
        """A dead notifier is logged, not raised — the chunk still reports."""
        notifier = MagicMock(side_effect=ConnectionError("slack down"))
        service = IngestionService(state_store=store, notifier=notifier)

        with caplog.at_level(logging.WARNING):
            summaries = list(
                service.ingest_backfill_chunked(1, 2, chunk_size=2, process_fn=_boom_on(1))
            )

        assert summaries[0]["errors"] == 1
        notifier.assert_called_once()
        assert any("notifier raised" in r.message for r in caplog.records)

    def test_backfill_validates_arguments_before_notifying(self, store, slack):
        service = IngestionService(state_store=store, notifier=slack.send_webhook)

        with patch(_POST) as post:
            with pytest.raises(ValueError):
                list(service.ingest_backfill_chunked(10, 1))
            with pytest.raises(ValueError):
                list(service.ingest_backfill_chunked(1, 10, chunk_size=0))

        post.assert_not_called()
