"""Tests for the Redis persistence health check (issue #956)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import redis

from astroml.cache.persistence_health import (
    PersistenceCheckError,
    check_persistence_health,
)


def _make_client(
    save_points: str,
    appendonly: str,
    rdb_last_bgsave_status: str | None = "ok",
    aof_last_bgrewrite_status: str | None = "ok",
) -> MagicMock:
    """Build a MagicMock standing in for a redis.Redis client."""
    client = MagicMock(spec=redis.Redis)

    def config_get(pattern: str) -> dict[str, str]:
        if pattern == "save":
            return {"save": save_points}
        if pattern == "appendonly":
            return {"appendonly": appendonly}
        return {}

    client.config_get.side_effect = config_get

    info: dict[str, str] = {}
    if rdb_last_bgsave_status is not None:
        info["rdb_last_bgsave_status"] = rdb_last_bgsave_status
    if appendonly == "yes" and aof_last_bgrewrite_status is not None:
        info["aof_last_bgrewrite_status"] = aof_last_bgrewrite_status
    client.info.return_value = info

    return client


class TestCheckPersistenceHealth:
    """Tests for check_persistence_health."""

    def test_healthy_with_rdb_only(self):
        """RDB save points configured, AOF off, last save ok -> healthy."""
        client = _make_client(save_points="3600 1 300 100", appendonly="no")

        result = check_persistence_health(client)

        assert result.healthy is True
        assert result.rdb_enabled is True
        assert result.aof_enabled is False
        assert result.last_bgsave_status == "ok"
        assert result.last_aof_rewrite_status is None
        assert result.issues == []

    def test_healthy_with_aof_only(self):
        """AOF enabled, no RDB save points, last rewrite ok -> healthy."""
        client = _make_client(save_points="", appendonly="yes")

        result = check_persistence_health(client)

        assert result.healthy is True
        assert result.rdb_enabled is False
        assert result.aof_enabled is True
        assert result.last_aof_rewrite_status == "ok"
        assert result.issues == []

    def test_healthy_with_both_enabled(self):
        """Both RDB and AOF enabled and healthy -> healthy."""
        client = _make_client(save_points="900 1", appendonly="yes")

        result = check_persistence_health(client)

        assert result.healthy is True
        assert result.rdb_enabled is True
        assert result.aof_enabled is True

    def test_unhealthy_when_neither_persistence_mechanism_enabled(self):
        """No save points and appendonly off -> unhealthy with a clear reason.

        This is the exact silent-data-loss scenario the check exists to
        catch: a Redis instance that looks fine (responds to PING, serves
        reads/writes) but loses everything on restart.
        """
        client = _make_client(save_points="", appendonly="no")

        result = check_persistence_health(client)

        assert result.healthy is False
        assert result.rdb_enabled is False
        assert result.aof_enabled is False
        assert len(result.issues) == 1
        assert "no persistence mechanism enabled" in result.issues[0]

    def test_unhealthy_when_last_bgsave_failed(self):
        """RDB enabled but the last background save errored -> unhealthy."""
        client = _make_client(
            save_points="3600 1",
            appendonly="no",
            rdb_last_bgsave_status="err",
        )

        result = check_persistence_health(client)

        assert result.healthy is False
        assert any("background save failed" in issue for issue in result.issues)

    def test_unhealthy_when_last_aof_rewrite_failed(self):
        """AOF enabled but the last background rewrite errored -> unhealthy."""
        client = _make_client(
            save_points="",
            appendonly="yes",
            aof_last_bgrewrite_status="err",
        )

        result = check_persistence_health(client)

        assert result.healthy is False
        assert any("rewrite failed" in issue for issue in result.issues)

    def test_reports_both_issues_when_both_present(self):
        """Both mechanisms enabled and both failing -> both issues reported."""
        client = _make_client(
            save_points="3600 1",
            appendonly="yes",
            rdb_last_bgsave_status="err",
            aof_last_bgrewrite_status="err",
        )

        result = check_persistence_health(client)

        assert result.healthy is False
        assert len(result.issues) == 2

    def test_fresh_instance_with_no_bgsave_yet_is_not_treated_as_a_failure(self):
        """A brand new server with no bgsave attempt yet must not be flagged.

        redis reports no rdb_last_bgsave_status field at all until the
        first save happens; that is not the same as a failed save.
        """
        client = _make_client(
            save_points="3600 1",
            appendonly="no",
            rdb_last_bgsave_status=None,
        )

        result = check_persistence_health(client)

        assert result.healthy is True
        assert result.last_bgsave_status is None

    def test_raises_persistence_check_error_on_connection_failure(self):
        """A Redis error while querying config surfaces as PersistenceCheckError.

        Distinguishes "the check itself could not run" from "the check ran
        and found a problem", since the two should page differently.
        """
        client = MagicMock(spec=redis.Redis)
        client.config_get.side_effect = redis.ConnectionError("connection refused")

        with pytest.raises(PersistenceCheckError, match="could not query"):
            check_persistence_health(client)

    def test_raises_persistence_check_error_when_config_command_disabled(self):
        """A managed Redis with CONFIG disabled surfaces the same error type."""
        client = MagicMock(spec=redis.Redis)
        client.config_get.side_effect = redis.ResponseError("ERR unknown command 'CONFIG'")

        with pytest.raises(PersistenceCheckError):
            check_persistence_health(client)
