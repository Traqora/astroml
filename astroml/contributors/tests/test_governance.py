"""Tests for astroml.contributors.governance."""
from __future__ import annotations

import pytest

from astroml.contributors.governance import (
    CONTRIBUTION_THRESHOLDS,
    ROLE_PERMISSIONS,
    ROLES,
    ContributorRecord,
    GovernanceSystem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gov() -> GovernanceSystem:
    """Return a fresh GovernanceSystem for each test."""
    return GovernanceSystem()


# ---------------------------------------------------------------------------
# add_contributor
# ---------------------------------------------------------------------------


class TestAddContributor:
    def test_default_role_is_contributor(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice")
        assert record.role == "contributor"

    def test_returns_contributor_record(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice")
        assert isinstance(record, ContributorRecord)
        assert record.github_login == "alice"

    def test_custom_role(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("bob", role="reviewer")
        assert record.role == "reviewer"

    def test_invalid_role_raises(self, gov: GovernanceSystem) -> None:
        with pytest.raises(ValueError, match="Unknown role"):
            gov.add_contributor("carol", role="admin")

    def test_idempotent_second_call(self, gov: GovernanceSystem) -> None:
        first = gov.add_contributor("alice")
        second = gov.add_contributor("alice")
        assert first is second

    def test_get_role_unknown_returns_none(self, gov: GovernanceSystem) -> None:
        assert gov.get_role("nobody") is None

    def test_get_role_known(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice")
        assert gov.get_role("alice") == "contributor"


# ---------------------------------------------------------------------------
# has_permission
# ---------------------------------------------------------------------------


class TestHasPermission:
    def test_contributor_can_create_pr(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        assert gov.has_permission("alice", "create_pr") is True

    def test_contributor_cannot_merge_pr(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        assert gov.has_permission("alice", "merge_pr") is False

    def test_reviewer_can_review_pr(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("bob", role="reviewer")
        assert gov.has_permission("bob", "review_pr") is True

    def test_reviewer_cannot_merge_pr(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("bob", role="reviewer")
        assert gov.has_permission("bob", "merge_pr") is False

    def test_maintainer_can_merge_pr(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("carol", role="maintainer")
        assert gov.has_permission("carol", "merge_pr") is True

    def test_maintainer_can_manage_roles(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("carol", role="maintainer")
        assert gov.has_permission("carol", "manage_roles") is True

    def test_all_roles_can_vote(self, gov: GovernanceSystem) -> None:
        for role in ROLES:
            gov.add_contributor(f"user_{role}", role=role)
            assert gov.has_permission(f"user_{role}", "vote") is True

    def test_unknown_contributor_has_no_permission(self, gov: GovernanceSystem) -> None:
        assert gov.has_permission("ghost", "vote") is False


# ---------------------------------------------------------------------------
# advance_role
# ---------------------------------------------------------------------------


class TestAdvanceRole:
    def test_no_advance_below_threshold(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice")
        result = gov.advance_role("alice")
        assert result is None
        assert gov.get_role("alice") == "contributor"

    def test_advance_to_reviewer_at_threshold(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice")
        record.contribution_count = CONTRIBUTION_THRESHOLDS["reviewer"]  # 20
        result = gov.advance_role("alice")
        assert result == "reviewer"
        assert gov.get_role("alice") == "reviewer"

    def test_advance_to_maintainer_at_threshold(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice")
        record.contribution_count = CONTRIBUTION_THRESHOLDS["maintainer"]  # 50
        result = gov.advance_role("alice")
        assert result == "maintainer"
        assert gov.get_role("alice") == "maintainer"

    def test_no_advance_when_already_maintainer(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice", role="maintainer")
        record.contribution_count = 200
        result = gov.advance_role("alice")
        assert result is None

    def test_increment_contributions_triggers_advance(self, gov: GovernanceSystem) -> None:
        record = gov.add_contributor("alice")
        record.contribution_count = CONTRIBUTION_THRESHOLDS["reviewer"] - 1  # 19

        count = gov.increment_contributions("alice")
        assert count == CONTRIBUTION_THRESHOLDS["reviewer"]
        assert gov.get_role("alice") == "reviewer"

    def test_increment_contributions_unknown_raises(self, gov: GovernanceSystem) -> None:
        with pytest.raises(KeyError, match="not found"):
            gov.increment_contributions("nobody")

    def test_advance_unknown_contributor_returns_none(self, gov: GovernanceSystem) -> None:
        assert gov.advance_role("nobody") is None


# ---------------------------------------------------------------------------
# vote
# ---------------------------------------------------------------------------


class TestVote:
    def test_contributor_can_vote(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        result = gov.vote("alice", "proposal-1", True)
        assert result["proposal_id"] == "proposal-1"
        assert result["votes_for"] == 1
        assert result["votes_against"] == 0

    def test_vote_counts_multiple_voters(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        gov.add_contributor("bob", role="reviewer")
        gov.vote("alice", "proposal-1", True)
        result = gov.vote("bob", "proposal-1", False)
        assert result["votes_for"] == 1
        assert result["votes_against"] == 1

    def test_voter_can_change_vote(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        gov.vote("alice", "proposal-1", True)
        result = gov.vote("alice", "proposal-1", False)
        # The contributor changed their vote to against
        assert result["votes_for"] == 0
        assert result["votes_against"] == 1

    def test_vote_requires_permission(self, gov: GovernanceSystem) -> None:
        """A role without 'vote' permission should be blocked — but per spec
        all defined roles have vote permission.  We test the guard by patching
        a record directly to a non-existent role that has no permissions."""
        gov.add_contributor("alice")
        # Force an invalid role to simulate the permission check path.
        gov._records["alice"].role = "observer"  # not in ROLE_PERMISSIONS
        with pytest.raises(PermissionError, match="does not have the 'vote' permission"):
            gov.vote("alice", "proposal-1", True)

    def test_vote_unknown_contributor_raises(self, gov: GovernanceSystem) -> None:
        with pytest.raises(KeyError, match="not found"):
            gov.vote("ghost", "proposal-1", True)


# ---------------------------------------------------------------------------
# get_governance_summary
# ---------------------------------------------------------------------------


class TestGovernanceSummary:
    def test_empty_system(self, gov: GovernanceSystem) -> None:
        summary = gov.get_governance_summary()
        assert summary["total_contributors"] == 0
        assert summary["by_role"] == {"maintainer": 0, "contributor": 0, "reviewer": 0}
        assert summary["recent_votes"] == []

    def test_counts_by_role(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("a", role="contributor")
        gov.add_contributor("b", role="contributor")
        gov.add_contributor("c", role="reviewer")
        gov.add_contributor("d", role="maintainer")

        summary = gov.get_governance_summary()
        assert summary["total_contributors"] == 4
        assert summary["by_role"]["contributor"] == 2
        assert summary["by_role"]["reviewer"] == 1
        assert summary["by_role"]["maintainer"] == 1

    def test_recent_votes_listed(self, gov: GovernanceSystem) -> None:
        gov.add_contributor("alice", role="contributor")
        gov.vote("alice", "prop-A", True)
        gov.vote("alice", "prop-B", False)

        summary = gov.get_governance_summary()
        assert set(summary["recent_votes"]) == {"prop-A", "prop-B"}
