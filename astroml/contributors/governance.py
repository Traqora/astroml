"""Contributor governance system for AstroML.

Defines roles, permissions, and contribution-based role advancement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLES: set[str] = {"maintainer", "contributor", "reviewer"}

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "maintainer": {"merge_pr", "close_issue", "assign_issue", "vote", "manage_roles"},
    "reviewer": {"review_pr", "vote", "close_issue"},
    "contributor": {"create_pr", "comment", "vote"},
}

# Number of merged PRs required to hold (or advance to) each role.
CONTRIBUTION_THRESHOLDS: dict[str, int] = {
    "contributor": 5,
    "reviewer": 20,
    "maintainer": 50,
}

# Ordered from least to most senior for advancement logic.
_ROLE_ORDER: list[str] = ["contributor", "reviewer", "maintainer"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ContributorRecord:
    """Persistent state for a single contributor."""

    github_login: str
    role: str
    contribution_count: int = 0
    joined_at: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Governance system
# ---------------------------------------------------------------------------


class GovernanceSystem:
    """Manage contributor roles, permissions, and voting."""

    def __init__(self) -> None:
        self._records: dict[str, ContributorRecord] = {}
        self._votes: dict[str, dict[str, bool]] = {}

    # ------------------------------------------------------------------
    # Contributor management
    # ------------------------------------------------------------------

    def add_contributor(
        self,
        github_login: str,
        role: str = "contributor",
    ) -> ContributorRecord:
        """Add a new contributor or return the existing record.

        Args:
            github_login: The contributor's GitHub username.
            role: Initial role.  Defaults to ``"contributor"``.

        Returns:
            The (possibly existing) ``ContributorRecord``.

        Raises:
            ValueError: If ``role`` is not in ``ROLES``.
        """
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}. Valid roles: {sorted(ROLES)}")

        if github_login not in self._records:
            self._records[github_login] = ContributorRecord(
                github_login=github_login,
                role=role,
            )
        return self._records[github_login]

    def get_role(self, github_login: str) -> Optional[str]:
        """Return the contributor's current role, or ``None`` if unknown."""
        record = self._records.get(github_login)
        return record.role if record else None

    # ------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------

    def has_permission(self, github_login: str, permission: str) -> bool:
        """Return whether the contributor holds the given permission.

        Returns ``False`` for unknown contributors.
        """
        role = self.get_role(github_login)
        if role is None:
            return False
        return permission in ROLE_PERMISSIONS.get(role, set())

    # ------------------------------------------------------------------
    # Role advancement
    # ------------------------------------------------------------------

    def advance_role(self, github_login: str) -> Optional[str]:
        """Advance the contributor's role if contribution thresholds are met.

        Roles advance in order: contributor -> reviewer -> maintainer.

        Args:
            github_login: The contributor's GitHub username.

        Returns:
            The new role string if the role was advanced, otherwise ``None``.
        """
        record = self._records.get(github_login)
        if record is None:
            return None

        current_index = _ROLE_ORDER.index(record.role) if record.role in _ROLE_ORDER else -1
        new_role: Optional[str] = None

        # Walk up from the next tier
        for candidate in _ROLE_ORDER[current_index + 1:]:
            threshold = CONTRIBUTION_THRESHOLDS[candidate]
            if record.contribution_count >= threshold:
                new_role = candidate
            else:
                break

        if new_role is not None and new_role != record.role:
            record.role = new_role
            return new_role

        return None

    def increment_contributions(self, github_login: str) -> int:
        """Increment the contributor's merged-PR count and attempt role advancement.

        Args:
            github_login: The contributor's GitHub username.

        Returns:
            The updated contribution count.

        Raises:
            KeyError: If the contributor is not registered.
        """
        record = self._records.get(github_login)
        if record is None:
            raise KeyError(f"Contributor {github_login!r} not found. Call add_contributor first.")

        record.contribution_count += 1
        self.advance_role(github_login)
        return record.contribution_count

    # ------------------------------------------------------------------
    # Voting
    # ------------------------------------------------------------------

    def vote(self, github_login: str, proposal_id: str, vote: bool) -> dict:
        """Record a vote on a proposal.

        Args:
            github_login: The contributor casting the vote.
            proposal_id: Unique identifier for the proposal.
            vote: ``True`` for in favour, ``False`` for against.

        Returns:
            A dict with ``proposal_id``, ``votes_for``, and ``votes_against``.

        Raises:
            PermissionError: If the contributor does not have the ``"vote"``
                permission.
            KeyError: If the contributor is not registered.
        """
        if github_login not in self._records:
            raise KeyError(f"Contributor {github_login!r} not found. Call add_contributor first.")

        if not self.has_permission(github_login, "vote"):
            raise PermissionError(
                f"{github_login!r} (role={self.get_role(github_login)!r}) "
                "does not have the 'vote' permission."
            )

        if proposal_id not in self._votes:
            self._votes[proposal_id] = {}
        self._votes[proposal_id][github_login] = vote

        ballots = self._votes[proposal_id]
        votes_for = sum(1 for v in ballots.values() if v)
        votes_against = sum(1 for v in ballots.values() if not v)

        return {
            "proposal_id": proposal_id,
            "votes_for": votes_for,
            "votes_against": votes_against,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_governance_summary(self) -> dict:
        """Return a high-level governance snapshot.

        Returns:
            A dict with ``total_contributors``, ``by_role`` counts, and
            ``recent_votes`` (all recorded proposal IDs).
        """
        by_role: dict[str, int] = {role: 0 for role in ROLES}
        for record in self._records.values():
            if record.role in by_role:
                by_role[record.role] += 1

        return {
            "total_contributors": len(self._records),
            "by_role": by_role,
            "recent_votes": list(self._votes.keys()),
        }
