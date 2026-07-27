"""Tests for tool permission system."""

import pytest
from astroml.llm.tools.permissions import PermissionChecker, PermissionDenied


class TestPermissionChecker:
    def setup_method(self):
        self.checker = PermissionChecker()

    def test_no_rules_allows_all(self):
        self.checker.check("any_tool", "user_1")

    def test_allow_all_users(self):
        self.checker.allow("restricted_tool")
        self.checker.check("restricted_tool", "user_1")
        self.checker.check("restricted_tool", "user_2")

    def test_allow_specific_user(self):
        self.checker.allow("restricted_tool", "user_1")
        self.checker.check("restricted_tool", "user_1")
        with pytest.raises(PermissionDenied):
            self.checker.check("restricted_tool", "user_2")

    def test_deny_after_allow(self):
        self.checker.allow("tool", "user_1")
        self.checker.check("tool", "user_1")
        self.checker.deny("tool", "user_1")
        with pytest.raises(PermissionDenied):
            self.checker.check("tool", "user_1")
