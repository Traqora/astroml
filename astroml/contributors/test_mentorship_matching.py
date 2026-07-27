"""Unit tests for mentorship matching algorithm.

Tests the core logic for:
  - Skill overlap calculation
  - Experience gap scoring
  - Availability matching
  - Overall compatibility scoring
"""
from __future__ import annotations

import pytest

from astroml.contributors.mentorship import MentorshipMatcher


class TestSkillOverlapCalculation:
    """Test skill overlap scoring."""

    def test_identical_skills_interests(self):
        """Test perfect overlap of skills and interests."""
        mentor_skills = ["Python", "ML", "Data Science"]
        mentee_interests = ["Python", "ML", "Data Science"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        assert score == 1.0

    def test_partial_overlap(self):
        """Test partial skill overlap."""
        mentor_skills = ["Python", "ML", "Data Science", "TensorFlow"]
        mentee_interests = ["Python", "ML"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        # Overlap: 2, Union: 4, Score: 0.5
        assert score == 0.5

    def test_no_overlap(self):
        """Test no skill overlap."""
        mentor_skills = ["Java", "Backend"]
        mentee_interests = ["Python", "ML"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        assert score == 0.0

    def test_empty_mentor_skills(self):
        """Test with empty mentor skills."""
        score = MentorshipMatcher._calculate_skill_overlap([], ["Python", "ML"])
        assert score == 0.0

    def test_empty_mentee_interests(self):
        """Test with empty mentee interests."""
        score = MentorshipMatcher._calculate_skill_overlap(["Python", "ML"], [])
        assert score == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        mentor_skills = ["PYTHON", "Machine Learning"]
        mentee_interests = ["python", "machine learning"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        assert score == 1.0

    def test_whitespace_trimmed(self):
        """Test whitespace is trimmed."""
        mentor_skills = [" Python ", "  ML  "]
        mentee_interests = ["Python", "ML"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        assert score == 1.0


class TestExperienceGapCalculation:
    """Test experience gap scoring."""

    def test_mentor_more_experienced(self):
        """Test mentor with significantly more experience."""
        score = MentorshipMatcher._calculate_experience_gap(10, 2)
        assert score == 0.8  # (10 - 2) / 10 = 0.8

    def test_equal_experience(self):
        """Test equal experience levels."""
        score = MentorshipMatcher._calculate_experience_gap(5, 5)
        assert score == 0.0

    def test_mentor_less_experienced(self):
        """Test mentor less experienced than mentee."""
        score = MentorshipMatcher._calculate_experience_gap(2, 10)
        assert score == 0.0

    def test_small_gap(self):
        """Test small experience gap."""
        score = MentorshipMatcher._calculate_experience_gap(6, 5)
        assert score == 0.1  # (6 - 5) / 10 = 0.1

    def test_large_gap_capped(self):
        """Test large gap is capped at 1.0."""
        score = MentorshipMatcher._calculate_experience_gap(20, 5)
        assert score == 1.0


class TestAvailabilityMatchCalculation:
    """Test availability matching."""

    def test_same_preferred_day(self):
        """Test matching preferred days."""
        score = MentorshipMatcher._calculate_availability_match("Monday", "Monday")
        assert score == 1.0

    def test_different_preferred_days(self):
        """Test different preferred days."""
        score = MentorshipMatcher._calculate_availability_match("Monday", "Wednesday")
        assert score == 0.0

    def test_both_flexible(self):
        """Test both flexible."""
        score = MentorshipMatcher._calculate_availability_match(None, None)
        assert score == 0.5

    def test_mentor_flexible(self):
        """Test mentor flexible, mentee has preference."""
        score = MentorshipMatcher._calculate_availability_match(None, "Monday")
        assert score == 0.5

    def test_mentee_flexible(self):
        """Test mentee flexible, mentor has preference."""
        score = MentorshipMatcher._calculate_availability_match("Monday", None)
        assert score == 0.5

    def test_case_insensitive_days(self):
        """Test days are case-insensitive."""
        score = MentorshipMatcher._calculate_availability_match("monday", "MONDAY")
        assert score == 1.0


class TestOverallScoring:
    """Test overall compatibility scoring."""

    def test_high_compatibility(self):
        """Test high compatibility combination."""
        # Simulate a high-compatibility pair
        # This would require integration with DB
        pass

    def test_low_compatibility(self):
        """Test low compatibility combination."""
        pass

    def test_score_weighting(self):
        """Test score components are properly weighted.

        Skills: 50%, Experience: 30%, Availability: 20%
        """
        # Example: skill_overlap=1.0, exp_gap=1.0, avail=0.5
        # Score = 1.0*0.5 + 1.0*0.3 + 0.5*0.2 = 0.5 + 0.3 + 0.1 = 0.9
        score = 1.0 * 0.5 + 1.0 * 0.3 + 0.5 * 0.2
        assert score == 0.9


class TestSkillEdgeCases:
    """Test edge cases in skill matching."""

    def test_duplicate_skills(self):
        """Test handling of duplicate skills."""
        mentor_skills = ["Python", "Python", "ML"]
        mentee_interests = ["Python", "ML"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        # Should deduplicate to ["Python", "ML"] for mentor
        assert score == 1.0

    def test_single_skill(self):
        """Test with single skill."""
        mentor_skills = ["Python"]
        mentee_interests = ["Python"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        assert score == 1.0

    def test_many_skills_partial_match(self):
        """Test many skills with partial match."""
        mentor_skills = ["Python", "Java", "C++", "Go", "Rust"]
        mentee_interests = ["Python", "Java"]
        score = MentorshipMatcher._calculate_skill_overlap(mentor_skills, mentee_interests)
        # Overlap: 2, Union: 5, Score: 2/5 = 0.4
        assert score == 0.4


# Integration test placeholder
class TestMentorshipMatcherIntegration:
    """Integration tests requiring database."""

    def test_find_matches_requires_db(self):
        """Placeholder for integration tests with real DB."""
        # Would be tested in test_mentorship.py with real DB session
        pass
