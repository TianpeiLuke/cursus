"""
Tests for workspace user experience validation functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.quality.user_experience_validator import UserExperienceValidator


class TestUserExperienceValidator(unittest.TestCase):
    """Test cases for UserExperienceValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.ux_validator = UserExperienceValidator(workspace_root=self.workspace_path)

    def test_ux_validator_initialization(self):
        """Test that UserExperienceValidator initializes correctly."""
        self.assertIsInstance(self.ux_validator, UserExperienceValidator)
        self.assertTrue(hasattr(self.ux_validator, "run_user_experience_assessment"))

    def test_run_user_experience_assessment(self):
        """Test comprehensive user experience assessment."""
        # Test the main assessment method
        ux_report = self.ux_validator.run_user_experience_assessment()

        # Check that we get a UserExperienceReport object
        self.assertIsNotNone(ux_report)
        self.assertTrue(hasattr(ux_report, "overall_score"))
        self.assertTrue(hasattr(ux_report, "overall_status"))
        self.assertTrue(hasattr(ux_report, "onboarding_score"))
        self.assertTrue(hasattr(ux_report, "api_usability_score"))

    def test_onboarding_assessment(self):
        """Test developer onboarding assessment."""
        # Test that onboarding assessment is included in main assessment
        ux_report = self.ux_validator.run_user_experience_assessment()

        self.assertTrue(hasattr(ux_report, "onboarding_result"))
        self.assertTrue(hasattr(ux_report, "onboarding_score"))
        self.assertGreaterEqual(ux_report.onboarding_score, 0.0)
        self.assertLessEqual(ux_report.onboarding_score, 100.0)

    def test_api_usability_assessment(self):
        """Test API usability assessment."""
        # Test that API usability is included in main assessment
        ux_report = self.ux_validator.run_user_experience_assessment()

        self.assertTrue(hasattr(ux_report, "api_tests"))
        self.assertTrue(hasattr(ux_report, "api_usability_score"))
        self.assertGreaterEqual(ux_report.api_usability_score, 0.0)
        self.assertLessEqual(ux_report.api_usability_score, 100.0)

    def test_error_handling_validation(self):
        """Test error handling validation."""
        # Test that error handling assessment is included
        ux_report = self.ux_validator.run_user_experience_assessment()

        self.assertTrue(hasattr(ux_report, "error_handling_score"))
        self.assertGreaterEqual(ux_report.error_handling_score, 0.0)
        self.assertLessEqual(ux_report.error_handling_score, 100.0)

    def test_documentation_effectiveness(self):
        """Test documentation effectiveness assessment."""
        # Test that documentation assessment is included
        ux_report = self.ux_validator.run_user_experience_assessment()

        self.assertTrue(hasattr(ux_report, "documentation_score"))
        self.assertGreaterEqual(ux_report.documentation_score, 0.0)
        self.assertLessEqual(ux_report.documentation_score, 100.0)

    def test_phase3_requirements_check(self):
        """Test Phase 3 requirements validation."""
        # Test that the report can check Phase 3 requirements
        ux_report = self.ux_validator.run_user_experience_assessment()

        self.assertTrue(hasattr(ux_report, "meets_phase3_requirements"))
        # Should return a boolean
        phase3_compliance = ux_report.meets_phase3_requirements
        self.assertIsInstance(phase3_compliance, bool)


if __name__ == "__main__":
    unittest.main()
