"""
Tests for workspace user experience validation functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.cursus.workspace.quality.user_experience_validator import UserExperienceValidator


class TestUserExperienceValidator(unittest.TestCase):
    """Test cases for UserExperienceValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.ux_validator = UserExperienceValidator()

    def test_ux_validator_initialization(self):
        """Test that UserExperienceValidator initializes correctly."""
        self.assertIsInstance(self.ux_validator, UserExperienceValidator)
        self.assertTrue(hasattr(self.ux_validator, 'validate_user_experience'))

    def test_validate_user_experience(self):
        """Test user experience validation."""
        # Test basic UX validation
        ux_report = self.ux_validator.validate_user_experience()
        self.assertIsInstance(ux_report, dict)
        self.assertIn('overall_ux_score', ux_report)

    def test_onboarding_simulation(self):
        """Test developer onboarding simulation."""
        # Test onboarding simulation
        onboarding_result = self.ux_validator.simulate_developer_onboarding()
        self.assertIsInstance(onboarding_result, dict)
        self.assertIn('onboarding_success_rate', onboarding_result)
        self.assertIn('time_to_first_success', onboarding_result)

    def test_api_usability_assessment(self):
        """Test API usability assessment."""
        # Test API usability
        usability_result = self.ux_validator.assess_api_usability()
        self.assertIsInstance(usability_result, dict)
        self.assertIn('api_intuitiveness_score', usability_result)

    def test_error_handling_validation(self):
        """Test error handling validation."""
        # Test error handling assessment
        error_handling_result = self.ux_validator.validate_error_handling()
        self.assertIsInstance(error_handling_result, dict)
        self.assertIn('error_clarity_score', error_handling_result)

    def test_documentation_effectiveness(self):
        """Test documentation effectiveness assessment."""
        # Test documentation effectiveness
        doc_result = self.ux_validator.assess_documentation_effectiveness()
        self.assertIsInstance(doc_result, dict)
        self.assertIn('documentation_score', doc_result)


if __name__ == '__main__':
    unittest.main()
