"""
Unit tests for config class name validation.
"""

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest

from src.cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestConfigClassNameValidation(unittest.TestCase):
    """Test config class name validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_valid_config_class_names(self):
        """Test valid config class names."""
        valid_names = [
            "XGBoostTrainingConfig",
            "TabularPreprocessingConfig",
            "ModelCalibrationConfig",
            "AConfig",  # Single letter + Config
            "ABC123Config"  # With numbers + Config
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_config_class_name(name)
                self.assertEqual(len(violations), 0, f"Valid config name '{name}' should not have violations")
    
    def test_missing_config_suffix(self):
        """Test config class names missing 'Config' suffix."""
        invalid_names = [
            "XGBoostTraining",
            "TabularPreprocessing",
            "ModelCalibration"
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_config_class_name(name)
                violation_types = [v.violation_type for v in violations]
                self.assertIn("config_suffix", violation_types)
    
    def test_invalid_base_name_pattern(self):
        """Test config class names with invalid base name patterns."""
        invalid_names = [
            "snake_caseConfig",
            "camelCaseConfig",
            "123NumberConfig",
            "with-hyphenConfig"
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_config_class_name(name)
                violation_types = [v.violation_type for v in violations]
                self.assertIn("pascal_case", violation_types)


if __name__ == '__main__':
    unittest.main()
