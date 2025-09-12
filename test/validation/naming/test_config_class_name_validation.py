"""
Pytest tests for config class name validation.
"""

import pytest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestConfigClassNameValidation:
    """Test config class name validation."""

    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()

    def test_valid_config_class_names(self, validator):
        """Test valid config class names."""
        valid_names = [
            "XGBoostTrainingConfig",
            "TabularPreprocessingConfig",
            "ModelCalibrationConfig",
            "AConfig",  # Single letter + Config
            "ABC123Config",  # With numbers + Config
        ]

        for name in valid_names:
            violations = validator._validate_config_class_name(name)
            assert (
                len(violations) == 0
            ), f"Valid config name '{name}' should not have violations"

    def test_missing_config_suffix(self, validator):
        """Test config class names missing 'Config' suffix."""
        invalid_names = ["XGBoostTraining", "TabularPreprocessing", "ModelCalibration"]

        for name in invalid_names:
            violations = validator._validate_config_class_name(name)
            violation_types = [v.violation_type for v in violations]
            assert "config_suffix" in violation_types

    def test_invalid_base_name_pattern(self, validator):
        """Test config class names with invalid base name patterns."""
        invalid_names = [
            "snake_caseConfig",
            "camelCaseConfig",
            "123NumberConfig",
            "with-hyphenConfig",
        ]

        for name in invalid_names:
            violations = validator._validate_config_class_name(name)
            violation_types = [v.violation_type for v in violations]
            assert "pascal_case" in violation_types
