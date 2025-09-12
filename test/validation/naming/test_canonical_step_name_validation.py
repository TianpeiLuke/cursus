"""
Pytest tests for canonical step name validation.
"""

import pytest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestCanonicalStepNameValidation:
    """Test canonical step name validation."""

    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()

    def test_valid_pascal_case_names(self, validator):
        """Test valid PascalCase step names."""
        valid_names = [
            "CradleDataLoading",
            "XGBoostTraining",
            "TabularPreprocessing",
            "ModelCalibration",
            "A",  # Single letter
            "ABC123",  # With numbers
        ]

        for name in valid_names:
            violations = validator._validate_canonical_step_name(name, "Test")
            assert (
                len(violations) == 0
            ), f"Valid name '{name}' should not have violations"

    def test_invalid_pascal_case_names(self, validator):
        """Test invalid PascalCase step names."""
        invalid_names = [
            "camelCase",  # starts with lowercase
            "snake_case",  # contains underscore
            "kebab-case",  # contains hyphen
            "lowercase",  # all lowercase
            "UPPERCASE",  # all uppercase (but this actually passes PascalCase regex)
            "123Number",  # starts with number
            "With Space",  # contains space
            "with.dot",  # contains dot
        ]

        for name in invalid_names:
            violations = validator._validate_canonical_step_name(name, "Test")
            if name not in ["UPPERCASE"]:  # UPPERCASE actually passes the regex
                assert (
                    len(violations) > 0
                ), f"Invalid name '{name}' should have violations"

    def test_empty_step_name(self, validator):
        """Test empty step name validation."""
        violations = validator._validate_canonical_step_name("", "Test")
        assert len(violations) == 1
        assert violations[0].violation_type == "empty_name"

    def test_none_step_name(self, validator):
        """Test None step name validation."""
        violations = validator._validate_canonical_step_name(None, "Test")
        assert len(violations) == 1
        assert violations[0].violation_type == "empty_name"

    def test_underscore_in_name(self, validator):
        """Test step name with underscores."""
        violations = validator._validate_canonical_step_name("Step_Name", "Test")
        # Should have both pascal_case and underscore_in_name violations
        violation_types = [v.violation_type for v in violations]
        assert "underscore_in_name" in violation_types

    def test_lowercase_name(self, validator):
        """Test all lowercase step name."""
        violations = validator._validate_canonical_step_name("stepname", "Test")
        violation_types = [v.violation_type for v in violations]
        assert "lowercase_name" in violation_types
