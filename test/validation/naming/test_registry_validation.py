"""
Pytest tests for registry validation functionality.
"""

import pytest
from unittest.mock import Mock, patch

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestRegistryValidation:
    """Test registry validation functionality."""

    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()

    @patch("cursus.validation.naming.naming_standard_validator.STEP_NAMES")
    def test_validate_all_registry_entries_valid(self, mock_step_names, validator):
        """Test validation of valid registry entries."""
        # Mock a valid registry - spec_type should match step name
        mock_step_names.items.return_value = [
            (
                "XGBoostTraining",
                {
                    "sagemaker_step_type": "Training",
                    "spec_type": "XGBoostTraining",  # Should match step name
                },
            ),
            (
                "TabularPreprocessing",
                {
                    "sagemaker_step_type": "Processing",
                    "spec_type": "TabularPreprocessing",  # Should match step name
                },
            ),
        ]

        violations = validator.validate_all_registry_entries()
        assert len(violations) == 0

    @patch("cursus.validation.naming.naming_standard_validator.STEP_NAMES")
    def test_validate_all_registry_entries_invalid_step_name(
        self, mock_step_names, validator
    ):
        """Test validation with invalid step names in registry."""
        # Mock registry with invalid step name
        mock_step_names.items.return_value = [
            (
                "invalid_step_name",
                {  # Invalid: snake_case instead of PascalCase
                    "sagemaker_step_type": "Training",
                    "spec_type": "TrainingSpec",
                },
            )
        ]

        violations = validator.validate_all_registry_entries()
        assert len(violations) > 0
        violation_types = [v.violation_type for v in violations]
        assert "pascal_case" in violation_types

    @patch("cursus.validation.naming.naming_standard_validator.STEP_NAMES")
    def test_validate_all_registry_entries_invalid_sagemaker_type(
        self, mock_step_names, validator
    ):
        """Test validation with invalid SageMaker step type."""
        # Mock registry with invalid SageMaker step type
        mock_step_names.items.return_value = [
            (
                "XGBoostTraining",
                {
                    "sagemaker_step_type": "InvalidType",  # Invalid SageMaker step type
                    "spec_type": "TrainingSpec",
                },
            )
        ]

        violations = validator.validate_all_registry_entries()
        assert len(violations) > 0
        violation_types = [v.violation_type for v in violations]
        assert "invalid_sagemaker_type" in violation_types

    @patch("cursus.validation.naming.naming_standard_validator.STEP_NAMES")
    def test_validate_all_registry_entries_spec_type_mismatch(
        self, mock_step_names, validator
    ):
        """Test validation with spec type mismatch."""
        # Mock registry with spec type that doesn't match step name
        mock_step_names.items.return_value = [
            (
                "XGBoostTraining",
                {
                    "sagemaker_step_type": "Training",
                    "spec_type": "ProcessingSpec",  # Should be TrainingSpec for Training step
                },
            )
        ]

        violations = validator.validate_all_registry_entries()
        assert len(violations) > 0
        violation_types = [v.violation_type for v in violations]
        assert "spec_type_mismatch" in violation_types

    def test_validate_registry_entry_valid(self, validator):
        """Test validation of a single valid registry entry."""
        violations = validator.validate_registry_entry(
            "XGBoostTraining",
            {
                "sagemaker_step_type": "Training",
                "spec_type": "XGBoostTraining",  # Should match step name
            },
        )
        assert len(violations) == 0

    def test_validate_registry_entry_missing_fields(self, validator):
        """Test validation of registry entry with missing fields."""
        violations = validator.validate_registry_entry(
            "XGBoostTraining", {}  # Missing required fields
        )
        # Missing fields don't generate violations in current implementation
        # The method only validates what's present
        assert len(violations) == 0

    def test_validate_sagemaker_step_type_valid(self, validator):
        """Test validation of valid SageMaker step types."""
        valid_types = [
            "Processing",
            "Training",
            "Transform",
            "CreateModel",
            "RegisterModel",
            "Lambda",
            "MimsModelRegistrationProcessing",
            "CradleDataLoading",
            "Base",
        ]

        for step_type in valid_types:
            violations = validator._validate_sagemaker_step_type(step_type, "Test")
            assert (
                len(violations) == 0
            ), f"Valid SageMaker step type '{step_type}' should not have violations"

    def test_validate_sagemaker_step_type_invalid(self, validator):
        """Test validation of invalid SageMaker step types."""
        invalid_types = [
            "InvalidType",
            "training",  # lowercase
            "TRAINING",  # uppercase
            "Custom",
        ]

        for step_type in invalid_types:
            violations = validator._validate_sagemaker_step_type(step_type, "Test")
            assert (
                len(violations) > 0
            ), f"Invalid SageMaker step type '{step_type}' should have violations"
