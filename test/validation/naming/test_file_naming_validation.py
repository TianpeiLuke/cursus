"""
Pytest tests for file naming validation.
"""

import pytest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestFileNamingValidation:
    """Test file naming validation."""

    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()

    def test_valid_builder_file_names(self, validator):
        """Test valid builder file names."""
        valid_names = [
            "builder_xgboost_training_step.py",
            "builder_tabular_preprocessing_step.py",
            "builder_model_calibration_step.py",
        ]

        for filename in valid_names:
            violations = validator.validate_file_naming(filename, "builder")
            assert (
                len(violations) == 0
            ), f"Valid builder file '{filename}' should not have violations"

    def test_valid_config_file_names(self, validator):
        """Test valid config file names."""
        valid_names = [
            "config_xgboost_training_step.py",
            "config_tabular_preprocessing_step.py",
            "config_model_calibration_step.py",
        ]

        for filename in valid_names:
            violations = validator.validate_file_naming(filename, "config")
            assert (
                len(violations) == 0
            ), f"Valid config file '{filename}' should not have violations"

    def test_valid_spec_file_names(self, validator):
        """Test valid spec file names."""
        valid_names = [
            "xgboost_training_spec.py",
            "tabular_preprocessing_spec.py",
            "model_calibration_spec.py",
        ]

        for filename in valid_names:
            violations = validator.validate_file_naming(filename, "spec")
            assert (
                len(violations) == 0
            ), f"Valid spec file '{filename}' should not have violations"

    def test_valid_contract_file_names(self, validator):
        """Test valid contract file names."""
        valid_names = [
            "xgboost_training_contract.py",
            "tabular_preprocessing_contract.py",
            "model_calibration_contract.py",
        ]

        for filename in valid_names:
            violations = validator.validate_file_naming(filename, "contract")
            assert (
                len(violations) == 0
            ), f"Valid contract file '{filename}' should not have violations"

    def test_invalid_builder_file_names(self, validator):
        """Test invalid builder file names."""
        invalid_names = [
            "xgboost_training_step.py",  # Missing 'builder_' prefix
            "builder_XGBoostTraining_step.py",  # PascalCase instead of snake_case
            "builder_xgboost_training.py",  # Missing '_step' suffix
            "builder_xgboost_training_step.txt",  # Wrong extension
        ]

        for filename in invalid_names:
            violations = validator.validate_file_naming(filename, "builder")
            assert (
                len(violations) > 0
            ), f"Invalid builder file '{filename}' should have violations"

    def test_invalid_config_file_names(self, validator):
        """Test invalid config file names."""
        invalid_names = [
            "xgboost_training_step.py",  # Missing 'config_' prefix
            "config_XGBoostTraining_step.py",  # PascalCase instead of snake_case
            "config_xgboost_training.py",  # Missing '_step' suffix
            "config_xgboost_training_step.txt",  # Wrong extension
        ]

        for filename in invalid_names:
            violations = validator.validate_file_naming(filename, "config")
            assert (
                len(violations) > 0
            ), f"Invalid config file '{filename}' should have violations"

    def test_invalid_spec_file_names(self, validator):
        """Test invalid spec file names."""
        invalid_names = [
            "xgboost_training.py",  # Missing '_spec' suffix
            "XGBoostTraining_spec.py",  # PascalCase instead of snake_case
            "xgboost_training_spec.txt",  # Wrong extension
        ]

        for filename in invalid_names:
            violations = validator.validate_file_naming(filename, "spec")
            assert (
                len(violations) > 0
            ), f"Invalid spec file '{filename}' should have violations"

    def test_invalid_contract_file_names(self, validator):
        """Test invalid contract file names."""
        invalid_names = [
            "xgboost_training.py",  # Missing '_contract' suffix
            "XGBoostTraining_contract.py",  # PascalCase instead of snake_case
            "xgboost_training_contract.txt",  # Wrong extension
        ]

        for filename in invalid_names:
            violations = validator.validate_file_naming(filename, "contract")
            assert (
                len(violations) > 0
            ), f"Invalid contract file '{filename}' should have violations"

    def test_unsupported_file_type(self, validator):
        """Test validation with unsupported file type."""
        violations = validator.validate_file_naming("test.py", "unsupported")
        # Unsupported file types return no violations (they're just ignored)
        assert len(violations) == 0
