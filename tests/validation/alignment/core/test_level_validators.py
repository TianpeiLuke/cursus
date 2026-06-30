"""
Test Level Validators Module

Tests for consolidated validation logic for all 4 validation levels.
Tests integration with existing alignment modules and step-type-specific validator support.
"""

import pytest
from unittest.mock import patch, MagicMock, Mock

from cursus.validation.alignment.core.level_validators import LevelValidators


class TestLevelValidators:
    """Test LevelValidators consolidated validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.workspace_dirs = ["/test/workspace"]
        self.level_validators = LevelValidators(self.workspace_dirs)

    @patch(
        "cursus.validation.alignment.core.script_contract_alignment.ScriptContractAlignmentTester"
    )
    def test_run_level_1_validation(self, mock_alignment_class):
        """Test Level 1: Script ↔ Contract validation."""
        # Mock the alignment class and its method
        mock_alignment = Mock()
        mock_alignment.validate_script_contract_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 1,
            "validation_type": "script_contract",
        }
        mock_alignment_class.return_value = mock_alignment

        result = self.level_validators.run_level_1_validation("test_script")

        # Verify alignment class was instantiated with correct workspace_dirs
        mock_alignment_class.assert_called_once_with(workspace_dirs=self.workspace_dirs)

        # Verify validation method was called
        mock_alignment.validate_script.assert_called_once_with("test_script")

        # Verify result structure
        assert result["level"] == 1
        assert result["step_name"] == "test_script"
        assert result["validation_type"] == "script_contract"
        assert result["status"] == "COMPLETED"
        assert "result" in result

    # test_run_level_2_validation removed — Level 2 (CONTRACT_SPEC) is gone (construction invariant;
    # FZ 31e1d3h/D5). Its property-path check moved to Level 3 (B2).

    @patch(
        "cursus.validation.alignment.core.spec_dependency_alignment.SpecificationDependencyAlignmentTester"
    )
    def test_run_level_3_validation(self, mock_alignment_class):
        """Test Level 3: Specification ↔ Dependencies validation."""
        # Mock the alignment class and its method
        mock_alignment = Mock()
        mock_alignment.validate_spec_dependency_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 3,
            "validation_type": "spec_dependency",
        }
        mock_alignment_class.return_value = mock_alignment

        result = self.level_validators.run_level_3_validation("test_spec")

        # Verify alignment class was instantiated with correct workspace_dirs
        mock_alignment_class.assert_called_once_with(workspace_dirs=self.workspace_dirs)

        # Verify validation method was called
        mock_alignment.validate_specification.assert_called_once_with("test_spec")

        # Verify result structure
        assert result["level"] == 3
        assert result["step_name"] == "test_spec"
        assert result["validation_type"] == "spec_dependency"
        assert result["status"] == "COMPLETED"
        assert "result" in result

    def test_run_level_4_validation_with_validator_class(self):
        """Test Level 4: Builder ↔ Configuration validation with specific validator."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_builder_config_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 4,
            "validation_type": "builder_config",
            "validator_class": "ProcessingStepBuilderValidator",
        }

        # Mock the _get_step_type_validator method
        with patch.object(
            self.level_validators,
            "_get_step_type_validator",
            return_value=mock_validator,
        ):
            result = self.level_validators.run_level_4_validation(
                "test_step", "ProcessingStepBuilderValidator"
            )

        # Verify validator method was called
        mock_validator.validate_builder_config_alignment.assert_called_once_with(
            "test_step"
        )

        # Verify result structure
        assert result["status"] == "COMPLETED"
        assert result["level"] == 4
        assert result["step_name"] == "test_step"
        assert result["validation_type"] == "builder_config"
        assert result["validator_class"] == "ProcessingStepBuilderValidator"
        assert "result" in result
        assert result["result"]["passed"] is True

    def test_run_level_4_validation_without_validator_class(self):
        """Test Level 4 validation without specific validator class."""
        result = self.level_validators.run_level_4_validation("test_step")

        # Should return skipped result when no validator class provided
        assert result["status"] == "SKIPPED"
        assert result["level"] == 4
        assert result["step_name"] == "test_step"
        assert result["validation_type"] == "builder_config"
        assert "reason" in result
        assert "No validator class specified" in result["reason"]

    def test_run_level_4_validation_with_invalid_validator_class(self):
        """Test Level 4 validation with invalid validator class."""
        # Mock _get_step_type_validator to return None (invalid validator)
        with patch.object(
            self.level_validators, "_get_step_type_validator", return_value=None
        ):
            result = self.level_validators.run_level_4_validation(
                "test_step", "InvalidValidator"
            )

        # Should return SKIPPED result when validator class is invalid (actual implementation behavior)
        assert result["status"] == "SKIPPED"
        assert result["level"] == 4
        assert result["step_name"] == "test_step"
        assert result["validation_type"] == "builder_config"
        assert "reason" in result
        assert "InvalidValidator is not yet implemented" in result["reason"]

    def test_get_step_type_validator_returns_registry_binding_validator(self):
        """FZ 31e1d3g3 Phase D3: Level-4 is now the single step-type-AGNOSTIC B3
        RegistryBindingValidator (replacing the per-step-type source-scanning validators that
        reported every shell FAILED). The ruleset validator_class string is ignored."""
        from cursus.validation.alignment.validators.registry_binding_validator import (
            RegistryBindingValidator,
        )

        validator = self.level_validators._get_step_type_validator(
            "ProcessingStepBuilderValidator"
        )
        assert isinstance(validator, RegistryBindingValidator)

    def test_get_step_type_validator_ignores_validator_class_name(self):
        """Any ruleset validator_class string yields the same B3 validator (type-agnostic)."""
        from cursus.validation.alignment.validators.registry_binding_validator import (
            RegistryBindingValidator,
        )

        for name in ("TrainingStepBuilderValidator", "InvalidValidator", ""):
            assert isinstance(
                self.level_validators._get_step_type_validator(name),
                RegistryBindingValidator,
            )

    @patch(
        "cursus.validation.alignment.validators.registry_binding_validator.RegistryBindingValidator"
    )
    def test_get_step_type_validator_exception_handling(self, mock_b3):
        """If B3 construction raises, _get_step_type_validator returns None (graceful)."""
        mock_b3.side_effect = Exception("B3 init failed")

        validator = self.level_validators._get_step_type_validator(
            "ProcessingStepBuilderValidator"
        )
        assert validator is None

    def test_integration_with_all_levels(self):
        """Test integration workflow with all validation levels."""
        step_name = "test_integration_step"

        # Mock the surviving alignment levels (Level 2 / CONTRACT_SPEC removed — FZ 31e1d3h/D5).
        with (
            patch(
                "cursus.validation.alignment.core.script_contract_alignment.ScriptContractAlignmentTester"
            ) as mock_level1,
            patch(
                "cursus.validation.alignment.core.spec_dependency_alignment.SpecificationDependencyAlignmentTester"
            ) as mock_level3,
        ):
            # Set up mock returns
            mock_level1.return_value.validate_script.return_value = {
                "passed": True,
                "level": 1,
            }
            mock_level3.return_value.validate_specification.return_value = {
                "passed": True,
                "level": 3,
            }

            # Run the surviving levels
            result1 = self.level_validators.run_level_1_validation(step_name)
            result3 = self.level_validators.run_level_3_validation(step_name)

            # Verify levels ran successfully - check the wrapped result structure
            assert result1["status"] == "COMPLETED" and result1["level"] == 1
            assert result3["status"] == "COMPLETED" and result3["level"] == 3
            assert result1["result"]["passed"] is True
            assert result3["result"]["passed"] is True

            # Verify alignment classes were instantiated with correct workspace_dirs
            mock_level1.assert_called_once_with(workspace_dirs=self.workspace_dirs)
            mock_level3.assert_called_once_with(workspace_dirs=self.workspace_dirs)

    def test_error_handling_in_validation_levels(self):
        """Test error handling when validation levels raise exceptions."""
        step_name = "test_error_step"

        # Test Level 1 error handling
        with patch(
            "cursus.validation.alignment.core.script_contract_alignment.ScriptContractAlignmentTester"
        ) as mock_level1:
            mock_level1.side_effect = Exception("Level 1 validation failed")

            result = self.level_validators.run_level_1_validation(step_name)
            assert result["status"] == "ERROR"
            assert "Level 1 validation failed" in result["error"]

        # (Level 2 / CONTRACT_SPEC removed — FZ 31e1d3h/D5; no run_level_2_validation to test.)

        # Test Level 3 error handling
        with patch(
            "cursus.validation.alignment.core.spec_dependency_alignment.SpecificationDependencyAlignmentTester"
        ) as mock_level3:
            mock_level3.side_effect = Exception("Level 3 validation failed")

            result = self.level_validators.run_level_3_validation(step_name)
            assert result["status"] == "ERROR"
            assert "Level 3 validation failed" in result["error"]

    def test_workspace_dirs_propagation(self):
        """Test that workspace_dirs are properly propagated to all alignment classes."""
        custom_workspace_dirs = ["/custom/workspace1", "/custom/workspace2"]
        custom_level_validators = LevelValidators(custom_workspace_dirs)

        # Test Level 1
        with patch(
            "cursus.validation.alignment.core.script_contract_alignment.ScriptContractAlignmentTester"
        ) as mock_level1:
            mock_level1.return_value.validate_script.return_value = {"passed": True}
            custom_level_validators.run_level_1_validation("test_step")
            mock_level1.assert_called_once_with(workspace_dirs=custom_workspace_dirs)

        # (Level 2 / CONTRACT_SPEC removed — FZ 31e1d3h/D5.)

        # Test Level 3
        with patch(
            "cursus.validation.alignment.core.spec_dependency_alignment.SpecificationDependencyAlignmentTester"
        ) as mock_level3:
            mock_level3.return_value.validate_specification.return_value = {
                "passed": True
            }
            custom_level_validators.run_level_3_validation("test_step")
            mock_level3.assert_called_once_with(workspace_dirs=custom_workspace_dirs)

        # Test Level 4 (the B3 RegistryBindingValidator — FZ 31e1d3g3 Phase D3)
        with patch(
            "cursus.validation.alignment.validators.registry_binding_validator.RegistryBindingValidator"
        ) as mock_b3:
            mock_b3.return_value = Mock()
            custom_level_validators._get_step_type_validator("TestValidator")
            mock_b3.assert_called_once_with(custom_workspace_dirs)

    def test_validation_result_structure(self):
        """Test that validation results have consistent structure across levels."""
        step_name = "test_structure_step"

        # Mock the surviving levels to return structured results (Level 2 / CONTRACT_SPEC removed).
        with (
            patch(
                "cursus.validation.alignment.core.script_contract_alignment.ScriptContractAlignmentTester"
            ) as mock_level1,
            patch(
                "cursus.validation.alignment.core.spec_dependency_alignment.SpecificationDependencyAlignmentTester"
            ) as mock_level3,
        ):
            # Set up consistent result structures
            level1_result = {
                "passed": True,
                "issues": [],
                "level": 1,
                "validation_type": "script_contract",
                "step_name": step_name,
            }
            level3_result = {
                "passed": True,
                "issues": [],
                "level": 3,
                "validation_type": "spec_dependency",
                "step_name": step_name,
            }

            mock_level1.return_value.validate_script.return_value = level1_result
            mock_level3.return_value.validate_specification.return_value = level3_result

            # Run validations
            result1 = self.level_validators.run_level_1_validation(step_name)
            result3 = self.level_validators.run_level_3_validation(step_name)

            # Verify result structures - these are wrapped in the level validator result structure
            for result in [result1, result3]:
                assert "status" in result
                assert "level" in result
                assert "validation_type" in result
                assert "step_name" in result
                assert "result" in result
                # Check the inner result structure
                inner_result = result["result"]
                assert "passed" in inner_result
                assert "issues" in inner_result

            # Verify specific values from inner results
            assert result1["result"]["passed"] is True and result1["level"] == 1
            assert result3["result"]["passed"] is True and result3["level"] == 3
