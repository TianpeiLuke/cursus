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

    @patch('cursus.validation.alignment.core.level_validators.ScriptContractAlignment')
    def test_run_level_1_validation(self, mock_alignment_class):
        """Test Level 1: Script ↔ Contract validation."""
        # Mock the alignment class and its method
        mock_alignment = Mock()
        mock_alignment.validate_script_contract_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 1,
            "validation_type": "script_contract"
        }
        mock_alignment_class.return_value = mock_alignment
        
        result = self.level_validators.run_level_1_validation("test_script")
        
        # Verify alignment class was instantiated with correct workspace_dirs
        mock_alignment_class.assert_called_once_with(workspace_dirs=self.workspace_dirs)
        
        # Verify validation method was called
        mock_alignment.validate_script_contract_alignment.assert_called_once_with("test_script")
        
        # Verify result
        assert result["passed"] is True
        assert result["level"] == 1
        assert result["validation_type"] == "script_contract"

    @patch('cursus.validation.alignment.core.level_validators.ContractSpecAlignment')
    def test_run_level_2_validation(self, mock_alignment_class):
        """Test Level 2: Contract ↔ Specification validation."""
        # Mock the alignment class and its method
        mock_alignment = Mock()
        mock_alignment.validate_contract_spec_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 2,
            "validation_type": "contract_spec"
        }
        mock_alignment_class.return_value = mock_alignment
        
        result = self.level_validators.run_level_2_validation("test_contract")
        
        # Verify alignment class was instantiated with correct workspace_dirs
        mock_alignment_class.assert_called_once_with(workspace_dirs=self.workspace_dirs)
        
        # Verify validation method was called
        mock_alignment.validate_contract_spec_alignment.assert_called_once_with("test_contract")
        
        # Verify result
        assert result["passed"] is True
        assert result["level"] == 2
        assert result["validation_type"] == "contract_spec"

    @patch('cursus.validation.alignment.core.level_validators.SpecDependencyAlignment')
    def test_run_level_3_validation(self, mock_alignment_class):
        """Test Level 3: Specification ↔ Dependencies validation."""
        # Mock the alignment class and its method
        mock_alignment = Mock()
        mock_alignment.validate_spec_dependency_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 3,
            "validation_type": "spec_dependency"
        }
        mock_alignment_class.return_value = mock_alignment
        
        result = self.level_validators.run_level_3_validation("test_spec")
        
        # Verify alignment class was instantiated with correct workspace_dirs
        mock_alignment_class.assert_called_once_with(workspace_dirs=self.workspace_dirs)
        
        # Verify validation method was called
        mock_alignment.validate_spec_dependency_alignment.assert_called_once_with("test_spec")
        
        # Verify result
        assert result["passed"] is True
        assert result["level"] == 3
        assert result["validation_type"] == "spec_dependency"

    def test_run_level_4_validation_with_validator_class(self):
        """Test Level 4: Builder ↔ Configuration validation with specific validator."""
        # Mock validator
        mock_validator = Mock()
        mock_validator.validate_builder_config_alignment.return_value = {
            "passed": True,
            "issues": [],
            "level": 4,
            "validation_type": "builder_config",
            "validator_class": "ProcessingStepBuilderValidator"
        }
        
        # Mock the _get_step_type_validator method
        with patch.object(self.level_validators, '_get_step_type_validator', return_value=mock_validator):
            result = self.level_validators.run_level_4_validation("test_step", "ProcessingStepBuilderValidator")
        
        # Verify validator method was called
        mock_validator.validate_builder_config_alignment.assert_called_once_with("test_step")
        
        # Verify result
        assert result["passed"] is True
        assert result["level"] == 4
        assert result["validation_type"] == "builder_config"
        assert result["validator_class"] == "ProcessingStepBuilderValidator"

    def test_run_level_4_validation_without_validator_class(self):
        """Test Level 4 validation without specific validator class."""
        result = self.level_validators.run_level_4_validation("test_step")
        
        # Should return error result when no validator class provided
        assert result["passed"] is False
        assert "error" in result
        assert "No validator class specified" in result["error"]

    def test_run_level_4_validation_with_invalid_validator_class(self):
        """Test Level 4 validation with invalid validator class."""
        # Mock _get_step_type_validator to return None (invalid validator)
        with patch.object(self.level_validators, '_get_step_type_validator', return_value=None):
            result = self.level_validators.run_level_4_validation("test_step", "InvalidValidator")
        
        # Should return error result when validator class is invalid
        assert result["passed"] is False
        assert "error" in result
        assert "Invalid validator class" in result["error"]

    @patch('cursus.validation.alignment.core.level_validators.ValidatorFactory')
    def test_get_step_type_validator_success(self, mock_factory_class):
        """Test successful step-type validator retrieval."""
        # Mock validator factory and validator
        mock_factory = Mock()
        mock_validator = Mock()
        mock_factory.get_validator.return_value = mock_validator
        mock_factory_class.return_value = mock_factory
        
        validator = self.level_validators._get_step_type_validator("ProcessingStepBuilderValidator")
        
        # Verify factory was instantiated with workspace_dirs
        mock_factory_class.assert_called_once_with(self.workspace_dirs)
        
        # Verify get_validator was called with correct class name
        mock_factory.get_validator.assert_called_once_with("ProcessingStepBuilderValidator")
        
        # Verify validator was returned
        assert validator == mock_validator

    @patch('cursus.validation.alignment.core.level_validators.ValidatorFactory')
    def test_get_step_type_validator_failure(self, mock_factory_class):
        """Test step-type validator retrieval failure."""
        # Mock validator factory to return None
        mock_factory = Mock()
        mock_factory.get_validator.return_value = None
        mock_factory_class.return_value = mock_factory
        
        validator = self.level_validators._get_step_type_validator("InvalidValidator")
        
        # Should return None for invalid validator
        assert validator is None

    @patch('cursus.validation.alignment.core.level_validators.ValidatorFactory')
    def test_get_step_type_validator_exception_handling(self, mock_factory_class):
        """Test step-type validator retrieval with exception."""
        # Mock validator factory to raise exception
        mock_factory_class.side_effect = Exception("Factory initialization failed")
        
        validator = self.level_validators._get_step_type_validator("ProcessingStepBuilderValidator")
        
        # Should return None when exception occurs
        assert validator is None

    def test_integration_with_all_levels(self):
        """Test integration workflow with all validation levels."""
        step_name = "test_integration_step"
        
        # Mock all alignment classes
        with patch('cursus.validation.alignment.core.level_validators.ScriptContractAlignment') as mock_level1, \
             patch('cursus.validation.alignment.core.level_validators.ContractSpecAlignment') as mock_level2, \
             patch('cursus.validation.alignment.core.level_validators.SpecDependencyAlignment') as mock_level3:
            
            # Set up mock returns
            mock_level1.return_value.validate_script_contract_alignment.return_value = {"passed": True, "level": 1}
            mock_level2.return_value.validate_contract_spec_alignment.return_value = {"passed": True, "level": 2}
            mock_level3.return_value.validate_spec_dependency_alignment.return_value = {"passed": True, "level": 3}
            
            # Run all levels
            result1 = self.level_validators.run_level_1_validation(step_name)
            result2 = self.level_validators.run_level_2_validation(step_name)
            result3 = self.level_validators.run_level_3_validation(step_name)
            
            # Verify all levels ran successfully
            assert result1["passed"] is True and result1["level"] == 1
            assert result2["passed"] is True and result2["level"] == 2
            assert result3["passed"] is True and result3["level"] == 3
            
            # Verify all alignment classes were instantiated with correct workspace_dirs
            mock_level1.assert_called_once_with(workspace_dirs=self.workspace_dirs)
            mock_level2.assert_called_once_with(workspace_dirs=self.workspace_dirs)
            mock_level3.assert_called_once_with(workspace_dirs=self.workspace_dirs)

    def test_error_handling_in_validation_levels(self):
        """Test error handling when validation levels raise exceptions."""
        step_name = "test_error_step"
        
        # Test Level 1 error handling
        with patch('cursus.validation.alignment.core.level_validators.ScriptContractAlignment') as mock_level1:
            mock_level1.side_effect = Exception("Level 1 validation failed")
            
            with pytest.raises(Exception, match="Level 1 validation failed"):
                self.level_validators.run_level_1_validation(step_name)
        
        # Test Level 2 error handling
        with patch('cursus.validation.alignment.core.level_validators.ContractSpecAlignment') as mock_level2:
            mock_level2.side_effect = Exception("Level 2 validation failed")
            
            with pytest.raises(Exception, match="Level 2 validation failed"):
                self.level_validators.run_level_2_validation(step_name)
        
        # Test Level 3 error handling
        with patch('cursus.validation.alignment.core.level_validators.SpecDependencyAlignment') as mock_level3:
            mock_level3.side_effect = Exception("Level 3 validation failed")
            
            with pytest.raises(Exception, match="Level 3 validation failed"):
                self.level_validators.run_level_3_validation(step_name)

    def test_workspace_dirs_propagation(self):
        """Test that workspace_dirs are properly propagated to all alignment classes."""
        custom_workspace_dirs = ["/custom/workspace1", "/custom/workspace2"]
        custom_level_validators = LevelValidators(custom_workspace_dirs)
        
        # Test Level 1
        with patch('cursus.validation.alignment.core.level_validators.ScriptContractAlignment') as mock_level1:
            mock_level1.return_value.validate_script_contract_alignment.return_value = {"passed": True}
            custom_level_validators.run_level_1_validation("test_step")
            mock_level1.assert_called_once_with(workspace_dirs=custom_workspace_dirs)
        
        # Test Level 2
        with patch('cursus.validation.alignment.core.level_validators.ContractSpecAlignment') as mock_level2:
            mock_level2.return_value.validate_contract_spec_alignment.return_value = {"passed": True}
            custom_level_validators.run_level_2_validation("test_step")
            mock_level2.assert_called_once_with(workspace_dirs=custom_workspace_dirs)
        
        # Test Level 3
        with patch('cursus.validation.alignment.core.level_validators.SpecDependencyAlignment') as mock_level3:
            mock_level3.return_value.validate_spec_dependency_alignment.return_value = {"passed": True}
            custom_level_validators.run_level_3_validation("test_step")
            mock_level3.assert_called_once_with(workspace_dirs=custom_workspace_dirs)
        
        # Test Level 4 (ValidatorFactory)
        with patch('cursus.validation.alignment.core.level_validators.ValidatorFactory') as mock_factory:
            mock_factory.return_value.get_validator.return_value = Mock()
            custom_level_validators._get_step_type_validator("TestValidator")
            mock_factory.assert_called_once_with(custom_workspace_dirs)

    def test_validation_result_structure(self):
        """Test that validation results have consistent structure across levels."""
        step_name = "test_structure_step"
        
        # Mock all levels to return structured results
        with patch('cursus.validation.alignment.core.level_validators.ScriptContractAlignment') as mock_level1, \
             patch('cursus.validation.alignment.core.level_validators.ContractSpecAlignment') as mock_level2, \
             patch('cursus.validation.alignment.core.level_validators.SpecDependencyAlignment') as mock_level3:
            
            # Set up consistent result structures
            level1_result = {
                "passed": True,
                "issues": [],
                "level": 1,
                "validation_type": "script_contract",
                "step_name": step_name
            }
            level2_result = {
                "passed": False,
                "issues": [{"severity": "ERROR", "message": "Test error"}],
                "level": 2,
                "validation_type": "contract_spec",
                "step_name": step_name
            }
            level3_result = {
                "passed": True,
                "issues": [],
                "level": 3,
                "validation_type": "spec_dependency",
                "step_name": step_name
            }
            
            mock_level1.return_value.validate_script_contract_alignment.return_value = level1_result
            mock_level2.return_value.validate_contract_spec_alignment.return_value = level2_result
            mock_level3.return_value.validate_spec_dependency_alignment.return_value = level3_result
            
            # Run validations
            result1 = self.level_validators.run_level_1_validation(step_name)
            result2 = self.level_validators.run_level_2_validation(step_name)
            result3 = self.level_validators.run_level_3_validation(step_name)
            
            # Verify result structures
            for result in [result1, result2, result3]:
                assert "passed" in result
                assert "issues" in result
                assert "level" in result
                assert "validation_type" in result
                assert "step_name" in result
            
            # Verify specific values
            assert result1["passed"] is True and result1["level"] == 1
            assert result2["passed"] is False and result2["level"] == 2
            assert result3["passed"] is True and result3["level"] == 3
