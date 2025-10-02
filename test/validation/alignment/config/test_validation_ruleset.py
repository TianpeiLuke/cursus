"""
Unit tests for validation ruleset configuration system.
"""

import pytest
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.config.validation_ruleset import (
    ValidationLevel,
    StepTypeCategory,
    ValidationRuleset,
    VALIDATION_RULESETS,
    get_validation_ruleset,
    is_validation_level_enabled,
    get_enabled_validation_levels,
    get_level_4_validator_class,
    is_step_type_excluded,
    get_step_types_by_category,
    get_all_step_types,
    validate_step_type_configuration,
    get_validation_ruleset_for_step_name,
    is_validation_level_enabled_for_step_name,
    get_enabled_validation_levels_for_step_name,
    is_step_name_excluded,
)


class TestValidationLevel:
    """Test ValidationLevel enum."""
    
    def test_validation_level_values(self):
        """Test that validation levels have correct values."""
        assert ValidationLevel.SCRIPT_CONTRACT.value == 1
        assert ValidationLevel.CONTRACT_SPEC.value == 2
        assert ValidationLevel.SPEC_DEPENDENCY.value == 3
        assert ValidationLevel.BUILDER_CONFIG.value == 4
    
    def test_validation_level_names(self):
        """Test that validation levels have correct names."""
        assert ValidationLevel.SCRIPT_CONTRACT.name == "SCRIPT_CONTRACT"
        assert ValidationLevel.CONTRACT_SPEC.name == "CONTRACT_SPEC"
        assert ValidationLevel.SPEC_DEPENDENCY.name == "SPEC_DEPENDENCY"
        assert ValidationLevel.BUILDER_CONFIG.name == "BUILDER_CONFIG"


class TestStepTypeCategory:
    """Test StepTypeCategory enum."""
    
    def test_step_type_category_values(self):
        """Test that step type categories have correct values."""
        assert StepTypeCategory.SCRIPT_BASED.value == "script_based"
        assert StepTypeCategory.CONTRACT_BASED.value == "contract_based"
        assert StepTypeCategory.NON_SCRIPT.value == "non_script"
        assert StepTypeCategory.CONFIG_ONLY.value == "config_only"
        assert StepTypeCategory.EXCLUDED.value == "excluded"


class TestValidationRuleset:
    """Test ValidationRuleset dataclass."""
    
    def test_validation_ruleset_creation(self):
        """Test creating a ValidationRuleset."""
        ruleset = ValidationRuleset(
            step_type="Processing",
            category=StepTypeCategory.SCRIPT_BASED,
            enabled_levels={ValidationLevel.SCRIPT_CONTRACT, ValidationLevel.CONTRACT_SPEC},
            level_4_validator_class="ProcessingStepBuilderValidator",
            skip_reason=None,
            examples=["TabularPreprocessing"]
        )
        
        assert ruleset.step_type == "Processing"
        assert ruleset.category == StepTypeCategory.SCRIPT_BASED
        assert ValidationLevel.SCRIPT_CONTRACT in ruleset.enabled_levels
        assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
        assert ruleset.level_4_validator_class == "ProcessingStepBuilderValidator"
        assert ruleset.examples == ["TabularPreprocessing"]


class TestValidationRulesets:
    """Test VALIDATION_RULESETS configuration."""
    
    def test_script_based_steps_configuration(self):
        """Test that script-based steps have full 4-level validation."""
        script_based_steps = ["Processing", "Training"]
        
        for step_type in script_based_steps:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.SCRIPT_BASED
            assert len(ruleset.enabled_levels) == 4
            assert ValidationLevel.SCRIPT_CONTRACT in ruleset.enabled_levels
            assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.level_4_validator_class is not None
    
    def test_contract_based_steps_configuration(self):
        """Test that contract-based steps skip Level 1."""
        contract_based_steps = ["CradleDataLoading", "MimsModelRegistrationProcessing"]
        
        for step_type in contract_based_steps:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.CONTRACT_BASED
            assert len(ruleset.enabled_levels) == 3
            assert ValidationLevel.SCRIPT_CONTRACT not in ruleset.enabled_levels
            assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.skip_reason is not None
    
    def test_non_script_steps_configuration(self):
        """Test that non-script steps skip Levels 1-2."""
        non_script_steps = ["CreateModel", "Transform", "RegisterModel"]
        
        for step_type in non_script_steps:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.NON_SCRIPT
            assert len(ruleset.enabled_levels) == 2
            assert ValidationLevel.SCRIPT_CONTRACT not in ruleset.enabled_levels
            assert ValidationLevel.CONTRACT_SPEC not in ruleset.enabled_levels
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.skip_reason is not None
    
    def test_config_only_steps_configuration(self):
        """Test that config-only steps have Level 3 (universal) and Level 4."""
        config_only_steps = ["Lambda"]
        
        for step_type in config_only_steps:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.CONFIG_ONLY
            assert len(ruleset.enabled_levels) == 2  # Level 3 + Level 4
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels  # Universal Level 3
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.skip_reason is not None
    
    def test_excluded_steps_configuration(self):
        """Test that excluded steps have no validation levels."""
        excluded_steps = ["Base", "Utility"]
        
        for step_type in excluded_steps:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.EXCLUDED
            assert len(ruleset.enabled_levels) == 0
            assert ruleset.skip_reason is not None
    
    def test_universal_level_3_validation(self):
        """Test that all non-excluded steps have Level 3 validation."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if ruleset.category != StepTypeCategory.EXCLUDED:
                assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels, \
                    f"Step type {step_type} missing universal Level 3 validation"


class TestConfigurationAPI:
    """Test configuration API functions."""
    
    def test_get_validation_ruleset(self):
        """Test getting validation ruleset for step type."""
        # Test existing step type
        ruleset = get_validation_ruleset("Processing")
        assert ruleset is not None
        assert ruleset.step_type == "Processing"
        
        # Test non-existing step type
        ruleset = get_validation_ruleset("NonExistentStep")
        assert ruleset is None
    
    def test_is_validation_level_enabled(self):
        """Test checking if validation level is enabled."""
        # Test script-based step
        assert is_validation_level_enabled("Processing", ValidationLevel.SCRIPT_CONTRACT) == True
        assert is_validation_level_enabled("Processing", ValidationLevel.CONTRACT_SPEC) == True
        assert is_validation_level_enabled("Processing", ValidationLevel.SPEC_DEPENDENCY) == True
        assert is_validation_level_enabled("Processing", ValidationLevel.BUILDER_CONFIG) == True
        
        # Test non-script step
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SCRIPT_CONTRACT) == False
        assert is_validation_level_enabled("CreateModel", ValidationLevel.CONTRACT_SPEC) == False
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SPEC_DEPENDENCY) == True
        assert is_validation_level_enabled("CreateModel", ValidationLevel.BUILDER_CONFIG) == True
        
        # Test excluded step
        assert is_validation_level_enabled("Base", ValidationLevel.SCRIPT_CONTRACT) == False
        assert is_validation_level_enabled("Base", ValidationLevel.SPEC_DEPENDENCY) == False
        
        # Test non-existing step type
        assert is_validation_level_enabled("NonExistentStep", ValidationLevel.SCRIPT_CONTRACT) == False
    
    def test_get_enabled_validation_levels(self):
        """Test getting enabled validation levels."""
        # Test script-based step
        levels = get_enabled_validation_levels("Processing")
        assert len(levels) == 4
        assert ValidationLevel.SCRIPT_CONTRACT in levels
        assert ValidationLevel.CONTRACT_SPEC in levels
        assert ValidationLevel.SPEC_DEPENDENCY in levels
        assert ValidationLevel.BUILDER_CONFIG in levels
        
        # Test non-script step
        levels = get_enabled_validation_levels("CreateModel")
        assert len(levels) == 2
        assert ValidationLevel.SPEC_DEPENDENCY in levels
        assert ValidationLevel.BUILDER_CONFIG in levels
        
        # Test excluded step
        levels = get_enabled_validation_levels("Base")
        assert len(levels) == 0
        
        # Test non-existing step type
        levels = get_enabled_validation_levels("NonExistentStep")
        assert len(levels) == 0
    
    def test_get_level_4_validator_class(self):
        """Test getting Level 4 validator class."""
        # Test step with validator class
        validator_class = get_level_4_validator_class("Processing")
        assert validator_class == "ProcessingStepBuilderValidator"
        
        validator_class = get_level_4_validator_class("Training")
        assert validator_class == "TrainingStepBuilderValidator"
        
        # Test non-existing step type
        validator_class = get_level_4_validator_class("NonExistentStep")
        assert validator_class is None
    
    def test_is_step_type_excluded(self):
        """Test checking if step type is excluded."""
        # Test excluded steps
        assert is_step_type_excluded("Base") == True
        assert is_step_type_excluded("Utility") == True
        
        # Test non-excluded steps
        assert is_step_type_excluded("Processing") == False
        assert is_step_type_excluded("Training") == False
        assert is_step_type_excluded("CreateModel") == False
        
        # Test non-existing step type
        assert is_step_type_excluded("NonExistentStep") == False
    
    def test_get_step_types_by_category(self):
        """Test getting step types by category."""
        # Test script-based category
        script_based = get_step_types_by_category(StepTypeCategory.SCRIPT_BASED)
        assert "Processing" in script_based
        assert "Training" in script_based
        
        # Test excluded category
        excluded = get_step_types_by_category(StepTypeCategory.EXCLUDED)
        assert "Base" in excluded
        assert "Utility" in excluded
        
        # Test non-script category
        non_script = get_step_types_by_category(StepTypeCategory.NON_SCRIPT)
        assert "CreateModel" in non_script
        assert "Transform" in non_script
        assert "RegisterModel" in non_script
    
    def test_get_all_step_types(self):
        """Test getting all configured step types."""
        all_types = get_all_step_types()
        
        # Check that all expected step types are present
        expected_types = [
            "Processing", "Training", "CradleDataLoading", "MimsModelRegistrationProcessing",
            "CreateModel", "Transform", "RegisterModel", "Lambda", "Base", "Utility"
        ]
        
        for step_type in expected_types:
            assert step_type in all_types
        
        # Check that we have the expected number of step types
        assert len(all_types) == len(expected_types)
    
    def test_validate_step_type_configuration(self):
        """Test configuration validation."""
        issues = validate_step_type_configuration()
        
        # Configuration should be valid with no issues
        assert len(issues) == 0, f"Configuration validation failed with issues: {issues}"


class TestRegistryIntegration:
    """Test registry integration functions."""
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_get_validation_ruleset_for_step_name_success(self, mock_get_step_type):
        """Test getting validation ruleset for step name with successful registry lookup."""
        mock_get_step_type.return_value = "Processing"
        
        ruleset = get_validation_ruleset_for_step_name("TabularPreprocessing")
        
        assert ruleset is not None
        assert ruleset.step_type == "Processing"
        mock_get_step_type.assert_called_once_with("TabularPreprocessing", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_get_validation_ruleset_for_step_name_fallback(self, mock_get_step_type):
        """Test fallback to Processing when registry lookup fails."""
        mock_get_step_type.side_effect = ValueError("Unknown step")
        
        ruleset = get_validation_ruleset_for_step_name("UnknownStep")
        
        assert ruleset is not None
        assert ruleset.step_type == "Processing"  # Fallback
        mock_get_step_type.assert_called_once_with("UnknownStep", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_is_validation_level_enabled_for_step_name_success(self, mock_get_step_type):
        """Test checking validation level for step name with successful registry lookup."""
        mock_get_step_type.return_value = "Processing"
        
        result = is_validation_level_enabled_for_step_name("TabularPreprocessing", ValidationLevel.SCRIPT_CONTRACT)
        
        assert result == True
        mock_get_step_type.assert_called_once_with("TabularPreprocessing", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_is_validation_level_enabled_for_step_name_fallback(self, mock_get_step_type):
        """Test fallback when registry lookup fails."""
        mock_get_step_type.side_effect = ImportError("Registry not available")
        
        result = is_validation_level_enabled_for_step_name("UnknownStep", ValidationLevel.SCRIPT_CONTRACT)
        
        assert result == True  # Processing fallback has all levels enabled
        mock_get_step_type.assert_called_once_with("UnknownStep", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_get_enabled_validation_levels_for_step_name_success(self, mock_get_step_type):
        """Test getting enabled levels for step name with successful registry lookup."""
        mock_get_step_type.return_value = "CreateModel"
        
        levels = get_enabled_validation_levels_for_step_name("XGBoostModel")
        
        assert len(levels) == 2
        assert ValidationLevel.SPEC_DEPENDENCY in levels
        assert ValidationLevel.BUILDER_CONFIG in levels
        mock_get_step_type.assert_called_once_with("XGBoostModel", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_get_enabled_validation_levels_for_step_name_fallback(self, mock_get_step_type):
        """Test fallback when registry lookup fails."""
        mock_get_step_type.side_effect = ValueError("Unknown step")
        
        levels = get_enabled_validation_levels_for_step_name("UnknownStep")
        
        assert len(levels) == 4  # Processing fallback has all 4 levels
        mock_get_step_type.assert_called_once_with("UnknownStep", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_is_step_name_excluded_success(self, mock_get_step_type):
        """Test checking if step name is excluded with successful registry lookup."""
        mock_get_step_type.return_value = "Base"
        
        result = is_step_name_excluded("BaseConfig")
        
        assert result == True
        mock_get_step_type.assert_called_once_with("BaseConfig", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_is_step_name_excluded_fallback(self, mock_get_step_type):
        """Test fallback when registry lookup fails."""
        mock_get_step_type.side_effect = ImportError("Registry not available")
        
        result = is_step_name_excluded("UnknownStep")
        
        assert result == False  # Fallback to not excluded
        mock_get_step_type.assert_called_once_with("UnknownStep", None)
    
    @patch('cursus.registry.step_names.get_sagemaker_step_type')
    def test_workspace_id_parameter_passed(self, mock_get_step_type):
        """Test that workspace_id parameter is passed to registry functions."""
        mock_get_step_type.return_value = "Processing"
        workspace_id = "test_workspace"
        
        get_validation_ruleset_for_step_name("TabularPreprocessing", workspace_id)
        
        mock_get_step_type.assert_called_once_with("TabularPreprocessing", workspace_id)


class TestConfigurationConsistency:
    """Test configuration consistency and edge cases."""
    
    def test_all_step_types_have_examples(self):
        """Test that all step types have example step names."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            assert ruleset.examples is not None, f"Step type {step_type} missing examples"
            assert len(ruleset.examples) > 0, f"Step type {step_type} has empty examples"
    
    def test_all_non_excluded_have_level_4_validator(self):
        """Test that all non-excluded step types have Level 4 validator class."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if ruleset.category != StepTypeCategory.EXCLUDED:
                assert ruleset.level_4_validator_class is not None, \
                    f"Step type {step_type} missing Level 4 validator class"
    
    def test_excluded_steps_have_skip_reason(self):
        """Test that excluded steps have skip reason."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if ruleset.category == StepTypeCategory.EXCLUDED:
                assert ruleset.skip_reason is not None, \
                    f"Excluded step type {step_type} missing skip reason"
    
    def test_step_type_matches_key(self):
        """Test that step_type field matches dictionary key."""
        for key, ruleset in VALIDATION_RULESETS.items():
            assert ruleset.step_type == key, \
                f"Step type mismatch: key={key}, step_type={ruleset.step_type}"
    
    def test_level_4_validator_naming_convention(self):
        """Test that Level 4 validator classes follow naming convention."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if ruleset.level_4_validator_class:
                expected_suffix = "StepBuilderValidator"
                assert ruleset.level_4_validator_class.endswith(expected_suffix), \
                    f"Validator class {ruleset.level_4_validator_class} doesn't follow naming convention"


if __name__ == "__main__":
    pytest.main([__file__])
