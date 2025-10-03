"""
Test module for validation ruleset configuration.

Tests the validation ruleset that defines step types, validation priorities,
and rule application logic.
"""

import pytest
from typing import Dict, Any, List

from cursus.validation.alignment.config.validation_ruleset import (
    VALIDATION_RULESETS,
    ValidationLevel,
    StepTypeCategory,
    ValidationRuleset,
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
    is_step_name_excluded
)


class TestValidationRuleset:
    """Test cases for validation ruleset configuration."""

    def test_validation_rulesets_structure(self):
        """Test that VALIDATION_RULESETS has correct structure."""
        assert isinstance(VALIDATION_RULESETS, dict)
        assert len(VALIDATION_RULESETS) > 0
        
        # Check that all rulesets have required structure
        for step_type, ruleset in VALIDATION_RULESETS.items():
            assert isinstance(step_type, str)
            assert isinstance(ruleset, ValidationRuleset)
            assert isinstance(ruleset.category, StepTypeCategory)
            assert isinstance(ruleset.enabled_levels, set)

    def test_step_type_categories_are_valid(self):
        """Test that all step type categories are valid enum values."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            assert isinstance(ruleset.category, StepTypeCategory)
            assert ruleset.category in [
                StepTypeCategory.SCRIPT_BASED,
                StepTypeCategory.CONTRACT_BASED,
                StepTypeCategory.NON_SCRIPT,
                StepTypeCategory.CONFIG_ONLY,
                StepTypeCategory.EXCLUDED
            ]

    def test_get_validation_ruleset(self):
        """Test get_validation_ruleset function."""
        # Test valid step type
        ruleset = get_validation_ruleset("Processing")
        assert isinstance(ruleset, ValidationRuleset)
        assert ruleset.step_type == "Processing"
        
        # Test invalid step type
        ruleset = get_validation_ruleset("InvalidStepType")
        assert ruleset is None

    def test_is_validation_level_enabled(self):
        """Test is_validation_level_enabled function."""
        # Test script-based step types (should have all levels enabled)
        assert is_validation_level_enabled("Processing", ValidationLevel.SCRIPT_CONTRACT) is True
        assert is_validation_level_enabled("Processing", ValidationLevel.CONTRACT_SPEC) is True
        assert is_validation_level_enabled("Processing", ValidationLevel.SPEC_DEPENDENCY) is True
        assert is_validation_level_enabled("Processing", ValidationLevel.BUILDER_CONFIG) is True
        
        # Test non-script step types (should skip script/contract levels)
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SCRIPT_CONTRACT) is False
        assert is_validation_level_enabled("CreateModel", ValidationLevel.CONTRACT_SPEC) is False
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SPEC_DEPENDENCY) is True
        assert is_validation_level_enabled("CreateModel", ValidationLevel.BUILDER_CONFIG) is True
        
        # Test excluded step types (should have no levels enabled)
        assert is_validation_level_enabled("Base", ValidationLevel.SCRIPT_CONTRACT) is False
        assert is_validation_level_enabled("Base", ValidationLevel.SPEC_DEPENDENCY) is False
        
        # Test invalid step type
        assert is_validation_level_enabled("InvalidStepType", ValidationLevel.SCRIPT_CONTRACT) is False

    def test_get_enabled_validation_levels(self):
        """Test get_enabled_validation_levels function."""
        # Test script-based step type
        processing_levels = get_enabled_validation_levels("Processing")
        assert isinstance(processing_levels, set)
        assert ValidationLevel.SCRIPT_CONTRACT in processing_levels
        assert ValidationLevel.CONTRACT_SPEC in processing_levels
        assert ValidationLevel.SPEC_DEPENDENCY in processing_levels
        assert ValidationLevel.BUILDER_CONFIG in processing_levels
        
        # Test non-script step type
        createmodel_levels = get_enabled_validation_levels("CreateModel")
        assert isinstance(createmodel_levels, set)
        assert ValidationLevel.SCRIPT_CONTRACT not in createmodel_levels
        assert ValidationLevel.CONTRACT_SPEC not in createmodel_levels
        assert ValidationLevel.SPEC_DEPENDENCY in createmodel_levels
        assert ValidationLevel.BUILDER_CONFIG in createmodel_levels
        
        # Test excluded step type
        base_levels = get_enabled_validation_levels("Base")
        assert isinstance(base_levels, set)
        assert len(base_levels) == 0
        
        # Test invalid step type
        invalid_levels = get_enabled_validation_levels("InvalidStepType")
        assert isinstance(invalid_levels, set)
        assert len(invalid_levels) == 0

    def test_is_step_type_excluded(self):
        """Test is_step_type_excluded function."""
        # Test excluded step types
        assert is_step_type_excluded("Base") is True
        assert is_step_type_excluded("Utility") is True
        
        # Test non-excluded step types
        assert is_step_type_excluded("Processing") is False
        assert is_step_type_excluded("Training") is False
        assert is_step_type_excluded("CreateModel") is False
        assert is_step_type_excluded("Transform") is False
        
        # Test invalid step type
        assert is_step_type_excluded("InvalidStepType") is False

    def test_get_all_step_types(self):
        """Test get_all_step_types function."""
        step_types = get_all_step_types()
        assert isinstance(step_types, list)
        assert len(step_types) > 0
        assert "Processing" in step_types
        assert "Training" in step_types
        assert "CreateModel" in step_types
        assert "Transform" in step_types
        assert "Base" in step_types
        assert "Utility" in step_types

    def test_validate_step_type_configuration(self):
        """Test validate_step_type_configuration function."""
        issues = validate_step_type_configuration()
        assert isinstance(issues, list)
        
        # Should have no consistency issues in well-formed ruleset
        if issues:
            # Print issues for debugging if any exist
            for issue in issues:
                print(f"Configuration issue: {issue}")

    def test_get_step_types_by_category(self):
        """Test get_step_types_by_category function."""
        # Test script-based step types
        script_based_types = get_step_types_by_category(StepTypeCategory.SCRIPT_BASED)
        assert isinstance(script_based_types, list)
        assert "Processing" in script_based_types
        assert "Training" in script_based_types
        
        # Test non-script step types
        non_script_types = get_step_types_by_category(StepTypeCategory.NON_SCRIPT)
        assert isinstance(non_script_types, list)
        assert "CreateModel" in non_script_types
        assert "Transform" in non_script_types
        
        # Test excluded step types
        excluded_types = get_step_types_by_category(StepTypeCategory.EXCLUDED)
        assert isinstance(excluded_types, list)
        assert "Base" in excluded_types
        assert "Utility" in excluded_types

    def test_get_level_4_validator_class(self):
        """Test get_level_4_validator_class function."""
        # Test step types with Level 4 validators
        processing_validator = get_level_4_validator_class("Processing")
        assert processing_validator == "ProcessingStepBuilderValidator"
        
        training_validator = get_level_4_validator_class("Training")
        assert training_validator == "TrainingStepBuilderValidator"
        
        createmodel_validator = get_level_4_validator_class("CreateModel")
        assert createmodel_validator == "CreateModelStepBuilderValidator"
        
        # Test excluded step type
        base_validator = get_level_4_validator_class("Base")
        assert base_validator is None
        
        # Test invalid step type
        invalid_validator = get_level_4_validator_class("InvalidStepType")
        assert invalid_validator is None

    def test_registry_integration_functions(self):
        """Test registry integration functions."""
        # Test get_validation_ruleset_for_step_name
        # Note: This may fall back to Processing if registry lookup fails
        ruleset = get_validation_ruleset_for_step_name("test_processing_step")
        assert isinstance(ruleset, ValidationRuleset)
        
        # Test is_validation_level_enabled_for_step_name
        enabled = is_validation_level_enabled_for_step_name("test_processing_step", ValidationLevel.SPEC_DEPENDENCY)
        assert isinstance(enabled, bool)
        
        # Test get_enabled_validation_levels_for_step_name
        levels = get_enabled_validation_levels_for_step_name("test_processing_step")
        assert isinstance(levels, set)
        
        # Test is_step_name_excluded
        excluded = is_step_name_excluded("test_processing_step")
        assert isinstance(excluded, bool)

    def test_validation_level_enum_values(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.SCRIPT_CONTRACT.value == 1
        assert ValidationLevel.CONTRACT_SPEC.value == 2
        assert ValidationLevel.SPEC_DEPENDENCY.value == 3
        assert ValidationLevel.BUILDER_CONFIG.value == 4

    def test_step_type_category_enum_values(self):
        """Test StepTypeCategory enum values."""
        assert StepTypeCategory.SCRIPT_BASED.value == "script_based"
        assert StepTypeCategory.CONTRACT_BASED.value == "contract_based"
        assert StepTypeCategory.NON_SCRIPT.value == "non_script"
        assert StepTypeCategory.CONFIG_ONLY.value == "config_only"
        assert StepTypeCategory.EXCLUDED.value == "excluded"

    def test_validation_ruleset_dataclass(self):
        """Test ValidationRuleset dataclass."""
        # Test creating a ValidationRuleset instance
        ruleset = ValidationRuleset(
            step_type="TestStep",
            category=StepTypeCategory.SCRIPT_BASED,
            enabled_levels={ValidationLevel.SCRIPT_CONTRACT, ValidationLevel.CONTRACT_SPEC},
            level_4_validator_class="TestValidator",
            skip_reason=None,
            examples=["TestExample"]
        )
        
        assert ruleset.step_type == "TestStep"
        assert ruleset.category == StepTypeCategory.SCRIPT_BASED
        assert ValidationLevel.SCRIPT_CONTRACT in ruleset.enabled_levels
        assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
        assert ruleset.level_4_validator_class == "TestValidator"
        assert ruleset.examples == ["TestExample"]

    def test_api_function_error_handling(self):
        """Test that API functions handle errors gracefully."""
        # Test with None input
        assert get_validation_ruleset(None) is None
        assert is_validation_level_enabled(None, ValidationLevel.SCRIPT_CONTRACT) is False
        assert get_enabled_validation_levels(None) == set()
        assert get_level_4_validator_class(None) is None
        assert is_step_type_excluded(None) is False
        
        # Test with empty string
        assert get_validation_ruleset("") is None
        assert is_validation_level_enabled("", ValidationLevel.SCRIPT_CONTRACT) is False
        assert get_enabled_validation_levels("") == set()
        assert get_level_4_validator_class("") is None
        assert is_step_type_excluded("") is False

    def test_ruleset_completeness(self):
        """Test that all rulesets are complete and consistent."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            # Check required fields
            assert isinstance(ruleset.step_type, str)
            assert isinstance(ruleset.category, StepTypeCategory)
            assert isinstance(ruleset.enabled_levels, set)
            
            # Check that excluded steps have no enabled levels
            if ruleset.category == StepTypeCategory.EXCLUDED:
                assert len(ruleset.enabled_levels) == 0
                assert ruleset.skip_reason is not None
            
            # Check that non-excluded steps have Level 3 enabled (universal requirement)
            if ruleset.category != StepTypeCategory.EXCLUDED:
                assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels
            
            # Check that Level 4 validator class is specified for non-excluded steps
            if ruleset.category != StepTypeCategory.EXCLUDED:
                assert ruleset.level_4_validator_class is not None
            
            # Check that examples are provided
            assert ruleset.examples is not None
            assert len(ruleset.examples) > 0
