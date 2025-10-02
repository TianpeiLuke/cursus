#!/usr/bin/env python3
"""
Comprehensive tests for the refactored validation alignment configuration system.

Tests the configuration-driven validation approach including:
- Validation ruleset configuration
- Step-type-aware validation
- Configuration API functions
- Integration with registry system
"""

import pytest
from unittest.mock import patch, MagicMock

from cursus.validation.alignment.config import (
    ValidationLevel,
    StepTypeCategory,
    ValidationRuleset,
    VALIDATION_RULESETS,
    get_validation_ruleset,
    get_enabled_validation_levels,
    is_validation_level_enabled,
    is_step_type_excluded,
    validate_step_type_configuration,
    get_universal_validation_rules,
    get_step_type_validation_rules,
)


class TestValidationRulesetConfiguration:
    """Test validation ruleset configuration system."""

    def test_validation_level_enum(self):
        """Test ValidationLevel enum values."""
        assert ValidationLevel.SCRIPT_CONTRACT.value == 1
        assert ValidationLevel.CONTRACT_SPEC.value == 2
        assert ValidationLevel.SPEC_DEPENDENCY.value == 3
        assert ValidationLevel.BUILDER_CONFIG.value == 4

    def test_step_type_category_enum(self):
        """Test StepTypeCategory enum values."""
        assert StepTypeCategory.SCRIPT_BASED.value == "script_based"
        assert StepTypeCategory.CONTRACT_BASED.value == "contract_based"
        assert StepTypeCategory.NON_SCRIPT.value == "non_script"
        assert StepTypeCategory.CONFIG_ONLY.value == "config_only"
        assert StepTypeCategory.EXCLUDED.value == "excluded"

    def test_validation_ruleset_dataclass(self):
        """Test ValidationRuleset dataclass."""
        ruleset = ValidationRuleset(
            step_type="Processing",
            category=StepTypeCategory.SCRIPT_BASED,
            enabled_levels={ValidationLevel.SCRIPT_CONTRACT, ValidationLevel.CONTRACT_SPEC},
            level_4_validator_class="ProcessingStepBuilderValidator",
            skip_reason=None,
            examples=["processing_script.py"]
        )
        
        assert ruleset.step_type == "Processing"
        assert ruleset.category == StepTypeCategory.SCRIPT_BASED
        assert len(ruleset.enabled_levels) == 2
        assert ValidationLevel.SCRIPT_CONTRACT in ruleset.enabled_levels
        assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
        assert ruleset.level_4_validator_class == "ProcessingStepBuilderValidator"
        assert ruleset.skip_reason is None
        assert "processing_script.py" in ruleset.examples

    def test_validation_rulesets_completeness(self):
        """Test that VALIDATION_RULESETS contains expected step types."""
        expected_step_types = [
            "Processing", "Training", "CradleDataLoading", "MimsModelRegistrationProcessing",
            "CreateModel", "Transform", "RegisterModel", "Lambda", "Base", "Utility"
        ]
        
        for step_type in expected_step_types:
            assert step_type in VALIDATION_RULESETS, f"Missing ruleset for {step_type}"
            
        # Verify total count
        assert len(VALIDATION_RULESETS) == len(expected_step_types)

    def test_script_based_step_types(self):
        """Test script-based step types have full validation."""
        script_based_types = ["Processing", "Training"]
        
        for step_type in script_based_types:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.SCRIPT_BASED
            assert len(ruleset.enabled_levels) == 4  # All levels enabled
            assert ValidationLevel.SCRIPT_CONTRACT in ruleset.enabled_levels
            assert ValidationLevel.CONTRACT_SPEC in ruleset.enabled_levels
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.level_4_validator_class is not None

    def test_non_script_step_types(self):
        """Test non-script step types skip script/contract validation."""
        non_script_types = ["CreateModel", "Transform", "RegisterModel"]
        
        for step_type in non_script_types:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.NON_SCRIPT
            assert ValidationLevel.SCRIPT_CONTRACT not in ruleset.enabled_levels
            assert ValidationLevel.CONTRACT_SPEC not in ruleset.enabled_levels
            assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels  # Universal
            assert ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels
            assert ruleset.level_4_validator_class is not None

    def test_excluded_step_types(self):
        """Test excluded step types have no validation."""
        excluded_types = ["Base", "Utility"]
        
        for step_type in excluded_types:
            ruleset = VALIDATION_RULESETS[step_type]
            assert ruleset.category == StepTypeCategory.EXCLUDED
            assert len(ruleset.enabled_levels) == 0
            assert ruleset.skip_reason is not None
            assert ruleset.level_4_validator_class is None

    def test_universal_level_3_requirement(self):
        """Test that all non-excluded step types have Level 3 enabled."""
        for step_type, ruleset in VALIDATION_RULESETS.items():
            if ruleset.category != StepTypeCategory.EXCLUDED:
                assert ValidationLevel.SPEC_DEPENDENCY in ruleset.enabled_levels, \
                    f"Step type {step_type} missing universal Level 3 validation"


class TestConfigurationAPI:
    """Test configuration API functions."""

    def test_get_validation_ruleset_existing(self):
        """Test getting validation ruleset for existing step type."""
        ruleset = get_validation_ruleset("Processing")
        assert ruleset is not None
        assert ruleset.step_type == "Processing"
        assert ruleset.category == StepTypeCategory.SCRIPT_BASED

    def test_get_validation_ruleset_nonexistent(self):
        """Test getting validation ruleset for non-existent step type."""
        ruleset = get_validation_ruleset("NonExistentStepType")
        assert ruleset is None

    def test_get_validation_ruleset_case_sensitivity(self):
        """Test that step type matching is case sensitive."""
        ruleset = get_validation_ruleset("processing")  # lowercase
        assert ruleset is None
        
        ruleset = get_validation_ruleset("Processing")  # correct case
        assert ruleset is not None

    def test_get_enabled_validation_levels(self):
        """Test getting enabled validation levels for step type."""
        # Test script-based step type
        levels = get_enabled_validation_levels("Processing")
        assert len(levels) == 4
        assert all(level in levels for level in ValidationLevel)
        
        # Test non-script step type
        levels = get_enabled_validation_levels("CreateModel")
        assert ValidationLevel.SCRIPT_CONTRACT not in levels
        assert ValidationLevel.CONTRACT_SPEC not in levels
        assert ValidationLevel.SPEC_DEPENDENCY in levels
        assert ValidationLevel.BUILDER_CONFIG in levels
        
        # Test excluded step type
        levels = get_enabled_validation_levels("Base")
        assert len(levels) == 0

    def test_is_validation_level_enabled(self):
        """Test checking if specific validation level is enabled."""
        # Test script-based step type
        assert is_validation_level_enabled("Processing", ValidationLevel.SCRIPT_CONTRACT) is True
        assert is_validation_level_enabled("Processing", ValidationLevel.SPEC_DEPENDENCY) is True
        
        # Test non-script step type
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SCRIPT_CONTRACT) is False
        assert is_validation_level_enabled("CreateModel", ValidationLevel.SPEC_DEPENDENCY) is True
        
        # Test excluded step type
        assert is_validation_level_enabled("Base", ValidationLevel.SCRIPT_CONTRACT) is False
        assert is_validation_level_enabled("Base", ValidationLevel.SPEC_DEPENDENCY) is False

    def test_is_step_type_excluded(self):
        """Test checking if step type is excluded from validation."""
        assert is_step_type_excluded("Processing") is False
        assert is_step_type_excluded("CreateModel") is False
        assert is_step_type_excluded("Base") is True
        assert is_step_type_excluded("Utility") is True
        assert is_step_type_excluded("NonExistentType") is False  # Default to not excluded

    def test_validate_step_type_configuration(self):
        """Test configuration validation function."""
        issues = validate_step_type_configuration()
        
        # Should return empty list for valid configuration
        assert isinstance(issues, list)
        # Configuration should be valid, so no issues
        assert len(issues) == 0

    def test_get_universal_validation_rules(self):
        """Test getting universal validation rules."""
        rules = get_universal_validation_rules()
        assert isinstance(rules, dict)
        assert "required_methods" in rules
        assert "inherited_methods" in rules
        
        # Check required methods
        required_methods = rules["required_methods"]
        expected_required = ["validate_configuration", "_get_inputs", "create_step"]
        for method in expected_required:
            assert method in required_methods

    def test_get_step_type_validation_rules(self):
        """Test getting step-type-specific validation rules."""
        rules = get_step_type_validation_rules()
        assert isinstance(rules, dict)
        
        # Check that major step types have rules
        expected_step_types = ["Training", "Processing", "CreateModel", "Transform"]
        for step_type in expected_step_types:
            assert step_type in rules
            assert "required_methods" in rules[step_type]


class TestConfigurationIntegration:
    """Test configuration system integration."""

    @patch('cursus.validation.alignment.config.validation_ruleset.get_sagemaker_step_type')
    def test_registry_integration(self, mock_get_step_type):
        """Test integration with registry system."""
        mock_get_step_type.return_value = "Processing"
        
        # Test that configuration works with registry
        ruleset = get_validation_ruleset("Processing")
        assert ruleset is not None
        assert ruleset.category == StepTypeCategory.SCRIPT_BASED

    def test_configuration_consistency(self):
        """Test configuration consistency across all step types."""
        issues = []
        
        for step_type, ruleset in VALIDATION_RULESETS.items():
            # Check that excluded steps have no enabled levels
            if ruleset.category == StepTypeCategory.EXCLUDED:
                if len(ruleset.enabled_levels) > 0:
                    issues.append(f"Excluded step type {step_type} has enabled levels")
                if not ruleset.skip_reason:
                    issues.append(f"Excluded step type {step_type} missing skip reason")
            
            # Check that non-excluded steps have Level 3 enabled
            else:
                if ValidationLevel.SPEC_DEPENDENCY not in ruleset.enabled_levels:
                    issues.append(f"Non-excluded step type {step_type} missing Level 3")
            
            # Check that step types with Level 4 have validator class
            if ValidationLevel.BUILDER_CONFIG in ruleset.enabled_levels:
                if not ruleset.level_4_validator_class:
                    issues.append(f"Step type {step_type} has Level 4 but no validator class")
        
        assert len(issues) == 0, f"Configuration consistency issues: {issues}"

    def test_performance_optimization_potential(self):
        """Test that configuration enables performance optimization."""
        # Count how many step types skip each level
        level_skip_counts = {level: 0 for level in ValidationLevel}
        
        for step_type, ruleset in VALIDATION_RULESETS.items():
            for level in ValidationLevel:
                if level not in ruleset.enabled_levels:
                    level_skip_counts[level] += 1
        
        # Verify that some step types skip expensive levels
        assert level_skip_counts[ValidationLevel.SCRIPT_CONTRACT] > 0, \
            "No step types skip Level 1 - missing optimization opportunity"
        assert level_skip_counts[ValidationLevel.CONTRACT_SPEC] > 0, \
            "No step types skip Level 2 - missing optimization opportunity"
        
        # Level 3 should be universal (only excluded types skip it)
        excluded_count = sum(1 for r in VALIDATION_RULESETS.values() 
                           if r.category == StepTypeCategory.EXCLUDED)
        assert level_skip_counts[ValidationLevel.SPEC_DEPENDENCY] == excluded_count, \
            "Level 3 should only be skipped by excluded step types"


class TestConfigurationEdgeCases:
    """Test configuration system edge cases."""

    def test_empty_step_type(self):
        """Test handling of empty step type."""
        ruleset = get_validation_ruleset("")
        assert ruleset is None
        
        levels = get_enabled_validation_levels("")
        assert len(levels) == 0
        
        assert is_step_type_excluded("") is False

    def test_none_step_type(self):
        """Test handling of None step type."""
        ruleset = get_validation_ruleset(None)
        assert ruleset is None
        
        levels = get_enabled_validation_levels(None)
        assert len(levels) == 0

    def test_whitespace_step_type(self):
        """Test handling of whitespace step type."""
        ruleset = get_validation_ruleset("  ")
        assert ruleset is None
        
        ruleset = get_validation_ruleset("\t\n")
        assert ruleset is None

    def test_special_characters_step_type(self):
        """Test handling of step types with special characters."""
        special_types = ["Step-Type", "Step_Type", "Step.Type", "Step/Type"]
        
        for step_type in special_types:
            ruleset = get_validation_ruleset(step_type)
            assert ruleset is None  # Should not match any existing rulesets

    def test_very_long_step_type(self):
        """Test handling of very long step type names."""
        long_step_type = "A" * 1000
        ruleset = get_validation_ruleset(long_step_type)
        assert ruleset is None

    def test_unicode_step_type(self):
        """Test handling of unicode step type names."""
        unicode_types = ["Procéssing", "Tráining", "Créate_Modél"]
        
        for step_type in unicode_types:
            ruleset = get_validation_ruleset(step_type)
            assert ruleset is None  # Should not match any existing rulesets


class TestConfigurationPerformance:
    """Test configuration system performance characteristics."""

    def test_configuration_lookup_performance(self):
        """Test that configuration lookups are fast."""
        import time
        
        # Test multiple lookups
        start_time = time.time()
        for _ in range(1000):
            get_validation_ruleset("Processing")
            get_validation_ruleset("CreateModel")
            get_validation_ruleset("Base")
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 1000 lookups)
        assert (end_time - start_time) < 1.0

    def test_level_checking_performance(self):
        """Test that level checking is fast."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            is_validation_level_enabled("Processing", ValidationLevel.SCRIPT_CONTRACT)
            is_validation_level_enabled("CreateModel", ValidationLevel.SPEC_DEPENDENCY)
            is_step_type_excluded("Base")
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 1.0

    def test_configuration_memory_usage(self):
        """Test that configuration doesn't use excessive memory."""
        import sys
        
        # Get size of configuration
        config_size = sys.getsizeof(VALIDATION_RULESETS)
        
        # Should be reasonable size (less than 10KB)
        assert config_size < 10240, f"Configuration too large: {config_size} bytes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
