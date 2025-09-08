"""
Unit tests for registry validation utilities.

Tests the simplified validation approach implemented in Phase 1 of the
Hybrid Registry Standardization Enforcement Implementation Plan.

Based on redundancy analysis findings:
- Focus on essential validation for new step creation
- Simple regex-based validation patterns
- Clear error messages with examples
- Auto-correction for common naming violations
"""

import pytest
from typing import Dict, List, Any

from cursus.registry.validation_utils import (
    validate_new_step_definition,
    auto_correct_step_definition,
    to_pascal_case,
    get_validation_errors_with_suggestions,
    register_step_with_validation,
    PASCAL_CASE_PATTERN,
    VALID_SAGEMAKER_TYPES
)

class TestValidateNewStepDefinition:
    """Test core validation function for new step definitions."""
    
    def test_valid_step_definition(self):
        """Test validation passes for compliant step definition."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        
        errors = validate_new_step_definition(step_data)
        assert errors == []
    
    def test_missing_step_name(self):
        """Test validation fails for missing step name."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        
        errors = validate_new_step_definition(step_data)
        assert len(errors) == 1
        assert "Step name is required" in errors[0]
    
    def test_invalid_step_name_format(self):
        """Test validation fails for non-PascalCase step names."""
        test_cases = [
            ("my_custom_step", "MyCustomStep"),   # snake_case
            ("myCustomStep", "MyCustomStep"),     # camelCase
            ("my-custom-step", "MyCustomStep"),   # kebab-case
            ("My Custom Step", "MyCustomStep"),   # spaces
            ("123CustomStep", "CustomStep"),      # starts with number - should remove leading digits
            ("my_CUSTOM_step", "MyCustomStep")    # mixed case
        ]
        
        for invalid_name, expected_correction in test_cases:
            step_data = {
                "name": invalid_name,
                "config_class": "MyCustomStepConfig",
                "builder_step_name": "MyCustomStepStepBuilder",
                "sagemaker_step_type": "Processing"
            }
            
            errors = validate_new_step_definition(step_data)
            assert len(errors) >= 1
            assert "must be PascalCase" in errors[0]
            assert expected_correction in errors[0]  # Should suggest correct correction
    
    def test_invalid_config_class_name(self):
        """Test validation fails for config classes not ending with 'Config'."""
        test_cases = [
            "MyCustomStepConfiguration",
            "MyCustomStepSettings", 
            "MyCustomStepParams",
            "MyCustomStep"
        ]
        
        for invalid_config in test_cases:
            step_data = {
                "name": "MyCustomStep",
                "config_class": invalid_config,
                "builder_step_name": "MyCustomStepStepBuilder",
                "sagemaker_step_type": "Processing"
            }
            
            errors = validate_new_step_definition(step_data)
            assert len(errors) >= 1
            assert any("must end with 'Config'" in error for error in errors)
    
    def test_invalid_builder_name(self):
        """Test validation fails for builder names not ending with 'StepBuilder'."""
        test_cases = [
            "MyCustomBuilder",      # Missing 'Step' - should be 'MyCustomStepBuilder'
            "MyCustomStepCreator",  # Wrong suffix
            "MyCustomStep",         # Missing 'Builder' suffix
            "MyCustomStepBuild"     # Incomplete suffix
        ]
        
        for invalid_builder in test_cases:
            step_data = {
                "name": "MyCustomStep",
                "config_class": "MyCustomStepConfig",
                "builder_step_name": invalid_builder,
                "sagemaker_step_type": "Processing"
            }
            
            errors = validate_new_step_definition(step_data)
            assert len(errors) >= 1
            assert any("must end with 'StepBuilder'" in error for error in errors)
    
    def test_invalid_sagemaker_step_type(self):
        """Test validation fails for invalid SageMaker step types."""
        invalid_types = [
            "InvalidType",
            "CustomProcessing",
            "TrainingStep",  # Should be 'Training', not 'TrainingStep'
            "ProcessingStep"  # Should be 'Processing', not 'ProcessingStep'
        ]
        
        for invalid_type in invalid_types:
            step_data = {
                "name": "MyCustomStep",
                "config_class": "MyCustomStepConfig", 
                "builder_step_name": "MyCustomStepStepBuilder",
                "sagemaker_step_type": invalid_type
            }
            
            errors = validate_new_step_definition(step_data)
            assert len(errors) >= 1
            assert any("is invalid" in error for error in errors)
            assert any("Valid types:" in error for error in errors)
    
    def test_multiple_validation_errors(self):
        """Test validation returns multiple errors for multiple violations."""
        step_data = {
            "name": "my_custom_step",  # Invalid PascalCase
            "config_class": "MyCustomStepConfiguration",  # Invalid suffix
            "builder_step_name": "MyCustomBuilder",  # Invalid suffix
            "sagemaker_step_type": "InvalidType"  # Invalid type
        }
        
        errors = validate_new_step_definition(step_data)
        assert len(errors) == 4  # Should have 4 errors
        
        # Check each type of error is present
        error_text = " ".join(errors)
        assert "must be PascalCase" in error_text
        assert "must end with 'Config'" in error_text
        assert "must end with 'StepBuilder'" in error_text
        assert "is invalid" in error_text
    
    def test_optional_fields_validation(self):
        """Test validation handles optional fields correctly."""
        # Test with minimal required fields
        step_data = {
            "name": "MyCustomStep"
        }
        
        errors = validate_new_step_definition(step_data)
        assert errors == []  # Should pass with just name
        
        # Test with empty optional fields
        step_data = {
            "name": "MyCustomStep",
            "config_class": "",
            "builder_step_name": "",
            "sagemaker_step_type": ""
        }
        
        errors = validate_new_step_definition(step_data)
        assert errors == []  # Should pass with empty optional fields

class TestAutoCorrectStepDefinition:
    """Test auto-correction functionality for step definitions."""
    
    def test_auto_correct_step_name(self):
        """Test auto-correction of step names to PascalCase."""
        test_cases = [
            ("my_custom_step", "MyCustomStep"),
            ("my-custom-step", "MyCustomStep"),
            ("my custom step", "MyCustomStep"),
            ("myCustomStep", "MyCustomStep"),  # camelCase to PascalCase
            ("CUSTOM_STEP", "CustomStep")
        ]
        
        for input_name, expected_name in test_cases:
            step_data = {
                "name": input_name,
                "config_class": "SomeConfig",
                "builder_step_name": "SomeBuilder"
            }
            
            corrected = auto_correct_step_definition(step_data)
            assert corrected["name"] == expected_name
    
    def test_auto_correct_config_class(self):
        """Test auto-correction of config class names."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfiguration",
            "builder_step_name": "MyCustomStepStepBuilder"
        }
        
        corrected = auto_correct_step_definition(step_data)
        assert corrected["config_class"] == "MyCustomStepConfig"
    
    def test_auto_correct_builder_name(self):
        """Test auto-correction of builder names."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomBuilder"
        }
        
        corrected = auto_correct_step_definition(step_data)
        assert corrected["builder_step_name"] == "MyCustomStepStepBuilder"
    
    def test_auto_correct_multiple_fields(self):
        """Test auto-correction of multiple fields simultaneously."""
        step_data = {
            "name": "my_custom_step",
            "config_class": "my_custom_configuration",
            "builder_step_name": "my_custom_builder"
        }
        
        corrected = auto_correct_step_definition(step_data)
        assert corrected["name"] == "MyCustomStep"
        assert corrected["config_class"] == "MyCustomStepConfig"
        assert corrected["builder_step_name"] == "MyCustomStepStepBuilder"
    
    def test_auto_correct_preserves_valid_fields(self):
        """Test auto-correction preserves already valid fields."""
        step_data = {
            "name": "MyCustomStep",  # Already valid
            "config_class": "MyCustomStepConfig",  # Already valid
            "builder_step_name": "InvalidBuilder",  # Needs correction
            "sagemaker_step_type": "Processing",  # Should be preserved
            "description": "Test step"  # Should be preserved
        }
        
        corrected = auto_correct_step_definition(step_data)
        assert corrected["name"] == "MyCustomStep"  # Unchanged
        assert corrected["config_class"] == "MyCustomStepConfig"  # Unchanged
        assert corrected["builder_step_name"] == "MyCustomStepStepBuilder"  # Corrected
        assert corrected["sagemaker_step_type"] == "Processing"  # Preserved
        assert corrected["description"] == "Test step"  # Preserved

class TestToPascalCase:
    """Test PascalCase conversion utility function."""
    
    def test_snake_case_conversion(self):
        """Test conversion from snake_case."""
        test_cases = [
            ("my_custom_step", "MyCustomStep"),
            ("xgboost_training", "XgboostTraining"),
            ("cradle_data_loading", "CradleDataLoading"),
            ("single", "Single"),
            ("", "")
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            assert result == expected
    
    def test_kebab_case_conversion(self):
        """Test conversion from kebab-case."""
        test_cases = [
            ("my-custom-step", "MyCustomStep"),
            ("xgboost-training", "XgboostTraining"),
            ("single-word", "SingleWord")
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            assert result == expected
    
    def test_space_separated_conversion(self):
        """Test conversion from space-separated words."""
        test_cases = [
            ("my custom step", "MyCustomStep"),
            ("XGBoost Training", "XgboostTraining"),
            ("single word", "SingleWord")
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            assert result == expected
    
    def test_mixed_separators(self):
        """Test conversion with mixed separators."""
        test_cases = [
            ("my_custom-step name", "MyCustomStepName"),
            ("test_case-with spaces", "TestCaseWithSpaces")
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            assert result == expected
    
    def test_edge_cases(self):
        """Test edge cases for PascalCase conversion."""
        test_cases = [
            ("", ""),  # Empty string
            ("a", "A"),  # Single character
            ("A", "A"),  # Already uppercase
            ("123", "123"),  # Numbers only
            ("_test_", "Test"),  # Leading/trailing underscores
            ("multiple___underscores", "MultipleUnderscores")  # Multiple separators
        ]
        
        for input_text, expected in test_cases:
            result = to_pascal_case(input_text)
            assert result == expected

class TestGetValidationErrorsWithSuggestions:
    """Test detailed error messages with suggestions."""
    
    def test_no_errors_returns_empty_list(self):
        """Test function returns empty list for valid step definition."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        
        errors = get_validation_errors_with_suggestions(step_data)
        assert errors == []
    
    def test_detailed_error_messages(self):
        """Test detailed error messages include helpful context."""
        step_data = {
            "name": "my_custom_step",
            "config_class": "MyCustomStepConfiguration",
            "builder_step_name": "MyCustomBuilder"
        }
        
        errors = get_validation_errors_with_suggestions(step_data)
        
        # Should have error markers
        assert any(error.startswith("❌") for error in errors)
        
        # Should have helpful examples
        error_text = " ".join(errors)
        assert "💡" in error_text  # Should have suggestion markers
        assert "examples:" in error_text.lower()
    
    def test_pascal_case_examples(self):
        """Test PascalCase examples are included for naming violations."""
        step_data = {
            "name": "my_custom_step"
        }
        
        errors = get_validation_errors_with_suggestions(step_data)
        error_text = " ".join(errors)
        
        assert "CradleDataLoading" in error_text
        assert "XGBoostTraining" in error_text
        assert "PyTorchModel" in error_text
    
    def test_config_class_examples(self):
        """Test config class examples are included for config violations."""
        step_data = {
            "name": "MyCustomStep",
            "config_class": "MyCustomStepConfiguration"
        }
        
        errors = get_validation_errors_with_suggestions(step_data)
        error_text = " ".join(errors)
        
        assert "CradleDataLoadConfig" in error_text
        assert "XGBoostTrainingConfig" in error_text
    
    def test_builder_name_examples(self):
        """Test builder name examples are included for builder violations."""
        step_data = {
            "name": "MyCustomStep",
            "builder_step_name": "MyCustomBuilder"
        }
        
        errors = get_validation_errors_with_suggestions(step_data)
        error_text = " ".join(errors)
        
        assert "CradleDataLoadingStepBuilder" in error_text
        assert "XGBoostTrainingStepBuilder" in error_text

class TestRegisterStepWithValidation:
    """Test step registration with validation integration."""
    
    def test_register_valid_step_warn_mode(self):
        """Test registering valid step in warn mode."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {}
        
        warnings = register_step_with_validation(
            "MyCustomStep", step_data, existing_steps, "warn"
        )
        
        assert warnings == []
    
    def test_register_duplicate_step_warn_mode(self):
        """Test registering duplicate step in warn mode."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {"MyCustomStep": {}}
        
        warnings = register_step_with_validation(
            "MyCustomStep", step_data, existing_steps, "warn"
        )
        
        assert len(warnings) == 1
        assert "already exists" in warnings[0]
    
    def test_register_duplicate_step_strict_mode(self):
        """Test registering duplicate step in strict mode raises error."""
        step_data = {
            "config_class": "MyCustomStepConfig",
            "builder_step_name": "MyCustomStepStepBuilder",
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {"MyCustomStep": {}}
        
        with pytest.raises(ValueError) as exc_info:
            register_step_with_validation(
                "MyCustomStep", step_data, existing_steps, "strict"
            )
        
        assert "already exists" in str(exc_info.value)
    
    def test_register_invalid_step_strict_mode(self):
        """Test registering invalid step in strict mode raises error."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",  # Invalid
            "builder_step_name": "MyCustomBuilder",  # Invalid
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {}
        
        with pytest.raises(ValueError) as exc_info:
            register_step_with_validation(
                "my_custom_step", step_data, existing_steps, "strict"
            )
        
        assert "validation failed" in str(exc_info.value)
    
    def test_register_invalid_step_auto_correct_mode(self):
        """Test registering invalid step in auto_correct mode."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",  # Will be corrected
            "builder_step_name": "MyCustomBuilder",  # Will be corrected
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {}
        
        warnings = register_step_with_validation(
            "my_custom_step", step_data, existing_steps, "auto_correct"
        )
        
        assert len(warnings) >= 1
        assert any("Auto-corrected" in warning for warning in warnings)
        assert any("Fixed:" in warning for warning in warnings)
    
    def test_register_invalid_step_warn_mode(self):
        """Test registering invalid step in warn mode."""
        step_data = {
            "config_class": "MyCustomStepConfiguration",  # Invalid
            "builder_step_name": "MyCustomBuilder",  # Invalid
            "sagemaker_step_type": "Processing"
        }
        existing_steps = {}
        
        warnings = register_step_with_validation(
            "my_custom_step", step_data, existing_steps, "warn"
        )
        
        assert len(warnings) >= 3  # Should have warnings for each violation
        assert all("Validation issue:" in warning for warning in warnings)

class TestValidationPatterns:
    """Test validation patterns and constants."""
    
    def test_pascal_case_pattern(self):
        """Test PascalCase regex pattern."""
        valid_names = [
            "MyCustomStep",
            "XGBoostTraining", 
            "CradleDataLoading",
            "A",
            "Step123",
            "PyTorchModel"
        ]
        
        invalid_names = [
            "myCustomStep",  # camelCase
            "my_custom_step",  # snake_case
            "my-custom-step",  # kebab-case
            "123Step",  # starts with number
            "My Custom Step",  # spaces
            ""  # empty
        ]
        
        for name in valid_names:
            assert PASCAL_CASE_PATTERN.match(name), f"'{name}' should be valid PascalCase"
        
        for name in invalid_names:
            assert not PASCAL_CASE_PATTERN.match(name), f"'{name}' should be invalid PascalCase"
    
    def test_valid_sagemaker_types(self):
        """Test valid SageMaker step types."""
        expected_types = {
            'Processing', 'Training', 'Transform', 'CreateModel', 'RegisterModel',
            'Base', 'Utility', 'Lambda', 'CradleDataLoading', 'MimsModelRegistrationProcessing'
        }
        
        assert VALID_SAGEMAKER_TYPES == expected_types
        
        # Test some specific types
        assert "Processing" in VALID_SAGEMAKER_TYPES
        assert "Training" in VALID_SAGEMAKER_TYPES
        assert "CradleDataLoading" in VALID_SAGEMAKER_TYPES
        assert "InvalidType" not in VALID_SAGEMAKER_TYPES

if __name__ == "__main__":
    pytest.main([__file__])
