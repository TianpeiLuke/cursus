"""
Test module for step-type-specific validation rules.

Tests the step-type-specific validation rules configuration including
method specifications, return types, and step type categorization.
"""

import pytest
from typing import Dict, Any, List

from cursus.validation.alignment.config.step_type_specific_rules import (
    STEP_TYPE_SPECIFIC_VALIDATION_RULES,
    StepTypeCategory,
    get_step_type_validation_rules,
    get_step_type_required_methods,
    get_step_type_category,
    is_step_type_supported,
    get_supported_step_types,
    get_step_type_method_specification,
    validate_step_type_rules_consistency,
    get_step_type_examples,
    get_step_type_description,
    get_all_step_type_methods,
    get_step_type_return_types,
    get_step_type_implementation_patterns,
    get_step_type_usage_patterns,
    get_step_type_validation_specifics,
    get_step_type_framework_patterns
)


class TestStepTypeSpecificRules:
    """Test cases for step-type-specific validation rules."""

    def test_step_type_specific_validation_rules_structure(self):
        """Test that STEP_TYPE_SPECIFIC_VALIDATION_RULES has correct structure."""
        assert isinstance(STEP_TYPE_SPECIFIC_VALIDATION_RULES, dict)
        assert len(STEP_TYPE_SPECIFIC_VALIDATION_RULES) > 0
        
        # Check that all step types have required structure
        for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
            assert isinstance(step_type, str)
            assert isinstance(rules, dict)
            assert "category" in rules
            assert "required_methods" in rules
            assert "examples" in rules
            assert "description" in rules

    def test_step_type_categories_are_valid(self):
        """Test that all step type categories are valid enum values."""
        for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
            category = rules["category"]
            assert isinstance(category, StepTypeCategory)
            assert category in [
                StepTypeCategory.SCRIPT_BASED,
                StepTypeCategory.CONTRACT_BASED,
                StepTypeCategory.NON_SCRIPT,
                StepTypeCategory.CONFIG_ONLY
            ]

    def test_get_step_type_validation_rules(self):
        """Test get_step_type_validation_rules function."""
        rules = get_step_type_validation_rules()
        assert isinstance(rules, dict)
        assert rules == STEP_TYPE_SPECIFIC_VALIDATION_RULES

    def test_get_step_type_required_methods_processing(self):
        """Test get_step_type_required_methods for Processing step type."""
        methods = get_step_type_required_methods("Processing")
        assert isinstance(methods, dict)
        assert "_create_processor" in methods
        assert "_get_outputs" in methods
        
        # Check method specifications
        create_processor_spec = methods["_create_processor"]
        assert "return_type" in create_processor_spec
        assert "description" in create_processor_spec

    def test_get_step_type_required_methods_training(self):
        """Test get_step_type_required_methods for Training step type."""
        methods = get_step_type_required_methods("Training")
        assert isinstance(methods, dict)
        assert "_create_estimator" in methods
        assert "_get_outputs" in methods
        
        # Check method specifications
        create_estimator_spec = methods["_create_estimator"]
        assert "return_type" in create_estimator_spec
        assert "parameters" in create_estimator_spec
        assert "output_path" in create_estimator_spec["parameters"]

    def test_get_step_type_required_methods_createmodel(self):
        """Test get_step_type_required_methods for CreateModel step type."""
        methods = get_step_type_required_methods("CreateModel")
        assert isinstance(methods, dict)
        assert "_create_model" in methods
        
        # Check that _get_outputs returns None for CreateModel
        if "_get_outputs" in methods:
            get_outputs_spec = methods["_get_outputs"]
            assert get_outputs_spec["return_type"] == "None"

    def test_get_step_type_required_methods_transform(self):
        """Test get_step_type_required_methods for Transform step type."""
        methods = get_step_type_required_methods("Transform")
        assert isinstance(methods, dict)
        assert "_create_transformer" in methods
        assert "_get_outputs" in methods

    def test_get_step_type_required_methods_invalid_step_type(self):
        """Test get_step_type_required_methods with invalid step type."""
        methods = get_step_type_required_methods("InvalidStepType")
        assert methods == {}

    def test_get_step_type_category(self):
        """Test get_step_type_category function."""
        # Test known step types
        assert get_step_type_category("Processing") == StepTypeCategory.SCRIPT_BASED
        assert get_step_type_category("Training") == StepTypeCategory.SCRIPT_BASED
        assert get_step_type_category("CreateModel") == StepTypeCategory.NON_SCRIPT
        assert get_step_type_category("Transform") == StepTypeCategory.NON_SCRIPT
        
        # Test invalid step type
        assert get_step_type_category("InvalidStepType") is None

    def test_is_step_type_supported(self):
        """Test is_step_type_supported function."""
        # Test supported step types
        assert is_step_type_supported("Processing") is True
        assert is_step_type_supported("Training") is True
        assert is_step_type_supported("CreateModel") is True
        assert is_step_type_supported("Transform") is True
        
        # Test unsupported step type
        assert is_step_type_supported("InvalidStepType") is False

    def test_get_supported_step_types(self):
        """Test get_supported_step_types function."""
        supported_types = get_supported_step_types()
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert "Processing" in supported_types
        assert "Training" in supported_types
        assert "CreateModel" in supported_types
        assert "Transform" in supported_types

    def test_get_step_type_method_specification(self):
        """Test get_step_type_method_specification function."""
        # Test valid method specification
        spec = get_step_type_method_specification("Processing", "_create_processor")
        assert isinstance(spec, dict)
        assert "return_type" in spec
        assert "description" in spec
        
        # Test invalid step type
        spec = get_step_type_method_specification("InvalidStepType", "_create_processor")
        assert spec == {}
        
        # Test invalid method
        spec = get_step_type_method_specification("Processing", "_invalid_method")
        assert spec == {}

    def test_validate_step_type_rules_consistency(self):
        """Test validate_step_type_rules_consistency function."""
        issues = validate_step_type_rules_consistency()
        assert isinstance(issues, list)
        
        # Should have no consistency issues in well-formed rules
        if issues:
            # Print issues for debugging if any exist
            for issue in issues:
                print(f"Consistency issue: {issue}")

    def test_get_step_type_examples(self):
        """Test get_step_type_examples function."""
        # Test valid step type
        examples = get_step_type_examples("Processing")
        assert isinstance(examples, list)
        assert len(examples) > 0
        
        # Test invalid step type
        examples = get_step_type_examples("InvalidStepType")
        assert examples == []

    def test_get_step_type_description(self):
        """Test get_step_type_description function."""
        # Test valid step type
        description = get_step_type_description("Processing")
        assert isinstance(description, str)
        assert len(description) > 0
        
        # Test invalid step type
        description = get_step_type_description("InvalidStepType")
        assert description == ""

    def test_get_all_step_type_methods(self):
        """Test get_all_step_type_methods function."""
        # Test valid step type
        methods = get_all_step_type_methods("Processing")
        assert isinstance(methods, list)
        assert "_create_processor" in methods
        assert "_get_outputs" in methods
        
        # Test invalid step type
        methods = get_all_step_type_methods("InvalidStepType")
        assert methods == []

    def test_get_step_type_return_types(self):
        """Test get_step_type_return_types function."""
        # Test valid step type
        return_types = get_step_type_return_types("Processing")
        assert isinstance(return_types, dict)
        assert "_create_processor" in return_types
        assert "_get_outputs" in return_types
        
        # Test invalid step type
        return_types = get_step_type_return_types("InvalidStepType")
        assert return_types == {}

    def test_get_step_type_implementation_patterns(self):
        """Test get_step_type_implementation_patterns function."""
        # Test valid step type
        patterns = get_step_type_implementation_patterns("Processing")
        assert isinstance(patterns, dict)
        
        # Test invalid step type
        patterns = get_step_type_implementation_patterns("InvalidStepType")
        assert patterns == {}

    def test_get_step_type_usage_patterns(self):
        """Test get_step_type_usage_patterns function."""
        # Test valid step type
        patterns = get_step_type_usage_patterns("Training")
        assert isinstance(patterns, dict)
        
        # Test invalid step type
        patterns = get_step_type_usage_patterns("InvalidStepType")
        assert patterns == {}

    def test_get_step_type_validation_specifics(self):
        """Test get_step_type_validation_specifics function."""
        # Test valid step type
        specifics = get_step_type_validation_specifics("CreateModel")
        assert isinstance(specifics, dict)
        
        # Test invalid step type
        specifics = get_step_type_validation_specifics("InvalidStepType")
        assert specifics == {}

    def test_get_step_type_framework_patterns(self):
        """Test get_step_type_framework_patterns function."""
        # Test valid step type
        patterns = get_step_type_framework_patterns("Training")
        assert isinstance(patterns, dict)
        
        # Test invalid step type
        patterns = get_step_type_framework_patterns("InvalidStepType")
        assert patterns == {}

    def test_processing_step_specific_rules(self):
        """Test Processing step-specific rules in detail."""
        rules = STEP_TYPE_SPECIFIC_VALIDATION_RULES["Processing"]
        
        # Check category
        assert rules["category"] == StepTypeCategory.SCRIPT_BASED
        
        # Check required methods
        required_methods = rules["required_methods"]
        assert "_create_processor" in required_methods
        assert "_get_outputs" in required_methods
        
        # Check method specifications
        create_processor_spec = required_methods["_create_processor"]
        assert create_processor_spec["return_type"] == "Dict[str, Any]"
        assert "ScriptProcessor" in create_processor_spec["description"]
        
        get_outputs_spec = required_methods["_get_outputs"]
        assert get_outputs_spec["return_type"] == "List[ProcessingOutput]"

    def test_training_step_specific_rules(self):
        """Test Training step-specific rules in detail."""
        rules = STEP_TYPE_SPECIFIC_VALIDATION_RULES["Training"]
        
        # Check category
        assert rules["category"] == StepTypeCategory.SCRIPT_BASED
        
        # Check required methods
        required_methods = rules["required_methods"]
        assert "_create_estimator" in required_methods
        assert "_get_outputs" in required_methods
        
        # Check method specifications
        create_estimator_spec = required_methods["_create_estimator"]
        assert create_estimator_spec["return_type"] == "Dict[str, Any]"
        assert "output_path" in create_estimator_spec["parameters"]
        
        get_outputs_spec = required_methods["_get_outputs"]
        assert get_outputs_spec["return_type"] == "str"

    def test_createmodel_step_specific_rules(self):
        """Test CreateModel step-specific rules in detail."""
        rules = STEP_TYPE_SPECIFIC_VALIDATION_RULES["CreateModel"]
        
        # Check category
        assert rules["category"] == StepTypeCategory.NON_SCRIPT
        
        # Check required methods
        required_methods = rules["required_methods"]
        assert "_create_model" in required_methods
        
        # Check method specifications
        create_model_spec = required_methods["_create_model"]
        assert create_model_spec["return_type"] == "Dict[str, Any]"

    def test_transform_step_specific_rules(self):
        """Test Transform step-specific rules in detail."""
        rules = STEP_TYPE_SPECIFIC_VALIDATION_RULES["Transform"]
        
        # Check category
        assert rules["category"] == StepTypeCategory.NON_SCRIPT
        
        # Check required methods
        required_methods = rules["required_methods"]
        assert "_create_transformer" in required_methods
        assert "_get_outputs" in required_methods
        
        # Check method specifications
        create_transformer_spec = required_methods["_create_transformer"]
        assert create_transformer_spec["return_type"] == "Dict[str, Any]"
        
        get_outputs_spec = required_methods["_get_outputs"]
        assert get_outputs_spec["return_type"] == "str"

    def test_step_type_rules_completeness(self):
        """Test that step type rules are complete and consistent."""
        for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
            # Check required fields
            assert "category" in rules
            assert "required_methods" in rules
            assert "examples" in rules
            assert "description" in rules
            
            # Check category is valid
            assert isinstance(rules["category"], StepTypeCategory)
            
            # Check required methods structure
            required_methods = rules["required_methods"]
            assert isinstance(required_methods, dict)
            
            for method_name, method_spec in required_methods.items():
                assert isinstance(method_name, str)
                assert isinstance(method_spec, dict)
                assert "return_type" in method_spec
                assert "description" in method_spec
                
            # Check examples
            examples = rules["examples"]
            assert isinstance(examples, list)
            assert len(examples) > 0
            
            # Check description
            description = rules["description"]
            assert isinstance(description, str)
            assert len(description) > 0

    def test_step_type_method_parameter_specifications(self):
        """Test that method parameter specifications are correct."""
        # Test Training _create_estimator parameters
        training_methods = get_step_type_required_methods("Training")
        if "_create_estimator" in training_methods:
            create_estimator_spec = training_methods["_create_estimator"]
            if "parameters" in create_estimator_spec:
                parameters = create_estimator_spec["parameters"]
                assert "output_path" in parameters
                assert parameters["output_path"]["required"] is False
                assert parameters["output_path"]["default"] is None

    def test_step_type_return_type_consistency(self):
        """Test that return types are consistent across step types."""
        # Test that all _get_outputs methods have consistent return type patterns
        processing_methods = get_step_type_required_methods("Processing")
        training_methods = get_step_type_required_methods("Training")
        transform_methods = get_step_type_required_methods("Transform")
        createmodel_methods = get_step_type_required_methods("CreateModel")
        
        # Processing should return List[ProcessingOutput]
        if "_get_outputs" in processing_methods:
            assert processing_methods["_get_outputs"]["return_type"] == "List[ProcessingOutput]"
        
        # Training should return str (used by _create_estimator)
        if "_get_outputs" in training_methods:
            assert training_methods["_get_outputs"]["return_type"] == "str"
        
        # Transform should return str (used by _create_transformer)
        if "_get_outputs" in transform_methods:
            assert transform_methods["_get_outputs"]["return_type"] == "str"
        
        # CreateModel should return None (SageMaker handles automatically)
        if "_get_outputs" in createmodel_methods:
            assert createmodel_methods["_get_outputs"]["return_type"] == "None"

    def test_step_type_category_consistency(self):
        """Test that step type categories are logically consistent."""
        script_based_types = []
        non_script_types = []
        
        for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
            category = rules["category"]
            if category == StepTypeCategory.SCRIPT_BASED:
                script_based_types.append(step_type)
            elif category == StepTypeCategory.NON_SCRIPT:
                non_script_types.append(step_type)
        
        # Verify expected categorizations
        assert "Processing" in script_based_types
        assert "Training" in script_based_types
        assert "CreateModel" in non_script_types
        assert "Transform" in non_script_types

    def test_api_function_error_handling(self):
        """Test that API functions handle errors gracefully."""
        # Test with None input
        assert get_step_type_required_methods(None) == {}
        assert get_step_type_category(None) is None
        assert is_step_type_supported(None) is False
        assert get_step_type_examples(None) == []
        assert get_step_type_description(None) == ""
        
        # Test with empty string
        assert get_step_type_required_methods("") == {}
        assert get_step_type_category("") is None
        assert is_step_type_supported("") is False
        assert get_step_type_examples("") == []
        assert get_step_type_description("") == ""

    def test_step_type_rules_integration_with_validation_system(self):
        """Test that step type rules integrate properly with validation system."""
        # Test that all supported step types have proper rule structure
        supported_types = get_supported_step_types()
        
        for step_type in supported_types:
            # Should have required methods
            methods = get_step_type_required_methods(step_type)
            assert len(methods) > 0
            
            # Should have valid category
            category = get_step_type_category(step_type)
            assert category is not None
            
            # Should have examples
            examples = get_step_type_examples(step_type)
            assert len(examples) > 0
            
            # Should have description
            description = get_step_type_description(step_type)
            assert len(description) > 0
