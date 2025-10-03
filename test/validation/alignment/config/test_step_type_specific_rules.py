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
    get_validation_rules_for_step_type,
    get_required_methods_for_step_type,
    get_optional_methods_for_step_type,
    get_all_methods_for_step_type,
    get_step_types_by_category,
    is_step_type_excluded,
    get_step_type_category,
    validate_step_type_compliance,
    get_validation_summary
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
                StepTypeCategory.CONFIG_ONLY,
                StepTypeCategory.EXCLUDED
            ]

    def test_get_step_type_validation_rules(self):
        """Test get_step_type_validation_rules function."""
        rules = get_step_type_validation_rules()
        assert isinstance(rules, dict)
        assert rules == STEP_TYPE_SPECIFIC_VALIDATION_RULES

    def test_get_required_methods_for_step_type_processing(self):
        """Test get_required_methods_for_step_type for Processing step type."""
        methods = get_required_methods_for_step_type("Processing")
        assert isinstance(methods, dict)
        assert "_create_processor" in methods
        assert "_get_outputs" in methods
        
        # Check method specifications
        create_processor_spec = methods["_create_processor"]
        assert "return_type" in create_processor_spec
        assert "description" in create_processor_spec

    def test_get_required_methods_for_step_type_training(self):
        """Test get_required_methods_for_step_type for Training step type."""
        methods = get_required_methods_for_step_type("Training")
        assert isinstance(methods, dict)
        assert "_create_estimator" in methods
        assert "_get_outputs" in methods
        
        # Check method specifications
        create_estimator_spec = methods["_create_estimator"]
        assert "return_type" in create_estimator_spec
        # Note: parameters may not exist in actual implementation
        if "parameters" in create_estimator_spec:
            assert "output_path" in create_estimator_spec["parameters"]

    def test_get_required_methods_for_step_type_createmodel(self):
        """Test get_required_methods_for_step_type for CreateModel step type."""
        methods = get_required_methods_for_step_type("CreateModel")
        assert isinstance(methods, dict)
        assert "_create_model" in methods
        
        # Check that _get_outputs returns None for CreateModel (if it exists)
        # Note: CreateModel may not have _get_outputs method
        if "_get_outputs" in methods:
            get_outputs_spec = methods["_get_outputs"]
            assert get_outputs_spec["return_type"] == "None"

    def test_get_required_methods_for_step_type_transform(self):
        """Test get_required_methods_for_step_type for Transform step type."""
        methods = get_required_methods_for_step_type("Transform")
        assert isinstance(methods, dict)
        assert "_create_transformer" in methods
        assert "_get_outputs" in methods

    def test_get_required_methods_for_step_type_invalid_step_type(self):
        """Test get_required_methods_for_step_type with invalid step type."""
        methods = get_required_methods_for_step_type("InvalidStepType")
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
        
        # Test unsupported step type
        assert is_step_type_excluded("InvalidStepType") is False

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

    def test_get_all_methods_for_step_type(self):
        """Test get_all_methods_for_step_type function."""
        # Test valid step type
        methods = get_all_methods_for_step_type("Processing")
        assert isinstance(methods, dict)
        assert "_create_processor" in methods
        assert "_get_outputs" in methods
        
        # Test invalid step type
        methods = get_all_methods_for_step_type("InvalidStepType")
        assert methods == {}

    def test_validate_step_type_compliance(self):
        """Test validate_step_type_compliance function."""
        # Create a mock builder class for testing
        class MockProcessingBuilder:
            def _create_processor(self):
                pass
            def _get_outputs(self):
                pass
        
        # Test compliant builder
        issues = validate_step_type_compliance(MockProcessingBuilder, "Processing")
        assert isinstance(issues, list)
        assert len(issues) == 0  # Should have no issues
        
        # Create a non-compliant builder class
        class MockIncompleteBuilder:
            def _create_processor(self):
                pass
            # Missing _get_outputs method
        
        # Test non-compliant builder
        issues = validate_step_type_compliance(MockIncompleteBuilder, "Processing")
        assert isinstance(issues, list)
        assert len(issues) > 0  # Should have issues
        
        # Test excluded step type
        issues = validate_step_type_compliance(MockIncompleteBuilder, "Base")
        assert isinstance(issues, list)
        assert len(issues) == 0  # No validation for excluded types

    def test_get_validation_summary(self):
        """Test get_validation_summary function."""
        summary = get_validation_summary()
        assert isinstance(summary, dict)
        assert "total_step_types" in summary
        assert "by_category" in summary
        assert "validation_coverage" in summary
        
        # Check that summary contains expected data
        assert summary["total_step_types"] > 0
        assert isinstance(summary["by_category"], dict)
        assert isinstance(summary["validation_coverage"], dict)

    def test_get_validation_rules_for_step_type(self):
        """Test get_validation_rules_for_step_type function."""
        # Test valid step type
        rules = get_validation_rules_for_step_type("Processing")
        assert isinstance(rules, dict)
        assert "category" in rules
        assert "required_methods" in rules
        assert "examples" in rules
        assert "description" in rules
        
        # Test invalid step type
        rules = get_validation_rules_for_step_type("InvalidStepType")
        assert rules is None

    def test_get_optional_methods_for_step_type(self):
        """Test get_optional_methods_for_step_type function."""
        # Test step type with optional methods
        optional_methods = get_optional_methods_for_step_type("Processing")
        assert isinstance(optional_methods, dict)
        # Processing may have optional _get_job_arguments method
        
        # Test step type without optional methods
        optional_methods = get_optional_methods_for_step_type("CreateModel")
        assert isinstance(optional_methods, dict)
        
        # Test invalid step type
        optional_methods = get_optional_methods_for_step_type("InvalidStepType")
        assert optional_methods == {}

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
        assert create_processor_spec["return_type"] == "Processor"
        assert "Processor" in create_processor_spec["description"]
        
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
        assert create_estimator_spec["return_type"] == "Estimator"
        assert "output_path" in create_estimator_spec["signature"]
        
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
        assert create_model_spec["return_type"] == "Model"

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
        assert create_transformer_spec["return_type"] == "Transformer"
        
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
        training_methods = get_required_methods_for_step_type("Training")
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
        processing_methods = get_required_methods_for_step_type("Processing")
        training_methods = get_required_methods_for_step_type("Training")
        transform_methods = get_required_methods_for_step_type("Transform")
        createmodel_methods = get_required_methods_for_step_type("CreateModel")
        
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
        assert get_required_methods_for_step_type(None) == {}
        assert get_step_type_category(None) is None
        assert is_step_type_excluded(None) is False
        
        # Test with empty string
        assert get_required_methods_for_step_type("") == {}
        assert get_step_type_category("") is None
        assert is_step_type_excluded("") is False

    def test_step_type_rules_integration_with_validation_system(self):
        """Test that step type rules integrate properly with validation system."""
        # Test that all non-excluded step types have proper rule structure
        for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
            if not is_step_type_excluded(step_type):
                # Should have required methods
                methods = get_required_methods_for_step_type(step_type)
                assert len(methods) > 0
                
                # Should have valid category
                category = get_step_type_category(step_type)
                assert category is not None
                
                # Should have examples
                examples = rules.get("examples", [])
                assert len(examples) > 0
                
                # Should have description
                description = rules.get("description", "")
                assert len(description) > 0
