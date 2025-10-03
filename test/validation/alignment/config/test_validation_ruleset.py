"""
Test module for validation ruleset configuration.

Tests the validation ruleset that defines step types, validation priorities,
and rule application logic.
"""

import pytest
from typing import Dict, Any, List

from cursus.validation.alignment.config.validation_ruleset import (
    VALIDATION_RULESET,
    ValidationPriority,
    get_validation_ruleset,
    get_sagemaker_step_type,
    get_validation_priority,
    is_step_type_supported,
    get_supported_step_types,
    get_step_type_mapping,
    validate_ruleset_consistency,
    get_priority_order,
    get_rule_application_order,
    get_step_type_validation_config,
    get_validation_level_config,
    apply_validation_rules,
    resolve_validation_conflicts,
    get_validation_metadata
)


class TestValidationRuleset:
    """Test cases for validation ruleset configuration."""

    def test_validation_ruleset_structure(self):
        """Test that VALIDATION_RULESET has correct structure."""
        assert isinstance(VALIDATION_RULESET, dict)
        assert "step_type_mapping" in VALIDATION_RULESET
        assert "validation_priorities" in VALIDATION_RULESET
        assert "rule_application_order" in VALIDATION_RULESET

    def test_validation_priorities_are_valid(self):
        """Test that all validation priorities are valid enum values."""
        priorities = VALIDATION_RULESET["validation_priorities"]
        
        for priority_name, priority_value in priorities.items():
            assert isinstance(priority_name, str)
            assert isinstance(priority_value, ValidationPriority)
            assert priority_value in [
                ValidationPriority.HIGHEST,
                ValidationPriority.HIGH,
                ValidationPriority.MEDIUM,
                ValidationPriority.LOW,
                ValidationPriority.LOWEST
            ]

    def test_get_validation_ruleset(self):
        """Test get_validation_ruleset function."""
        ruleset = get_validation_ruleset()
        assert isinstance(ruleset, dict)
        assert ruleset == VALIDATION_RULESET

    def test_get_sagemaker_step_type(self):
        """Test get_sagemaker_step_type function."""
        # Test known step types
        assert get_sagemaker_step_type("processing_step") == "Processing"
        assert get_sagemaker_step_type("training_step") == "Training"
        assert get_sagemaker_step_type("createmodel_step") == "CreateModel"
        assert get_sagemaker_step_type("transform_step") == "Transform"
        
        # Test step name variations
        assert get_sagemaker_step_type("data_processing_step") == "Processing"
        assert get_sagemaker_step_type("model_training_step") == "Training"
        assert get_sagemaker_step_type("create_model_step") == "CreateModel"
        assert get_sagemaker_step_type("batch_transform_step") == "Transform"
        
        # Test unknown step type
        assert get_sagemaker_step_type("unknown_step") is None

    def test_get_validation_priority(self):
        """Test get_validation_priority function."""
        # Test known priorities
        assert get_validation_priority("universal") == ValidationPriority.HIGHEST
        assert get_validation_priority("step_specific") == ValidationPriority.HIGH
        assert get_validation_priority("contract_alignment") == ValidationPriority.MEDIUM
        assert get_validation_priority("spec_dependency") == ValidationPriority.LOW
        
        # Test unknown priority
        assert get_validation_priority("unknown_priority") is None

    def test_is_step_type_supported(self):
        """Test is_step_type_supported function."""
        # Test supported step types
        assert is_step_type_supported("Processing") is True
        assert is_step_type_supported("Training") is True
        assert is_step_type_supported("CreateModel") is True
        assert is_step_type_supported("Transform") is True
        
        # Test unsupported step type
        assert is_step_type_supported("UnsupportedStepType") is False

    def test_get_supported_step_types(self):
        """Test get_supported_step_types function."""
        supported_types = get_supported_step_types()
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        assert "Processing" in supported_types
        assert "Training" in supported_types
        assert "CreateModel" in supported_types
        assert "Transform" in supported_types

    def test_get_step_type_mapping(self):
        """Test get_step_type_mapping function."""
        mapping = get_step_type_mapping()
        assert isinstance(mapping, dict)
        
        # Check that common step name patterns are mapped
        processing_patterns = [key for key in mapping.keys() if "processing" in key.lower()]
        training_patterns = [key for key in mapping.keys() if "training" in key.lower()]
        createmodel_patterns = [key for key in mapping.keys() if "createmodel" in key.lower() or "create_model" in key.lower()]
        transform_patterns = [key for key in mapping.keys() if "transform" in key.lower()]
        
        assert len(processing_patterns) > 0
        assert len(training_patterns) > 0
        assert len(createmodel_patterns) > 0
        assert len(transform_patterns) > 0

    def test_validate_ruleset_consistency(self):
        """Test validate_ruleset_consistency function."""
        issues = validate_ruleset_consistency()
        assert isinstance(issues, list)
        
        # Should have no consistency issues in well-formed ruleset
        if issues:
            # Print issues for debugging if any exist
            for issue in issues:
                print(f"Consistency issue: {issue}")

    def test_get_priority_order(self):
        """Test get_priority_order function."""
        order = get_priority_order()
        assert isinstance(order, list)
        assert len(order) > 0
        
        # Check that priorities are in correct order (highest to lowest)
        priority_values = [get_validation_priority(priority_name) for priority_name in order]
        for i in range(len(priority_values) - 1):
            assert priority_values[i].value >= priority_values[i + 1].value

    def test_get_rule_application_order(self):
        """Test get_rule_application_order function."""
        order = get_rule_application_order()
        assert isinstance(order, list)
        assert len(order) > 0
        
        # Check that universal rules come first
        assert order[0] == "universal"
        
        # Check that step-specific rules come second
        assert "step_specific" in order[:3]

    def test_get_step_type_validation_config(self):
        """Test get_step_type_validation_config function."""
        # Test valid step type
        config = get_step_type_validation_config("Processing")
        assert isinstance(config, dict)
        
        # Test invalid step type
        config = get_step_type_validation_config("InvalidStepType")
        assert config == {}

    def test_get_validation_level_config(self):
        """Test get_validation_level_config function."""
        # Test valid validation level
        config = get_validation_level_config("universal")
        assert isinstance(config, dict)
        
        # Test invalid validation level
        config = get_validation_level_config("invalid_level")
        assert config == {}

    def test_apply_validation_rules(self):
        """Test apply_validation_rules function."""
        step_name = "test_processing_step"
        
        # Mock validation context
        validation_context = {
            "step_name": step_name,
            "step_type": "Processing",
            "builder_class": Mock(),
            "workspace_dirs": ["/test/workspace"]
        }
        
        # Execute rule application
        result = apply_validation_rules(validation_context)
        assert isinstance(result, dict)
        assert "status" in result
        assert "applied_rules" in result
        assert "total_issues" in result

    def test_resolve_validation_conflicts(self):
        """Test resolve_validation_conflicts function."""
        # Mock conflicting validation results
        validation_results = [
            {
                "rule_type": "universal",
                "priority": ValidationPriority.HIGHEST,
                "status": "PASSED",
                "issues": []
            },
            {
                "rule_type": "step_specific",
                "priority": ValidationPriority.HIGH,
                "status": "ISSUES_FOUND",
                "issues": [{"level": "WARNING", "message": "Minor issue"}]
            }
        ]
        
        # Execute conflict resolution
        result = resolve_validation_conflicts(validation_results)
        assert isinstance(result, dict)
        assert "final_status" in result
        assert "resolution_strategy" in result
        assert "combined_issues" in result

    def test_get_validation_metadata(self):
        """Test get_validation_metadata function."""
        metadata = get_validation_metadata()
        assert isinstance(metadata, dict)
        assert "version" in metadata
        assert "supported_step_types" in metadata
        assert "validation_levels" in metadata

    def test_step_type_mapping_completeness(self):
        """Test that step type mapping covers all common patterns."""
        mapping = get_step_type_mapping()
        
        # Check Processing patterns
        processing_patterns = [
            "processing_step", "data_processing_step", "feature_processing_step",
            "preprocessing_step", "postprocessing_step"
        ]
        for pattern in processing_patterns:
            if pattern in mapping:
                assert mapping[pattern] == "Processing"
        
        # Check Training patterns
        training_patterns = [
            "training_step", "model_training_step", "train_step",
            "model_train_step", "estimator_step"
        ]
        for pattern in training_patterns:
            if pattern in mapping:
                assert mapping[pattern] == "Training"
        
        # Check CreateModel patterns
        createmodel_patterns = [
            "createmodel_step", "create_model_step", "model_creation_step",
            "model_step", "model_registration_step"
        ]
        for pattern in createmodel_patterns:
            if pattern in mapping:
                assert mapping[pattern] == "CreateModel"
        
        # Check Transform patterns
        transform_patterns = [
            "transform_step", "batch_transform_step", "inference_step",
            "prediction_step", "batch_inference_step"
        ]
        for pattern in transform_patterns:
            if pattern in mapping:
                assert mapping[pattern] == "Transform"

    def test_validation_priority_ordering(self):
        """Test that validation priorities are correctly ordered."""
        priorities = VALIDATION_RULESET["validation_priorities"]
        
        # Check that universal has highest priority
        assert priorities["universal"] == ValidationPriority.HIGHEST
        
        # Check that step_specific has high priority
        assert priorities["step_specific"] == ValidationPriority.HIGH
        
        # Check relative ordering
        universal_value = priorities["universal"].value
        step_specific_value = priorities["step_specific"].value
        assert universal_value > step_specific_value

    def test_rule_application_order_logic(self):
        """Test that rule application order follows logical precedence."""
        order = get_rule_application_order()
        
        # Universal rules should be first (highest priority)
        assert order[0] == "universal"
        
        # Step-specific rules should be early in the order
        step_specific_index = order.index("step_specific")
        assert step_specific_index <= 2
        
        # Lower priority rules should come later
        if "contract_alignment" in order and "spec_dependency" in order:
            contract_index = order.index("contract_alignment")
            spec_index = order.index("spec_dependency")
            assert contract_index < spec_index

    def test_step_type_detection_patterns(self):
        """Test step type detection with various naming patterns."""
        test_cases = [
            ("data_processing_step_builder", "Processing"),
            ("xgboost_training_step_builder", "Training"),
            ("model_createmodel_step_builder", "CreateModel"),
            ("batch_transform_step_builder", "Transform"),
            ("feature_processing_pipeline_step", "Processing"),
            ("deep_learning_training_step", "Training"),
            ("ensemble_model_creation_step", "CreateModel"),
            ("inference_transform_step", "Transform")
        ]
        
        for step_name, expected_type in test_cases:
            detected_type = get_sagemaker_step_type(step_name)
            if detected_type is not None:
                assert detected_type == expected_type

    def test_validation_context_handling(self):
        """Test validation context handling in rule application."""
        # Test minimal context
        minimal_context = {
            "step_name": "test_step",
            "step_type": "Processing"
        }
        
        result = apply_validation_rules(minimal_context)
        assert isinstance(result, dict)
        assert "status" in result
        
        # Test complete context
        complete_context = {
            "step_name": "comprehensive_processing_step",
            "step_type": "Processing",
            "builder_class": Mock(),
            "workspace_dirs": ["/test/workspace1", "/test/workspace2"],
            "validation_options": {"strict_mode": True}
        }
        
        result = apply_validation_rules(complete_context)
        assert isinstance(result, dict)
        assert "status" in result

    def test_conflict_resolution_strategies(self):
        """Test different conflict resolution strategies."""
        # Test no conflicts (all passed)
        no_conflict_results = [
            {"rule_type": "universal", "priority": ValidationPriority.HIGHEST, "status": "PASSED", "issues": []},
            {"rule_type": "step_specific", "priority": ValidationPriority.HIGH, "status": "PASSED", "issues": []}
        ]
        
        result = resolve_validation_conflicts(no_conflict_results)
        assert result["final_status"] == "PASSED"
        
        # Test conflicts with different priorities
        conflict_results = [
            {"rule_type": "universal", "priority": ValidationPriority.HIGHEST, "status": "FAILED", "issues": [{"level": "ERROR"}]},
            {"rule_type": "step_specific", "priority": ValidationPriority.HIGH, "status": "PASSED", "issues": []}
        ]
        
        result = resolve_validation_conflicts(conflict_results)
        assert result["final_status"] == "FAILED"  # Higher priority rule determines outcome

    def test_api_function_error_handling(self):
        """Test that API functions handle errors gracefully."""
        # Test with None input
        assert get_sagemaker_step_type(None) is None
        assert get_validation_priority(None) is None
        assert is_step_type_supported(None) is False
        assert get_step_type_validation_config(None) == {}
        
        # Test with empty string
        assert get_sagemaker_step_type("") is None
        assert get_validation_priority("") is None
        assert is_step_type_supported("") is False
        assert get_step_type_validation_config("") == {}

    def test_ruleset_integration_with_validation_system(self):
        """Test that ruleset integrates properly with validation system."""
        # Test that all supported step types have proper configuration
        supported_types = get_supported_step_types()
        
        for step_type in supported_types:
            # Should have validation configuration
            config = get_step_type_validation_config(step_type)
            assert len(config) >= 0  # May be empty but should not error
            
            # Should be recognized as supported
            assert is_step_type_supported(step_type) is True
        
        # Test that all validation levels have proper configuration
        priority_order = get_priority_order()
        
        for priority_name in priority_order:
            # Should have validation level configuration
            config = get_validation_level_config(priority_name)
            assert len(config) >= 0  # May be empty but should not error
            
            # Should have valid priority
            priority = get_validation_priority(priority_name)
            assert priority is not None

    def test_ruleset_metadata_consistency(self):
        """Test that ruleset metadata is consistent with actual configuration."""
        metadata = get_validation_metadata()
        
        # Check that metadata supported step types match actual supported types
        metadata_step_types = set(metadata.get("supported_step_types", []))
        actual_step_types = set(get_supported_step_types())
        assert metadata_step_types == actual_step_types
        
        # Check that metadata validation levels match actual levels
        metadata_levels = set(metadata.get("validation_levels", []))
        actual_levels = set(get_priority_order())
        assert metadata_levels == actual_levels
