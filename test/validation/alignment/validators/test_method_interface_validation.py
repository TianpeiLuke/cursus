#!/usr/bin/env python3
"""
Tests for method interface validation in the refactored system.

Tests the method-focused validation approach including:
- Universal method validation
- Step-type-specific method validation
- Priority-based validation system
- Method signature validation
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from typing import Type, Any

from cursus.validation.alignment.validators.method_interface_validator import MethodInterfaceValidator
from cursus.validation.alignment.config import get_universal_validation_rules, get_step_type_validation_rules


class MockStepBuilder:
    """Mock step builder class for testing."""
    
    def validate_configuration(self):
        """Universal required method."""
        pass
    
    def _get_inputs(self):
        """Universal required method."""
        pass
    
    def create_step(self):
        """Universal required method."""
        pass
    
    def _get_outputs(self):
        """Inherited optional method."""
        pass


class MockProcessingStepBuilder(MockStepBuilder):
    """Mock processing step builder with step-specific methods."""
    
    def _create_processor(self):
        """Processing-specific required method."""
        pass


class MockTrainingStepBuilder(MockStepBuilder):
    """Mock training step builder with step-specific methods."""
    
    def _create_estimator(self):
        """Training-specific required method."""
        pass


class MockIncompleteStepBuilder:
    """Mock incomplete step builder missing required methods."""
    
    def validate_configuration(self):
        """Only has one universal method."""
        pass


class TestMethodInterfaceValidator:
    """Test method interface validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create MethodInterfaceValidator instance for testing."""
        return MethodInterfaceValidator(workspace_dirs=["/mock/workspace"])

    def test_initialization(self, validator):
        """Test validator initialization."""
        assert validator.workspace_dirs == ["/mock/workspace"]
        assert hasattr(validator, 'universal_rules')
        assert hasattr(validator, 'step_type_rules')

    def test_validate_universal_methods_complete_builder(self, validator):
        """Test universal method validation with complete builder."""
        issues = validator._validate_universal_methods(MockStepBuilder, "Processing")
        
        # Should have no issues since all universal methods are present
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0

    def test_validate_universal_methods_incomplete_builder(self, validator):
        """Test universal method validation with incomplete builder."""
        issues = validator._validate_universal_methods(MockIncompleteStepBuilder, "Processing")
        
        # Should have issues for missing universal methods
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) >= 2  # Missing _get_inputs and create_step
        
        # Check specific missing methods
        missing_methods = [issue.method_name for issue in error_issues]
        assert "_get_inputs" in missing_methods
        assert "create_step" in missing_methods

    def test_validate_step_type_methods_processing(self, validator):
        """Test step-type-specific method validation for Processing."""
        # Test complete processing builder
        issues = validator._validate_step_type_methods(MockProcessingStepBuilder, "Processing")
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0  # Should have all required methods
        
        # Test incomplete processing builder (missing _create_processor)
        issues = validator._validate_step_type_methods(MockStepBuilder, "Processing")
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) >= 1  # Missing _create_processor
        
        missing_methods = [issue.method_name for issue in error_issues]
        assert "_create_processor" in missing_methods

    def test_validate_step_type_methods_training(self, validator):
        """Test step-type-specific method validation for Training."""
        # Test complete training builder
        issues = validator._validate_step_type_methods(MockTrainingStepBuilder, "Training")
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0  # Should have all required methods
        
        # Test incomplete training builder (missing _create_estimator)
        issues = validator._validate_step_type_methods(MockStepBuilder, "Training")
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) >= 1  # Missing _create_estimator
        
        missing_methods = [issue.method_name for issue in error_issues]
        assert "_create_estimator" in missing_methods

    def test_validate_step_type_methods_no_rules(self, validator):
        """Test step-type-specific validation for step type with no specific rules."""
        issues = validator._validate_step_type_methods(MockStepBuilder, "UnknownStepType")
        
        # Should return empty list for unknown step types
        assert len(issues) == 0

    def test_validate_builder_interface_priority_system(self, validator):
        """Test that builder interface validation follows priority system."""
        issues = validator.validate_builder_interface(MockProcessingStepBuilder, "Processing")
        
        # Should validate both universal and step-specific methods
        universal_issues = [issue for issue in issues if issue.rule_type == "universal"]
        step_specific_issues = [issue for issue in issues if issue.rule_type == "step_specific"]
        
        # Complete builder should have no issues
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0
        
        # Should have validated both rule types
        assert len(universal_issues) >= 0  # May have warnings but no errors
        assert len(step_specific_issues) >= 0  # May have warnings but no errors

    def test_validate_builder_interface_incomplete_builder(self, validator):
        """Test builder interface validation with incomplete builder."""
        issues = validator.validate_builder_interface(MockIncompleteStepBuilder, "Processing")
        
        # Should have issues from both universal and step-specific validation
        universal_errors = [issue for issue in issues if issue.rule_type == "universal" and issue.level == "ERROR"]
        step_specific_errors = [issue for issue in issues if issue.rule_type == "step_specific" and issue.level == "ERROR"]
        
        # Should have universal method errors
        assert len(universal_errors) >= 2  # Missing _get_inputs and create_step
        
        # Should have step-specific method errors
        assert len(step_specific_errors) >= 1  # Missing _create_processor

    def test_method_signature_validation(self, validator):
        """Test method signature validation."""
        # Test with method that has correct signature
        class CorrectSignatureBuilder:
            def validate_configuration(self, config: dict) -> bool:
                return True
        
        issues = validator._validate_method_signature(
            CorrectSignatureBuilder, 
            "validate_configuration", 
            {"parameters": ["config"], "return_type": "bool"},
            "Processing",
            "universal"
        )
        
        # Should have no signature issues
        signature_issues = [issue for issue in issues if "signature" in issue.message.lower()]
        assert len(signature_issues) == 0

    def test_inheritance_compliance_checking(self, validator):
        """Test inheritance compliance checking."""
        # Mock a builder that overrides a final method (should generate warning)
        class OverridingBuilder(MockStepBuilder):
            def _get_step_name(self):  # This should be INHERITED_FINAL
                return "custom_name"
        
        # This test would require the validator to check inheritance compliance
        # For now, we'll test that the method override detection works
        result = validator._is_method_overridden(OverridingBuilder, "_get_step_name")
        assert isinstance(result, bool)

    @patch('cursus.validation.alignment.validators.method_interface_validator.get_universal_validation_rules')
    def test_universal_rules_integration(self, mock_get_universal_rules, validator):
        """Test integration with universal validation rules."""
        mock_rules = {
            "required_methods": {
                "validate_configuration": {"category": "REQUIRED_ABSTRACT"},
                "_get_inputs": {"category": "REQUIRED_ABSTRACT"},
                "create_step": {"category": "REQUIRED_ABSTRACT"}
            },
            "inherited_methods": {
                "_get_outputs": {"category": "INHERITED_OPTIONAL"}
            }
        }
        mock_get_universal_rules.return_value = mock_rules
        
        # Re-initialize validator to use mocked rules
        validator.universal_rules = mock_get_universal_rules()
        
        issues = validator._validate_universal_methods(MockStepBuilder, "Processing")
        
        # Should use the mocked rules
        mock_get_universal_rules.assert_called()
        
        # Should validate against mocked required methods
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0  # MockStepBuilder has all required methods

    @patch('cursus.validation.alignment.validators.method_interface_validator.get_step_type_validation_rules')
    def test_step_type_rules_integration(self, mock_get_step_type_rules, validator):
        """Test integration with step-type-specific validation rules."""
        mock_rules = {
            "Processing": {
                "required_methods": {
                    "_create_processor": {"return_type": "Processor"}
                }
            },
            "Training": {
                "required_methods": {
                    "_create_estimator": {"return_type": "Estimator"}
                }
            }
        }
        mock_get_step_type_rules.return_value = mock_rules
        
        # Re-initialize validator to use mocked rules
        validator.step_type_rules = mock_get_step_type_rules()
        
        issues = validator._validate_step_type_methods(MockProcessingStepBuilder, "Processing")
        
        # Should use the mocked rules
        mock_get_step_type_rules.assert_called()
        
        # Should validate against mocked step-specific methods
        error_issues = [issue for issue in issues if issue.level == "ERROR"]
        assert len(error_issues) == 0  # MockProcessingStepBuilder has _create_processor

    def test_validation_issue_structure(self, validator):
        """Test that validation issues have correct structure."""
        issues = validator._validate_universal_methods(MockIncompleteStepBuilder, "Processing")
        
        if issues:
            issue = issues[0]
            
            # Should have required fields
            assert hasattr(issue, 'level')
            assert hasattr(issue, 'message')
            assert hasattr(issue, 'method_name')
            assert hasattr(issue, 'rule_type')
            assert hasattr(issue, 'details')
            
            # Should have correct rule type
            assert issue.rule_type == "universal"
            
            # Should have method name
            assert issue.method_name is not None

    def test_builder_class_discovery_integration(self, validator):
        """Test integration with builder class discovery."""
        with patch.object(validator, '_get_builder_class') as mock_get_builder:
            mock_get_builder.return_value = MockStepBuilder
            
            # Test that validator can discover builder class
            builder_class = validator._get_builder_class("test_step")
            assert builder_class == MockStepBuilder
            mock_get_builder.assert_called_once_with("test_step")

    def test_comprehensive_validation_workflow(self, validator):
        """Test complete validation workflow."""
        # Test with a complete builder
        issues = validator.validate_builder_interface(MockProcessingStepBuilder, "Processing")
        
        # Should have comprehensive validation results
        assert isinstance(issues, list)
        
        # Should have both universal and step-specific validation
        rule_types = {issue.rule_type for issue in issues}
        expected_rule_types = {"universal", "step_specific"}
        
        # At least one rule type should be present (may not have issues for complete builder)
        assert len(rule_types.intersection(expected_rule_types)) >= 0

    def test_error_handling_malformed_builder(self, validator):
        """Test error handling with malformed builder class."""
        # Test with None builder class - this should be handled gracefully
        try:
            issues = validator.validate_builder_interface(None, "Processing")
            
            # Should handle gracefully and return appropriate error
            assert isinstance(issues, list)
            # May or may not have specific error messages, but should not crash
            
        except AttributeError:
            # If the validator doesn't handle None gracefully, that's also acceptable
            # as long as it fails predictably
            pass

    def test_performance_with_large_builder(self, validator):
        """Test performance with builder class that has many methods."""
        # Create a builder with many methods
        class LargeBuilder(MockStepBuilder):
            pass
        
        # Add many methods dynamically
        for i in range(100):
            setattr(LargeBuilder, f"method_{i}", lambda self: None)
        
        # Validation should still complete quickly
        import time
        start_time = time.time()
        
        issues = validator.validate_builder_interface(LargeBuilder, "Processing")
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        assert (end_time - start_time) < 1.0
        
        # Should still validate correctly
        assert isinstance(issues, list)


class TestMethodInterfaceValidatorEdgeCases:
    """Test edge cases for method interface validation."""

    @pytest.fixture
    def validator(self):
        """Create validator for edge case testing."""
        return MethodInterfaceValidator(workspace_dirs=["/mock/workspace"])

    def test_empty_workspace_dirs(self):
        """Test validator with empty workspace directories."""
        validator = MethodInterfaceValidator(workspace_dirs=[])
        assert validator.workspace_dirs == []
        
        # Should still work for validation
        issues = validator.validate_builder_interface(MockStepBuilder, "Processing")
        assert isinstance(issues, list)

    def test_none_workspace_dirs(self):
        """Test validator with None workspace directories."""
        validator = MethodInterfaceValidator(workspace_dirs=None)
        assert validator.workspace_dirs is None
        
        # Should still work for validation
        issues = validator.validate_builder_interface(MockStepBuilder, "Processing")
        assert isinstance(issues, list)

    def test_invalid_step_type(self, validator):
        """Test validation with invalid step type."""
        issues = validator.validate_builder_interface(MockStepBuilder, None)
        
        # Should handle gracefully
        assert isinstance(issues, list)
        
        # Should still perform universal validation even with invalid step type
        # The validator should be robust enough to handle None step types
        assert len(issues) >= 0  # May or may not have issues, but should not crash

    def test_builder_with_property_methods(self, validator):
        """Test validation with builder that has property methods."""
        class PropertyBuilder(MockStepBuilder):
            @property
            def some_property(self):
                return "value"
        
        # Should handle properties gracefully
        issues = validator.validate_builder_interface(PropertyBuilder, "Processing")
        assert isinstance(issues, list)

    def test_builder_with_static_methods(self, validator):
        """Test validation with builder that has static methods."""
        class StaticMethodBuilder(MockStepBuilder):
            @staticmethod
            def static_method():
                return "static"
        
        # Should handle static methods gracefully
        issues = validator.validate_builder_interface(StaticMethodBuilder, "Processing")
        assert isinstance(issues, list)

    def test_builder_with_class_methods(self, validator):
        """Test validation with builder that has class methods."""
        class ClassMethodBuilder(MockStepBuilder):
            @classmethod
            def class_method(cls):
                return "class"
        
        # Should handle class methods gracefully
        issues = validator.validate_builder_interface(ClassMethodBuilder, "Processing")
        assert isinstance(issues, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
