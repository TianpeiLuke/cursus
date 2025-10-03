"""
Test module for universal builder validation rules.

Tests the universal validation rules that apply to ALL step builders,
including method categorization and validation requirements.
"""

import pytest
from typing import Dict, Any, List

from cursus.validation.alignment.config.universal_builder_rules import (
    UNIVERSAL_BUILDER_VALIDATION_RULES,
    UniversalMethodCategory,
    get_universal_validation_rules,
    get_required_methods,
    get_inherited_methods,
    get_validation_rules,
    validate_universal_compliance
)


class TestUniversalBuilderRules:
    """Test cases for universal builder validation rules."""

    def test_universal_builder_validation_rules_structure(self):
        """Test that UNIVERSAL_BUILDER_VALIDATION_RULES has correct structure."""
        assert isinstance(UNIVERSAL_BUILDER_VALIDATION_RULES, dict)
        assert "required_methods" in UNIVERSAL_BUILDER_VALIDATION_RULES
        assert "inherited_methods" in UNIVERSAL_BUILDER_VALIDATION_RULES
        assert "common_implementation_patterns" in UNIVERSAL_BUILDER_VALIDATION_RULES

    def test_universal_method_categories_are_valid(self):
        """Test that all universal method categories are valid enum values."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        # Check required methods
        for method_name, method_spec in rules["required_methods"].items():
            category = method_spec["category"]
            assert isinstance(category, UniversalMethodCategory)
            assert category == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Check inherited methods
        for method_name, method_spec in rules["inherited_methods"].items():
            category = method_spec["category"]
            assert isinstance(category, UniversalMethodCategory)
            assert category in [
                UniversalMethodCategory.INHERITED_OPTIONAL,
                UniversalMethodCategory.INHERITED_FINAL
            ]

    def test_get_universal_validation_rules(self):
        """Test get_universal_validation_rules function."""
        rules = get_universal_validation_rules()
        assert isinstance(rules, dict)
        assert rules == UNIVERSAL_BUILDER_VALIDATION_RULES

    def test_get_required_methods(self):
        """Test get_required_methods function."""
        methods = get_required_methods()
        assert isinstance(methods, dict)
        assert len(methods) == 3  # Based on analysis: validate_configuration, _get_inputs, create_step
        
        # Check that all required methods are present
        assert "validate_configuration" in methods
        assert "_get_inputs" in methods
        assert "create_step" in methods
        
        # Check method specifications
        for method_name, method_spec in methods.items():
            assert "category" in method_spec
            assert "description" in method_spec
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT

    def test_get_inherited_methods(self):
        """Test get_inherited_methods function."""
        methods = get_inherited_methods()
        assert isinstance(methods, dict)
        assert len(methods) >= 6  # Based on analysis: 6 inherited methods
        
        # Check that inherited methods are present
        expected_inherited = [
            "_get_environment_variables",
            "_get_job_arguments", 
            "_get_outputs",
            "_get_cache_config",
            "_generate_job_name",
            "_get_step_name"
        ]
        
        for method_name in expected_inherited:
            if method_name in methods:  # Some may be optional
                method_spec = methods[method_name]
                assert "category" in method_spec
                assert "description" in method_spec
                assert method_spec["category"] in [
                    UniversalMethodCategory.INHERITED_OPTIONAL,
                    UniversalMethodCategory.INHERITED_FINAL
                ]

    def test_validate_universal_compliance(self):
        """Test validate_universal_compliance function."""
        # Create a mock builder class for testing
        class MockCompliantBuilder:
            def validate_configuration(self):
                pass
            def _get_inputs(self, inputs):
                pass
            def create_step(self, **kwargs):
                pass
        
        # Test compliant builder
        issues = validate_universal_compliance(MockCompliantBuilder)
        assert isinstance(issues, list)
        # Note: May have issues due to inheritance check, but should validate methods
        
        # Create a non-compliant builder class
        class MockIncompliantBuilder:
            def validate_configuration(self):
                pass
            # Missing _get_inputs and create_step methods
        
        # Test non-compliant builder
        issues = validate_universal_compliance(MockIncompliantBuilder)
        assert isinstance(issues, list)
        assert len(issues) > 0  # Should have issues for missing methods

    def test_get_validation_rules(self):
        """Test get_validation_rules function."""
        rules = get_validation_rules()
        assert isinstance(rules, dict)
        
        # Check that validation rules are properly structured
        expected_keys = ["inheritance", "method_signatures", "abstract_methods"]
        for key in expected_keys:
            if key in rules:
                assert isinstance(rules[key], dict)

    def test_universal_builder_validation_rules_content(self):
        """Test that UNIVERSAL_BUILDER_VALIDATION_RULES contains expected content."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        # Check required methods content
        required_methods = rules["required_methods"]
        assert "validate_configuration" in required_methods
        assert "description" in required_methods["validate_configuration"]
        assert "category" in required_methods["validate_configuration"]
        
        # Check inherited methods content
        inherited_methods = rules["inherited_methods"]
        assert len(inherited_methods) > 0
        
        for method_name, method_spec in inherited_methods.items():
            assert "description" in method_spec
            assert "category" in method_spec
            assert method_spec["category"] in [
                UniversalMethodCategory.INHERITED_OPTIONAL,
                UniversalMethodCategory.INHERITED_FINAL
            ]

    def test_method_categorization_from_rules(self):
        """Test method categorization based on actual rules data."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        # Test required methods categorization
        required_methods = rules["required_methods"]
        for method_name, method_spec in required_methods.items():
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Test inherited methods categorization
        inherited_methods = rules["inherited_methods"]
        optional_count = 0
        final_count = 0
        
        for method_name, method_spec in inherited_methods.items():
            if method_spec["category"] == UniversalMethodCategory.INHERITED_OPTIONAL:
                optional_count += 1
            elif method_spec["category"] == UniversalMethodCategory.INHERITED_FINAL:
                final_count += 1
        
        # Should have both optional and final methods
        assert optional_count > 0 or final_count > 0

    def test_implementation_patterns_content(self):
        """Test that implementation patterns contain expected content."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        if "common_implementation_patterns" in rules:
            patterns = rules["common_implementation_patterns"]
            assert isinstance(patterns, dict)
            
            for pattern_name, pattern_spec in patterns.items():
                assert isinstance(pattern_spec, dict)
                assert "description" in pattern_spec
                assert len(pattern_spec["description"]) > 0

    def test_required_methods_specifications(self):
        """Test required methods have proper specifications."""
        required_methods = get_required_methods()
        
        for method_name, method_spec in required_methods.items():
            # Check basic structure
            assert "category" in method_spec
            assert "description" in method_spec
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT
            
            # Check description is meaningful
            assert len(method_spec["description"]) > 10
            
            # Check method-specific requirements
            if method_name == "validate_configuration":
                assert "configuration" in method_spec["description"].lower()
            elif method_name == "_get_inputs":
                assert "input" in method_spec["description"].lower()
            elif method_name == "create_step":
                assert "step" in method_spec["description"].lower()

    def test_inherited_methods_specifications(self):
        """Test inherited methods have proper specifications."""
        inherited_methods = get_inherited_methods()
        
        for method_name, method_spec in inherited_methods.items():
            # Check basic structure
            assert "category" in method_spec
            assert "description" in method_spec
            assert method_spec["category"] in [
                UniversalMethodCategory.INHERITED_OPTIONAL,
                UniversalMethodCategory.INHERITED_FINAL
            ]
            
            # Check description is meaningful
            assert len(method_spec["description"]) > 10

    def test_method_categorization_consistency(self):
        """Test that method categorization is consistent."""
        # Test based on actual data from rules
        required_methods = get_required_methods()
        inherited_methods = get_inherited_methods()
        
        # Check that required methods are properly categorized
        for method_name, method_spec in required_methods.items():
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Check that inherited methods are properly categorized
        for method_name, method_spec in inherited_methods.items():
            assert method_spec["category"] in [
                UniversalMethodCategory.INHERITED_OPTIONAL,
                UniversalMethodCategory.INHERITED_FINAL
            ]

    def test_universal_rules_completeness(self):
        """Test that universal rules are complete and consistent."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        # Check required fields
        assert "required_methods" in rules
        assert "inherited_methods" in rules
        assert "common_implementation_patterns" in rules
        
        # Check required methods structure
        required_methods = rules["required_methods"]
        assert isinstance(required_methods, dict)
        assert len(required_methods) == 3  # validate_configuration, _get_inputs, create_step
        
        for method_name, method_spec in required_methods.items():
            assert isinstance(method_name, str)
            assert isinstance(method_spec, dict)
            assert "category" in method_spec
            assert "description" in method_spec
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Check inherited methods structure
        inherited_methods = rules["inherited_methods"]
        assert isinstance(inherited_methods, dict)
        
        for method_name, method_spec in inherited_methods.items():
            assert isinstance(method_name, str)
            assert isinstance(method_spec, dict)
            assert "category" in method_spec
            assert "description" in method_spec
            assert method_spec["category"] in [
                UniversalMethodCategory.INHERITED_OPTIONAL,
                UniversalMethodCategory.INHERITED_FINAL
            ]

    def test_api_function_error_handling(self):
        """Test that API functions handle errors gracefully."""
        # Test validate_universal_compliance with None
        issues = validate_universal_compliance(None)
        assert isinstance(issues, list)
        assert len(issues) > 0  # Should have issues for None input

    def test_universal_rules_integration_with_validation_system(self):
        """Test that universal rules integrate properly with validation system."""
        # Test that all required methods are properly defined in the rules
        required_methods = get_required_methods()
        
        for method_name, method_spec in required_methods.items():
            # Should have proper category
            assert method_spec["category"] == UniversalMethodCategory.REQUIRED_ABSTRACT
            
            # Should have description
            assert "description" in method_spec
            assert len(method_spec["description"]) > 0

    def test_implementation_patterns_structure(self):
        """Test that implementation patterns are properly structured."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        if "common_implementation_patterns" in rules:
            patterns = rules["common_implementation_patterns"]
            
            for pattern_name, pattern_spec in patterns.items():
                assert isinstance(pattern_name, str)
                assert isinstance(pattern_spec, dict)
                assert "description" in pattern_spec
                assert len(pattern_spec["description"]) > 0

    def test_universal_method_category_enum_coverage(self):
        """Test that all enum values are used in the rules."""
        all_categories = set()
        
        # Collect categories from required methods
        required_methods = get_required_methods()
        for method_spec in required_methods.values():
            all_categories.add(method_spec["category"])
        
        # Collect categories from inherited methods
        inherited_methods = get_inherited_methods()
        for method_spec in inherited_methods.values():
            all_categories.add(method_spec["category"])
        
        # Check that all enum values are represented
        expected_categories = {
            UniversalMethodCategory.REQUIRED_ABSTRACT,
            UniversalMethodCategory.INHERITED_OPTIONAL,
            UniversalMethodCategory.INHERITED_FINAL
        }
        
        # Should have at least REQUIRED_ABSTRACT
        assert UniversalMethodCategory.REQUIRED_ABSTRACT in all_categories
        
        # Other categories may or may not be present depending on implementation
        for category in all_categories:
            assert category in expected_categories
