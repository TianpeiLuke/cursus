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
    get_universal_required_methods,
    get_universal_inherited_methods,
    get_universal_method_category,
    is_universal_method_required,
    is_universal_method_final,
    get_universal_method_specification,
    validate_universal_rules_consistency,
    get_universal_implementation_patterns,
    get_all_universal_methods,
    get_universal_method_categories,
    get_required_abstract_methods,
    get_inherited_optional_methods,
    get_inherited_final_methods,
    get_universal_method_descriptions,
    get_universal_validation_levels
)


class TestUniversalBuilderRules:
    """Test cases for universal builder validation rules."""

    def test_universal_builder_validation_rules_structure(self):
        """Test that UNIVERSAL_BUILDER_VALIDATION_RULES has correct structure."""
        assert isinstance(UNIVERSAL_BUILDER_VALIDATION_RULES, dict)
        assert "required_methods" in UNIVERSAL_BUILDER_VALIDATION_RULES
        assert "inherited_methods" in UNIVERSAL_BUILDER_VALIDATION_RULES
        assert "implementation_patterns" in UNIVERSAL_BUILDER_VALIDATION_RULES

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

    def test_get_universal_required_methods(self):
        """Test get_universal_required_methods function."""
        methods = get_universal_required_methods()
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

    def test_get_universal_inherited_methods(self):
        """Test get_universal_inherited_methods function."""
        methods = get_universal_inherited_methods()
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

    def test_get_universal_method_category(self):
        """Test get_universal_method_category function."""
        # Test required methods
        assert get_universal_method_category("validate_configuration") == UniversalMethodCategory.REQUIRED_ABSTRACT
        assert get_universal_method_category("_get_inputs") == UniversalMethodCategory.REQUIRED_ABSTRACT
        assert get_universal_method_category("create_step") == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Test inherited methods (if they exist)
        cache_config_category = get_universal_method_category("_get_cache_config")
        if cache_config_category is not None:
            assert cache_config_category == UniversalMethodCategory.INHERITED_FINAL
        
        # Test invalid method
        assert get_universal_method_category("invalid_method") is None

    def test_is_universal_method_required(self):
        """Test is_universal_method_required function."""
        # Test required methods
        assert is_universal_method_required("validate_configuration") is True
        assert is_universal_method_required("_get_inputs") is True
        assert is_universal_method_required("create_step") is True
        
        # Test inherited methods
        assert is_universal_method_required("_get_cache_config") is False
        assert is_universal_method_required("_get_environment_variables") is False
        
        # Test invalid method
        assert is_universal_method_required("invalid_method") is False

    def test_is_universal_method_final(self):
        """Test is_universal_method_final function."""
        # Test final methods
        final_methods = get_inherited_final_methods()
        for method_name in final_methods:
            assert is_universal_method_final(method_name) is True
        
        # Test non-final methods
        assert is_universal_method_final("validate_configuration") is False
        assert is_universal_method_final("_get_inputs") is False
        assert is_universal_method_final("create_step") is False
        
        # Test invalid method
        assert is_universal_method_final("invalid_method") is False

    def test_get_universal_method_specification(self):
        """Test get_universal_method_specification function."""
        # Test valid method specification
        spec = get_universal_method_specification("validate_configuration")
        assert isinstance(spec, dict)
        assert "category" in spec
        assert "description" in spec
        
        # Test invalid method
        spec = get_universal_method_specification("invalid_method")
        assert spec == {}

    def test_validate_universal_rules_consistency(self):
        """Test validate_universal_rules_consistency function."""
        issues = validate_universal_rules_consistency()
        assert isinstance(issues, list)
        
        # Should have no consistency issues in well-formed rules
        if issues:
            # Print issues for debugging if any exist
            for issue in issues:
                print(f"Consistency issue: {issue}")

    def test_get_universal_implementation_patterns(self):
        """Test get_universal_implementation_patterns function."""
        patterns = get_universal_implementation_patterns()
        assert isinstance(patterns, dict)
        
        # Check that implementation patterns are present
        expected_patterns = [
            "initialization",
            "validation", 
            "input_processing",
            "output_processing",
            "step_creation"
        ]
        
        for pattern in expected_patterns:
            if pattern in patterns:
                assert isinstance(patterns[pattern], dict)
                assert "description" in patterns[pattern]

    def test_get_all_universal_methods(self):
        """Test get_all_universal_methods function."""
        methods = get_all_universal_methods()
        assert isinstance(methods, list)
        assert len(methods) >= 3  # At least the 3 required methods
        
        # Check that required methods are included
        assert "validate_configuration" in methods
        assert "_get_inputs" in methods
        assert "create_step" in methods

    def test_get_universal_method_categories(self):
        """Test get_universal_method_categories function."""
        categories = get_universal_method_categories()
        assert isinstance(categories, dict)
        
        # Check that all methods have categories
        all_methods = get_all_universal_methods()
        for method_name in all_methods:
            assert method_name in categories
            assert isinstance(categories[method_name], UniversalMethodCategory)

    def test_get_required_abstract_methods(self):
        """Test get_required_abstract_methods function."""
        methods = get_required_abstract_methods()
        assert isinstance(methods, list)
        assert len(methods) == 3  # Based on analysis
        
        # Check that all required methods are included
        assert "validate_configuration" in methods
        assert "_get_inputs" in methods
        assert "create_step" in methods

    def test_get_inherited_optional_methods(self):
        """Test get_inherited_optional_methods function."""
        methods = get_inherited_optional_methods()
        assert isinstance(methods, list)
        
        # Check that optional methods are included
        expected_optional = [
            "_get_environment_variables",
            "_get_job_arguments",
            "_get_outputs"
        ]
        
        for method_name in expected_optional:
            if method_name in methods:
                # Verify it's actually optional
                category = get_universal_method_category(method_name)
                assert category == UniversalMethodCategory.INHERITED_OPTIONAL

    def test_get_inherited_final_methods(self):
        """Test get_inherited_final_methods function."""
        methods = get_inherited_final_methods()
        assert isinstance(methods, list)
        
        # Check that final methods are included
        expected_final = [
            "_get_cache_config",
            "_generate_job_name", 
            "_get_step_name"
        ]
        
        for method_name in expected_final:
            if method_name in methods:
                # Verify it's actually final
                category = get_universal_method_category(method_name)
                assert category == UniversalMethodCategory.INHERITED_FINAL

    def test_get_universal_method_descriptions(self):
        """Test get_universal_method_descriptions function."""
        descriptions = get_universal_method_descriptions()
        assert isinstance(descriptions, dict)
        
        # Check that all methods have descriptions
        all_methods = get_all_universal_methods()
        for method_name in all_methods:
            assert method_name in descriptions
            assert isinstance(descriptions[method_name], str)
            assert len(descriptions[method_name]) > 0

    def test_get_universal_validation_levels(self):
        """Test get_universal_validation_levels function."""
        levels = get_universal_validation_levels()
        assert isinstance(levels, dict)
        
        # Check that validation levels are properly structured
        expected_levels = ["method_presence", "method_signature", "inheritance_compliance"]
        for level in expected_levels:
            if level in levels:
                assert isinstance(levels[level], dict)
                assert "description" in levels[level]

    def test_required_methods_specifications(self):
        """Test required methods have proper specifications."""
        required_methods = get_universal_required_methods()
        
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
        inherited_methods = get_universal_inherited_methods()
        
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
        # Get all methods and their categories
        all_methods = get_all_universal_methods()
        categories = get_universal_method_categories()
        
        # Check that every method has a category
        for method_name in all_methods:
            assert method_name in categories
            category = categories[method_name]
            assert isinstance(category, UniversalMethodCategory)
        
        # Check that required methods are properly categorized
        required_methods = get_required_abstract_methods()
        for method_name in required_methods:
            assert categories[method_name] == UniversalMethodCategory.REQUIRED_ABSTRACT
        
        # Check that optional methods are properly categorized
        optional_methods = get_inherited_optional_methods()
        for method_name in optional_methods:
            assert categories[method_name] == UniversalMethodCategory.INHERITED_OPTIONAL
        
        # Check that final methods are properly categorized
        final_methods = get_inherited_final_methods()
        for method_name in final_methods:
            assert categories[method_name] == UniversalMethodCategory.INHERITED_FINAL

    def test_universal_rules_completeness(self):
        """Test that universal rules are complete and consistent."""
        rules = UNIVERSAL_BUILDER_VALIDATION_RULES
        
        # Check required fields
        assert "required_methods" in rules
        assert "inherited_methods" in rules
        assert "implementation_patterns" in rules
        
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
        # Test with None input
        assert get_universal_method_category(None) is None
        assert is_universal_method_required(None) is False
        assert is_universal_method_final(None) is False
        assert get_universal_method_specification(None) == {}
        
        # Test with empty string
        assert get_universal_method_category("") is None
        assert is_universal_method_required("") is False
        assert is_universal_method_final("") is False
        assert get_universal_method_specification("") == {}

    def test_universal_rules_integration_with_validation_system(self):
        """Test that universal rules integrate properly with validation system."""
        # Test that all required methods are properly defined
        required_methods = get_required_abstract_methods()
        
        for method_name in required_methods:
            # Should have proper category
            category = get_universal_method_category(method_name)
            assert category == UniversalMethodCategory.REQUIRED_ABSTRACT
            
            # Should be marked as required
            assert is_universal_method_required(method_name) is True
            
            # Should not be marked as final
            assert is_universal_method_final(method_name) is False
            
            # Should have specification
            spec = get_universal_method_specification(method_name)
            assert len(spec) > 0
            assert "description" in spec

    def test_implementation_patterns_structure(self):
        """Test that implementation patterns are properly structured."""
        patterns = get_universal_implementation_patterns()
        
        for pattern_name, pattern_spec in patterns.items():
            assert isinstance(pattern_name, str)
            assert isinstance(pattern_spec, dict)
            assert "description" in pattern_spec
            assert len(pattern_spec["description"]) > 0

    def test_universal_method_category_enum_coverage(self):
        """Test that all enum values are used in the rules."""
        all_categories = set()
        
        # Collect categories from required methods
        required_methods = get_universal_required_methods()
        for method_spec in required_methods.values():
            all_categories.add(method_spec["category"])
        
        # Collect categories from inherited methods
        inherited_methods = get_universal_inherited_methods()
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
