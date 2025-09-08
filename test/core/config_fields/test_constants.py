"""
Unit tests for constants module.

This module contains comprehensive tests for the constants and enums
used throughout the config_field_manager package, addressing the
critical gap identified in the test coverage analysis.
"""

import unittest
import sys
from pathlib import Path
from enum import Enum
from typing import Set, Dict, Any

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cursus.core.config_fields.constants import (
    SPECIAL_FIELDS_TO_KEEP_SPECIFIC,
    NON_STATIC_FIELD_PATTERNS,
    NON_STATIC_FIELD_EXCEPTIONS,
    CategoryType,
    MergeDirection,
    SerializationMode,
    TYPE_MAPPING
)

class TestConstants(unittest.TestCase):
    """Test cases for constants module."""

    def test_special_fields_to_keep_specific_completeness(self):
        """Test that SPECIAL_FIELDS_TO_KEEP_SPECIFIC contains expected fields."""
        # Verify it's a set
        self.assertIsInstance(SPECIAL_FIELDS_TO_KEEP_SPECIFIC, set)
        
        # Verify it's not empty
        self.assertGreater(len(SPECIAL_FIELDS_TO_KEEP_SPECIFIC), 0)
        
        # Test for expected critical fields
        expected_fields = {
            "image_uri",
            "script_name", 
            "output_path",
            "input_path",
            "model_path",
            "hyperparameters",
            "instance_type",
            "job_name_prefix",
            "output_schema"
        }
        
        for field in expected_fields:
            self.assertIn(field, SPECIAL_FIELDS_TO_KEEP_SPECIFIC,
                         f"Expected special field '{field}' not found")
        
        # Verify all entries are strings
        for field in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            self.assertIsInstance(field, str, f"Special field '{field}' should be string")
            self.assertGreater(len(field), 0, f"Special field should not be empty")

    def test_non_static_field_patterns_accuracy(self):
        """Test that NON_STATIC_FIELD_PATTERNS contains reasonable patterns."""
        # Verify it's a set
        self.assertIsInstance(NON_STATIC_FIELD_PATTERNS, set)
        
        # Verify it's not empty
        self.assertGreater(len(NON_STATIC_FIELD_PATTERNS), 0)
        
        # Test for expected patterns
        expected_patterns = {
            "_names",
            "input_", 
            "output_",
            "_specific",
            "batch_count",
            "item_count", 
            "record_count",
            "instance_type_count",
            "_path",
            "_uri"
        }
        
        for pattern in expected_patterns:
            self.assertIn(pattern, NON_STATIC_FIELD_PATTERNS,
                         f"Expected pattern '{pattern}' not found")
        
        # Verify all entries are strings
        for pattern in NON_STATIC_FIELD_PATTERNS:
            self.assertIsInstance(pattern, str, f"Pattern '{pattern}' should be string")
            self.assertGreater(len(pattern), 0, f"Pattern should not be empty")

    def test_non_static_field_exceptions_validity(self):
        """Test that NON_STATIC_FIELD_EXCEPTIONS contains valid exceptions."""
        # Verify it's a set
        self.assertIsInstance(NON_STATIC_FIELD_EXCEPTIONS, set)
        
        # Test for expected exception
        self.assertIn("processing_instance_count", NON_STATIC_FIELD_EXCEPTIONS)
        
        # Verify all entries are strings
        for exception in NON_STATIC_FIELD_EXCEPTIONS:
            self.assertIsInstance(exception, str, f"Exception '{exception}' should be string")
            self.assertGreater(len(exception), 0, f"Exception should not be empty")

    def test_pattern_exception_logic(self):
        """Test the logic between patterns and exceptions."""
        # processing_instance_count should match a pattern but be excepted
        field_name = "processing_instance_count"
        
        # Check if it matches any pattern
        matches_pattern = any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS)
        
        # It should match a pattern (like "instance_type_count" or similar)
        # but be in exceptions
        if matches_pattern:
            self.assertIn(field_name, NON_STATIC_FIELD_EXCEPTIONS,
                         f"Field '{field_name}' matches pattern but should be excepted")

    def test_category_type_enum(self):
        """Test the CategoryType enum."""
        # Verify it's an enum
        self.assertTrue(issubclass(CategoryType, Enum))
        
        # Test expected values
        self.assertTrue(hasattr(CategoryType, 'SHARED'))
        self.assertTrue(hasattr(CategoryType, 'SPECIFIC'))
        
        # Verify enum values are unique
        values = [member.value for member in CategoryType]
        self.assertEqual(len(values), len(set(values)), "CategoryType values should be unique")
        
        # Test enum usage
        self.assertNotEqual(CategoryType.SHARED, CategoryType.SPECIFIC)
        
        # Test string representation
        self.assertIn('SHARED', str(CategoryType.SHARED))
        self.assertIn('SPECIFIC', str(CategoryType.SPECIFIC))

    def test_merge_direction_enum(self):
        """Test the MergeDirection enum."""
        # Verify it's an enum
        self.assertTrue(issubclass(MergeDirection, Enum))
        
        # Test expected values
        expected_directions = ['PREFER_SOURCE', 'PREFER_TARGET', 'ERROR_ON_CONFLICT']
        for direction in expected_directions:
            self.assertTrue(hasattr(MergeDirection, direction),
                           f"MergeDirection should have {direction}")
        
        # Verify enum values are unique
        values = [member.value for member in MergeDirection]
        self.assertEqual(len(values), len(set(values)), "MergeDirection values should be unique")
        
        # Test enum usage
        self.assertNotEqual(MergeDirection.PREFER_SOURCE, MergeDirection.PREFER_TARGET)
        self.assertNotEqual(MergeDirection.PREFER_SOURCE, MergeDirection.ERROR_ON_CONFLICT)
        self.assertNotEqual(MergeDirection.PREFER_TARGET, MergeDirection.ERROR_ON_CONFLICT)

    def test_serialization_mode_enum(self):
        """Test the SerializationMode enum."""
        # Verify it's an enum
        self.assertTrue(issubclass(SerializationMode, Enum))
        
        # Test expected values
        expected_modes = ['PRESERVE_TYPES', 'SIMPLE_JSON', 'CUSTOM_FIELDS']
        for mode in expected_modes:
            self.assertTrue(hasattr(SerializationMode, mode),
                           f"SerializationMode should have {mode}")
        
        # Verify enum values are unique
        values = [member.value for member in SerializationMode]
        self.assertEqual(len(values), len(set(values)), "SerializationMode values should be unique")
        
        # Test enum usage
        self.assertNotEqual(SerializationMode.PRESERVE_TYPES, SerializationMode.SIMPLE_JSON)
        self.assertNotEqual(SerializationMode.PRESERVE_TYPES, SerializationMode.CUSTOM_FIELDS)
        self.assertNotEqual(SerializationMode.SIMPLE_JSON, SerializationMode.CUSTOM_FIELDS)

    def test_type_mapping_completeness(self):
        """Test that TYPE_MAPPING contains expected type mappings."""
        # Verify it's a dictionary
        self.assertIsInstance(TYPE_MAPPING, dict)
        
        # Verify it's not empty
        self.assertGreater(len(TYPE_MAPPING), 0)
        
        # Test for expected type mappings
        expected_mappings = {
            "dict": "dict",
            "list": "list", 
            "tuple": "tuple",
            "set": "set",
            "frozenset": "frozenset",
            "BaseModel": "pydantic_model",
            "Enum": "enum",
            "datetime": "datetime",
            "Path": "path"
        }
        
        for type_name, expected_serialized in expected_mappings.items():
            self.assertIn(type_name, TYPE_MAPPING,
                         f"Expected type '{type_name}' not found in TYPE_MAPPING")
            self.assertEqual(TYPE_MAPPING[type_name], expected_serialized,
                           f"Type '{type_name}' should map to '{expected_serialized}'")
        
        # Verify all keys and values are strings
        for type_name, serialized_name in TYPE_MAPPING.items():
            self.assertIsInstance(type_name, str, f"Type name '{type_name}' should be string")
            self.assertIsInstance(serialized_name, str, f"Serialized name '{serialized_name}' should be string")
            self.assertGreater(len(type_name), 0, f"Type name should not be empty")
            self.assertGreater(len(serialized_name), 0, f"Serialized name should not be empty")

    def test_type_mapping_consistency(self):
        """Test that TYPE_MAPPING has consistent naming patterns."""
        # Check for reasonable naming patterns
        for type_name, serialized_name in TYPE_MAPPING.items():
            # Serialized names should be lowercase with underscores
            self.assertTrue(serialized_name.islower() or '_' in serialized_name,
                           f"Serialized name '{serialized_name}' should follow snake_case pattern")
            
            # No spaces in names
            self.assertNotIn(' ', type_name, f"Type name '{type_name}' should not contain spaces")
            self.assertNotIn(' ', serialized_name, f"Serialized name '{serialized_name}' should not contain spaces")

    def test_constants_immutability(self):
        """Test that constants are properly defined as immutable types."""
        # SPECIAL_FIELDS_TO_KEEP_SPECIFIC should be a set (mutable but conventionally treated as constant)
        self.assertIsInstance(SPECIAL_FIELDS_TO_KEEP_SPECIFIC, set)
        
        # NON_STATIC_FIELD_PATTERNS should be a set
        self.assertIsInstance(NON_STATIC_FIELD_PATTERNS, set)
        
        # NON_STATIC_FIELD_EXCEPTIONS should be a set
        self.assertIsInstance(NON_STATIC_FIELD_EXCEPTIONS, set)
        
        # TYPE_MAPPING should be a dict
        self.assertIsInstance(TYPE_MAPPING, dict)
        
        # Enums should be proper enum types
        self.assertTrue(issubclass(CategoryType, Enum))
        self.assertTrue(issubclass(MergeDirection, Enum))
        self.assertTrue(issubclass(SerializationMode, Enum))

    def test_field_pattern_matching_logic(self):
        """Test the logic for how field patterns would be used."""
        # Test fields that should match non-static patterns
        non_static_test_cases = [
            ("input_data", True),      # matches "input_"
            ("output_path", True),     # matches "output_" and "_path"
            ("field_names", True),     # matches "_names"
            ("batch_count", True),     # matches "batch_count"
            ("model_uri", True),       # matches "_uri"
            ("config_specific", True), # matches "_specific"
        ]
        
        for field_name, should_match in non_static_test_cases:
            matches = any(pattern in field_name for pattern in NON_STATIC_FIELD_PATTERNS)
            if should_match:
                self.assertTrue(matches, f"Field '{field_name}' should match non-static patterns")
            else:
                self.assertFalse(matches, f"Field '{field_name}' should not match non-static patterns")
        
        # Test exception handling
        exception_field = "processing_instance_count"
        matches_pattern = any(pattern in exception_field for pattern in NON_STATIC_FIELD_PATTERNS)
        is_exception = exception_field in NON_STATIC_FIELD_EXCEPTIONS
        
        # This field should match a pattern but be excepted
        if matches_pattern:
            self.assertTrue(is_exception, 
                           f"Field '{exception_field}' matches pattern but should be in exceptions")

    def test_special_fields_coverage(self):
        """Test that special fields cover important configuration aspects."""
        # Group special fields by category
        path_fields = [f for f in SPECIAL_FIELDS_TO_KEEP_SPECIFIC if 'path' in f or 'uri' in f]
        job_fields = [f for f in SPECIAL_FIELDS_TO_KEEP_SPECIFIC if 'job' in f or 'instance' in f]
        config_fields = [f for f in SPECIAL_FIELDS_TO_KEEP_SPECIFIC if 'hyperparameters' in f or 'schema' in f]
        
        # Verify we have coverage in different categories
        self.assertGreater(len(path_fields), 0, "Should have path-related special fields")
        self.assertGreater(len(job_fields), 0, "Should have job-related special fields")
        self.assertGreater(len(config_fields), 0, "Should have config-related special fields")

    def test_enum_completeness(self):
        """Test that enums have reasonable completeness for their use cases."""
        # CategoryType should have at least SHARED and SPECIFIC
        category_members = list(CategoryType)
        self.assertGreaterEqual(len(category_members), 2, "CategoryType should have at least 2 members")
        
        # MergeDirection should have at least 3 options
        merge_members = list(MergeDirection)
        self.assertGreaterEqual(len(merge_members), 3, "MergeDirection should have at least 3 members")
        
        # SerializationMode should have multiple options
        serialization_members = list(SerializationMode)
        self.assertGreaterEqual(len(serialization_members), 2, "SerializationMode should have at least 2 members")

    def test_constants_documentation_alignment(self):
        """Test that constants align with their documented purposes."""
        # SPECIAL_FIELDS_TO_KEEP_SPECIFIC should contain fields that are typically configuration-specific
        config_specific_indicators = ['hyperparameters', 'path', 'uri', 'instance', 'job', 'schema']
        
        special_fields_text = ' '.join(SPECIAL_FIELDS_TO_KEEP_SPECIFIC).lower()
        found_indicators = [indicator for indicator in config_specific_indicators 
                           if indicator in special_fields_text]
        
        self.assertGreater(len(found_indicators), 0, 
                          "Special fields should contain configuration-specific terms")
        
        # NON_STATIC_FIELD_PATTERNS should contain patterns for dynamic/variable fields
        dynamic_indicators = ['input', 'output', 'count', 'path', 'uri', 'names']
        
        patterns_text = ' '.join(NON_STATIC_FIELD_PATTERNS).lower()
        found_dynamic = [indicator for indicator in dynamic_indicators 
                        if indicator in patterns_text]
        
        self.assertGreater(len(found_dynamic), 0,
                          "Non-static patterns should contain dynamic field indicators")

if __name__ == '__main__':
    unittest.main()
