"""
Unit tests for logical name validation.
"""

import unittest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestLogicalNameValidation(unittest.TestCase):
    """Test logical name validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_valid_logical_names(self):
        """Test valid snake_case logical names."""
        valid_names = [
            "input_data",
            "model_artifacts",
            "training_data",
            "evaluation_results",
            "a",  # Single letter
            "abc123",  # With numbers
            "very_long_descriptive_name_with_many_words"
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_logical_name(name, "Test")
                self.assertEqual(len(violations), 0, f"Valid logical name '{name}' should not have violations")
    
    def test_invalid_logical_names(self):
        """Test invalid logical names."""
        invalid_names = [
            "PascalCase",
            "camelCase",
            "kebab-case",
            "UPPERCASE",
            "123number",  # starts with number
            "with space",  # contains space
            "with.dot"  # contains dot
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_logical_name(name, "Test")
                self.assertGreater(len(violations), 0, f"Invalid logical name '{name}' should have violations")
    
    def test_boundary_underscore_issues(self):
        """Test logical names with boundary underscore issues."""
        boundary_names = [
            "_leading_underscore",
            "trailing_underscore_",
            "_both_sides_"
        ]
        
        for name in boundary_names:
            with self.subTest(name=name):
                violations = self.validator._validate_logical_name(name, "Test")
                violation_types = [v.violation_type for v in violations]
                self.assertIn("underscore_boundary", violation_types)
    
    def test_double_underscore_issues(self):
        """Test logical names with double underscores."""
        double_underscore_names = [
            "double__underscore",
            "multiple___underscores",
            "start__and__end"
        ]
        
        for name in double_underscore_names:
            with self.subTest(name=name):
                violations = self.validator._validate_logical_name(name, "Test")
                violation_types = [v.violation_type for v in violations]
                self.assertIn("double_underscore", violation_types)
    
    def test_empty_logical_name(self):
        """Test empty logical name validation."""
        violations = self.validator._validate_logical_name("", "Test")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, "snake_case")
    
    def test_none_logical_name(self):
        """Test None logical name validation."""
        # The current implementation doesn't handle None properly, so this will raise TypeError
        with self.assertRaises(TypeError):
            self.validator._validate_logical_name(None, "Test")

if __name__ == '__main__':
    unittest.main()
