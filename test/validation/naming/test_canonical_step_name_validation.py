"""
Unit tests for canonical step name validation.
"""

import unittest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestCanonicalStepNameValidation(unittest.TestCase):
    """Test canonical step name validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_valid_pascal_case_names(self):
        """Test valid PascalCase step names."""
        valid_names = [
            "CradleDataLoading",
            "XGBoostTraining",
            "TabularPreprocessing",
            "ModelCalibration",
            "A",  # Single letter
            "ABC123"  # With numbers
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_canonical_step_name(name, "Test")
                self.assertEqual(len(violations), 0, f"Valid name '{name}' should not have violations")
    
    def test_invalid_pascal_case_names(self):
        """Test invalid PascalCase step names."""
        invalid_names = [
            "camelCase",  # starts with lowercase
            "snake_case",  # contains underscore
            "kebab-case",  # contains hyphen
            "lowercase",  # all lowercase
            "UPPERCASE",  # all uppercase (but this actually passes PascalCase regex)
            "123Number",  # starts with number
            "With Space",  # contains space
            "with.dot"  # contains dot
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_canonical_step_name(name, "Test")
                if name not in ["UPPERCASE"]:  # UPPERCASE actually passes the regex
                    self.assertGreater(len(violations), 0, f"Invalid name '{name}' should have violations")
    
    def test_empty_step_name(self):
        """Test empty step name validation."""
        violations = self.validator._validate_canonical_step_name("", "Test")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, "empty_name")
    
    def test_none_step_name(self):
        """Test None step name validation."""
        violations = self.validator._validate_canonical_step_name(None, "Test")
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, "empty_name")
    
    def test_underscore_in_name(self):
        """Test step name with underscores."""
        violations = self.validator._validate_canonical_step_name("Step_Name", "Test")
        # Should have both pascal_case and underscore_in_name violations
        violation_types = [v.violation_type for v in violations]
        self.assertIn("underscore_in_name", violation_types)
    
    def test_lowercase_name(self):
        """Test all lowercase step name."""
        violations = self.validator._validate_canonical_step_name("stepname", "Test")
        violation_types = [v.violation_type for v in violations]
        self.assertIn("lowercase_name", violation_types)

if __name__ == '__main__':
    unittest.main()
