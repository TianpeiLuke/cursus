"""
Unit tests for builder class name validation.
"""

import unittest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestBuilderClassNameValidation(unittest.TestCase):
    """Test builder class name validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_valid_builder_class_names(self):
        """Test valid builder class names."""
        valid_names = [
            "XGBoostTrainingStepBuilder",
            "TabularPreprocessingStepBuilder",
            "ModelCalibrationStepBuilder",
            "AStepBuilder",  # Single letter + StepBuilder
            "ABC123StepBuilder"  # With numbers + StepBuilder
        ]
        
        for name in valid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_builder_class_name(name)
                self.assertEqual(len(violations), 0, f"Valid builder name '{name}' should not have violations")
    
    def test_missing_stepbuilder_suffix(self):
        """Test builder class names missing 'StepBuilder' suffix."""
        invalid_names = [
            "XGBoostTraining",
            "TabularPreprocessing",
            "XGBoostTrainingBuilder"  # Missing 'Step'
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_builder_class_name(name)
                violation_types = [v.violation_type for v in violations]
                self.assertIn("builder_suffix", violation_types)
    
    def test_invalid_base_name_pattern(self):
        """Test builder class names with invalid base name patterns."""
        invalid_names = [
            "snake_caseStepBuilder",
            "camelCaseStepBuilder",
            "123NumberStepBuilder",
            "with-hyphenStepBuilder"
        ]
        
        for name in invalid_names:
            with self.subTest(name=name):
                violations = self.validator._validate_builder_class_name(name)
                violation_types = [v.violation_type for v in violations]
                self.assertIn("pascal_case", violation_types)

if __name__ == '__main__':
    unittest.main()
