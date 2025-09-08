"""
Unit tests for class validation functionality.
"""

import unittest
from unittest.mock import Mock, MagicMock

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestClassValidation(unittest.TestCase):
    """Test class validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_validate_config_class_valid(self):
        """Test validation of a valid config class."""
        # Create a mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTrainingConfig"
        mock_config_class.__fields__ = {
            "input_data": "field1",
            "model_artifacts": "field2", 
            "training_params": "field3"
        }
        
        violations = self.validator.validate_config_class(mock_config_class)
        self.assertEqual(len(violations), 0)
    
    def test_validate_config_class_invalid_name(self):
        """Test validation of config class with invalid name."""
        # Create a mock config class with invalid name
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTraining"  # Missing 'Config' suffix
        mock_config_class.__fields__ = {
            "input_data": "field1"
        }
        
        violations = self.validator.validate_config_class(mock_config_class)
        violation_types = [v.violation_type for v in violations]
        self.assertIn("config_suffix", violation_types)
    
    def test_validate_config_class_invalid_field_names(self):
        """Test validation of config class with invalid field names."""
        # Create a mock config class with invalid field names
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTrainingConfig"
        mock_config_class.__fields__ = {
            "InputData": "field1",  # PascalCase instead of snake_case
            "modelArtifacts": "field2",  # camelCase instead of snake_case
            "training_params": "field3"  # Valid
        }
        
        violations = self.validator.validate_config_class(mock_config_class)
        # Should have violations for the invalid field names
        self.assertGreater(len(violations), 0)
    
    def test_validate_builder_class_valid(self):
        """Test validation of a valid builder class."""
        # Create a mock builder class
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "XGBoostTrainingStepBuilder"
        
        # Mock the methods
        mock_method1 = Mock()
        mock_method1.__name__ = "build_step"
        mock_method2 = Mock()
        mock_method2.__name__ = "validate_config"
        
        # Mock inspect.getmembers to return our mock methods
        with unittest.mock.patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = [
                ("build_step", mock_method1),
                ("validate_config", mock_method2),
                ("__init__", Mock()),  # Should be ignored
                ("_private_method", Mock())  # Should be ignored
            ]
            
            violations = self.validator.validate_step_builder_class(mock_builder_class)
            self.assertEqual(len(violations), 0)
    
    def test_validate_builder_class_invalid_name(self):
        """Test validation of builder class with invalid name."""
        # Create a mock builder class with invalid name
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "XGBoostTrainingBuilder"  # Missing 'Step'
        
        with unittest.mock.patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = []
            
            violations = self.validator.validate_step_builder_class(mock_builder_class)
            violation_types = [v.violation_type for v in violations]
            self.assertIn("builder_suffix", violation_types)
    
    def test_validate_builder_class_invalid_method_names(self):
        """Test validation of builder class with invalid method names."""
        # Create a real class with invalid method names for testing
        class TestBuilderClass:
            def __init__(self):
                pass
            
            def BuildStep(self):  # PascalCase - invalid
                pass
                
            def validateConfig(self):  # camelCase - invalid
                pass
                
            def _private_method(self):  # Private - should be ignored
                pass
        
        TestBuilderClass.__name__ = "XGBoostTrainingStepBuilder"
        
        violations = self.validator.validate_step_builder_class(TestBuilderClass)
        # Should have violations for the invalid method names
        method_violations = [v for v in violations if v.violation_type == "method_snake_case"]
        self.assertGreater(len(method_violations), 0)

if __name__ == '__main__':
    unittest.main()
