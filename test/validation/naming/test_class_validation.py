"""
Pytest tests for class validation functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestClassValidation:
    """Test class validation functionality."""
    
    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()
    
    def test_validate_config_class_valid(self, validator):
        """Test validation of a valid config class."""
        # Create a mock config class
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTrainingConfig"
        mock_config_class.__fields__ = {
            "input_data": "field1",
            "model_artifacts": "field2", 
            "training_params": "field3"
        }
        
        violations = validator.validate_config_class(mock_config_class)
        assert len(violations) == 0
    
    def test_validate_config_class_invalid_name(self, validator):
        """Test validation of config class with invalid name."""
        # Create a mock config class with invalid name
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTraining"  # Missing 'Config' suffix
        mock_config_class.__fields__ = {
            "input_data": "field1"
        }
        
        violations = validator.validate_config_class(mock_config_class)
        violation_types = [v.violation_type for v in violations]
        assert "config_suffix" in violation_types
    
    def test_validate_config_class_invalid_field_names(self, validator):
        """Test validation of config class with invalid field names."""
        # Create a mock config class with invalid field names
        mock_config_class = Mock()
        mock_config_class.__name__ = "XGBoostTrainingConfig"
        mock_config_class.__fields__ = {
            "InputData": "field1",  # PascalCase instead of snake_case
            "modelArtifacts": "field2",  # camelCase instead of snake_case
            "training_params": "field3"  # Valid
        }
        
        violations = validator.validate_config_class(mock_config_class)
        # Should have violations for the invalid field names
        assert len(violations) > 0
    
    def test_validate_builder_class_valid(self, validator):
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
        with patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = [
                ("build_step", mock_method1),
                ("validate_config", mock_method2),
                ("__init__", Mock()),  # Should be ignored
                ("_private_method", Mock())  # Should be ignored
            ]
            
            violations = validator.validate_step_builder_class(mock_builder_class)
            assert len(violations) == 0
    
    def test_validate_builder_class_invalid_name(self, validator):
        """Test validation of builder class with invalid name."""
        # Create a mock builder class with invalid name
        mock_builder_class = Mock()
        mock_builder_class.__name__ = "XGBoostTrainingBuilder"  # Missing 'Step'
        
        with patch('inspect.getmembers') as mock_getmembers:
            mock_getmembers.return_value = []
            
            violations = validator.validate_step_builder_class(mock_builder_class)
            violation_types = [v.violation_type for v in violations]
            assert "builder_suffix" in violation_types
    
    def test_validate_builder_class_invalid_method_names(self, validator):
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
        
        violations = validator.validate_step_builder_class(TestBuilderClass)
        # Should have violations for the invalid method names
        method_violations = [v for v in violations if v.violation_type == "method_snake_case"]
        assert len(method_violations) > 0
