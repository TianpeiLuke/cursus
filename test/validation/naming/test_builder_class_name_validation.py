"""
Pytest tests for builder class name validation.
"""

import pytest

from cursus.validation.naming.naming_standard_validator import NamingStandardValidator

class TestBuilderClassNameValidation:
    """Test builder class name validation."""
    
    @pytest.fixture
    def validator(self):
        """Set up test fixtures."""
        return NamingStandardValidator()
    
    def test_valid_builder_class_names(self, validator):
        """Test valid builder class names."""
        valid_names = [
            "XGBoostTrainingStepBuilder",
            "TabularPreprocessingStepBuilder",
            "ModelCalibrationStepBuilder",
            "AStepBuilder",  # Single letter + StepBuilder
            "ABC123StepBuilder"  # With numbers + StepBuilder
        ]
        
        for name in valid_names:
            violations = validator._validate_builder_class_name(name)
            assert len(violations) == 0, f"Valid builder name '{name}' should not have violations"
    
    def test_missing_stepbuilder_suffix(self, validator):
        """Test builder class names missing 'StepBuilder' suffix."""
        invalid_names = [
            "XGBoostTraining",
            "TabularPreprocessing",
            "XGBoostTrainingBuilder"  # Missing 'Step'
        ]
        
        for name in invalid_names:
            violations = validator._validate_builder_class_name(name)
            violation_types = [v.violation_type for v in violations]
            assert "builder_suffix" in violation_types
    
    def test_invalid_base_name_pattern(self, validator):
        """Test builder class names with invalid base name patterns."""
        invalid_names = [
            "snake_caseStepBuilder",
            "camelCaseStepBuilder",
            "123NumberStepBuilder",
            "with-hyphenStepBuilder"
        ]
        
        for name in invalid_names:
            violations = validator._validate_builder_class_name(name)
            violation_types = [v.violation_type for v in violations]
            assert "pascal_case" in violation_types
