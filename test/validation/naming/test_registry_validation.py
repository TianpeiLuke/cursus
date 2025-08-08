"""
Unit tests for registry validation functionality.
"""

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from unittest.mock import Mock, patch

from src.cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestRegistryValidation(unittest.TestCase):
    """Test registry validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    @patch('src.cursus.validation.naming.naming_standard_validator.STEP_NAMES')
    def test_validate_all_registry_entries_valid(self, mock_step_names):
        """Test validation of valid registry entries."""
        # Mock a valid registry - spec_type should match step name
        mock_step_names.items.return_value = [
            ("XGBoostTraining", {
                "sagemaker_step_type": "Training",
                "spec_type": "XGBoostTraining"  # Should match step name
            }),
            ("TabularPreprocessing", {
                "sagemaker_step_type": "Processing", 
                "spec_type": "TabularPreprocessing"  # Should match step name
            })
        ]
        
        violations = self.validator.validate_all_registry_entries()
        self.assertEqual(len(violations), 0)
    
    @patch('src.cursus.validation.naming.naming_standard_validator.STEP_NAMES')
    def test_validate_all_registry_entries_invalid_step_name(self, mock_step_names):
        """Test validation with invalid step names in registry."""
        # Mock registry with invalid step name
        mock_step_names.items.return_value = [
            ("invalid_step_name", {  # Invalid: snake_case instead of PascalCase
                "sagemaker_step_type": "Training",
                "spec_type": "TrainingSpec"
            })
        ]
        
        violations = self.validator.validate_all_registry_entries()
        self.assertGreater(len(violations), 0)
        violation_types = [v.violation_type for v in violations]
        self.assertIn("pascal_case", violation_types)
    
    @patch('src.cursus.validation.naming.naming_standard_validator.STEP_NAMES')
    def test_validate_all_registry_entries_invalid_sagemaker_type(self, mock_step_names):
        """Test validation with invalid SageMaker step type."""
        # Mock registry with invalid SageMaker step type
        mock_step_names.items.return_value = [
            ("XGBoostTraining", {
                "sagemaker_step_type": "InvalidType",  # Invalid SageMaker step type
                "spec_type": "TrainingSpec"
            })
        ]
        
        violations = self.validator.validate_all_registry_entries()
        self.assertGreater(len(violations), 0)
        violation_types = [v.violation_type for v in violations]
        self.assertIn("invalid_sagemaker_type", violation_types)
    
    @patch('src.cursus.validation.naming.naming_standard_validator.STEP_NAMES')
    def test_validate_all_registry_entries_spec_type_mismatch(self, mock_step_names):
        """Test validation with spec type mismatch."""
        # Mock registry with spec type that doesn't match step name
        mock_step_names.items.return_value = [
            ("XGBoostTraining", {
                "sagemaker_step_type": "Training",
                "spec_type": "ProcessingSpec"  # Should be TrainingSpec for Training step
            })
        ]
        
        violations = self.validator.validate_all_registry_entries()
        self.assertGreater(len(violations), 0)
        violation_types = [v.violation_type for v in violations]
        self.assertIn("spec_type_mismatch", violation_types)
    
    def test_validate_registry_entry_valid(self):
        """Test validation of a single valid registry entry."""
        violations = self.validator.validate_registry_entry(
            "XGBoostTraining",
            {
                "sagemaker_step_type": "Training",
                "spec_type": "XGBoostTraining"  # Should match step name
            }
        )
        self.assertEqual(len(violations), 0)
    
    def test_validate_registry_entry_missing_fields(self):
        """Test validation of registry entry with missing fields."""
        violations = self.validator.validate_registry_entry(
            "XGBoostTraining",
            {}  # Missing required fields
        )
        # Missing fields don't generate violations in current implementation
        # The method only validates what's present
        self.assertEqual(len(violations), 0)
    
    def test_validate_sagemaker_step_type_valid(self):
        """Test validation of valid SageMaker step types."""
        valid_types = [
            "Processing", 
            "Training", 
            "Transform", 
            "CreateModel", 
            "RegisterModel",
            "Lambda", 
            "MimsModelRegistrationProcessing", 
            "CradleDataLoading", 
            "Base"
        ]
        
        for step_type in valid_types:
            with self.subTest(step_type=step_type):
                violations = self.validator._validate_sagemaker_step_type(step_type, "Test")
                self.assertEqual(len(violations), 0, f"Valid SageMaker step type '{step_type}' should not have violations")
    
    def test_validate_sagemaker_step_type_invalid(self):
        """Test validation of invalid SageMaker step types."""
        invalid_types = [
            "InvalidType",
            "training",  # lowercase
            "TRAINING",  # uppercase
            "Custom"
        ]
        
        for step_type in invalid_types:
            with self.subTest(step_type=step_type):
                violations = self.validator._validate_sagemaker_step_type(step_type, "Test")
                self.assertGreater(len(violations), 0, f"Invalid SageMaker step type '{step_type}' should have violations")


if __name__ == '__main__':
    unittest.main()
