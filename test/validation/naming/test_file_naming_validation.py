"""
Unit tests for file naming validation.
"""

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest

from src.cursus.validation.naming.naming_standard_validator import NamingStandardValidator


class TestFileNamingValidation(unittest.TestCase):
    """Test file naming validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = NamingStandardValidator()
    
    def test_valid_builder_file_names(self):
        """Test valid builder file names."""
        valid_names = [
            "builder_xgboost_training_step.py",
            "builder_tabular_preprocessing_step.py",
            "builder_model_calibration_step.py"
        ]
        
        for filename in valid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "builder")
                self.assertEqual(len(violations), 0, f"Valid builder file '{filename}' should not have violations")
    
    def test_valid_config_file_names(self):
        """Test valid config file names."""
        valid_names = [
            "config_xgboost_training_step.py",
            "config_tabular_preprocessing_step.py",
            "config_model_calibration_step.py"
        ]
        
        for filename in valid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "config")
                self.assertEqual(len(violations), 0, f"Valid config file '{filename}' should not have violations")
    
    def test_valid_spec_file_names(self):
        """Test valid spec file names."""
        valid_names = [
            "xgboost_training_spec.py",
            "tabular_preprocessing_spec.py",
            "model_calibration_spec.py"
        ]
        
        for filename in valid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "spec")
                self.assertEqual(len(violations), 0, f"Valid spec file '{filename}' should not have violations")
    
    def test_valid_contract_file_names(self):
        """Test valid contract file names."""
        valid_names = [
            "xgboost_training_contract.py",
            "tabular_preprocessing_contract.py",
            "model_calibration_contract.py"
        ]
        
        for filename in valid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "contract")
                self.assertEqual(len(violations), 0, f"Valid contract file '{filename}' should not have violations")
    
    def test_invalid_builder_file_names(self):
        """Test invalid builder file names."""
        invalid_names = [
            "xgboost_training_step.py",  # Missing 'builder_' prefix
            "builder_XGBoostTraining_step.py",  # PascalCase instead of snake_case
            "builder_xgboost_training.py",  # Missing '_step' suffix
            "builder_xgboost_training_step.txt"  # Wrong extension
        ]
        
        for filename in invalid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "builder")
                self.assertGreater(len(violations), 0, f"Invalid builder file '{filename}' should have violations")
    
    def test_invalid_config_file_names(self):
        """Test invalid config file names."""
        invalid_names = [
            "xgboost_training_step.py",  # Missing 'config_' prefix
            "config_XGBoostTraining_step.py",  # PascalCase instead of snake_case
            "config_xgboost_training.py",  # Missing '_step' suffix
            "config_xgboost_training_step.txt"  # Wrong extension
        ]
        
        for filename in invalid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "config")
                self.assertGreater(len(violations), 0, f"Invalid config file '{filename}' should have violations")
    
    def test_invalid_spec_file_names(self):
        """Test invalid spec file names."""
        invalid_names = [
            "xgboost_training.py",  # Missing '_spec' suffix
            "XGBoostTraining_spec.py",  # PascalCase instead of snake_case
            "xgboost_training_spec.txt"  # Wrong extension
        ]
        
        for filename in invalid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "spec")
                self.assertGreater(len(violations), 0, f"Invalid spec file '{filename}' should have violations")
    
    def test_invalid_contract_file_names(self):
        """Test invalid contract file names."""
        invalid_names = [
            "xgboost_training.py",  # Missing '_contract' suffix
            "XGBoostTraining_contract.py",  # PascalCase instead of snake_case
            "xgboost_training_contract.txt"  # Wrong extension
        ]
        
        for filename in invalid_names:
            with self.subTest(filename=filename):
                violations = self.validator.validate_file_naming(filename, "contract")
                self.assertGreater(len(violations), 0, f"Invalid contract file '{filename}' should have violations")
    
    def test_unsupported_file_type(self):
        """Test validation with unsupported file type."""
        violations = self.validator.validate_file_naming("test.py", "unsupported")
        # Unsupported file types return no violations (they're just ignored)
        self.assertEqual(len(violations), 0)


if __name__ == '__main__':
    unittest.main()
