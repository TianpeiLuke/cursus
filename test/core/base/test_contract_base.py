import unittest
from unittest.mock import Mock, patch, mock_open
from typing import Dict, List
import logging
import tempfile
import os

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.cursus.core.base.contract_base import (
    ScriptContract, ValidationResult, ScriptAnalyzer
)


class TestValidationResult(unittest.TestCase):
    """Test cases for ValidationResult class."""
    
    def test_init_valid(self):
        """Test initialization with valid result."""
        result = ValidationResult(is_valid=True)
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
    
    def test_init_invalid_with_errors(self):
        """Test initialization with invalid result and errors."""
        errors = ["Error 1", "Error 2"]
        warnings = ["Warning 1"]
        
        result = ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings
        )
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, errors)
        self.assertEqual(result.warnings, warnings)
    
    def test_success_class_method(self):
        """Test success class method."""
        result = ValidationResult.success("Test success")
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.warnings, [])
    
    def test_error_class_method_with_list(self):
        """Test error class method with list of errors."""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult.error(errors)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, errors)
    
    def test_error_class_method_with_string(self):
        """Test error class method with single error string."""
        error = "Single error"
        result = ValidationResult.error(error)
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.errors, [error])
    
    def test_combine_class_method(self):
        """Test combine class method."""
        result1 = ValidationResult(is_valid=True, warnings=["Warning 1"])
        result2 = ValidationResult(is_valid=False, errors=["Error 1"], warnings=["Warning 2"])
        result3 = ValidationResult(is_valid=False, errors=["Error 2"])
        
        combined = ValidationResult.combine([result1, result2, result3])
        
        self.assertFalse(combined.is_valid)
        self.assertEqual(combined.errors, ["Error 1", "Error 2"])
        self.assertEqual(combined.warnings, ["Warning 1", "Warning 2"])
    
    def test_combine_all_valid(self):
        """Test combine with all valid results."""
        result1 = ValidationResult(is_valid=True)
        result2 = ValidationResult(is_valid=True, warnings=["Warning 1"])
        
        combined = ValidationResult.combine([result1, result2])
        
        self.assertTrue(combined.is_valid)
        self.assertEqual(combined.errors, [])
        self.assertEqual(combined.warnings, ["Warning 1"])


class TestScriptContract(unittest.TestCase):
    """Test cases for ScriptContract class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_contract_data = {
            "entry_point": "train.py",
            "expected_input_paths": {
                "training_data": "/opt/ml/processing/input/train",
                "validation_data": "/opt/ml/processing/input/validation"
            },
            "expected_output_paths": {
                "model": "/opt/ml/processing/output/model",
                "metrics": "/opt/ml/processing/output/metrics"
            },
            "required_env_vars": ["MODEL_TYPE", "LEARNING_RATE"],
            "optional_env_vars": {"BATCH_SIZE": "32", "EPOCHS": "10"},
            "expected_arguments": {
                "input-path": "/opt/ml/processing/input/train",
                "output-path": "/opt/ml/processing/output/model"
            },
            "framework_requirements": {"xgboost": "1.5.0"},
            "description": "Training script for XGBoost model"
        }
    
    def test_init_with_valid_data(self):
        """Test initialization with valid data."""
        contract = ScriptContract(**self.valid_contract_data)
        
        self.assertEqual(contract.entry_point, "train.py")
        self.assertEqual(contract.expected_input_paths["training_data"], "/opt/ml/processing/input/train")
        self.assertEqual(contract.expected_output_paths["model"], "/opt/ml/processing/output/model")
        self.assertEqual(contract.required_env_vars, ["MODEL_TYPE", "LEARNING_RATE"])
        self.assertEqual(contract.optional_env_vars["BATCH_SIZE"], "32")
        self.assertEqual(contract.expected_arguments["input-path"], "/opt/ml/processing/input/train")
        self.assertEqual(contract.framework_requirements["xgboost"], "1.5.0")
        self.assertEqual(contract.description, "Training script for XGBoost model")
    
    def test_init_with_minimal_data(self):
        """Test initialization with minimal required data."""
        minimal_data = {
            "entry_point": "script.py",
            "expected_input_paths": {},
            "expected_output_paths": {},
            "required_env_vars": []
        }
        
        contract = ScriptContract(**minimal_data)
        
        self.assertEqual(contract.entry_point, "script.py")
        self.assertEqual(contract.expected_input_paths, {})
        self.assertEqual(contract.expected_output_paths, {})
        self.assertEqual(contract.required_env_vars, [])
        self.assertEqual(contract.optional_env_vars, {})
        self.assertEqual(contract.expected_arguments, {})
        self.assertEqual(contract.framework_requirements, {})
        self.assertEqual(contract.description, "")
    
    def test_validate_entry_point_invalid(self):
        """Test entry point validation with invalid file."""
        invalid_data = self.valid_contract_data.copy()
        invalid_data["entry_point"] = "script.txt"
        
        with self.assertRaises(ValueError) as context:
            ScriptContract(**invalid_data)
        
        self.assertIn("Entry point must be a Python file", str(context.exception))
    
    def test_validate_input_paths_invalid(self):
        """Test input path validation with invalid paths."""
        invalid_data = self.valid_contract_data.copy()
        invalid_data["expected_input_paths"] = {
            "data": "/invalid/path"
        }
        
        with self.assertRaises(ValueError) as context:
            ScriptContract(**invalid_data)
        
        self.assertIn("must start with /opt/ml/processing/input", str(context.exception))
    
    def test_validate_input_paths_generated_payload_samples(self):
        """Test input path validation for GeneratedPayloadSamples."""
        valid_data = self.valid_contract_data.copy()
        valid_data["expected_input_paths"] = {
            "GeneratedPayloadSamples": "/opt/ml/processing/payload/samples"
        }
        
        # Should not raise an exception
        contract = ScriptContract(**valid_data)
        self.assertEqual(contract.expected_input_paths["GeneratedPayloadSamples"], "/opt/ml/processing/payload/samples")
    
    def test_validate_output_paths_invalid(self):
        """Test output path validation with invalid paths."""
        invalid_data = self.valid_contract_data.copy()
        invalid_data["expected_output_paths"] = {
            "result": "/invalid/path"
        }
        
        with self.assertRaises(ValueError) as context:
            ScriptContract(**invalid_data)
        
        self.assertIn("must start with /opt/ml/processing/output", str(context.exception))
    
    def test_validate_arguments_invalid_characters(self):
        """Test argument validation with invalid characters."""
        invalid_data = self.valid_contract_data.copy()
        invalid_data["expected_arguments"] = {
            "input_path!": "/some/path"
        }
        
        with self.assertRaises(ValueError) as context:
            ScriptContract(**invalid_data)
        
        self.assertIn("contains invalid characters", str(context.exception))
    
    def test_validate_arguments_uppercase(self):
        """Test argument validation with uppercase characters."""
        invalid_data = self.valid_contract_data.copy()
        invalid_data["expected_arguments"] = {
            "INPUT-PATH": "/some/path"
        }
        
        with self.assertRaises(ValueError) as context:
            ScriptContract(**invalid_data)
        
        self.assertIn("should be lowercase", str(context.exception))
    
    def test_validate_implementation_file_not_found(self):
        """Test implementation validation with non-existent file."""
        contract = ScriptContract(**self.valid_contract_data)
        
        result = contract.validate_implementation("/nonexistent/script.py")
        
        self.assertFalse(result.is_valid)
        self.assertIn("Script file not found", result.errors[0])
    
    @patch('src.cursus.core.base.contract_base.ScriptAnalyzer')
    def test_validate_implementation_success(self, mock_analyzer_class):
        """Test successful implementation validation."""
        # Mock analyzer
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {"/opt/ml/processing/input/train", "/opt/ml/processing/input/validation"}
        mock_analyzer.get_output_paths.return_value = {"/opt/ml/processing/output/model", "/opt/ml/processing/output/metrics"}
        mock_analyzer.get_env_var_usage.return_value = {"MODEL_TYPE", "LEARNING_RATE"}
        mock_analyzer.get_argument_usage.return_value = {"input-path", "output-path"}
        mock_analyzer_class.return_value = mock_analyzer
        
        contract = ScriptContract(**self.valid_contract_data)
        
        with patch('os.path.exists', return_value=True):
            result = contract.validate_implementation("/test/script.py")
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])
    
    @patch('src.cursus.core.base.contract_base.ScriptAnalyzer')
    def test_validate_implementation_missing_paths(self, mock_analyzer_class):
        """Test implementation validation with missing paths."""
        # Mock analyzer with missing paths
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {"/opt/ml/processing/input/train"}  # Missing validation
        mock_analyzer.get_output_paths.return_value = {"/opt/ml/processing/output/model"}  # Missing metrics
        mock_analyzer.get_env_var_usage.return_value = {"MODEL_TYPE"}  # Missing LEARNING_RATE
        mock_analyzer.get_argument_usage.return_value = {"input-path"}  # Missing output-path
        mock_analyzer_class.return_value = mock_analyzer
        
        contract = ScriptContract(**self.valid_contract_data)
        
        with patch('os.path.exists', return_value=True):
            result = contract.validate_implementation("/test/script.py")
        
        self.assertFalse(result.is_valid)
        self.assertTrue(any("validation" in error for error in result.errors))
        self.assertTrue(any("metrics" in error for error in result.errors))
        self.assertTrue(any("LEARNING_RATE" in error for error in result.errors))
    
    @patch('src.cursus.core.base.contract_base.ScriptAnalyzer')
    def test_validate_implementation_with_warnings(self, mock_analyzer_class):
        """Test implementation validation with warnings."""
        # Mock analyzer with extra paths
        mock_analyzer = Mock()
        mock_analyzer.get_input_paths.return_value = {
            "/opt/ml/processing/input/train", 
            "/opt/ml/processing/input/validation",
            "/opt/ml/processing/input/extra"  # Extra path
        }
        mock_analyzer.get_output_paths.return_value = {"/opt/ml/processing/output/model", "/opt/ml/processing/output/metrics"}
        mock_analyzer.get_env_var_usage.return_value = {"MODEL_TYPE", "LEARNING_RATE"}
        mock_analyzer.get_argument_usage.return_value = {"input-path"}  # Missing output-path
        mock_analyzer_class.return_value = mock_analyzer
        
        contract = ScriptContract(**self.valid_contract_data)
        
        with patch('os.path.exists', return_value=True):
            result = contract.validate_implementation("/test/script.py")
        
        self.assertTrue(result.is_valid)
        self.assertTrue(any("undeclared input path" in warning for warning in result.warnings))
        self.assertTrue(any("output-path" in warning for warning in result.warnings))


class TestScriptAnalyzer(unittest.TestCase):
    """Test cases for ScriptAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_script = '''
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    # Input paths
    train_data = "/opt/ml/processing/input/train"
    val_data = "/opt/ml/processing/input/validation"
    
    # Output paths
    model_path = "/opt/ml/processing/output/model"
    metrics_path = "/opt/ml/processing/output/metrics"
    
    # Environment variables
    model_type = os.environ["MODEL_TYPE"]
    learning_rate = os.environ.get("LEARNING_RATE", "0.1")
    batch_size = os.getenv("BATCH_SIZE", "32")
    
    print(f"Training with {model_type}")

if __name__ == "__main__":
    main()
'''
    
    def test_get_input_paths(self):
        """Test extraction of input paths."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                input_paths = analyzer.get_input_paths()
                
                expected_paths = {
                    "/opt/ml/processing/input/train",
                    "/opt/ml/processing/input/validation"
                }
                self.assertEqual(input_paths, expected_paths)
            finally:
                os.unlink(f.name)
    
    def test_get_output_paths(self):
        """Test extraction of output paths."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                output_paths = analyzer.get_output_paths()
                
                expected_paths = {
                    "/opt/ml/processing/output/model",
                    "/opt/ml/processing/output/metrics"
                }
                self.assertEqual(output_paths, expected_paths)
            finally:
                os.unlink(f.name)
    
    def test_get_env_var_usage(self):
        """Test extraction of environment variable usage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                env_vars = analyzer.get_env_var_usage()
                
                expected_vars = {"MODEL_TYPE", "LEARNING_RATE", "BATCH_SIZE"}
                self.assertEqual(env_vars, expected_vars)
            finally:
                os.unlink(f.name)
    
    def test_get_argument_usage(self):
        """Test extraction of argument usage."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                arguments = analyzer.get_argument_usage()
                
                expected_args = {"input-path", "output-path", "v"}  # Script analyzer finds short form "v"
                self.assertEqual(arguments, expected_args)
            finally:
                os.unlink(f.name)
    
    def test_ast_tree_lazy_loading(self):
        """Test that AST tree is lazily loaded."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                
                # AST should not be loaded yet
                self.assertIsNone(analyzer._ast_tree)
                
                # Access AST tree
                ast_tree = analyzer.ast_tree
                
                # Should now be loaded
                self.assertIsNotNone(ast_tree)
                self.assertEqual(analyzer._ast_tree, ast_tree)
                
                # Second access should return same object
                ast_tree2 = analyzer.ast_tree
                self.assertEqual(ast_tree, ast_tree2)
            finally:
                os.unlink(f.name)
    
    def test_caching_behavior(self):
        """Test that analysis results are cached."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(self.sample_script)
            f.flush()
            
            try:
                analyzer = ScriptAnalyzer(f.name)
                
                # First call should populate cache
                input_paths1 = analyzer.get_input_paths()
                self.assertIsNotNone(analyzer._input_paths)
                
                # Second call should use cache
                input_paths2 = analyzer.get_input_paths()
                self.assertEqual(input_paths1, input_paths2)
                
                # Same for other methods
                env_vars1 = analyzer.get_env_var_usage()
                env_vars2 = analyzer.get_env_var_usage()
                self.assertEqual(env_vars1, env_vars2)
            finally:
                os.unlink(f.name)


if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
