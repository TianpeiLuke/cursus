"""Unit tests for error handling utilities."""

import unittest

from src.cursus.validation.runtime.utils.error_handling import (
    ScriptExecutionError,
    ScriptImportError,
    DataFlowError,
    ConfigurationError,
    ValidationError
)


class TestErrorHandling(unittest.TestCase):
    """Test cases for custom error classes."""
    
    def test_script_execution_error_creation(self):
        """Test ScriptExecutionError creation and inheritance."""
        error_message = "Script execution failed with runtime error"
        error = ScriptExecutionError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertIsInstance(error, ScriptExecutionError)
        self.assertEqual(str(error), error_message)
    
    def test_script_execution_error_empty_message(self):
        """Test ScriptExecutionError with empty message."""
        error = ScriptExecutionError("")
        
        self.assertIsInstance(error, ScriptExecutionError)
        self.assertEqual(str(error), "")
    
    def test_script_execution_error_with_cause(self):
        """Test ScriptExecutionError with underlying cause."""
        original_error = ValueError("Original error")
        error = ScriptExecutionError("Script failed")
        error.__cause__ = original_error
        
        self.assertIsInstance(error, ScriptExecutionError)
        self.assertEqual(str(error), "Script failed")
        self.assertEqual(error.__cause__, original_error)
        self.assertIsInstance(error.__cause__, ValueError)
    
    def test_script_import_error_creation(self):
        """Test ScriptImportError creation and inheritance."""
        error_message = "Failed to import script module"
        error = ScriptImportError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertIsInstance(error, ScriptImportError)
        self.assertEqual(str(error), error_message)
    
    def test_script_import_error_with_details(self):
        """Test ScriptImportError with detailed message."""
        script_path = "/path/to/script.py"
        error_details = "Module not found: missing_dependency"
        error_message = f"Failed to import script {script_path}: {error_details}"
        
        error = ScriptImportError(error_message)
        
        self.assertIsInstance(error, ScriptImportError)
        self.assertIn(script_path, str(error))
        self.assertIn(error_details, str(error))
    
    def test_data_flow_error_creation(self):
        """Test DataFlowError creation and inheritance."""
        error_message = "Data flow validation failed"
        error = DataFlowError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertIsInstance(error, DataFlowError)
        self.assertEqual(str(error), error_message)
    
    def test_data_flow_error_path_specific(self):
        """Test DataFlowError with path-specific information."""
        path = "/path/to/missing/file.csv"
        error_message = f"Required input file does not exist: {path}"
        error = DataFlowError(error_message)
        
        self.assertIsInstance(error, DataFlowError)
        self.assertIn(path, str(error))
    
    def test_configuration_error_creation(self):
        """Test ConfigurationError creation and inheritance."""
        error_message = "Invalid configuration parameter"
        error = ConfigurationError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertIsInstance(error, ConfigurationError)
        self.assertEqual(str(error), error_message)
    
    def test_configuration_error_parameter_specific(self):
        """Test ConfigurationError with parameter-specific information."""
        param_name = "data_source"
        param_value = "invalid_source"
        error_message = f"Invalid value '{param_value}' for parameter '{param_name}'"
        error = ConfigurationError(error_message)
        
        self.assertIsInstance(error, ConfigurationError)
        self.assertIn(param_name, str(error))
        self.assertIn(param_value, str(error))
    
    def test_validation_error_creation(self):
        """Test ValidationError creation and inheritance."""
        error_message = "Data validation failed"
        error = ValidationError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertIsInstance(error, ValidationError)
        self.assertEqual(str(error), error_message)
    
    def test_validation_error_schema_specific(self):
        """Test ValidationError with schema-specific information."""
        field_name = "age"
        expected_type = "integer"
        actual_type = "string"
        error_message = f"Field '{field_name}' expected {expected_type}, got {actual_type}"
        error = ValidationError(error_message)
        
        self.assertIsInstance(error, ValidationError)
        self.assertIn(field_name, str(error))
        self.assertIn(expected_type, str(error))
        self.assertIn(actual_type, str(error))
    
    def test_error_inheritance_hierarchy(self):
        """Test that all custom errors inherit from Exception."""
        errors = [
            ScriptExecutionError("test"),
            ScriptImportError("test"),
            DataFlowError("test"),
            ConfigurationError("test"),
            ValidationError("test")
        ]
        
        for error in errors:
            self.assertIsInstance(error, Exception)
            # Verify they can be caught as generic Exception
            try:
                raise error
            except Exception as e:
                self.assertEqual(e, error)
    
    def test_error_distinctness(self):
        """Test that different error types are distinct."""
        script_exec_error = ScriptExecutionError("test")
        script_import_error = ScriptImportError("test")
        data_flow_error = DataFlowError("test")
        config_error = ConfigurationError("test")
        validation_error = ValidationError("test")
        
        # Test that they are different types
        self.assertNotIsInstance(script_exec_error, ScriptImportError)
        self.assertNotIsInstance(script_import_error, DataFlowError)
        self.assertNotIsInstance(data_flow_error, ConfigurationError)
        self.assertNotIsInstance(config_error, ValidationError)
        self.assertNotIsInstance(validation_error, ScriptExecutionError)
    
    def test_error_catching_specificity(self):
        """Test that specific error types can be caught individually."""
        # Test ScriptExecutionError catching
        try:
            raise ScriptExecutionError("execution failed")
        except ScriptExecutionError as e:
            self.assertIsInstance(e, ScriptExecutionError)
            self.assertEqual(str(e), "execution failed")
        except Exception:
            self.fail("Should have caught ScriptExecutionError specifically")
        
        # Test ScriptImportError catching
        try:
            raise ScriptImportError("import failed")
        except ScriptImportError as e:
            self.assertIsInstance(e, ScriptImportError)
            self.assertEqual(str(e), "import failed")
        except Exception:
            self.fail("Should have caught ScriptImportError specifically")
        
        # Test DataFlowError catching
        try:
            raise DataFlowError("data flow failed")
        except DataFlowError as e:
            self.assertIsInstance(e, DataFlowError)
            self.assertEqual(str(e), "data flow failed")
        except Exception:
            self.fail("Should have caught DataFlowError specifically")
        
        # Test ConfigurationError catching
        try:
            raise ConfigurationError("config failed")
        except ConfigurationError as e:
            self.assertIsInstance(e, ConfigurationError)
            self.assertEqual(str(e), "config failed")
        except Exception:
            self.fail("Should have caught ConfigurationError specifically")
        
        # Test ValidationError catching
        try:
            raise ValidationError("validation failed")
        except ValidationError as e:
            self.assertIsInstance(e, ValidationError)
            self.assertEqual(str(e), "validation failed")
        except Exception:
            self.fail("Should have caught ValidationError specifically")
    
    def test_error_with_complex_messages(self):
        """Test errors with complex, multi-line messages."""
        complex_message = """
        Script execution failed with multiple issues:
        1. Missing required input file: /path/to/input.csv
        2. Invalid configuration parameter: batch_size = -1
        3. Memory allocation error: insufficient memory
        
        Stack trace:
        File "script.py", line 42, in main
            process_data(input_file)
        File "script.py", line 15, in process_data
            data = pd.read_csv(input_file)
        """
        
        error = ScriptExecutionError(complex_message)
        
        self.assertIsInstance(error, ScriptExecutionError)
        self.assertIn("Script execution failed", str(error))
        self.assertIn("Missing required input file", str(error))
        self.assertIn("Stack trace", str(error))
    
    def test_error_chaining(self):
        """Test error chaining with __cause__ and __context__."""
        # Create a chain of errors
        original_error = FileNotFoundError("File not found")
        
        try:
            raise original_error
        except FileNotFoundError as e:
            import_error = ScriptImportError("Failed to import script")
            import_error.__cause__ = e
            
            try:
                raise import_error
            except ScriptImportError as e2:
                execution_error = ScriptExecutionError("Script execution failed")
                execution_error.__cause__ = e2
                
                # Verify the error chain
                self.assertIsInstance(execution_error, ScriptExecutionError)
                self.assertIsInstance(execution_error.__cause__, ScriptImportError)
                self.assertIsInstance(execution_error.__cause__.__cause__, FileNotFoundError)
                
                self.assertEqual(str(execution_error.__cause__.__cause__), "File not found")
    
    def test_error_args_attribute(self):
        """Test that error args attribute is properly set."""
        message = "Test error message"
        
        errors = [
            ScriptExecutionError(message),
            ScriptImportError(message),
            DataFlowError(message),
            ConfigurationError(message),
            ValidationError(message)
        ]
        
        for error in errors:
            self.assertEqual(error.args, (message,))
            self.assertEqual(error.args[0], message)


if __name__ == '__main__':
    unittest.main()
