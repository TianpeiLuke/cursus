"""Unit tests for PipelineScriptExecutor."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path
import argparse
import os

from src.cursus.validation.runtime.core.pipeline_script_executor import PipelineScriptExecutor
from src.cursus.validation.runtime.utils.result_models import TestResult, ExecutionResult
from src.cursus.validation.runtime.utils.execution_context import ExecutionContext
from src.cursus.validation.runtime.utils.error_handling import (
    ScriptExecutionError, ScriptImportError, ConfigurationError
)


class TestPipelineScriptExecutor(unittest.TestCase):
    """Test cases for PipelineScriptExecutor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.executor = PipelineScriptExecutor(workspace_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_workspace_directory(self):
        """Test that initialization creates workspace directory."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertTrue((Path(self.temp_dir) / "logs").exists())
        
    def test_init_initializes_managers(self):
        """Test that initialization creates required managers."""
        self.assertIsNotNone(self.executor.script_manager)
        self.assertIsNotNone(self.executor.data_manager)
        self.assertIsNotNone(self.executor.local_data_manager)
        self.assertEqual(self.executor.execution_history, [])
    
    @patch('pathlib.Path.exists')
    def test_discover_script_path_success(self, mock_exists):
        """Test successful script path discovery."""
        def mock_exists_side_effect():
            return True
        
        # Mock the first path to return True
        mock_exists.side_effect = [True]
        
        result = self.executor._discover_script_path("test_script")
        self.assertEqual(result, "src/cursus/steps/scripts/test_script.py")
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.Path.exists')
    def test_discover_script_path_not_found(self, mock_exists):
        """Test script path discovery when script not found."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError) as context:
            self.executor._discover_script_path("nonexistent_script")
        
        self.assertIn("Script not found: nonexistent_script", str(context.exception))
    
    def test_prepare_basic_execution_context_synthetic(self):
        """Test preparation of execution context with synthetic data."""
        context = self.executor._prepare_basic_execution_context("test_script", "synthetic")
        
        self.assertIsInstance(context, ExecutionContext)
        self.assertIn("input", context.input_paths)
        self.assertIn("output", context.output_paths)
        self.assertIsInstance(context.environ_vars, dict)
        self.assertIsInstance(context.job_args, argparse.Namespace)
        self.assertTrue(context.job_args.verbose)
    
    def test_prepare_basic_execution_context_local(self):
        """Test preparation of execution context with local data."""
        # Mock local data manager methods
        self.executor.local_data_manager.get_data_for_script = Mock(return_value={"data.csv": "/path/to/data.csv"})
        self.executor.local_data_manager.prepare_data_for_execution = Mock()
        
        # Call the actual method
        context = self.executor._prepare_basic_execution_context("test_script", "local")
        
        # Verify local data manager was called
        self.executor.local_data_manager.get_data_for_script.assert_called_once_with("test_script")
        self.executor.local_data_manager.prepare_data_for_execution.assert_called_once()
        
        # Verify context was created properly
        self.assertIsInstance(context, ExecutionContext)
        self.assertIn("input", context.input_paths)
        self.assertIn("output", context.output_paths)
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_success(self, mock_discover):
        """Test successful script isolation testing."""
        mock_discover.return_value = "test/path/script.py"
        
        # Mock script manager
        mock_main_func = Mock(return_value={"status": "success"})
        self.executor.script_manager.import_script_main = Mock(return_value=mock_main_func)
        
        mock_execution_result = ExecutionResult(
            success=True,
            execution_time=1.5,
            memory_usage=100,
            result_data={"status": "success"}
        )
        self.executor.script_manager.execute_script_main = Mock(return_value=mock_execution_result)
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.script_name, "test_script")
        self.assertEqual(result.status, "PASS")
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.memory_usage, 100)
        self.assertIsNone(result.error_message)
        
        # Verify execution history was recorded
        self.assertEqual(len(self.executor.execution_history), 1)
        self.assertEqual(self.executor.execution_history[0]["script_name"], "test_script")
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_script_import_error(self, mock_discover):
        """Test script isolation with import error."""
        mock_discover.side_effect = ScriptImportError("Failed to import script")
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.script_name, "test_script")
        self.assertEqual(result.status, "FAIL")
        self.assertIn("ScriptImportError", result.error_message)
        self.assertIn("Check if the script has syntax errors", result.recommendations[0])
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_script_execution_error(self, mock_discover):
        """Test script isolation with execution error."""
        mock_discover.return_value = "test/path/script.py"
        
        # Mock script manager to raise execution error
        self.executor.script_manager.import_script_main = Mock(return_value=Mock())
        self.executor.script_manager.execute_script_main = Mock(
            side_effect=ScriptExecutionError("Runtime error")
        )
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("ScriptExecutionError", result.error_message)
        self.assertIn("Review script logic for runtime errors", result.recommendations[0])
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_configuration_error(self, mock_discover):
        """Test script isolation with configuration error."""
        mock_discover.return_value = "test/path/script.py"
        
        # Mock script manager to raise configuration error
        self.executor.script_manager.import_script_main = Mock(return_value=Mock())
        self.executor.script_manager.execute_script_main = Mock(
            side_effect=ConfigurationError("Invalid configuration")
        )
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("ConfigurationError", result.error_message)
        self.assertIn("Check data source configuration", result.recommendations[0])
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_file_not_found_error(self, mock_discover):
        """Test script isolation with file not found error."""
        mock_discover.side_effect = FileNotFoundError("Script not found")
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("Script not found", result.error_message)
        self.assertIn("Verify script 'test_script' exists", result.recommendations[0])
    
    @patch('src.cursus.validation.runtime.core.pipeline_script_executor.PipelineScriptExecutor._discover_script_path')
    def test_test_script_isolation_unexpected_error(self, mock_discover):
        """Test script isolation with unexpected error."""
        mock_discover.side_effect = RuntimeError("Unexpected error")
        
        result = self.executor.test_script_isolation("test_script")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("Unexpected error", result.error_message)
        self.assertIn("Check system logs for details", result.recommendations[0])
    
    def test_test_script_isolation_unsupported_data_source(self):
        """Test script isolation with unsupported data source."""
        result = self.executor.test_script_isolation("test_script", data_source="s3")
        
        self.assertIsInstance(result, TestResult)
        self.assertEqual(result.status, "FAIL")
        self.assertIn("ConfigurationError", result.error_message)
        self.assertIn("Data source 's3' not yet implemented", result.error_message)
    
    def test_test_pipeline_e2e_not_implemented(self):
        """Test that pipeline end-to-end testing raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as context:
            self.executor.test_pipeline_e2e({})
        
        self.assertIn("Pipeline end-to-end testing will be implemented in Phase 2", str(context.exception))
    
    def test_generate_basic_recommendations_performance(self):
        """Test basic recommendations generation for performance issues."""
        execution_result = ExecutionResult(
            success=True,
            execution_time=65.0,  # > 60 seconds
            memory_usage=1500,    # > 1GB
            result_data=None
        )
        
        recommendations = self.executor._generate_basic_recommendations(execution_result)
        
        self.assertIn("Consider optimizing script performance", recommendations[0])
        self.assertIn("Consider optimizing memory usage", recommendations[1])
    
    def test_generate_basic_recommendations_execution_error(self):
        """Test basic recommendations generation for execution errors."""
        execution_result = ExecutionResult(
            success=False,
            execution_time=10.0,
            memory_usage=100,
            error_message="Division by zero"
        )
        
        recommendations = self.executor._generate_basic_recommendations(execution_result)
        
        self.assertIn("Address execution error: Division by zero", recommendations[0])
    
    def test_generate_error_specific_recommendations_script_import_error(self):
        """Test error-specific recommendations for ScriptImportError."""
        error = ScriptImportError("Import failed")
        recommendations = self.executor._generate_error_specific_recommendations(error, "ScriptImportError")
        
        self.assertIn("Check if the script has syntax errors", recommendations[0])
        self.assertIn("Verify the script has a 'main' function defined", recommendations[1])
    
    def test_generate_error_specific_recommendations_script_execution_error(self):
        """Test error-specific recommendations for ScriptExecutionError."""
        error = ScriptExecutionError("Execution failed")
        recommendations = self.executor._generate_error_specific_recommendations(error, "ScriptExecutionError")
        
        self.assertIn("Review script logic for runtime errors", recommendations[0])
        self.assertIn("Check input data format and availability", recommendations[1])
    
    def test_generate_error_specific_recommendations_configuration_error(self):
        """Test error-specific recommendations for ConfigurationError."""
        error = ConfigurationError("Config invalid")
        recommendations = self.executor._generate_error_specific_recommendations(error, "ConfigurationError")
        
        self.assertIn("Check data source configuration", recommendations[0])
        self.assertIn("Verify workspace directory permissions", recommendations[1])
    
    def test_generate_error_specific_recommendations_unknown_error(self):
        """Test error-specific recommendations for unknown error type."""
        error = ValueError("Unknown error")
        recommendations = self.executor._generate_error_specific_recommendations(error, "ValueError")
        
        self.assertIn("Review error details for ValueError", recommendations[0])


if __name__ == '__main__':
    unittest.main()
