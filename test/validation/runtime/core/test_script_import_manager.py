"""Unit tests for ScriptImportManager."""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import tempfile
import shutil
import os
import sys
import argparse
from pathlib import Path

from src.cursus.validation.runtime.core.script_import_manager import ScriptImportManager
from src.cursus.validation.runtime.utils.execution_context import ExecutionContext
from src.cursus.validation.runtime.utils.result_models import ExecutionResult
from src.cursus.validation.runtime.utils.error_handling import ScriptExecutionError, ScriptImportError


class TestScriptImportManager(unittest.TestCase):
    """Test cases for ScriptImportManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ScriptImportManager()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clean up any modules added to sys.modules during testing
        if "script_module" in sys.modules:
            del sys.modules["script_module"]
    
    def test_init(self):
        """Test ScriptImportManager initialization."""
        self.assertEqual(self.manager._imported_modules, {})
        self.assertEqual(self.manager._script_cache, {})
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_script_main_success(self, mock_module_from_spec, mock_spec_from_file):
        """Test successful script import."""
        # Setup mocks
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock()
        mock_main_func = Mock()
        mock_module.main = mock_main_func
        mock_module_from_spec.return_value = mock_module
        
        script_path = "test_script.py"
        
        # Test import
        result = self.manager.import_script_main(script_path)
        
        # Verify calls
        mock_spec_from_file.assert_called_once_with("script_module", script_path)
        mock_module_from_spec.assert_called_once_with(mock_spec)
        mock_loader.exec_module.assert_called_once_with(mock_module)
        
        # Verify result
        self.assertEqual(result, mock_main_func)
        self.assertIn(script_path, self.manager._script_cache)
        self.assertIn(script_path, self.manager._imported_modules)
    
    @patch('importlib.util.spec_from_file_location')
    def test_import_script_main_no_spec(self, mock_spec_from_file):
        """Test script import when spec cannot be created."""
        mock_spec_from_file.return_value = None
        
        with self.assertRaises(ScriptImportError) as context:
            self.manager.import_script_main("nonexistent_script.py")
        
        self.assertIn("Cannot load script from nonexistent_script.py", str(context.exception))
    
    @patch('importlib.util.spec_from_file_location')
    def test_import_script_main_no_loader(self, mock_spec_from_file):
        """Test script import when spec has no loader."""
        mock_spec = Mock()
        mock_spec.loader = None
        mock_spec_from_file.return_value = mock_spec
        
        with self.assertRaises(ScriptImportError) as context:
            self.manager.import_script_main("script_without_loader.py")
        
        self.assertIn("Cannot load script from script_without_loader.py", str(context.exception))
    
    @patch('importlib.util.spec_from_file_location')
    @patch('importlib.util.module_from_spec')
    def test_import_script_main_no_main_function(self, mock_module_from_spec, mock_spec_from_file):
        """Test script import when script has no main function."""
        # Setup mocks
        mock_spec = Mock()
        mock_loader = Mock()
        mock_spec.loader = mock_loader
        mock_spec_from_file.return_value = mock_spec
        
        mock_module = Mock(spec=[])  # Empty spec means no attributes
        # Explicitly remove main attribute if it exists
        if hasattr(mock_module, 'main'):
            delattr(mock_module, 'main')
        mock_module_from_spec.return_value = mock_module
        
        script_path = "script_without_main.py"
        
        with self.assertRaises(ScriptImportError) as context:
            self.manager.import_script_main(script_path)
        
        self.assertIn("does not have a 'main' function", str(context.exception))
    
    @patch('importlib.util.spec_from_file_location')
    def test_import_script_main_import_error(self, mock_spec_from_file):
        """Test script import with import error."""
        mock_spec_from_file.side_effect = ImportError("Module not found")
        
        with self.assertRaises(ScriptImportError) as context:
            self.manager.import_script_main("failing_script.py")
        
        self.assertIn("Failed to import script failing_script.py", str(context.exception))
    
    def test_import_script_main_caching(self):
        """Test that imported scripts are cached."""
        script_path = "cached_script.py"
        mock_main_func = Mock()
        
        # Add to cache
        self.manager._script_cache[script_path] = mock_main_func
        
        result = self.manager.import_script_main(script_path)
        
        self.assertEqual(result, mock_main_func)
    
    @patch('time.time')
    @patch('src.cursus.validation.runtime.core.script_import_manager.ScriptImportManager._get_memory_usage')
    def test_execute_script_main_success(self, mock_memory, mock_time):
        """Test successful script execution."""
        # Setup mocks
        mock_time.side_effect = [1000.0, 1001.5]  # start_time, end_time
        mock_memory.side_effect = [100, 150]  # start_memory, end_memory
        
        mock_main_func = Mock(return_value={"status": "success"})
        
        context = ExecutionContext(
            input_paths={"input": "/path/to/input"},
            output_paths={"output": "/path/to/output"},
            environ_vars={"VAR": "value"},
            job_args=argparse.Namespace(verbose=True)
        )
        
        result = self.manager.execute_script_main(mock_main_func, context)
        
        # Verify function was called with correct parameters
        mock_main_func.assert_called_once_with(
            input_paths={"input": "/path/to/input"},
            output_paths={"output": "/path/to/output"},
            environ_vars={"VAR": "value"},
            job_args=argparse.Namespace(verbose=True)
        )
        
        # Verify result
        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.execution_time, 1.5)
        self.assertEqual(result.memory_usage, 50)
        self.assertEqual(result.result_data, {"status": "success"})
        self.assertIsNone(result.error_message)
    
    def test_execute_script_main_none_function(self):
        """Test execution with None function."""
        context = ExecutionContext(
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args=argparse.Namespace()
        )
        
        with self.assertRaises(ScriptExecutionError) as context_manager:
            self.manager.execute_script_main(None, context)
        
        self.assertIn("Main function cannot be None", str(context_manager.exception))
    
    def test_execute_script_main_none_context(self):
        """Test execution with None context."""
        mock_main_func = Mock()
        
        with self.assertRaises(ScriptExecutionError) as context:
            self.manager.execute_script_main(mock_main_func, None)
        
        self.assertIn("Execution context cannot be None", str(context.exception))
    
    @patch('time.time')
    @patch('src.cursus.validation.runtime.core.script_import_manager.ScriptImportManager._get_memory_usage')
    def test_execute_script_main_script_execution_error(self, mock_memory, mock_time):
        """Test execution with ScriptExecutionError."""
        # Setup mocks
        mock_time.side_effect = [1000.0, 1001.0]
        mock_memory.side_effect = [100, 100]
        
        mock_main_func = Mock(side_effect=ScriptExecutionError("Script failed"))
        
        context = ExecutionContext(
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args=argparse.Namespace()
        )
        
        with self.assertRaises(ScriptExecutionError) as context_manager:
            self.manager.execute_script_main(mock_main_func, context)
        
        self.assertIn("Script failed", str(context_manager.exception))
    
    @patch('time.time')
    @patch('src.cursus.validation.runtime.core.script_import_manager.ScriptImportManager._get_memory_usage')
    def test_execute_script_main_generic_error(self, mock_memory, mock_time):
        """Test execution with generic error that gets converted to ScriptExecutionError."""
        # Setup mocks
        mock_time.side_effect = [1000.0, 1001.0]
        mock_memory.side_effect = [100, 100]
        
        mock_main_func = Mock(side_effect=ValueError("Generic error"))
        
        context = ExecutionContext(
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args=argparse.Namespace()
        )
        
        with self.assertRaises(ScriptExecutionError) as context_manager:
            self.manager.execute_script_main(mock_main_func, context)
        
        self.assertIn("Script execution failed: Generic error", str(context_manager.exception))
        # Verify original exception is preserved
        self.assertIsInstance(context_manager.exception.__cause__, ValueError)
    
    @patch('psutil.Process')
    @patch('os.getpid')
    def test_get_memory_usage_with_psutil(self, mock_getpid, mock_process_class):
        """Test memory usage calculation with psutil available."""
        mock_getpid.return_value = 1234
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        result = self.manager._get_memory_usage()
        
        mock_getpid.assert_called_once()
        mock_process_class.assert_called_once_with(1234)
        self.assertEqual(result, 100)  # 100 MB
    
    @patch('src.cursus.validation.runtime.core.script_import_manager.psutil', None)
    def test_get_memory_usage_without_psutil(self):
        """Test memory usage calculation when psutil is not available."""
        result = self.manager._get_memory_usage()
        self.assertEqual(result, 0)
    
    @patch('psutil.Process')
    @patch('os.getpid')
    def test_get_memory_usage_with_exception(self, mock_getpid, mock_process_class):
        """Test memory usage calculation when psutil raises exception."""
        mock_getpid.return_value = 1234
        mock_process_class.side_effect = Exception("Process error")
        
        result = self.manager._get_memory_usage()
        self.assertEqual(result, 0)
    
    @patch('time.time')
    @patch('src.cursus.validation.runtime.core.script_import_manager.ScriptImportManager._get_memory_usage')
    def test_execute_script_main_memory_calculation_edge_case(self, mock_memory, mock_time):
        """Test memory calculation when end memory is less than start memory."""
        # Setup mocks
        mock_time.side_effect = [1000.0, 1001.0]
        mock_memory.side_effect = [150, 100]  # end_memory < start_memory
        
        mock_main_func = Mock(return_value={"status": "success"})
        
        context = ExecutionContext(
            input_paths={},
            output_paths={},
            environ_vars={},
            job_args=argparse.Namespace()
        )
        
        result = self.manager.execute_script_main(mock_main_func, context)
        
        # Memory usage should be 0 when end < start (max(end - start, 0))
        self.assertEqual(result.memory_usage, 0)


if __name__ == '__main__':
    unittest.main()
