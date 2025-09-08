"""
Unit tests for runtime testing CLI

Tests the Click-based CLI interface for runtime testing functionality.
"""

import sys
from pathlib import Path

)

import unittest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

from cursus.cli.runtime_testing_cli import runtime
from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult
)

class TestRuntimeTestingCLI(unittest.TestCase):
    """Test runtime testing CLI commands"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock test results
        self.mock_script_result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=1.5,
            has_main_function=True,
            error_message=None
        )
        
        self.mock_compatibility_result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            compatibility_issues=[],
            data_format_a="csv",
            data_format_b="csv"
        )
    
    def test_cli_help(self):
        """Test CLI help message"""
        result = self.runner.invoke(runtime, ['--help'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Pipeline Runtime Testing CLI", result.output)
        self.assertIn("test-script", result.output)
        self.assertIn("test-compatibility", result.output)
        self.assertIn("test-pipeline", result.output)
    
    def test_cli_version(self):
        """Test CLI version"""
        result = self.runner.invoke(runtime, ['--version'])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.1.0", result.output)
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_test_script_success(self, mock_tester_class):
        """Test successful script testing"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester.test_script.return_value = self.mock_script_result
        
        result = self.runner.invoke(runtime, [
            'test-script', 'test_script',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("test_script", result.output)
        self.assertIn("PASS", result.output)
        mock_tester.test_script.assert_called_once_with('test_script')
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_test_script_failure(self, mock_tester_class):
        """Test script testing failure"""
        failed_result = ScriptTestResult(
            script_name="test_script",
            success=False,
            execution_time=0.5,
            has_main_function=False,
            error_message="Script missing main() function"
        )
        
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester.test_script.return_value = failed_result
        
        result = self.runner.invoke(runtime, [
            'test-script', 'test_script',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL", result.output)
        self.assertIn("Script missing main() function", result.output)
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_test_compatibility_success(self, mock_tester_class):
        """Test successful data compatibility testing"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester._generate_sample_data.return_value = {"col1": [1, 2], "col2": ["a", "b"]}
        mock_tester.test_data_compatibility.return_value = self.mock_compatibility_result
        
        result = self.runner.invoke(runtime, [
            'test-compatibility', 'script_a', 'script_b',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("script_a -> script_b", result.output)
        self.assertIn("PASS", result.output)
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_test_compatibility_failure(self, mock_tester_class):
        """Test data compatibility testing failure"""
        incompatible_result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            compatibility_issues=["Column mismatch", "Type error"],
            data_format_a="csv",
            data_format_b="json"
        )
        
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester._generate_sample_data.return_value = {"col1": [1, 2]}
        mock_tester.test_data_compatibility.return_value = incompatible_result
        
        result = self.runner.invoke(runtime, [
            'test-compatibility', 'script_a', 'script_b',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("FAIL", result.output)
        self.assertIn("Column mismatch", result.output)
        self.assertIn("Type error", result.output)
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    @patch('builtins.open')
    @patch('pathlib.Path.exists', return_value=True)
    def test_test_pipeline_success(self, mock_exists, mock_open, mock_tester_class):
        """Test successful pipeline testing"""
        pipeline_config = {
            "steps": {
                "step1": {"script": "script1.py"},
                "step2": {"script": "script2.py"}
            }
        }
        
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(pipeline_config)
        
        pipeline_result = {
            "pipeline_success": True,
            "script_results": {
                "step1": self.mock_script_result,
                "step2": self.mock_script_result
            },
            "data_flow_results": {
                "step1->step2": self.mock_compatibility_result
            },
            "errors": []
        }
        
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester.test_pipeline_flow.return_value = pipeline_result
        
        result = self.runner.invoke(runtime, [
            'test-pipeline', 'pipeline.json',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 0)
        self.assertIn("pipeline.json", result.output)
        self.assertIn("PASS", result.output)
    
    @patch('pathlib.Path.exists', return_value=False)
    def test_test_pipeline_file_not_found(self, mock_exists):
        """Test pipeline testing with missing file"""
        result = self.runner.invoke(runtime, [
            'test-pipeline', 'nonexistent.json',
            '--workspace-dir', self.temp_dir
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("not found", result.output)
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_exception_handling(self, mock_tester_class):
        """Test CLI exception handling"""
        mock_tester_class.side_effect = Exception("Test exception")
        
        result = self.runner.invoke(runtime, [
            'test-script', 'test_script'
        ])
        
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Error", result.output)

class TestRuntimeTestingCLIIntegration(unittest.TestCase):
    """Integration tests for runtime testing CLI"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    @patch('cursus.cli.runtime_testing_cli.RuntimeTester')
    def test_full_workflow_simulation(self, mock_tester_class):
        """Test a complete workflow simulation"""
        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        
        # Mock script test results
        script_result_a = ScriptTestResult(
            script_name="script_a",
            success=True,
            execution_time=1.0,
            has_main_function=True
        )
        
        script_result_b = ScriptTestResult(
            script_name="script_b", 
            success=True,
            execution_time=1.5,
            has_main_function=True
        )
        
        # Mock compatibility result
        compatibility_result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            compatibility_issues=[],
            data_format_a="csv",
            data_format_b="csv"
        )
        
        # Set up mock returns
        mock_tester.test_script.side_effect = [script_result_a, script_result_b]
        mock_tester._generate_sample_data.return_value = {"col1": [1, 2]}
        mock_tester.test_data_compatibility.return_value = compatibility_result
        
        # Test individual scripts
        result1 = self.runner.invoke(runtime, [
            'test-script', 'script_a',
            '--workspace-dir', self.temp_dir
        ])
        self.assertEqual(result1.exit_code, 0)
        
        result2 = self.runner.invoke(runtime, [
            'test-script', 'script_b', 
            '--workspace-dir', self.temp_dir
        ])
        self.assertEqual(result2.exit_code, 0)
        
        # Test compatibility
        result3 = self.runner.invoke(runtime, [
            'test-compatibility', 'script_a', 'script_b',
            '--workspace-dir', self.temp_dir
        ])
        self.assertEqual(result3.exit_code, 0)

if __name__ == '__main__':
    unittest.main()
