"""
Unit tests for the alignment CLI module.

This module tests all functionality of the alignment command-line interface,
including argument parsing, command execution, output formatting, and JSON serialization.
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import unittest
from unittest.mock import patch, MagicMock, call, mock_open
from io import StringIO
import json
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner

from src.cursus.cli.alignment_cli import alignment
from src.cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester


class TestAlignmentCLI(unittest.TestCase):
    """Test the alignment CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        
        # Set up stdout mocking for tests that need it
        self.stdout_patcher = patch('sys.stdout', new_callable=StringIO)
        self.mock_stdout = self.stdout_patcher.start()
        
    def tearDown(self):
        """Clean up after tests."""
        # Clean up stdout patcher
        self.stdout_patcher.stop()
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_validate_single_script_success(self):
        """Test validating a single script successfully."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            # Test the CLI command using CliRunner
            result = self.runner.invoke(alignment, ['validate', 'payload'])
            
            self.assertEqual(result.exit_code, 0)
            mock_tester.validate_specific_script.assert_called_once_with('payload')
            self.assertIn('✅ payload: PASSING', result.output)
    
    def test_validate_single_script_failure(self):
        """Test validating a single script with failures."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'FAILING',
            'level1': {'passed': False, 'issues': [{'severity': 'ERROR', 'message': 'Test error'}]},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, ['validate', 'payload'])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('❌ payload: FAILING', result.output)
    
    def test_validate_single_script_json_output(self):
        """Test validating a single script with JSON output."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        output_file = os.path.join(self.temp_dir, 'payload_alignment_report.json')
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, [
                'validate', 'payload', 
                '--format', 'json',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify JSON file was created
            self.assertTrue(os.path.exists(output_file))
            
            # Verify JSON content
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data['script_name'], 'payload')
            self.assertEqual(saved_data['overall_status'], 'PASSING')
    
    def test_validate_all_scripts_success(self):
        """Test validating all scripts successfully."""
        # Mock the script discovery and validation results
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock validate_specific_script to return success for each script
            mock_tester.validate_specific_script.side_effect = [
                {'script_name': 'payload', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                {'script_name': 'package', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                {'script_name': 'dummy_training', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}}
            ]
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects with proper stem attributes
                mock_files = []
                for name in ['payload', 'package', 'dummy_training']:
                    mock_file = MagicMock()
                    mock_file.name = f'{name}.py'
                    mock_file.stem = name
                    mock_files.append(mock_file)
                
                mock_glob.return_value = mock_files
                
                result = self.runner.invoke(alignment, ['validate-all'])
                
                self.assertEqual(result.exit_code, 0)
                self.assertIn('🎉 All 3 scripts passed alignment validation!', result.output)
    
    def test_validate_all_scripts_with_failures(self):
        """Test validating all scripts with some failures."""
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock mixed results - one passing, one failing
            mock_tester.validate_specific_script.side_effect = [
                {'script_name': 'payload', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                {'script_name': 'package', 'overall_status': 'FAILING', 'level1': {'passed': False, 'issues': [{'severity': 'ERROR', 'message': 'Test error'}]}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}}
            ]
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects with proper stem attributes
                mock_files = []
                for name in ['payload', 'package']:
                    mock_file = MagicMock()
                    mock_file.name = f'{name}.py'
                    mock_file.stem = name
                    mock_files.append(mock_file)
                
                mock_glob.return_value = mock_files
                
                result = self.runner.invoke(alignment, ['validate-all'])
                
                self.assertEqual(result.exit_code, 1)
                self.assertIn('❌ Failed: 1 (50.0%)', result.output)
    
    def test_validate_all_scripts_json_output_with_summary(self):
        """Test validating all scripts with JSON output and summary file."""
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock validate_specific_script to return success
            mock_tester.validate_specific_script.return_value = {
                'script_name': 'payload', 
                'overall_status': 'PASSING',
                'level1': {'passed': True, 'issues': []},
                'level2': {'passed': True, 'issues': []},
                'level3': {'passed': True, 'issues': []},
                'level4': {'passed': True, 'issues': []}
            }
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects with proper stem attributes
                mock_file = MagicMock()
                mock_file.name = 'payload.py'
                mock_file.stem = 'payload'
                mock_glob.return_value = [mock_file]
                
                result = self.runner.invoke(alignment, [
                    'validate-all', 
                    '--format', 'json',
                    '--output-dir', self.temp_dir
                ])
                
                self.assertEqual(result.exit_code, 0)
                
                # Verify summary file was created
                summary_file = os.path.join(self.temp_dir, 'validation_summary.json')
                self.assertTrue(os.path.exists(summary_file))
                
                # Verify summary content
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                self.assertEqual(summary_data['total_scripts'], 1)
                self.assertEqual(summary_data['passed_scripts'], 1)
                self.assertEqual(summary_data['failed_scripts'], 0)
    
    def test_json_serialization_with_complex_objects(self):
        """Test that JSON serialization works with complex Python objects."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level4': {
                'passed': True,
                'issues': [],
                'config_analysis': {
                    'fields': {
                        'test_field': {
                            'type': str,  # This is a type object
                            'required': True
                        }
                    },
                    'default_values': {
                        'computed_property': property(lambda self: "test"),  # This is a property
                        'model_fields': {
                            'field1': "annotation=str required=True"
                        }
                    }
                }
            }
        }
        
        output_file = os.path.join(self.temp_dir, 'payload_alignment_report.json')
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            # This should not raise a JSON serialization error
            result = self.runner.invoke(alignment, [
                'validate', 'payload',
                '--format', 'json',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            # Verify complex objects were converted to strings
            self.assertIn("str", json.dumps(saved_data))
            self.assertIn("property", json.dumps(saved_data))
    
    def test_verbose_output(self):
        """Test verbose output mode."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': [
                {'severity': 'WARNING', 'message': 'Test warning', 'category': 'test'}
            ]}
        }
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, ['validate', 'payload', '--verbose'])
            
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Test warning', result.output)
    
    def test_invalid_script_name(self):
        """Test validation with invalid script name."""
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.side_effect = Exception("Script not found")
            
            result = self.runner.invoke(alignment, ['validate', 'nonexistent_script'])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('❌ Error validating nonexistent_script', result.output)
    
    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'new_dir')
        
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, [
                'validate', 'payload',
                '--format', 'json',
                '--output-dir', non_existent_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify directory was created
            self.assertTrue(os.path.exists(non_existent_dir))
            
            # Verify file was created in the new directory
            output_file = os.path.join(non_existent_dir, 'payload_alignment_report.json')
            self.assertTrue(os.path.exists(output_file))
    
    def test_argument_parsing_errors(self):
        """Test various argument parsing error cases."""
        # Test missing script name for validate command
        result = self.runner.invoke(alignment, ['validate'])
        self.assertEqual(result.exit_code, 2)  # click error
        
        # Test invalid format
        result = self.runner.invoke(alignment, ['validate', 'payload', '--format', 'invalid'])
        self.assertEqual(result.exit_code, 2)  # click error
    
    def test_help_command(self):
        """Test help command output."""
        result = self.runner.invoke(alignment, ['--help'])
        
        # Click exits with code 0 for help
        self.assertEqual(result.exit_code, 0)
        
        self.assertIn('Unified Alignment Tester for Cursus Scripts', result.output)
        self.assertIn('validate', result.output)
        self.assertIn('validate-all', result.output)
    
    def test_error_handling_in_comprehensive_validation(self):
        """Test error handling during comprehensive validation."""
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock an exception during validation
            mock_tester.validate_specific_script.side_effect = Exception("Validation error")
            
            # Mock the scripts directory to contain test scripts
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = [
                    MagicMock(name='payload.py', stem='payload')
                ]
                
                result = self.runner.invoke(alignment, ['validate-all'])
                
                self.assertEqual(result.exit_code, 1)
                self.assertIn('❌ Failed to validate payload', result.output)
    
    def test_json_serialization_edge_cases(self):
        """Test JSON serialization with various edge cases."""
        mock_result = {
            'script_name': 'test',
            'overall_status': 'PASSING',
            'complex_data': {
                'none_value': None,
                'empty_list': [],
                'empty_dict': {},
                'nested_complex': {
                    'type_obj': type,
                    'property_obj': property(lambda x: x),
                    'function_obj': lambda x: x,
                    'class_obj': str
                }
            }
        }
        
        output_file = os.path.join(self.temp_dir, 'test_alignment_report.json')
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, [
                'validate', 'test',
                '--format', 'json',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            # Verify the data structure is preserved
            self.assertEqual(saved_data['script_name'], 'test')
            self.assertIsNone(saved_data['complex_data']['none_value'])
            self.assertEqual(saved_data['complex_data']['empty_list'], [])
            self.assertEqual(saved_data['complex_data']['empty_dict'], {})


class TestAlignmentCLIIntegration(unittest.TestCase):
    """Integration tests for the alignment CLI with real components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.stdout_patcher = patch('sys.stdout', new_callable=StringIO)
        self.mock_stdout = self.stdout_patcher.start()
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        self.stdout_patcher.stop()
        
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cli_with_real_unified_tester(self):
        """Test CLI integration with real UnifiedAlignmentTester (mocked dependencies)."""
        # Mock the individual level testers to avoid file system dependencies
        with patch('src.cursus.validation.alignment.script_contract_alignment.ScriptContractAlignmentTester'), \
             patch('src.cursus.validation.alignment.contract_spec_alignment.ContractSpecificationAlignmentTester'), \
             patch('src.cursus.validation.alignment.spec_dependency_alignment.SpecificationDependencyAlignmentTester'), \
             patch('src.cursus.validation.alignment.builder_config_alignment.BuilderConfigurationAlignmentTester'):
            
            # Create a real UnifiedAlignmentTester but with mocked dependencies
            tester = UnifiedAlignmentTester()
            
            # Mock the validate_specific_script method
            with patch.object(tester, 'validate_specific_script') as mock_validate:
                mock_validate.return_value = {
                    'script_name': 'payload',
                    'overall_status': 'PASSING',
                    'level1': {'passed': True, 'issues': []},
                    'level2': {'passed': True, 'issues': []},
                    'level3': {'passed': True, 'issues': []},
                    'level4': {'passed': True, 'issues': []}
                }
                
                with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester', return_value=tester):
                    runner = CliRunner()
                    result = runner.invoke(alignment, ['validate', 'payload'])
                    
                    self.assertEqual(result.exit_code, 0)
                    mock_validate.assert_called_once_with('payload')
    
    def test_cli_json_output_file_structure(self):
        """Test that CLI creates proper JSON file structure."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []},
            'metadata': {
                'validation_timestamp': '2025-08-11T22:58:04.535968',
                'validator_version': '1.0.0'
            }
        }
        
        with patch('src.cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            runner = CliRunner()
            result = runner.invoke(alignment, [
                'validate', 'payload',
                '--format', 'json',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify file structure
            output_file = os.path.join(self.temp_dir, 'payload_alignment_report.json')
            self.assertTrue(os.path.exists(output_file))
            
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Verify required fields are present
            self.assertIn('script_name', data)
            self.assertIn('overall_status', data)
            self.assertIn('level1', data)
            self.assertIn('level2', data)
            self.assertIn('level3', data)
            self.assertIn('level4', data)
            self.assertIn('metadata', data)


if __name__ == '__main__':
    unittest.main()
