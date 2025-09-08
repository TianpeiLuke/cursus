"""
Unit tests for the alignment CLI module.

This module tests all functionality of the alignment command-line interface,
including argument parsing, command execution, output formatting, and JSON serialization.
"""

import sys
import os

import unittest
from unittest.mock import patch, MagicMock, call, mock_open
from io import StringIO
import json
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner

from cursus.cli.alignment_cli import alignment
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_scorer import AlignmentScorer

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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            result = self.runner.invoke(alignment, ['validate', 'payload', '--verbose'])
            
            self.assertEqual(result.exit_code, 0)
            self.assertIn('Test warning', result.output)
    
    def test_invalid_script_name(self):
        """Test validation with invalid script name."""
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock an exception during validation
            mock_tester.validate_specific_script.side_effect = Exception("Validation error")
            
            # Mock the scripts directory to contain test scripts
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                mock_file = MagicMock()
                mock_file.name = 'payload.py'
                mock_file.stem = 'payload'
                mock_glob.return_value = [mock_file]
                
                result = self.runner.invoke(alignment, ['validate-all'])
                
                # The CLI exits with 0 when using continue-on-error by default
                # and shows error summary in the final output
                self.assertEqual(result.exit_code, 0)
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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

class TestAlignmentCLIVisualization(unittest.TestCase):
    """Test the alignment CLI visualization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_visualize_single_script_success(self):
        """Test visualizing a single script successfully."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.return_value = Path(self.temp_dir) / "payload_alignment_chart.png"
            mock_scorer.generate_scoring_report.return_value = {"score": 100.0}
            
            result = self.runner.invoke(alignment, [
                'visualize', 'payload',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            mock_tester.validate_specific_script.assert_called_once_with('payload')
            mock_scorer.generate_chart.assert_called_once()
            self.assertIn('✅ Visualization generation complete for payload!', result.output)
    
    def test_visualize_single_script_failure(self):
        """Test visualizing a single script with validation failure."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'FAILING',
            'level1': {'passed': False, 'issues': [{'severity': 'ERROR', 'message': 'Test error'}]},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 45.0
            mock_scorer.get_quality_rating.return_value = "Needs Work"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.return_value = Path(self.temp_dir) / "payload_alignment_chart.png"
            mock_scorer.generate_scoring_report.return_value = {"score": 45.0}
            
            result = self.runner.invoke(alignment, [
                'visualize', 'payload',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 0)  # Visualization still succeeds
            self.assertIn('✅ Visualization generation complete for payload!', result.output)
    
    def test_visualize_all_scripts_success(self):
        """Test visualizing all scripts successfully."""
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock validation results for multiple scripts
            mock_tester.validate_specific_script.side_effect = [
                {'script_name': 'payload', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                {'script_name': 'package', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}}
            ]
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.return_value = Path(self.temp_dir) / "test_chart.png"
            mock_scorer.generate_scoring_report.return_value = {"score": 100.0}
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects
                mock_files = []
                for name in ['payload', 'package']:
                    mock_file = MagicMock()
                    mock_file.name = f'{name}.py'
                    mock_file.stem = name
                    mock_files.append(mock_file)
                
                mock_glob.return_value = mock_files
                
                result = self.runner.invoke(alignment, [
                    'visualize-all',
                    '--output-dir', self.temp_dir
                ])
                
                self.assertEqual(result.exit_code, 0)
                self.assertIn('🎉 All 2 visualizations generated successfully!', result.output)
                # Verify scorer was called for each script
                self.assertEqual(mock_scorer.generate_chart.call_count, 2)
    
    def test_visualize_all_scripts_with_errors(self):
        """Test visualizing all scripts with some errors."""
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # Mock validation results - first script (package) succeeds, second script (payload) fails
            mock_tester.validate_specific_script.side_effect = [
                {'script_name': 'package', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                Exception("Validation error for payload")
            ]
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.return_value = Path(self.temp_dir) / "test_chart.png"
            mock_scorer.generate_scoring_report.return_value = {"score": 100.0}
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects - note: scripts are processed in sorted order
                mock_files = []
                for name in ['package', 'payload']:  # Sorted order: package, payload
                    mock_file = MagicMock()
                    mock_file.name = f'{name}.py'
                    mock_file.stem = name
                    mock_files.append(mock_file)
                
                mock_glob.return_value = mock_files
                
                result = self.runner.invoke(alignment, [
                    'visualize-all',
                    '--output-dir', self.temp_dir,
                    '--continue-on-error'
                ])
                
                self.assertEqual(result.exit_code, 1)  # Should exit with error code due to failures
                self.assertIn('❌ Failed to generate visualization for payload', result.output)
    
    def test_visualize_chart_generation_failure(self):
        """Test handling of chart generation failures."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.side_effect = Exception("Chart generation failed")
            
            result = self.runner.invoke(alignment, [
                'visualize', 'payload',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('❌ Error generating visualization for payload', result.output)
    
    def test_show_scoring_flag_integration(self):
        """Test the --show-scoring flag integration."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 95.5
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {
                'level1_script_contract': 100.0,
                'level2_contract_spec': 100.0,
                'level3_spec_dependencies': 100.0,
                'level4_builder_config': 82.0
            }
            
            result = self.runner.invoke(alignment, [
                'validate', 'payload',
                '--show-scoring'
            ])
            
            self.assertEqual(result.exit_code, 0)
            mock_scorer.calculate_overall_score.assert_called_once()
            mock_scorer.get_quality_rating.assert_called_once()
            self.assertIn('95.5/100', result.output)
            self.assertIn('Excellent', result.output)

    def test_visualize_invalid_script(self):
        """Test visualization with invalid script name."""
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.side_effect = Exception("Script not found")
            
            result = self.runner.invoke(alignment, [
                'visualize', 'nonexistent_script',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 1)
            self.assertIn('❌ Error generating visualization for nonexistent_script', result.output)
    
    def test_visualize_output_directory_creation(self):
        """Test that visualization creates output directory if it doesn't exist."""
        non_existent_dir = os.path.join(self.temp_dir, 'new_viz_dir')
        
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            mock_scorer.generate_chart.return_value = Path(non_existent_dir) / "payload_alignment_chart.png"
            mock_scorer.generate_scoring_report.return_value = {"score": 100.0}
            
            result = self.runner.invoke(alignment, [
                'visualize', 'payload',
                '--output-dir', non_existent_dir
            ])
            
            self.assertEqual(result.exit_code, 0)
            
            # Verify directory was created
            self.assertTrue(os.path.exists(non_existent_dir))
    
    def test_visualize_all_no_scripts_found(self):
        """Test visualize-all when no scripts are found."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.glob', return_value=[]):
            
            result = self.runner.invoke(alignment, [
                'visualize-all',
                '--output-dir', self.temp_dir
            ])
            
            self.assertEqual(result.exit_code, 1)
            # The actual error message may vary, so check for the general error pattern
            self.assertIn('❌ Fatal error during visualization generation:', result.output)
    
    def test_visualize_all_continue_on_error_disabled(self):
        """Test visualize-all without continue-on-error flag."""
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            
            # First script succeeds, second fails
            mock_tester.validate_specific_script.side_effect = [
                {'script_name': 'payload', 'overall_status': 'PASSING', 'level1': {'passed': True, 'issues': []}, 'level2': {'passed': True, 'issues': []}, 'level3': {'passed': True, 'issues': []}, 'level4': {'passed': True, 'issues': []}},
                Exception("Validation error for package")
            ]
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.generate_visualization.return_value = True
            
            # Mock the Path.exists() and Path.glob() methods
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.glob') as mock_glob:
                
                # Create mock file objects
                mock_files = []
                for name in ['payload', 'package']:
                    mock_file = MagicMock()
                    mock_file.name = f'{name}.py'
                    mock_file.stem = name
                    mock_files.append(mock_file)
                
                mock_glob.return_value = mock_files
                
                result = self.runner.invoke(alignment, [
                    'visualize-all',
                    '--output-dir', self.temp_dir
                ])
                
                self.assertEqual(result.exit_code, 1)  # Should exit on first error
                self.assertIn('❌ Failed to generate visualization for package', result.output)

class TestAlignmentCLIScoring(unittest.TestCase):
    """Test the alignment CLI scoring functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
    def test_scoring_calculation_integration(self):
        """Test that scoring calculation works correctly with CLI."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'PASSING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': True, 'issues': []},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': True, 'issues': []}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 100.0
            mock_scorer.get_quality_rating.return_value = "Excellent"
            mock_scorer.get_level_scores.return_value = {}
            
            result = self.runner.invoke(alignment, [
                'validate', 'payload',
                '--show-scoring'
            ])
            
            self.assertEqual(result.exit_code, 0)
            # Verify scorer methods were called
            mock_scorer.calculate_overall_score.assert_called_once()
            mock_scorer.get_quality_rating.assert_called_once()
            self.assertIn('100.0/100', result.output)
            self.assertIn('Excellent', result.output)
    
    def test_scoring_with_partial_failures(self):
        """Test scoring calculation with partial failures."""
        mock_result = {
            'script_name': 'payload',
            'overall_status': 'FAILING',
            'level1': {'passed': True, 'issues': []},
            'level2': {'passed': False, 'issues': [{'severity': 'ERROR', 'message': 'Level 2 error'}]},
            'level3': {'passed': True, 'issues': []},
            'level4': {'passed': False, 'issues': [{'severity': 'WARNING', 'message': 'Level 4 warning'}]}
        }
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class, \
             patch('cursus.cli.alignment_cli.AlignmentScorer') as mock_scorer_class:
            
            mock_tester = MagicMock()
            mock_tester_class.return_value = mock_tester
            mock_tester.validate_specific_script.return_value = mock_result
            
            mock_scorer = MagicMock()
            mock_scorer_class.return_value = mock_scorer
            mock_scorer.calculate_overall_score.return_value = 42.5
            mock_scorer.get_quality_rating.return_value = "Needs Work"
            mock_scorer.get_level_scores.return_value = {}
            
            result = self.runner.invoke(alignment, [
                'validate', 'payload',
                '--show-scoring'
            ])
            
            self.assertEqual(result.exit_code, 1)  # Script failed validation
            self.assertIn('42.5/100', result.output)
            self.assertIn('Needs Work', result.output)

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
        with patch('cursus.validation.alignment.script_contract_alignment.ScriptContractAlignmentTester'), \
             patch('cursus.validation.alignment.contract_spec_alignment.ContractSpecificationAlignmentTester'), \
             patch('cursus.validation.alignment.spec_dependency_alignment.SpecificationDependencyAlignmentTester'), \
             patch('cursus.validation.alignment.builder_config_alignment.BuilderConfigurationAlignmentTester'):
            
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
                
                with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester', return_value=tester):
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
        
        with patch('cursus.cli.alignment_cli.UnifiedAlignmentTester') as mock_tester_class:
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
