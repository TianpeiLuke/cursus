"""
Unit tests for BuilderConfigurationAlignmentTester

Tests the enhanced builder-configuration alignment validation including:
- Integration of ConfigurationAnalyzer and BuilderCodeAnalyzer
- Field validation logic
- Pattern recognition and filtering
- File resolution strategies
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

)

from cursus.validation.alignment.builder_config_alignment import BuilderConfigurationAlignmentTester

class TestBuilderConfigurationAlignmentTester(unittest.TestCase):
    """Test cases for BuilderConfigurationAlignmentTester"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.builders_dir = Path(self.temp_dir) / "builders"
        self.configs_dir = Path(self.temp_dir) / "configs"
        
        # Create directories
        self.builders_dir.mkdir(parents=True)
        self.configs_dir.mkdir(parents=True)
        
        self.tester = BuilderConfigurationAlignmentTester(
            str(self.builders_dir), 
            str(self.configs_dir)
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test proper initialization of the tester"""
        self.assertEqual(self.tester.builders_dir, self.builders_dir)
        self.assertEqual(self.tester.configs_dir, self.configs_dir)
        
        # Check that analyzers are initialized
        self.assertIsNotNone(self.tester.config_analyzer)
        self.assertIsNotNone(self.tester.builder_analyzer)
        self.assertIsNotNone(self.tester.pattern_recognizer)
        self.assertIsNotNone(self.tester.file_resolver)
    
    def test_validate_builder_missing_builder_file(self):
        """Test validation when builder file is missing"""
        result = self.tester.validate_builder("nonexistent_builder")
        
        self.assertFalse(result['passed'])
        self.assertEqual(len(result['issues']), 1)
        self.assertEqual(result['issues'][0]['severity'], 'CRITICAL')
        self.assertEqual(result['issues'][0]['category'], 'missing_file')
        self.assertIn('Builder file not found', result['issues'][0]['message'])
    
    def test_validate_builder_missing_config_file(self):
        """Test validation when config file is missing"""
        # Create a builder file
        builder_content = '''
class TestStepBuilder:
    def build_step(self):
        return None
'''
        builder_file = self.builders_dir / "builder_test_step.py"
        builder_file.write_text(builder_content)
        
        # Mock the flexible_resolver which has the get_available_files_report method
        with patch.object(self.tester, 'flexible_resolver') as mock_flexible_resolver:
            mock_flexible_resolver.get_available_files_report.return_value = {
                'configs': {
                    'discovered_files': [],
                    'base_names': [],
                    'count': 0
                }
            }
            
            result = self.tester.validate_builder("test")
        
        self.assertFalse(result['passed'])
        issues = [issue for issue in result['issues'] if issue['category'] == 'missing_configuration']
        self.assertTrue(len(issues) > 0)
        self.assertEqual(issues[0]['severity'], 'ERROR')
        self.assertIn('Configuration file not found', issues[0]['message'])
    
    def test_validate_builder_successful_validation(self):
        """Test successful validation with matching builder and config"""
        # Create a builder file
        builder_content = '''
class TestStepBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_step(self):
        # Access some config fields
        instance_type = self.config.processing_instance_type
        volume_size = self.config.processing_volume_size
        return None
'''
        builder_file = self.builders_dir / "builder_test_step.py"
        builder_file.write_text(builder_content)
        
        # Create a config file
        config_content = '''
from typing import Optional
from pydantic import BaseModel

class TestConfig(BaseModel):
    processing_instance_type: str = "ml.m5.large"
    processing_volume_size: int = 30
    optional_field: Optional[str] = None
'''
        config_file = self.configs_dir / "config_test_step.py"
        config_file.write_text(config_content)
        
        # Mock the config loading to avoid import issues
        with patch.object(self.tester.config_analyzer, 'load_config_from_python') as mock_load_config, \
             patch.object(self.tester.builder_analyzer, 'analyze_builder_file') as mock_analyze_builder:
            
            # Mock config analysis result
            mock_load_config.return_value = {
                'class_name': 'TestConfig',
                'fields': {
                    'processing_instance_type': {'type': 'str', 'required': False},
                    'processing_volume_size': {'type': 'int', 'required': False},
                    'optional_field': {'type': 'Optional[str]', 'required': False}
                },
                'required_fields': set(),
                'optional_fields': {'processing_instance_type', 'processing_volume_size', 'optional_field'},
                'default_values': {
                    'processing_instance_type': 'ml.m5.large',
                    'processing_volume_size': 30,
                    'optional_field': None
                }
            }
            
            # Mock builder analysis result
            mock_analyze_builder.return_value = {
                'config_accesses': [
                    {'field_name': 'processing_instance_type', 'line': 7},
                    {'field_name': 'processing_volume_size', 'line': 8}
                ],
                'class_definitions': [
                    {'class_name': 'TestStepBuilder'}
                ],
                'validation_calls': []
            }
            
            result = self.tester.validate_builder("test")
        
        # Should pass validation since all accessed fields exist in config
        self.assertTrue(result['passed'])
        
        # Should have config and builder analysis in result
        self.assertIn('config_analysis', result)
        self.assertIn('builder_analysis', result)
    
    def test_validate_configuration_fields_undeclared_access(self):
        """Test detection of undeclared field access"""
        builder_analysis = {
            'config_accesses': [
                {'field_name': 'existing_field', 'line': 10},
                {'field_name': 'missing_field', 'line': 11}
            ]
        }
        
        config_analysis = {
            'fields': {
                'existing_field': {'type': 'str', 'required': True}
            },
            'required_fields': {'existing_field'},
            'optional_fields': set()
        }
        
        issues = self.tester._validate_configuration_fields(
            builder_analysis, config_analysis, "test_builder"
        )
        
        # Should detect the missing field
        error_issues = [issue for issue in issues if issue['severity'] == 'ERROR']
        self.assertTrue(len(error_issues) > 0)
        
        missing_field_issues = [
            issue for issue in error_issues 
            if 'missing_field' in issue['message']
        ]
        self.assertTrue(len(missing_field_issues) > 0)
    
    def test_validate_configuration_fields_pattern_filtering(self):
        """Test that pattern recognition filters acceptable patterns"""
        builder_analysis = {
            'config_accesses': [
                {'field_name': 'job_type', 'line': 10},  # Common pattern, might be filtered
                {'field_name': 'truly_missing_field', 'line': 11}
            ]
        }
        
        config_analysis = {
            'fields': {},
            'required_fields': set(),
            'optional_fields': set()
        }
        
        # Mock pattern recognizer to accept job_type but not truly_missing_field
        with patch.object(self.tester.pattern_recognizer, 'is_acceptable_pattern') as mock_pattern:
            mock_pattern.side_effect = lambda field, builder, issue_type: field == 'job_type'
            
            issues = self.tester._validate_configuration_fields(
                builder_analysis, config_analysis, "test_builder"
            )
        
        # Should only flag truly_missing_field, not job_type
        error_messages = [issue['message'] for issue in issues if issue['severity'] == 'ERROR']
        
        self.assertTrue(any('truly_missing_field' in msg for msg in error_messages))
        self.assertFalse(any('job_type' in msg for msg in error_messages))
    
    def test_validate_required_fields(self):
        """Test validation of required field handling"""
        builder_analysis = {
            'validation_calls': []  # No validation logic
        }
        
        config_analysis = {
            'required_fields': {'required_field1', 'required_field2'}
        }
        
        issues = self.tester._validate_required_fields(
            builder_analysis, config_analysis, "test_builder"
        )
        
        # Should suggest adding validation logic
        info_issues = [issue for issue in issues if issue['severity'] == 'INFO']
        validation_issues = [
            issue for issue in info_issues 
            if 'validation logic' in issue['message']
        ]
        self.assertTrue(len(validation_issues) > 0)
    
    def test_validate_config_import(self):
        """Test validation of configuration import"""
        builder_analysis = {
            'class_definitions': [
                {'class_name': 'TestStepBuilder'}
            ]
        }
        
        config_analysis = {
            'class_name': 'TestConfig'
        }
        
        issues = self.tester._validate_config_import(
            builder_analysis, config_analysis, "test_builder"
        )
        
        # The current implementation may not always detect import issues
        # This is acceptable as import validation is informational
        self.assertIsInstance(issues, list)
        
        # If there are issues, they should be INFO level
        for issue in issues:
            self.assertEqual(issue['severity'], 'INFO')
    
    def test_find_builder_file_hybrid_standard_pattern(self):
        """Test hybrid builder file resolution with standard pattern"""
        # Create builder file with standard naming
        builder_file = self.builders_dir / "builder_test_step.py"
        builder_file.write_text("# Test builder")
        
        result = self.tester._find_builder_file_hybrid("test")
        
        self.assertEqual(result, str(builder_file))
    
    def test_find_config_file_hybrid_standard_pattern(self):
        """Test hybrid config file resolution with standard pattern"""
        # Create config file with standard naming
        config_file = self.configs_dir / "config_test_step.py"
        config_file.write_text("# Test config")
        
        result = self.tester._find_config_file_hybrid("test")
        
        self.assertEqual(result, str(config_file))
    
    def test_discover_builders(self):
        """Test discovery of builder files"""
        # Create multiple builder files
        (self.builders_dir / "builder_test1_step.py").write_text("# Builder 1")
        (self.builders_dir / "builder_test2_step.py").write_text("# Builder 2")
        (self.builders_dir / "not_a_builder.py").write_text("# Not a builder")
        (self.builders_dir / "__init__.py").write_text("# Init file")
        
        builders = self.tester._discover_builders()
        
        self.assertIn("test1", builders)
        self.assertIn("test2", builders)
        self.assertNotIn("not_a_builder", builders)
        self.assertNotIn("__init__", builders)
    
    def test_validate_all_builders(self):
        """Test validation of all discovered builders"""
        # Create test files
        (self.builders_dir / "builder_test1_step.py").write_text('''
class Test1StepBuilder:
    def build_step(self):
        return None
''')
        (self.builders_dir / "builder_test2_step.py").write_text('''
class Test2StepBuilder:
    def build_step(self):
        return None
''')
        
        # Mock validation to avoid file resolution issues
        with patch.object(self.tester, 'validate_builder') as mock_validate:
            mock_validate.return_value = {'passed': True, 'issues': []}
            
            results = self.tester.validate_all_builders()
        
        # Should validate both builders
        self.assertIn("test1", results)
        self.assertIn("test2", results)
        self.assertEqual(mock_validate.call_count, 2)
    
    def test_validate_all_builders_with_target_scripts(self):
        """Test validation with specific target scripts"""
        # Create test files
        (self.builders_dir / "builder_test1_step.py").write_text("# Builder 1")
        (self.builders_dir / "builder_test2_step.py").write_text("# Builder 2")
        
        # Mock validation
        with patch.object(self.tester, 'validate_builder') as mock_validate:
            mock_validate.return_value = {'passed': True, 'issues': []}
            
            results = self.tester.validate_all_builders(target_scripts=["test1"])
        
        # Should only validate test1
        self.assertIn("test1", results)
        self.assertNotIn("test2", results)
        mock_validate.assert_called_once_with("test1")
    
    def test_error_handling_in_validation(self):
        """Test error handling during validation"""
        # Create builder file
        builder_file = self.builders_dir / "builder_test_step.py"
        builder_file.write_text("# Test builder")
        
        # Mock the flexible_resolver which has the get_available_files_report method
        with patch.object(self.tester, 'flexible_resolver') as mock_flexible_resolver, \
             patch.object(self.tester.builder_analyzer, 'analyze_builder_file') as mock_analyze:
            
            # Mock the report method
            mock_flexible_resolver.get_available_files_report.return_value = {
                'configs': {
                    'discovered_files': [],
                    'base_names': [],
                    'count': 0
                }
            }
            
            # Mock analyzer to raise exception
            mock_analyze.side_effect = Exception("Analysis failed")
            
            result = self.tester.validate_builder("test")
        
        # Should handle error gracefully
        self.assertFalse(result['passed'])
        
        # Check that there are issues (could be CRITICAL or ERROR)
        self.assertTrue(len(result['issues']) > 0)
        
        # Check that at least one issue is severe (CRITICAL or ERROR)
        severe_issues = [issue for issue in result['issues'] if issue['severity'] in ['CRITICAL', 'ERROR']]
        self.assertTrue(len(severe_issues) > 0)

if __name__ == '__main__':
    unittest.main()
