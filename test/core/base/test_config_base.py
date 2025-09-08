import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any
import logging
from datetime import datetime

from cursus.core.base.config_base import BasePipelineConfig

class TestBasePipelineConfig(unittest.TestCase):
    """Test cases for BasePipelineConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.valid_config_data = {
            'author': 'test_author',
            'bucket': 'test-bucket',
            'role': 'arn:aws:iam::123456789012:role/TestRole',
            'region': 'NA',
            'service_name': 'test_service',
            'pipeline_version': '1.0.0'
        }
    
    def test_init_with_required_fields(self):
        """Test initialization with all required fields."""
        config = BasePipelineConfig(**self.valid_config_data)
        
        # Verify required fields
        self.assertEqual(config.author, 'test_author')
        self.assertEqual(config.bucket, 'test-bucket')
        self.assertEqual(config.role, 'arn:aws:iam::123456789012:role/TestRole')
        self.assertEqual(config.region, 'NA')
        self.assertEqual(config.service_name, 'test_service')
        self.assertEqual(config.pipeline_version, '1.0.0')
        
        # Verify default fields
        self.assertEqual(config.model_class, 'xgboost')
        self.assertEqual(config.framework_version, '2.1.0')
        self.assertEqual(config.py_version, 'py310')
        self.assertIsNone(config.source_dir)
        
        # Verify current_date is set
        self.assertIsInstance(config.current_date, str)
        self.assertTrue(len(config.current_date) > 0)
    
    def test_init_with_optional_fields(self):
        """Test initialization with optional fields."""
        config_data = self.valid_config_data.copy()
        config_data.update({
            'model_class': 'pytorch',
            'framework_version': '1.8.0',
            'py_version': 'py39',
            'source_dir': '/test/source'
        })
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            config = BasePipelineConfig(**config_data)
            
            self.assertEqual(config.model_class, 'pytorch')
            self.assertEqual(config.framework_version, '1.8.0')
            self.assertEqual(config.py_version, 'py39')
            self.assertEqual(config.source_dir, '/test/source')
    
    def test_derived_properties(self):
        """Test derived properties are calculated correctly."""
        config = BasePipelineConfig(**self.valid_config_data)
        
        # Test aws_region
        self.assertEqual(config.aws_region, 'us-east-1')
        
        # Test pipeline_name
        expected_name = 'test_author-test_service-xgboost-NA'
        self.assertEqual(config.pipeline_name, expected_name)
        
        # Test pipeline_description
        expected_desc = 'test_service xgboost Model NA'
        self.assertEqual(config.pipeline_description, expected_desc)
        
        # Test pipeline_s3_loc
        expected_s3_loc = 's3://test-bucket/MODS/test_author-test_service-xgboost-NA_1.0.0'
        self.assertEqual(config.pipeline_s3_loc, expected_s3_loc)
    
    def test_region_validation(self):
        """Test region validation."""
        # Test valid regions
        for region in ['NA', 'EU', 'FE']:
            config_data = self.valid_config_data.copy()
            config_data['region'] = region
            config = BasePipelineConfig(**config_data)
            self.assertEqual(config.region, region)
        
        # Test invalid region
        config_data = self.valid_config_data.copy()
        config_data['region'] = 'INVALID'
        
        with self.assertRaises(ValueError) as context:
            BasePipelineConfig(**config_data)
        
        self.assertIn("Invalid custom region code", str(context.exception))
    
    def test_source_dir_validation(self):
        """Test source_dir validation."""
        config_data = self.valid_config_data.copy()
        config_data['source_dir'] = '/nonexistent/path'
        
        # Test non-existent local path
        with patch('pathlib.Path.exists', return_value=False):
            with self.assertRaises(ValueError) as context:
                BasePipelineConfig(**config_data)
            
            self.assertIn("Local source directory does not exist", str(context.exception))
        
        # Test path that exists but is not a directory
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=False):
            with self.assertRaises(ValueError) as context:
                BasePipelineConfig(**config_data)
            
            self.assertIn("Local source_dir is not a directory", str(context.exception))
        
        # Test S3 path (should not be validated)
        config_data['source_dir'] = 's3://bucket/path'
        config = BasePipelineConfig(**config_data)
        self.assertEqual(config.source_dir, 's3://bucket/path')
    
    def test_model_dump_includes_derived_properties(self):
        """Test that model_dump includes derived properties."""
        config = BasePipelineConfig(**self.valid_config_data)
        data = config.model_dump()
        
        # Check that derived properties are included
        self.assertIn('aws_region', data)
        self.assertIn('pipeline_name', data)
        self.assertIn('pipeline_description', data)
        self.assertIn('pipeline_s3_loc', data)
        
        # Verify values
        self.assertEqual(data['aws_region'], 'us-east-1')
        self.assertEqual(data['pipeline_name'], 'test_author-test_service-xgboost-NA')
    
    def test_categorize_fields(self):
        """Test field categorization."""
        config = BasePipelineConfig(**self.valid_config_data)
        categories = config.categorize_fields()
        
        # Check that all categories exist
        self.assertIn('essential', categories)
        self.assertIn('system', categories)
        self.assertIn('derived', categories)
        
        # Check essential fields (required, no defaults)
        essential_fields = set(categories['essential'])
        expected_essential = {'author', 'bucket', 'role', 'region', 'service_name', 'pipeline_version'}
        self.assertEqual(essential_fields, expected_essential)
        
        # Check system fields (have defaults)
        system_fields = set(categories['system'])
        expected_system = {'model_class', 'current_date', 'framework_version', 'py_version', 'source_dir'}
        self.assertEqual(system_fields, expected_system)
        
        # Check derived fields (properties)
        derived_fields = set(categories['derived'])
        expected_derived = {'aws_region', 'pipeline_name', 'pipeline_description', 'pipeline_s3_loc', 'script_contract', 'model_extra', 'model_fields_set'}
        self.assertEqual(derived_fields, expected_derived)
    
    def test_get_public_init_fields(self):
        """Test getting public initialization fields."""
        config = BasePipelineConfig(**self.valid_config_data)
        init_fields = config.get_public_init_fields()
        
        # Should include all essential fields
        for field in ['author', 'bucket', 'role', 'region', 'service_name', 'pipeline_version']:
            self.assertIn(field, init_fields)
            self.assertEqual(init_fields[field], getattr(config, field))
        
        # Should include non-None system fields
        for field in ['model_class', 'current_date', 'framework_version', 'py_version']:
            self.assertIn(field, init_fields)
        
        # Should not include None fields
        if config.source_dir is None:
            self.assertNotIn('source_dir', init_fields)
    
    def test_from_base_config(self):
        """Test creating config from base config."""
        base_config = BasePipelineConfig(**self.valid_config_data)
        
        # Create derived config with additional fields
        derived_config = BasePipelineConfig.from_base_config(
            base_config,
            model_class='pytorch',
            framework_version='1.8.0'
        )
        
        # Should inherit base fields
        self.assertEqual(derived_config.author, base_config.author)
        self.assertEqual(derived_config.bucket, base_config.bucket)
        
        # Should override with new values
        self.assertEqual(derived_config.model_class, 'pytorch')
        self.assertEqual(derived_config.framework_version, '1.8.0')
    
    def test_get_step_name_class_method(self):
        """Test get_step_name class method."""
        # This tests the class method that looks up step names
        step_name = BasePipelineConfig.get_step_name('TestConfig')
        # Should return the input if not found in registry
        self.assertEqual(step_name, 'TestConfig')
    
    def test_get_config_class_name_class_method(self):
        """Test get_config_class_name class method."""
        # This tests the reverse lookup
        config_class = BasePipelineConfig.get_config_class_name('TestStep')
        # Should return the input if not found in reverse mapping
        self.assertEqual(config_class, 'TestStep')
    
    def test_get_script_contract_default(self):
        """Test get_script_contract default implementation."""
        config = BasePipelineConfig(**self.valid_config_data)
        contract = config.get_script_contract()
        
        # Base implementation should return None
        self.assertIsNone(contract)
    
    def test_get_script_path_default(self):
        """Test get_script_path with default."""
        config = BasePipelineConfig(**self.valid_config_data)
        
        # Should return default when no contract or script_path
        default_path = '/test/default/script.py'
        script_path = config.get_script_path(default_path)
        self.assertEqual(script_path, default_path)
        
        # Should return None when no default provided
        script_path = config.get_script_path()
        self.assertIsNone(script_path)
    
    def test_string_representation(self):
        """Test string representation."""
        config = BasePipelineConfig(**self.valid_config_data)
        str_repr = str(config)
        
        # Should contain class name
        self.assertIn('BasePipelineConfig', str_repr)
        
        # Should contain field categories
        self.assertIn('Essential User Inputs', str_repr)
        self.assertIn('System Inputs', str_repr)
        self.assertIn('Derived Fields', str_repr)
        
        # Should contain some field values
        self.assertIn('test_author', str_repr)
        self.assertIn('test-bucket', str_repr)
    
    def test_print_config_method(self):
        """Test print_config method."""
        config = BasePipelineConfig(**self.valid_config_data)
        
        # Should not raise any exceptions
        try:
            config.print_config()
        except Exception as e:
            self.fail(f"print_config raised an exception: {e}")
    
    def test_region_mapping(self):
        """Test region mapping for all supported regions."""
        region_tests = [
            ('NA', 'us-east-1'),
            ('EU', 'eu-west-1'),
            ('FE', 'us-west-2')
        ]
        
        for region_code, expected_aws_region in region_tests:
            config_data = self.valid_config_data.copy()
            config_data['region'] = region_code
            config = BasePipelineConfig(**config_data)
            
            self.assertEqual(config.aws_region, expected_aws_region)
    
    def test_derived_fields_caching(self):
        """Test that derived fields are cached."""
        config = BasePipelineConfig(**self.valid_config_data)
        
        # Access derived property multiple times
        first_access = config.pipeline_name
        second_access = config.pipeline_name
        
        # Should return the same value (testing caching behavior)
        self.assertEqual(first_access, second_access)
        self.assertEqual(first_access, 'test_author-test_service-xgboost-NA')
    
    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        config_data = self.valid_config_data.copy()
        config_data['extra_field'] = 'extra_value'
        
        # Should not raise an exception
        config = BasePipelineConfig(**config_data)
        
        # Extra field should be accessible
        self.assertEqual(config.extra_field, 'extra_value')

if __name__ == '__main__':
    # Set up logging for tests
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
