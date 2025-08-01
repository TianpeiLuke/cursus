"""
Tests for the cradle_config_factory module.

These tests verify that the helper functions for creating CradleDataLoadConfig
objects work correctly with minimal required inputs.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

# Add the path for importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create mocks for the imported modules that might be missing in test environment
class MockConstants:
    """Mock for secure_ai_sandbox_workflow_python_sdk.utils.constants"""
    OUTPUT_TYPE_DATA = "DATA"
    OUTPUT_TYPE_METADATA = "METADATA"
    OUTPUT_TYPE_SIGNATURE = "SIGNATURE"

# Set up mocks for imports
sys.modules['secure_ai_sandbox_workflow_python_sdk'] = MagicMock()
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils'] = MagicMock()
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils.constants'] = MockConstants

# Import after setting up mocks
from src.pipeline_steps.config_base import BasePipelineConfig

from src.config_field_manager.cradle_config_factory import (
    create_cradle_data_load_config,
    create_training_and_calibration_configs,
    _map_region_to_aws_region,
    _create_field_schema,
    _create_edx_manifest,
    _create_edx_manifest_from_key,
    _generate_transform_sql,
    _get_all_fields
)

from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig,
    DataSourcesSpecificationConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig,
    JobSplitOptionsConfig
)


# Set environment variable for testing before any classes are instantiated
os.environ["MODS_SKIP_PATH_VALIDATION"] = "true"

class TestCradleConfigFactory(unittest.TestCase):
    """Tests for the cradle_config_factory module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample field list
        self.mds_field_list = [
            'field1',
            'field2',
            'field3',
            'cat_field1',
            'cat_field2',
            'objectId',
            'label',
            'id'
        ]
        
        # Create a base config for testing
        self.base_config = BasePipelineConfig(
            role='TestRole',
            region='NA',
            author='test-author',
            bucket='test-bucket',
            service_name='TestService',
            pipeline_version='1.0.0',
            source_dir='/mock/path/to/source'
        )
        
        # Sample configuration parameters for cradle data load using base_config
        self.config_params = {
            'base_config': self.base_config,
            'job_type': 'training',
            'mds_field_list': self.mds_field_list,
            'start_date': '2025-01-01T00:00:00',
            'end_date': '2025-04-17T00:00:00',
            'tag_edx_provider': 'test-provider',
            'tag_edx_subject': 'test-subject',
            'tag_edx_dataset': 'test-dataset',
            'etl_job_id': '12345'
        }
        
        # Set environment variable for testing
        os.environ["MODS_SKIP_PATH_VALIDATION"] = "true"
    
    def test_map_region_to_aws_region(self):
        """Test the region mapping function."""
        self.assertEqual(_map_region_to_aws_region('NA'), 'us-east-1')
        self.assertEqual(_map_region_to_aws_region('EU'), 'eu-west-1')
        self.assertEqual(_map_region_to_aws_region('FE'), 'us-west-2')
        
        # Test invalid region
        with self.assertRaises(ValueError):
            _map_region_to_aws_region('INVALID')
    
    def test_create_field_schema(self):
        """Test the field schema creation function."""
        fields = ['field1', 'field2', 'field3']
        expected = [
            {'field_name': 'field1', 'field_type': 'STRING'},
            {'field_name': 'field2', 'field_type': 'STRING'},
            {'field_name': 'field3', 'field_type': 'STRING'}
        ]
        
        result = _create_field_schema(fields)
        self.assertEqual(result, expected)
    
    def test_create_edx_manifest(self):
        """Test the EDX manifest creation function."""
        manifest = _create_edx_manifest(
            provider='test-provider',
            subject='test-subject',
            dataset='test-dataset',
            etl_job_id='12345',
            start_date='2025-01-01T00:00:00',
            end_date='2025-04-17T00:00:00',
            region='NA'
        )
        
        expected = (
            'arn:amazon:edx:iad::manifest/'
            'test-provider/test-subject/test-dataset/'
            '["12345",2025-01-01T00:00:00Z,2025-04-17T00:00:00Z,"NA"]'
        )
        
        self.assertEqual(manifest, expected)
    
    def test_create_edx_manifest_from_key(self):
        """Test the EDX manifest creation from key function."""
        manifest_key = '["12345","current"]'
        
        manifest = _create_edx_manifest_from_key(
            provider='test-provider',
            subject='test-subject',
            dataset='test-dataset',
            manifest_key=manifest_key
        )
        
        expected = (
            'arn:amazon:edx:iad::manifest/'
            'test-provider/test-subject/test-dataset/["12345","current"]'
        )
        
        self.assertEqual(manifest, expected)
    
    def test_generate_transform_sql(self):
        """Test the SQL generation function."""
        mds_source_name = 'RAW_MDS_NA'
        edx_source_name = 'TAGS'
        mds_field_list = ['objectId', 'field1', 'field2']
        tag_schema = ['order_id', 'tag_value']
        
        sql = _generate_transform_sql(
            mds_source_name=mds_source_name,
            edx_source_name=edx_source_name,
            mds_field_list=mds_field_list,
            tag_schema=tag_schema
        )
        
        # Check that the SQL contains the expected patterns
        self.assertIn('SELECT', sql)
        self.assertIn('RAW_MDS_NA.objectId', sql)
        self.assertIn('RAW_MDS_NA.field1', sql)
        self.assertIn('RAW_MDS_NA.field2', sql)
        self.assertIn('TAGS.order_id', sql)
        self.assertIn('TAGS.tag_value', sql)
        self.assertIn('FROM RAW_MDS_NA', sql)
        self.assertIn('JOIN TAGS', sql)
        self.assertIn('ON RAW_MDS_NA.objectId=TAGS.order_id', sql)
    
    def test_generate_transform_sql_with_duplicates(self):
        """Test the SQL generation function with duplicate fields."""
        mds_source_name = 'RAW_MDS_NA'
        edx_source_name = 'TAGS'
        # Include 'objectId' in both MDS and tags (via order_id)
        mds_field_list = ['objectId', 'field1', 'field2', 'duplicate_field']
        tag_schema = ['order_id', 'tag_value', 'duplicate_field']
        
        sql = _generate_transform_sql(
            mds_source_name=mds_source_name,
            edx_source_name=edx_source_name,
            mds_field_list=mds_field_list,
            tag_schema=tag_schema
        )
        
        # Count occurrences to ensure there are no duplicates except order_id
        self.assertEqual(sql.count('duplicate_field'), 1)
        self.assertEqual(sql.count('RAW_MDS_NA.duplicate_field'), 1)
        self.assertEqual(sql.count('TAGS.duplicate_field'), 0)  # Should be excluded
        
        # Make sure order_id is included from tags even though objectId exists in MDS
        # Note: In the current implementation, order_id appears in the JOIN condition and SELECT clause
        self.assertGreaterEqual(sql.count('TAGS.order_id'), 1)
    
    def test_generate_transform_sql_with_custom_join(self):
        """Test the SQL generation with custom join parameters."""
        mds_source_name = 'RAW_MDS_NA'
        edx_source_name = 'TAGS'
        mds_field_list = ['custom_key', 'field1', 'field2']
        tag_schema = ['alternate_key', 'tag_value']
        
        sql = _generate_transform_sql(
            mds_source_name=mds_source_name,
            edx_source_name=edx_source_name,
            mds_field_list=mds_field_list,
            tag_schema=tag_schema,
            mds_join_key='custom_key',
            edx_join_key='alternate_key',
            join_type='LEFT JOIN'
        )
        
        # Check that the SQL contains the custom join type and keys
        self.assertIn('LEFT JOIN', sql)
        self.assertIn('RAW_MDS_NA.custom_key=TAGS.alternate_key', sql)
        
        # Verify field selection still works correctly
        self.assertIn('RAW_MDS_NA.custom_key', sql)
        self.assertIn('TAGS.alternate_key', sql)
        self.assertIn('TAGS.tag_value', sql)
    
    def test_get_all_fields(self):
        """Test the field combination function."""
        mds_fields = ['objectId', 'field1', 'field2', 'duplicate']
        tag_fields = ['order_id', 'tag_value', 'duplicate']
        
        result = _get_all_fields(mds_fields, tag_fields)
        
        # Check that the result has all unique fields
        # The fields are: 'objectId', 'field1', 'field2', 'duplicate', 'order_id', 'tag_value'
        self.assertEqual(len(result), 6)  # 6 unique fields
        self.assertIn('objectId', result)
        self.assertIn('field1', result)
        self.assertIn('field2', result)
        self.assertIn('order_id', result)
        self.assertIn('tag_value', result)
        self.assertIn('duplicate', result)
        
        # Check that the result is sorted
        self.assertEqual(result, sorted(result))
    
    def test_create_cradle_data_load_config(self):
        """Test the main CradleDataLoadConfig creation function."""
        config = create_cradle_data_load_config(**self.config_params)
        
        # Check that the config is the correct type
        self.assertIsInstance(config, CradleDataLoadConfig)
        
        # Check that fields are properly inherited from base_config
        self.assertEqual(config.role, self.base_config.role)
        self.assertEqual(config.region, self.base_config.region)
        self.assertEqual(config.author, self.base_config.author)
        self.assertEqual(config.bucket, self.base_config.bucket)
        self.assertEqual(config.service_name, self.base_config.service_name)
        self.assertEqual(config.source_dir, self.base_config.source_dir)
        
        # Check that job-specific fields are set correctly
        self.assertEqual(config.job_type, self.config_params['job_type'])
        
        # Check that data sources specification is created
        self.assertIsInstance(config.data_sources_spec, DataSourcesSpecificationConfig)
        self.assertEqual(config.data_sources_spec.start_date, self.config_params['start_date'])
        self.assertEqual(config.data_sources_spec.end_date, self.config_params['end_date'])
        self.assertEqual(len(config.data_sources_spec.data_sources), 2)  # MDS and EDX
        
        # Check that transform specification is created
        self.assertIsInstance(config.transform_spec, TransformSpecificationConfig)
        self.assertIsNotNone(config.transform_spec.transform_sql)
        self.assertIsInstance(config.transform_spec.job_split_options, JobSplitOptionsConfig)
        
        # Check that output specification is created
        self.assertIsInstance(config.output_spec, OutputSpecificationConfig)
        self.assertEqual(config.output_spec.job_type, self.config_params['job_type'])
        
        # Check that cradle job specification is created
        self.assertIsInstance(config.cradle_job_spec, CradleJobSpecificationConfig)
        
        # Check derived fields are correctly calculated based on inherited values
        self.assertEqual(config.aws_region, 'us-east-1')  # Derived from region 'NA'
        self.assertTrue(config.pipeline_s3_loc.startswith(f"s3://{self.base_config.bucket}/MODS/"))
    
    def test_create_cradle_data_load_config_with_custom_transform_sql(self):
        """Test creating config with custom transform SQL."""
        params = self.config_params.copy()
        params['transform_sql'] = 'SELECT * FROM custom_source'
        
        config = create_cradle_data_load_config(**params)
        
        # Check that the custom SQL was used
        self.assertEqual(config.transform_spec.transform_sql, params['transform_sql'])
    
    def test_create_cradle_data_load_config_with_split_job(self):
        """Test creating config with job splitting enabled."""
        params = self.config_params.copy()
        params['split_job'] = True
        params['days_per_split'] = 5
        params['merge_sql'] = 'SELECT * FROM INPUT'
        
        config = create_cradle_data_load_config(**params)
        
        # Check split job options
        self.assertTrue(config.transform_spec.job_split_options.split_job)
        self.assertEqual(config.transform_spec.job_split_options.days_per_split, 5)
        self.assertEqual(config.transform_spec.job_split_options.merge_sql, 'SELECT * FROM INPUT')
        
    def test_create_cradle_data_load_config_with_custom_org_id(self):
        """Test creating config with a custom org_id."""
        params = self.config_params.copy()
        params['org_id'] = 123  # Custom org ID
        
        config = create_cradle_data_load_config(**params)
        
        # Check that org_id is set correctly in MDS data source
        self.assertEqual(config.data_sources_spec.data_sources[0].mds_data_source_properties.org_id, 123)
        
        # Check default org_id
        config_default = create_cradle_data_load_config(**self.config_params)
        self.assertEqual(config_default.data_sources_spec.data_sources[0].mds_data_source_properties.org_id, 0)
    
    def test_create_training_and_calibration_configs(self):
        """Test creating both training and calibration configs."""
        params = {
            'base_config': self.base_config,
            'mds_field_list': self.mds_field_list,
            'tag_edx_provider': 'test-provider',
            'tag_edx_subject': 'test-subject',
            'tag_edx_dataset': 'test-dataset',
            'etl_job_id': '12345',
            'training_start_date': '2025-01-01T00:00:00',
            'training_end_date': '2025-04-17T00:00:00',
            'calibration_start_date': '2025-04-18T00:00:00',
            'calibration_end_date': '2025-05-01T00:00:00'
        }
        
        configs = create_training_and_calibration_configs(**params)
        
        # Check that both configs were created
        self.assertIn('training', configs)
        self.assertIn('calibration', configs)
        
        # Check training config
        training = configs['training']
        self.assertIsInstance(training, CradleDataLoadConfig)
        self.assertEqual(training.job_type, 'training')
        self.assertEqual(training.data_sources_spec.start_date, params['training_start_date'])
        self.assertEqual(training.data_sources_spec.end_date, params['training_end_date'])
        
        # Check that training config inherits from base_config
        self.assertEqual(training.role, self.base_config.role)
        self.assertEqual(training.author, self.base_config.author)
        self.assertEqual(training.bucket, self.base_config.bucket)
        self.assertEqual(training.service_name, self.base_config.service_name)
        
        # Check calibration config
        calibration = configs['calibration']
        self.assertIsInstance(calibration, CradleDataLoadConfig)
        self.assertEqual(calibration.job_type, 'calibration')
        self.assertEqual(calibration.data_sources_spec.start_date, params['calibration_start_date'])
        self.assertEqual(calibration.data_sources_spec.end_date, params['calibration_end_date'])
        
        # Check that calibration config inherits from base_config
        self.assertEqual(calibration.role, self.base_config.role)
        self.assertEqual(calibration.author, self.base_config.author)
        self.assertEqual(calibration.bucket, self.base_config.bucket)
        self.assertEqual(calibration.service_name, self.base_config.service_name)
        
        # Check that the EDX manifest keys are properly formatted with Z suffixes
        self.assertIn("Z,", training.data_sources_spec.data_sources[1].edx_data_source_properties.edx_manifest)
        self.assertIn("Z,", calibration.data_sources_spec.data_sources[1].edx_data_source_properties.edx_manifest)


if __name__ == '__main__':
    unittest.main()
