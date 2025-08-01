"""
Test file for CradleDataLoadConfig.from_base_config functionality
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the path for importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock the secure_ai_sandbox_workflow_python_sdk imports
sys.modules['secure_ai_sandbox_workflow_python_sdk'] = MagicMock()
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils'] = MagicMock()
sys.modules['secure_ai_sandbox_workflow_python_sdk.utils.constants'] = MagicMock(
    OUTPUT_TYPE_DATA="DATA",
    OUTPUT_TYPE_METADATA="METADATA",
    OUTPUT_TYPE_SIGNATURE="SIGNATURE"
)

# Import after setting up mocks
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig,
    DataSourcesSpecificationConfig,
    TransformSpecificationConfig, 
    OutputSpecificationConfig,
    CradleJobSpecificationConfig
)


class TestCradleConfigFromBase(unittest.TestCase):
    """Test cases for CradleDataLoadConfig.from_base_config functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a base config
        self.base_config = BasePipelineConfig(
            author="testuser",
            bucket="test-bucket",
            role="TestRole",
            region="NA",
            service_name="TestService",
            pipeline_version="1.0.0"
        )
        
        # Create minimal specs needed for CradleDataLoadConfig
        self.data_sources_spec = DataSourcesSpecificationConfig(
            start_date="2025-01-01T00:00:00",
            end_date="2025-04-17T00:00:00",
            data_sources=[]  # Empty for simplicity in this test
        )
        
        self.transform_spec = TransformSpecificationConfig(
            transform_sql="SELECT * FROM test"
        )
        
        self.output_spec = OutputSpecificationConfig(
            output_schema=["field1", "field2"],
            job_type="training"
        )
        
        self.cradle_job_spec = CradleJobSpecificationConfig(
            cradle_account="Buyer-Abuse-RnD-Dev"
        )
        
        # Dictionary of configs to pass to from_base_config
        self.cradle_config_dict = {
            "job_type": "training",
            "data_sources_spec": self.data_sources_spec,
            "transform_spec": self.transform_spec,
            "output_spec": self.output_spec,
            "cradle_job_spec": self.cradle_job_spec,
        }
    
    def test_from_base_config(self):
        """Test that from_base_config correctly builds a valid CradleDataLoadConfig."""
        # Create a CradleDataLoadConfig from the base config
        cradle_config = CradleDataLoadConfig.from_base_config(
            self.base_config,
            **self.cradle_config_dict
        )
        
        # Verify that essential fields from base config were inherited
        self.assertEqual(cradle_config.author, self.base_config.author)
        self.assertEqual(cradle_config.bucket, self.base_config.bucket)
        self.assertEqual(cradle_config.role, self.base_config.role)
        self.assertEqual(cradle_config.region, self.base_config.region)
        self.assertEqual(cradle_config.service_name, self.base_config.service_name)
        
        # Verify that fields from the cradle config were set
        self.assertEqual(cradle_config.job_type, "training")
        self.assertEqual(cradle_config.data_sources_spec, self.data_sources_spec)
        self.assertEqual(cradle_config.transform_spec, self.transform_spec)
        
        # Verify that output_spec has pipeline_s3_loc set
        self.assertIsNotNone(cradle_config.output_spec.pipeline_s3_loc)
        self.assertEqual(
            cradle_config.output_spec.pipeline_s3_loc, 
            cradle_config.pipeline_s3_loc
        )
        
        # Verify that output_path is properly set using pipeline_s3_loc
        expected_path = f"{cradle_config.pipeline_s3_loc}/data-load/training"
        self.assertEqual(cradle_config.output_spec.output_path, expected_path)


if __name__ == '__main__':
    unittest.main()
