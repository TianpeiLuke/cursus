"""
Test for the field flattening functionality in CradleDataLoadConfig.

This test validates that get_all_tiered_fields() correctly flattens
the nested structure of configuration objects.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add the path for importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create a mock for the constants that might be missing in test environment
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
from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig, 
    DataSourcesSpecificationConfig,
    DataSourceConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig,
    JobSplitOptionsConfig,
    get_flattened_fields
)


class TestFieldFlattening(unittest.TestCase):
    """Tests for the field flattening functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create MDS data source config
        self.mds_config = MdsDataSourceConfig(
            service_name="orders",
            region="NA",
            output_schema=[
                {"field_name": "order_id", "field_type": "STRING"},
                {"field_name": "customer_id", "field_type": "STRING"}
            ]
        )
        
        # Create EDX data source config
        self.edx_config = EdxDataSourceConfig(
            edx_provider="amazon",
            edx_subject="buyerabuse",
            edx_dataset="features",
            edx_manifest_key='["current"]',
            schema_overrides=[
                {"field_name": "feature_id", "field_type": "STRING"},
                {"field_name": "feature_value", "field_type": "FLOAT"}
            ]
        )
        
        # Create data source configs
        self.mds_data_source = DataSourceConfig(
            data_source_name="RAW_MDS_NA",
            data_source_type="MDS",
            mds_data_source_properties=self.mds_config
        )
        
        self.edx_data_source = DataSourceConfig(
            data_source_name="FEATURES",
            data_source_type="EDX",
            edx_data_source_properties=self.edx_config
        )
        
        # Create data sources specification
        self.data_sources_spec = DataSourcesSpecificationConfig(
            start_date="2025-01-01T00:00:00",
            end_date="2025-01-31T23:59:59",
            data_sources=[self.mds_data_source, self.edx_data_source]
        )
        
        # Create transform specification with job split options
        self.transform_spec = TransformSpecificationConfig(
            transform_sql="SELECT * FROM orders LEFT JOIN features ON orders.order_id = features.order_id",
            job_split_options=JobSplitOptionsConfig(
                split_job=True,
                days_per_split=5,
                merge_sql="SELECT * FROM INPUT"
            )
        )
        
        # Create output specification
        self.output_spec = OutputSpecificationConfig(
            output_schema=["order_id", "customer_id", "feature_id", "feature_value"],
            job_type="training"
        )
        
        # Create Cradle job specification
        self.cradle_job_spec = CradleJobSpecificationConfig(
            cradle_account="Buyer-Abuse-RnD-Dev",
            cluster_type="MEDIUM"
        )
        
        # Create top-level config
        self.config = CradleDataLoadConfig(
            role="ML-Developer",
            region="us-east-1",
            job_type="training",
            data_sources_spec=self.data_sources_spec,
            transform_spec=self.transform_spec,
            output_spec=self.output_spec,
            cradle_job_spec=self.cradle_job_spec,
            pipeline_s3_loc="s3://my-bucket/pipeline"  # For derived output_path
        )
        
        # Initialize derived fields
        if hasattr(self.config, 'initialize_derived_fields'):
            self.config.initialize_derived_fields()
    
    def test_get_all_tiered_fields(self):
        """Test that get_all_tiered_fields returns the flattened fields."""
        flattened_fields = self.config.get_all_tiered_fields()
        
        # Check that the result has the expected structure
        self.assertIn('essential', flattened_fields)
        self.assertIn('system', flattened_fields)
        self.assertIn('derived', flattened_fields)
        
        # Check for key fields from different levels of nesting
        essential_fields = flattened_fields['essential']
        self.assertIn('job_type', essential_fields)
        self.assertIn('data_sources_spec.start_date', essential_fields)
        self.assertIn('data_sources_spec.end_date', essential_fields)
        
        # Check for fields from data sources (which are in a list)
        found_mds = False
        found_edx = False
        for field in essential_fields:
            if 'data_sources[' in field and 'data_source_name' in field:
                found_mds = found_mds or "RAW_MDS_NA" in self.config.data_sources_spec.data_sources[0].data_source_name
                found_edx = found_edx or "FEATURES" in self.config.data_sources_spec.data_sources[1].data_source_name
        self.assertTrue(found_mds, "Could not find MDS data source in flattened fields")
        self.assertTrue(found_edx, "Could not find EDX data source in flattened fields")
        
        # Check for derived fields
        derived_fields = flattened_fields['derived']
        # The output_path is derived from job_type and pipeline_s3_loc
        self.assertIn('output_spec.output_path', derived_fields)
        
        # Check for EDX manifest which is derived from other fields
        found_manifest = False
        for field in derived_fields:
            if 'edx_manifest' in field:
                found_manifest = True
                break
        self.assertTrue(found_manifest, "Could not find edx_manifest in derived fields")
    
    def test_flattened_fields_count(self):
        """Test that the flattened fields contain all expected fields."""
        flattened_fields = self.config.get_all_tiered_fields()
        
        # Get the total number of fields across all categories
        total_fields = (len(flattened_fields['essential']) +
                        len(flattened_fields['system']) +
                        len(flattened_fields['derived']))
        
        # We should have a substantial number of fields from all the nested objects
        # The exact number will depend on the implementation, but we can check it's reasonable
        self.assertGreater(total_fields, 20)
        
        # Check specific counts for each tier
        self.assertGreaterEqual(len(flattened_fields['essential']), 10)  # At least 10 essential fields
        self.assertGreaterEqual(len(flattened_fields['system']), 5)      # At least 5 system fields
        self.assertGreaterEqual(len(flattened_fields['derived']), 1)     # At least 1 derived field


if __name__ == '__main__':
    unittest.main()
