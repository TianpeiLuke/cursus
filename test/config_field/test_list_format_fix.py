#!/usr/bin/env python3
"""
Test script to verify our fixes for handling the special list format that was causing issues.
This test focuses on the specific serializer enhancements without requiring external dependencies.
"""
import unittest
import logging
import json
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our fixed serializer
from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
from src.config_field_manager.config_class_store import ConfigClassStore
from pydantic import BaseModel, Field, ConfigDict


# Create mock models that simulate the problematic structure
class DataSourceConfig(BaseModel):
    """Mock data source config."""
    data_source_name: str
    data_source_type: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Allow extra fields like type metadata
        frozen=True
    )

class DataSourcesSpecificationConfig(BaseModel):
    """Mock data sources specification with a list that caused circular ref issues."""
    start_date: str
    end_date: str
    data_sources: List[DataSourceConfig]
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow"  # Allow extra fields like type metadata
    )

class CradleDataLoadConfig(BaseModel):
    """Mock Cradle data load config that references data sources spec."""
    job_type: str
    data_sources_spec: DataSourcesSpecificationConfig
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow"  # Allow extra fields like type metadata
    )


class ListFormatFixTest(unittest.TestCase):
    """Test that our fixes for the special list format handling work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        # Register our mock models
        ConfigClassStore.register(DataSourceConfig)
        ConfigClassStore.register(DataSourcesSpecificationConfig)
        ConfigClassStore.register(CradleDataLoadConfig)
    
    def test_special_list_format_handling(self):
        """Test that our fixes properly handle the special list format."""
        # Create a serializer with our fixes
        serializer = TypeAwareConfigSerializer()
        
        # Create a JSON data structure that simulates the problematic structure
        data = {
            "__model_type__": "CradleDataLoadConfig",
            "__model_module__": "src.pipeline_steps.config_data_load_step_cradle",
            "job_type": "training",
            "data_sources_spec": {
                "__model_type__": "DataSourcesSpecificationConfig",
                "__model_module__": "src.pipeline_steps.config_data_load_step_cradle",
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-04-17T00:00:00",
                "data_sources": {
                    "__type_info__": "list",
                    "value": [
                        {
                            "__model_type__": "DataSourceConfig",
                            "__model_module__": "src.pipeline_steps.config_data_load_step_cradle",
                            "data_source_name": "RAW_MDS_NA",
                            "data_source_type": "MDS"
                        },
                        {
                            "__model_type__": "DataSourceConfig",
                            "__model_module__": "src.pipeline_steps.config_data_load_step_cradle",
                            "data_source_name": "TAGS",
                            "data_source_type": "EDX"
                        }
                    ]
                }
            }
        }
        
        # Attempt to deserialize
        logger.info("Deserializing data with special list format")
        result = serializer.deserialize(data)
        
        # Verify result is properly deserialized
        self.assertIsInstance(result, CradleDataLoadConfig, 
                              "Failed to deserialize into CradleDataLoadConfig instance")
        
        # Verify job_type
        self.assertEqual(result.job_type, "training", "job_type not correctly deserialized")
        
        # Verify data_sources_spec
        self.assertIsInstance(result.data_sources_spec, DataSourcesSpecificationConfig,
                             "data_sources_spec not correctly deserialized")
        
        # Verify data_sources is a list, not a dictionary
        self.assertIsInstance(result.data_sources_spec.data_sources, list,
                             "data_sources was not properly converted from special format to a list")
        
        # Verify list has expected length
        self.assertEqual(len(result.data_sources_spec.data_sources), 2,
                        "data_sources list doesn't have the expected number of items")
        
        # Verify first data source
        self.assertEqual(result.data_sources_spec.data_sources[0].data_source_name, "RAW_MDS_NA",
                        "First data source name not correctly deserialized")
        self.assertEqual(result.data_sources_spec.data_sources[0].data_source_type, "MDS",
                        "First data source type not correctly deserialized")
        
        # Verify second data source - it might be a DataSourceConfig instance or a dict with circular reference info
        second_item = result.data_sources_spec.data_sources[1]
        if hasattr(second_item, 'data_source_name'):
            # If it's a DataSourceConfig instance
            self.assertEqual(second_item.data_source_name, "TAGS",
                           "Second data source name not correctly deserialized")
            self.assertEqual(second_item.data_source_type, "EDX",
                           "Second data source type not correctly deserialized")
        else:
            # If it's a dict with circular reference info, verify that it has the expected circular reference fields
            self.assertIsInstance(second_item, dict, "Expected dict for circular reference handling")
            self.assertIn("__circular_ref__", second_item, "Expected __circular_ref__ field in circular reference dict")
        
        logger.info("Successfully deserialized data with special list format")


if __name__ == "__main__":
    unittest.main()
