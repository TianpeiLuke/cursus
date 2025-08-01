#!/usr/bin/env python3
"""
Test script to verify our fixes for handling circular references with enhanced placeholders.
This test focuses on the improved handling of required fields in circular references.
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


class EnhancedPlaceholdersTest(unittest.TestCase):
    """Test that our enhanced placeholders for circular references work correctly."""
    
    def setUp(self):
        """Set up test environment."""
        # Register our mock models
        ConfigClassStore.register(DataSourceConfig)
        ConfigClassStore.register(DataSourcesSpecificationConfig)
        ConfigClassStore.register(CradleDataLoadConfig)
    
    def test_enhanced_placeholders_for_circular_refs(self):
        """
        Test that our enhanced placeholders include required fields to prevent validation errors.
        
        This test verifies that when a circular reference is detected, the placeholder
        includes the required fields 'data_source_name' and 'data_source_type', which
        should allow validation to pass without ERROR messages.
        """
        # Create a serializer with our fixes
        serializer = TypeAwareConfigSerializer()
        
        # Manually create a circular reference test case
        # First, create some objects that reference each other
        ds_spec = DataSourcesSpecificationConfig(
            start_date="2025-01-01T00:00:00",
            end_date="2025-04-17T00:00:00",
            data_sources=[]  # Start with empty list
        )
        
        # Create first data source
        ds1 = DataSourceConfig(
            data_source_name="SOURCE1",
            data_source_type="TEST"
        )
        
        # Force a circular reference by adding the second one as a dict with a reference to self
        ds2_dict = {
            "__model_type__": "DataSourceConfig",
            "__model_module__": "src.pipeline_steps.config_data_load_step_cradle",
            # These fields would normally be missing in a circular reference
            "data_source_name": "SOURCE2",
            "data_source_type": "TEST",
            # Add a self-reference that would create a circular structure
            "circular_self_ref": ds_spec  # This creates a circular reference
        }
        
        # Set up the deserialization with the circular reference
        # We'll simulate what happens when the deserializer encounters this structure
        serialized = {
            "__model_type__": "DataSourcesSpecificationConfig",
            "__model_module__": "test.config_field_manager.test_enhanced_placeholders",
            "start_date": "2025-01-01T00:00:00",
            "end_date": "2025-04-17T00:00:00",
            "data_sources": {
                "__type_info__": "list",
                "value": [
                    {
                        "__model_type__": "DataSourceConfig",
                        "__model_module__": "test.config_field_manager.test_enhanced_placeholders",
                        "data_source_name": "SOURCE1",
                        "data_source_type": "TEST"
                    },
                    # Second item is where the circular reference would happen
                    ds2_dict
                ]
            }
        }
        
        # Create a special test logger to track if any ERROR messages are produced
        test_handler = TestLogHandler()
        serializer.logger.addHandler(test_handler)
        
        # Attempt to deserialize
        logger.info("Deserializing data with circular references")
        result = serializer.deserialize(serialized)
        
        # Check that no validation errors were produced during deserialization
        self.assertFalse(test_handler.has_error_logs, 
                         "ERROR log messages were produced during deserialization")
        
        # Verify result is properly deserialized
        self.assertIsInstance(result, DataSourcesSpecificationConfig, 
                              "Failed to deserialize into DataSourcesSpecificationConfig instance")
        
        # Verify data_sources list
        self.assertIsInstance(result.data_sources, list,
                             "data_sources was not properly converted to a list")
        self.assertEqual(len(result.data_sources), 2,
                        "data_sources list doesn't have the expected number of items")
        
        # Verify first data source
        self.assertEqual(result.data_sources[0].data_source_name, "SOURCE1",
                        "First data source name not correctly deserialized")
        self.assertEqual(result.data_sources[0].data_source_type, "TEST",
                        "First data source type not correctly deserialized")
        
        # Verify second item (should be a CircularRef object or data source with required fields)
        second_item = result.data_sources[1]
        
        # It doesn't matter if it's a circular ref stub or normal object,
        # as long as it has the required fields with valid values
        self.assertTrue(hasattr(second_item, "data_source_name"),
                       "Second item doesn't have data_source_name field")
        self.assertTrue(hasattr(second_item, "data_source_type"),
                       "Second item doesn't have data_source_type field")
                       
        # Verify that data_source_type is a valid value (MDS, EDX, or ANDES)
        if hasattr(second_item, "data_source_type"):
            self.assertIn(second_item.data_source_type, {"MDS", "EDX", "ANDES"},
                         f"data_source_type has invalid value: {second_item.data_source_type}")
        
        logger.info("Successfully verified enhanced placeholders for circular references")


class TestLogHandler(logging.Handler):
    """A custom log handler that tracks if any ERROR messages were produced."""
    
    def __init__(self):
        super().__init__()
        self.has_error_logs = False
        
    def emit(self, record):
        if record.levelname == 'ERROR':
            self.has_error_logs = True


if __name__ == "__main__":
    unittest.main()
