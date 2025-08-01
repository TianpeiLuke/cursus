#!/usr/bin/env python3
"""
Test script to verify our fixes for false positive detection in circular references.
This test focuses on improved ID generation for list items to avoid false positives.
"""
import unittest
import logging
import json
from typing import Dict, Any, List, Optional, Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our fixed circular reference tracker
from src.config_field_manager.circular_reference_tracker import CircularReferenceTracker
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


class FixedCircularDetectionTest(unittest.TestCase):
    """Test that our improved circular reference detection avoids false positives."""
    
    def setUp(self):
        """Set up test environment."""
        # Register our mock models
        ConfigClassStore.register(DataSourceConfig)
        ConfigClassStore.register(DataSourcesSpecificationConfig)
        ConfigClassStore.register(CradleDataLoadConfig)
    
    def test_list_items_avoid_false_positives(self):
        """
        Test that list items are properly distinguished to avoid false positives.
        
        This test verifies that two distinct DataSourceConfig objects in a list
        are treated as separate objects even though they share the same class.
        """
        # Create a serializer
        serializer = TypeAwareConfigSerializer()
        
        # Create a test config JSON with multiple DataSourceConfig objects in a list
        config_json = {
            "__model_type__": "DataSourcesSpecificationConfig",
            "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
            "start_date": "2025-01-01T00:00:00",
            "end_date": "2025-04-17T00:00:00",
            "data_sources": {
                "__type_info__": "list",
                "value": [
                    {
                        "__model_type__": "DataSourceConfig",
                        "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
                        "data_source_name": "SOURCE1",
                        "data_source_type": "MDS"
                    },
                    {
                        "__model_type__": "DataSourceConfig",
                        "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
                        "data_source_name": "SOURCE2",
                        "data_source_type": "EDX"
                    }
                ]
            }
        }
        
        # Create a special test logger to track warnings
        test_handler = TestLogHandler()
        serializer.logger.addHandler(test_handler)
        
        # Attempt to deserialize with improved circular reference detection
        logger.info("Deserializing DataSourcesSpecificationConfig with list of DataSourceConfig objects")
        result = serializer.deserialize(config_json)
        
        # Check that no circular reference warnings were produced
        circular_refs = [log for log in test_handler.logs if "Circular reference detected" in log]
        self.assertEqual(len(circular_refs), 0, 
                         f"Detected {len(circular_refs)} false positive circular references: {circular_refs}")
        
        # Verify result is properly deserialized
        self.assertIsInstance(result, DataSourcesSpecificationConfig, 
                              "Failed to deserialize into DataSourcesSpecificationConfig instance")
        
        # Verify data_sources list has the correct items
        self.assertIsInstance(result.data_sources, list,
                             "data_sources was not properly converted to a list")
        self.assertEqual(len(result.data_sources), 2,
                        "data_sources list doesn't have the expected number of items")
        
        # Verify first data source
        self.assertEqual(result.data_sources[0].data_source_name, "SOURCE1",
                        "First data source name not correctly deserialized")
        self.assertEqual(result.data_sources[0].data_source_type, "MDS",
                        "First data source type not correctly deserialized")
        
        # Verify second data source
        self.assertEqual(result.data_sources[1].data_source_name, "SOURCE2",
                        "Second data source name not correctly deserialized")
        self.assertEqual(result.data_sources[1].data_source_type, "EDX",
                        "Second data source type not correctly deserialized")
        
        logger.info("Successfully verified list items are properly distinguished")
        
    def test_nested_complex_structure(self):
        """
        Test a more complex nested structure with multiple layers, similar to the real config.
        
        This test verifies that our enhanced detection can handle complex nested structures
        with multiple levels of nesting without false positives.
        """
        # Create a serializer
        serializer = TypeAwareConfigSerializer()
        
        # Create a test config JSON with CradleDataLoadConfig containing DataSourcesSpecificationConfig
        # which contains multiple DataSourceConfig objects in a list
        config_json = {
            "__model_type__": "CradleDataLoadConfig",
            "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
            "job_type": "training",
            "data_sources_spec": {
                "__model_type__": "DataSourcesSpecificationConfig",
                "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-04-17T00:00:00",
                "data_sources": {
                    "__type_info__": "list",
                    "value": [
                        {
                            "__model_type__": "DataSourceConfig",
                            "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
                            "data_source_name": "RAW_MDS_NA",
                            "data_source_type": "MDS"
                        },
                        {
                            "__model_type__": "DataSourceConfig",
                            "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
                            "data_source_name": "TAGS",
                            "data_source_type": "EDX"
                        }
                    ]
                }
            }
        }
        
        # Create a special test logger to track warnings
        test_handler = TestLogHandler()
        serializer.logger.addHandler(test_handler)
        
        # Attempt to deserialize with improved circular reference detection
        logger.info("Deserializing complex nested structure")
        result = serializer.deserialize(config_json)
        
        # Check that no circular reference warnings were produced
        circular_refs = [log for log in test_handler.logs if "Circular reference detected" in log]
        self.assertEqual(len(circular_refs), 0, 
                         f"Detected {len(circular_refs)} false positive circular references: {circular_refs}")
        
        # Verify result is properly deserialized
        self.assertIsInstance(result, CradleDataLoadConfig, 
                              "Failed to deserialize into CradleDataLoadConfig instance")
        
        # Verify nested data_sources_spec
        self.assertIsInstance(result.data_sources_spec, DataSourcesSpecificationConfig,
                             "data_sources_spec was not properly deserialized")
        
        # Verify data_sources list
        self.assertIsInstance(result.data_sources_spec.data_sources, list,
                             "data_sources was not properly converted to a list")
        self.assertEqual(len(result.data_sources_spec.data_sources), 2,
                        "data_sources list doesn't have the expected number of items")
        
        # Verify both data sources
        ds1 = result.data_sources_spec.data_sources[0]
        ds2 = result.data_sources_spec.data_sources[1]
        
        self.assertEqual(ds1.data_source_name, "RAW_MDS_NA")
        self.assertEqual(ds1.data_source_type, "MDS")
        self.assertEqual(ds2.data_source_name, "TAGS")
        self.assertEqual(ds2.data_source_type, "EDX")
        
        logger.info("Successfully verified complex nested structure deserialization")
    
    def test_true_circular_references_still_detected(self):
        """
        Test that true circular references are still properly detected.
        
        This test verifies that our enhanced detection still correctly identifies
        actual circular references while avoiding false positives.
        """
        # Create a serializer with our improved detection
        serializer = TypeAwareConfigSerializer()
        
        # Create a complex object with a true circular reference
        # We'll create an object that references itself directly
        config_with_circular_ref = {
            "__model_type__": "CradleDataLoadConfig",
            "__model_module__": "test.config_field_manager.test_fixed_circular_detection",
            "job_type": "training"
        }
        
        # Add a circular reference to itself
        config_with_circular_ref["circular_ref"] = config_with_circular_ref
        
        # Create a special test logger to track warnings
        test_handler = TestLogHandler()
        serializer.logger.addHandler(test_handler)
        
        # Attempt to deserialize - this should detect the circular reference
        result = serializer.deserialize(config_with_circular_ref)
        
        # Check that circular reference warnings or depth errors were produced
        circular_refs = [log for log in test_handler.logs if 
                        "Circular reference detected" in log or 
                        "Maximum recursion depth" in log]
        self.assertGreater(len(circular_refs), 0, 
                         "No circular reference warnings or max depth errors detected for a true circular reference")
        
        logger.info("Successfully verified true circular references are still detected")


class TestLogHandler(logging.Handler):
    """A custom log handler that tracks log messages."""
    
    def __init__(self):
        super().__init__()
        self.logs = []
        
    def emit(self, record):
        self.logs.append(record.getMessage())


if __name__ == "__main__":
    unittest.main()
