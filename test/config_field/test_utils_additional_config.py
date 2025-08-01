#!/usr/bin/env python3
"""
Test script to verify our fixes for creating additional config instances
in src/pipeline_steps/utils.py with special list format handling.
"""
import unittest
import logging
import json
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
import tempfile

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
from src.pipeline_steps.utils import load_configs


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


class UtilsAdditionalConfigTest(unittest.TestCase):
    """Test that our fix for creating additional config instances works correctly."""
    
    def setUp(self):
        """Set up test environment."""
        # Register our mock models
        ConfigClassStore.register(DataSourceConfig)
        ConfigClassStore.register(DataSourcesSpecificationConfig)
        ConfigClassStore.register(CradleDataLoadConfig)
    
    def test_additional_config_with_special_list(self):
        """Test that our fix properly handles creating additional configs with special list format."""
        # Create config classes dict
        config_classes = {
            "CradleDataLoadConfig": CradleDataLoadConfig,
            "DataSourcesSpecificationConfig": DataSourcesSpecificationConfig,
            "DataSourceConfig": DataSourceConfig
        }
        
        # Create a mock config JSON file with metadata and both shared and specific sections
        config_json = {
            "metadata": {
                "config_types": {
                    "CradleDataLoading_training": "CradleDataLoadConfig",
                    "CradleDataLoading_calibration": "CradleDataLoadConfig"
                }
            },
            "configuration": {
                "shared": {
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
                },
                "specific": {
                    "CradleDataLoading_training": {
                        "job_type": "training"
                    },
                    "CradleDataLoading_calibration": {
                        "job_type": "calibration"
                    }
                }
            }
        }
        
        # Create a temporary file to load from
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(config_json, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Now use our fixed load_configs function to load this configuration
            logger.info(f"Loading configs from temporary file {tmp_path}")
            loaded_configs = load_configs(tmp_path, config_classes)
            
            # Verify we got both config instances
            self.assertIn("CradleDataLoading_training", loaded_configs, 
                         "Missing CradleDataLoading_training in loaded configs")
            self.assertIn("CradleDataLoading_calibration", loaded_configs,
                         "Missing CradleDataLoading_calibration in loaded configs")
            
            # Verify the training config
            training_config = loaded_configs["CradleDataLoading_training"]
            self.assertEqual(training_config.job_type, "training",
                            "job_type not correctly set for training config")
            
            # Verify the calibration config
            calibration_config = loaded_configs["CradleDataLoading_calibration"]
            self.assertEqual(calibration_config.job_type, "calibration",
                            "job_type not correctly set for calibration config")
            
            # Verify that data_sources_spec was properly deserialized in both configs
            for config_name, config in loaded_configs.items():
                # Verify data_sources_spec
                self.assertTrue(hasattr(config, "data_sources_spec"),
                              f"data_sources_spec missing from {config_name}")
                self.assertIsInstance(config.data_sources_spec, DataSourcesSpecificationConfig,
                                   f"data_sources_spec wrong type in {config_name}")
                
                # Verify data_sources list (special format correctly processed)
                self.assertTrue(hasattr(config.data_sources_spec, "data_sources"),
                              f"data_sources missing from {config_name}")
                self.assertIsInstance(config.data_sources_spec.data_sources, list,
                                   f"data_sources not properly converted to list in {config_name}")
                self.assertEqual(len(config.data_sources_spec.data_sources), 2,
                               f"data_sources wrong length in {config_name}")
            
            logger.info("Successfully loaded and validated all additional config instances")
            
        finally:
            # Clean up the temporary file
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
