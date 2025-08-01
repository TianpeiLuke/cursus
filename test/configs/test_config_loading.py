#!/usr/bin/env python3
"""
Test script to verify that the config loading issues have been fixed.
"""

import sys
import os
import logging
import json
import tempfile
from pathlib import Path
import unittest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Import the necessary modules
from src.pipeline_steps.utils import load_configs, build_complete_config_classes

class TestConfigLoading(unittest.TestCase):
    """Tests for the configuration loading functionality."""
    
    def setUp(self):
        """Set up test environment with valid local paths."""
        # Get the project root path
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Source directory to use for testing
        self.source_dir = str(self.project_root / "src" / "pipeline_scripts")
        self.processing_script_dir = str(self.project_root / "src" / "pipeline_scripts")
        
        # Create a temporary directory for test configs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_config_path = Path(self.temp_dir.name) / "test_config.json"
        
        # Create test config with valid paths
        self.create_test_config()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def create_test_config(self):
        """Create a test configuration file with valid local paths."""
        # Create a minimal config structure
        config = {
            "configuration": {
                "shared": {
                    "bucket": "test-bucket",
                    "author": "test-author",
                    "region": "NA",
                    "pipeline_name": "test-pipeline",
                    "pipeline_description": "Test pipeline description",
                    "pipeline_version": "0.1.0",
                    "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
                    "source_dir": self.source_dir
                },
                "specific": {
                    "Payload": {
                        "processing_source_dir": self.processing_script_dir,
                        "processing_entry_point": "mims_payload.py",
                        "model_registration_objective": "test-objective"
                    },
                    "CradleDataLoad_calibration": {
                        "job_type": "calibration",
                        "data_sources_spec": {
                            "start_date": "2025-01-01T00:00:00",
                            "end_date": "2025-01-02T00:00:00",
                            "data_sources": []
                        },
                        "transform_spec": {
                            "transform_sql": "SELECT * FROM data",
                            "job_split_options": {
                                "split_job": False
                            }
                        },
                        "output_spec": {
                            "output_schema": ["col1", "col2"],
                            "output_path": "s3://test-bucket/output"
                        },
                        "cradle_job_spec": {
                            "cradle_account": "test-account"
                        }
                    },
                    "CradleDataLoad_training": {
                        "job_type": "training",
                        "data_sources_spec": {
                            "start_date": "2025-01-01T00:00:00",
                            "end_date": "2025-01-02T00:00:00",
                            "data_sources": []
                        },
                        "transform_spec": {
                            "transform_sql": "SELECT * FROM data",
                            "job_split_options": {
                                "split_job": False
                            }
                        },
                        "output_spec": {
                            "output_schema": ["col1", "col2"],
                            "output_path": "s3://test-bucket/output"
                        },
                        "cradle_job_spec": {
                            "cradle_account": "test-account"
                        }
                    },
                    "ModelRegistration": {
                        "model_registration_objective": "test-objective",
                        "source_dir": self.source_dir
                    },
                    "PackageStep": {
                        "processing_source_dir": self.processing_script_dir,
                        "processing_entry_point": "package_dummy.py"
                    }
                }
            },
            "metadata": {
                "step_names": {
                    "CradleDataLoad_calibration": "CradleDataLoad",
                    "CradleDataLoad_training": "CradleDataLoad",
                    "ModelRegistration": "ModelRegistration",
                    "Payload": "Payload",
                    "PackageStep": "PackageStep"
                },
                "config_types": {
                    "CradleDataLoad_calibration": "CradleDataLoadConfig",
                    "CradleDataLoad_training": "CradleDataLoadConfig",
                    "ModelRegistration": "ModelRegistrationConfig",
                    "Payload": "PayloadConfig",
                    "PackageStep": "PackageStepConfig"
                }
            }
        }
        
        # Write the config to the temporary file
        with open(self.test_config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Created test config at {self.test_config_path}")
    
    def test_load_configs_with_recursion_detection(self):
        """Test that configs can be loaded with recursion detection in place."""
        # Build the complete list of config classes
        CONFIG_CLASSES = build_complete_config_classes()
        
        # Log the number of config classes
        logger.info(f"Loading configs from {self.test_config_path}")
        logger.info(f"Number of config classes: {len(CONFIG_CLASSES)}")
        
        # Load the configs
        loaded_configs = load_configs(str(self.test_config_path), CONFIG_CLASSES)
        
        # Log success and number of configs loaded
        logger.info(f"Successfully loaded {len(loaded_configs)} configs")
        
        # Log the loaded config names
        logger.info("Loaded configs:")
        for name, config in loaded_configs.items():
            logger.info(f"  {name}: {config.__class__.__name__}")
            
        # Check that we have at least some configs
        self.assertGreater(len(loaded_configs), 0, "Should load at least some configurations")
        
        # Check for Payload config that previously had recursion issues
        self.assertIn("Payload", loaded_configs, "Payload config should be loaded successfully")
        
        # Check for expected configs
        expected_configs = ["CradleDataLoad_calibration", "CradleDataLoad_training", 
                          "ModelRegistration", "PackageStep"]
        
        # Check that expected configs are there
        for expected in expected_configs:
            self.assertIn(expected, loaded_configs, f"{expected} config should be loaded")
        

if __name__ == "__main__":
    unittest.main()
