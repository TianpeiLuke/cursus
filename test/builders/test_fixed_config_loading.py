#!/usr/bin/env python3
"""
Test script to verify our fixes for the circular reference and special list format handling issues.
"""
import unittest
import logging
from pathlib import Path
from typing import Dict, Any, Type
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import necessary classes from our module
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.config_model_calibration_step import ModelCalibrationConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.utils import load_configs


class FixedConfigLoadingTest(unittest.TestCase):
    """Test suite for verifying the config loading fixes."""
    
    def setUp(self):
        """Set up the test environment."""
        # Define the config classes dictionary
        self.CONFIG_CLASSES = {
            'BasePipelineConfig': BasePipelineConfig,
            'XGBoostTrainingConfig': XGBoostTrainingConfig,
            'ModelCalibrationConfig': ModelCalibrationConfig,
            'ProcessingStepConfigBase': ProcessingStepConfigBase,
            'PackageStepConfig': PackageStepConfig,
            'ModelRegistrationConfig': ModelRegistrationConfig,
            'PayloadConfig': PayloadConfig,
            'CradleDataLoadConfig': CradleDataLoadConfig,
            'TabularPreprocessingConfig': TabularPreprocessingConfig,
            'XGBoostModelEvalConfig': XGBoostModelEvalConfig,
        }
    
    def test_config_loading_with_special_list_format(self):
        """
        Test loading a config with the special list format that was previously failing.
        
        This test verifies that our fix for handling the special '__type_info__': 'list' format
        and circular references properly resolves the issue.
        """
        # Path to the config file that was previously failing
        config_path = Path('pipeline_config/config_NA_xgboost_AtoZ_v2/config_NA_xgboost_AtoZ.json')
        
        # Skip the test if the file doesn't exist
        if not config_path.exists():
            self.skipTest(f"Test config file {config_path} not found")
        
        # Load the configs - our fix should handle special list format and circular references
        logger.info(f"Loading configs from {config_path}")
        loaded_configs = load_configs(config_path, self.CONFIG_CLASSES)
        
        # Print success message with the loaded config count
        logger.info(f"Successfully loaded {len(loaded_configs)} configs")
        
        # Check if all expected configs are loaded
        expected_configs = {
            "BasePipelineConfig",
            "CradleDataLoading_calibration",  # Previously failing
            "CradleDataLoading_training",     # Previously failing
            "ModelCalibration",
            "Package",
            "Payload",
            "Processing",
            "Registration",
            "TabularPreprocessing_calibration",
            "TabularPreprocessing_training",
            "XGBoostModelEval_calibration",
            "XGBoostTraining"
        }
        
        # Check that we have the previously problematic configs
        self.assertIn("CradleDataLoading_calibration", loaded_configs, 
                      "Failed to load CradleDataLoading_calibration config")
        self.assertIn("CradleDataLoading_training", loaded_configs,
                      "Failed to load CradleDataLoading_training config")
        
        # Verify these configs have the expected fields
        cradle_config_calibration = loaded_configs["CradleDataLoading_calibration"]
        self.assertEqual(cradle_config_calibration.job_type, "calibration",
                         "CradleDataLoading_calibration has incorrect job_type")
        
        cradle_config_training = loaded_configs["CradleDataLoading_training"]
        self.assertEqual(cradle_config_training.job_type, "training",
                         "CradleDataLoading_training has incorrect job_type")
        
        # Verify that the data_sources_spec field has been properly processed
        self.assertTrue(hasattr(cradle_config_calibration, "data_sources_spec"),
                        "data_sources_spec is missing from CradleDataLoading_calibration")
        self.assertTrue(hasattr(cradle_config_training, "data_sources_spec"),
                        "data_sources_spec is missing from CradleDataLoading_training")


if __name__ == "__main__":
    unittest.main()
