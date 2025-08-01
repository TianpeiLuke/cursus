import unittest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
import os

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_builder.template_pipeline_xgboost_dataload_preprocess import (
    create_pipeline_from_template,
    BUILDER_MAP,
    _find_config_key
)
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_builder.pipeline_dag import PipelineDAG
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate


class TestXGBoostDataloadPreprocessTemplateBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        
        self.mock_cradle_train_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_train_cfg.job_type = "training"
        
        self.mock_cradle_test_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_test_cfg.job_type = "testing"
        
        self.mock_tp_train_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_train_cfg.job_type = "training"
        self.mock_tp_train_cfg.input_names = {
            "data_input": "training_data",
            "metadata_input": "training_metadata"
        }
        self.mock_tp_train_cfg.output_names = {
            "processed_data": "processed_training_data"
        }
        self.mock_tp_train_cfg.get_input_names = MagicMock(return_value=self.mock_tp_train_cfg.input_names)
        self.mock_tp_train_cfg.get_output_names = MagicMock(return_value=self.mock_tp_train_cfg.output_names)
        
        self.mock_tp_test_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_test_cfg.job_type = "testing"
        self.mock_tp_test_cfg.input_names = {
            "data_input": "testing_data",
            "metadata_input": "testing_metadata"
        }
        self.mock_tp_test_cfg.output_names = {
            "processed_data": "processed_testing_data"
        }
        self.mock_tp_test_cfg.get_input_names = MagicMock(return_value=self.mock_tp_test_cfg.input_names)
        self.mock_tp_test_cfg.get_output_names = MagicMock(return_value=self.mock_tp_test_cfg.output_names)
        
        # Mock configs dictionary
        self.mock_configs = {
            'Base': self.mock_base_config,
            'CradleDataLoading_training': self.mock_cradle_train_cfg,
            'CradleDataLoading_testing': self.mock_cradle_test_cfg,
            'TabularPreprocessing_training': self.mock_tp_train_cfg,
            'TabularPreprocessing_testing': self.mock_tp_test_cfg
        }
        
        # Patch load_configs
        self.load_configs_patch = patch('src.pipeline_builder.template_pipeline_xgboost_dataload_preprocess.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        self.mock_load_configs.return_value = self.mock_configs
        
        # Patch PipelineBuilderTemplate
        self.template_patch = patch('src.pipeline_builder.template_pipeline_xgboost_dataload_preprocess.PipelineBuilderTemplate')
        self.mock_template_cls = self.template_patch.start()
        self.mock_template = MagicMock()
        self.mock_template_cls.return_value = self.mock_template
        self.mock_template.generate_pipeline.return_value = MagicMock(name="pipeline")
        
        # Patch PipelineDAG
        self.dag_patch = patch('src.pipeline_builder.template_pipeline_xgboost_dataload_preprocess.PipelineDAG')
        self.mock_dag_cls = self.dag_patch.start()
        self.mock_dag = MagicMock()
        self.mock_dag_cls.return_value = self.mock_dag
        
        # Patch isinstance to return True for our mocks
        self.original_isinstance = isinstance
        
        def patched_isinstance(obj, classinfo):
            if obj is self.mock_cradle_train_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_cradle_test_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_tp_train_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if obj is self.mock_tp_test_cfg and classinfo is TabularPreprocessingConfig:
                return True
            return self.original_isinstance(obj, classinfo)
        
        self.builtins_patch = patch('builtins.isinstance', patched_isinstance)
        self.builtins_patch.start()

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.template_patch.stop()
        self.dag_patch.stop()
        self.builtins_patch.stop()

    def test_find_config_key(self):
        """Test that _find_config_key returns the correct key."""
        # Test finding a config key with job_type=training
        key = _find_config_key(self.mock_configs, 'CradleDataLoadConfig', job_type='training')
        self.assertEqual(key, 'CradleDataLoading_training')
        
        # Test finding a config key with job_type=testing
        key = _find_config_key(self.mock_configs, 'CradleDataLoadConfig', job_type='testing')
        self.assertEqual(key, 'CradleDataLoading_testing')
        
        # Test error when no matching config is found
        with self.assertRaises(ValueError):
            _find_config_key(self.mock_configs, 'CradleDataLoadConfig', job_type='nonexistent')
        
        # Test error when multiple matching configs are found
        test_configs = {
            'CradleDataLoading_training_1': self.mock_cradle_train_cfg,
            'CradleDataLoading_training_2': self.mock_cradle_train_cfg
        }
        with self.assertRaises(ValueError):
            _find_config_key(test_configs, 'CradleDataLoadConfig', job_type='training')

    def test_create_pipeline_from_template(self):
        """Test that create_pipeline_from_template creates a pipeline correctly."""
        # Call the function
        pipeline = create_pipeline_from_template(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            notebook_root=Path("/dummy/notebook/root")
        )
        
        # Verify load_configs was called with the correct parameters
        self.mock_load_configs.assert_called_once()
        
        # Verify PipelineDAG was created
        self.mock_dag_cls.assert_called_once()
        
        # Verify PipelineBuilderTemplate was created with the right parameters
        self.mock_template_cls.assert_called_once()
        
        # Verify generate_pipeline was called with the right name
        self.mock_template.generate_pipeline.assert_called_once_with(
            "test-pipeline-loadprep"
        )
        
        # Verify the pipeline was returned
        self.assertEqual(pipeline, self.mock_template.generate_pipeline.return_value)

    def test_builder_map(self):
        """Test that BUILDER_MAP contains the expected step types."""
        self.assertIn("CradleDataLoadingStep", BUILDER_MAP)
        self.assertIn("TabularPreprocessingStep", BUILDER_MAP)


if __name__ == '__main__':
    unittest.main()
