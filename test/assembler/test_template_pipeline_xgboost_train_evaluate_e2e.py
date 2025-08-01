import unittest
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path
import os

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_builder.template_pipeline_xgboost_train_evaluate_e2e import (
    XGBoostTrainEvaluateE2ETemplateBuilder,
    CONFIG_CLASSES
)
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig
from src.pipeline_builder.pipeline_dag import PipelineDAG
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate


class TestXGBoostTrainEvaluateE2ETemplateBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        
        self.mock_cradle_train_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_train_cfg.job_type = "training"
        
        self.mock_cradle_calib_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_calib_cfg.job_type = "calibration"
        
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
        
        self.mock_tp_calib_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_calib_cfg.job_type = "calibration"
        self.mock_tp_calib_cfg.input_names = {
            "data_input": "calibration_data",
            "metadata_input": "calibration_metadata"
        }
        self.mock_tp_calib_cfg.output_names = {
            "processed_data": "processed_calibration_data"
        }
        self.mock_tp_calib_cfg.get_input_names = MagicMock(return_value=self.mock_tp_calib_cfg.input_names)
        self.mock_tp_calib_cfg.get_output_names = MagicMock(return_value=self.mock_tp_calib_cfg.output_names)
        
        self.mock_xgb_train_cfg = MagicMock(spec=XGBoostTrainingConfig)
        
        self.mock_xgb_eval_cfg = MagicMock(spec=XGBoostModelEvalConfig)
        self.mock_xgb_eval_cfg.input_names = {
            "model_input": "model_input",
            "eval_data_input": "eval_data_input"
        }
        self.mock_xgb_eval_cfg.output_names = {
            "eval_output": "eval_output",
            "metrics_output": "metrics_output"
        }
        self.mock_xgb_eval_cfg.get_input_names = MagicMock(return_value=self.mock_xgb_eval_cfg.input_names)
        self.mock_xgb_eval_cfg.get_output_names = MagicMock(return_value=self.mock_xgb_eval_cfg.output_names)
        
        self.mock_package_cfg = MagicMock(spec=PackageStepConfig)
        
        self.mock_registration_cfg = MagicMock(spec=ModelRegistrationConfig)
        self.mock_registration_cfg.region = "us-west-2"
        
        self.mock_payload_cfg = MagicMock(spec=PayloadConfig)
        
        # Mock configs dictionary
        self.mock_configs = {
            'Base': self.mock_base_config,
            'CradleDataLoading_training': self.mock_cradle_train_cfg,
            'CradleDataLoading_calibration': self.mock_cradle_calib_cfg,
            'TabularPreprocessing_training': self.mock_tp_train_cfg,
            'TabularPreprocessing_calibration': self.mock_tp_calib_cfg,
            'XGBoostTraining': self.mock_xgb_train_cfg,
            'XGBoostModelEval': self.mock_xgb_eval_cfg,
            'Package': self.mock_package_cfg,
            'Registration': self.mock_registration_cfg,
            'Payload': self.mock_payload_cfg
        }
        
        # Patch load_configs
        self.load_configs_patch = patch('src.pipeline_builder.template_pipeline_xgboost_train_evaluate_e2e.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        self.mock_load_configs.return_value = self.mock_configs
        
        # Patch PipelineBuilderTemplate
        self.template_patch = patch('src.pipeline_builder.template_pipeline_xgboost_train_evaluate_e2e.PipelineBuilderTemplate')
        self.mock_template_cls = self.template_patch.start()
        self.mock_template = MagicMock()
        self.mock_template_cls.return_value = self.mock_template
        self.mock_template.generate_pipeline.return_value = MagicMock(name="pipeline")
        
        # Patch PipelineDAG
        self.dag_patch = patch('src.pipeline_builder.template_pipeline_xgboost_train_evaluate_e2e.PipelineDAG')
        self.mock_dag_cls = self.dag_patch.start()
        self.mock_dag = MagicMock()
        self.mock_dag_cls.return_value = self.mock_dag
        
        # Patch isinstance to return True for our mocks
        self.original_isinstance = isinstance
        
        def patched_isinstance(obj, classinfo):
            if obj is self.mock_cradle_train_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_cradle_calib_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_tp_train_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if obj is self.mock_tp_calib_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if obj is self.mock_xgb_train_cfg and classinfo is XGBoostTrainingConfig:
                return True
            if obj is self.mock_xgb_eval_cfg and classinfo is XGBoostModelEvalConfig:
                return True
            if obj is self.mock_package_cfg and classinfo is PackageStepConfig:
                return True
            if obj is self.mock_registration_cfg and classinfo is ModelRegistrationConfig:
                return True
            if obj is self.mock_payload_cfg and classinfo is PayloadConfig:
                return True
            return self.original_isinstance(obj, classinfo)
        
        self.builtins_patch = patch('builtins.isinstance', patched_isinstance)
        self.builtins_patch.start()
        
        # Create the builder instance
        self.builder = XGBoostTrainEvaluateE2ETemplateBuilder(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            notebook_root=Path("/dummy/notebook/root")
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.template_patch.stop()
        self.dag_patch.stop()
        self.builtins_patch.stop()

    def test_initialization(self):
        """Test that the builder initializes correctly."""
        # Verify load_configs was called with the correct parameters
        self.mock_load_configs.assert_called_once_with(
            "dummy/path/to/config.json", 
            CONFIG_CLASSES
        )
        
        # Verify configs were extracted correctly
        self.assertEqual(self.builder.base_config, self.mock_base_config)
        self.assertEqual(self.builder.cradle_train_cfg, self.mock_cradle_train_cfg)
        self.assertEqual(self.builder.cradle_calib_cfg, self.mock_cradle_calib_cfg)
        self.assertEqual(self.builder.tp_train_cfg, self.mock_tp_train_cfg)
        self.assertEqual(self.builder.tp_calib_cfg, self.mock_tp_calib_cfg)
        self.assertEqual(self.builder.xgb_train_cfg, self.mock_xgb_train_cfg)
        self.assertEqual(self.builder.xgb_eval_cfg, self.mock_xgb_eval_cfg)
        self.assertEqual(self.builder.package_cfg, self.mock_package_cfg)
        self.assertEqual(self.builder.registration_cfg, self.mock_registration_cfg)
        self.assertEqual(self.builder.payload_cfg, self.mock_payload_cfg)

    def test_get_pipeline_parameters(self):
        """Test that _get_pipeline_parameters returns the correct parameters."""
        params = self.builder._get_pipeline_parameters()
        
        # Verify the parameters list contains the expected parameters
        self.assertEqual(len(params), 4)
        param_names = [p.name for p in params]
        self.assertIn("PipelineExecutionTempDir", param_names)
        self.assertIn("KMSEncryptionKey", param_names)
        self.assertIn("SecurityGroupId", param_names)
        self.assertIn("VPCEndpointSubnet", param_names)

    def test_create_step_builder_map(self):
        """Test that _create_step_builder_map returns the correct mapping."""
        builder_map = self.builder._create_step_builder_map()
        
        # Verify the builder map contains the expected step types
        self.assertIn("CradleDataLoadStep", builder_map)
        self.assertIn("TabularPreprocessingStep", builder_map)
        self.assertIn("XGBoostTrainingStep", builder_map)
        self.assertIn("PackagingStep", builder_map)
        self.assertIn("RegistrationStep", builder_map)
        self.assertIn("ModelEvalStep", builder_map)

    def test_create_config_map(self):
        """Test that _create_config_map returns the correct mapping."""
        config_map = self.builder._create_config_map()
        
        # Verify the config map contains the expected step names
        self.assertIn("train_data_load", config_map)
        self.assertIn("train_preprocess", config_map)
        self.assertIn("xgboost_train", config_map)
        self.assertIn("model_packaging", config_map)
        self.assertIn("model_registration", config_map)
        self.assertIn("calib_data_load", config_map)
        self.assertIn("calib_preprocess", config_map)
        self.assertIn("model_evaluation", config_map)
        
        # Verify the config map contains the correct config instances
        self.assertEqual(config_map["train_data_load"], self.mock_cradle_train_cfg)
        self.assertEqual(config_map["train_preprocess"], self.mock_tp_train_cfg)
        self.assertEqual(config_map["xgboost_train"], self.mock_xgb_train_cfg)
        self.assertEqual(config_map["model_packaging"], self.mock_package_cfg)
        self.assertEqual(config_map["model_registration"], self.mock_registration_cfg)
        self.assertEqual(config_map["calib_data_load"], self.mock_cradle_calib_cfg)
        self.assertEqual(config_map["calib_preprocess"], self.mock_tp_calib_cfg)
        self.assertEqual(config_map["model_evaluation"], self.mock_xgb_eval_cfg)

    def test_create_pipeline_dag(self):
        """Test that _create_pipeline_dag creates the correct DAG structure."""
        dag = self.builder._create_pipeline_dag()
        
        # Verify the DAG was created
        self.mock_dag_cls.assert_called_once()
        
        # Verify nodes were added
        self.mock_dag.add_node.assert_any_call("train_data_load")
        self.mock_dag.add_node.assert_any_call("train_preprocess")
        self.mock_dag.add_node.assert_any_call("xgboost_train")
        self.mock_dag.add_node.assert_any_call("model_packaging")
        self.mock_dag.add_node.assert_any_call("payload_test")
        self.mock_dag.add_node.assert_any_call("model_registration")
        self.mock_dag.add_node.assert_any_call("calib_data_load")
        self.mock_dag.add_node.assert_any_call("calib_preprocess")
        self.mock_dag.add_node.assert_any_call("model_evaluation")
        
        # Verify edges were added
        self.mock_dag.add_edge.assert_any_call("train_data_load", "train_preprocess")
        self.mock_dag.add_edge.assert_any_call("train_preprocess", "xgboost_train")
        self.mock_dag.add_edge.assert_any_call("xgboost_train", "model_packaging")
        self.mock_dag.add_edge.assert_any_call("model_packaging", "payload_test")
        self.mock_dag.add_edge.assert_any_call("payload_test", "model_registration")
        self.mock_dag.add_edge.assert_any_call("calib_data_load", "calib_preprocess")
        self.mock_dag.add_edge.assert_any_call("xgboost_train", "model_evaluation")
        self.mock_dag.add_edge.assert_any_call("calib_preprocess", "model_evaluation")
        
        # Verify the DAG was returned
        self.assertEqual(dag, self.mock_dag)

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline correctly."""
        # Mock the internal methods
        with patch.object(self.builder, '_create_pipeline_dag') as mock_create_dag:
            with patch.object(self.builder, '_create_config_map') as mock_create_config_map:
                with patch.object(self.builder, '_create_step_builder_map') as mock_create_builder_map:
                    with patch.object(self.builder, '_get_pipeline_parameters') as mock_get_params:
                        # Set up mock return values
                        mock_create_dag.return_value = self.mock_dag
                        mock_create_config_map.return_value = {"step1": MagicMock()}
                        mock_create_builder_map.return_value = {"StepType1": MagicMock()}
                        mock_get_params.return_value = [MagicMock()]
                        
                        # Call the method
                        pipeline = self.builder.generate_pipeline()
                        
                        # Verify the internal methods were called
                        mock_create_dag.assert_called_once()
                        mock_create_config_map.assert_called_once()
                        mock_create_builder_map.assert_called_once()
                        mock_get_params.assert_called_once()
                        
                        # Verify PipelineBuilderTemplate was created with the right parameters
                        self.mock_template_cls.assert_called_once_with(
                            dag=self.mock_dag,
                            config_map=mock_create_config_map.return_value,
                            step_builder_map=mock_create_builder_map.return_value,
                            sagemaker_session=self.builder.session,
                            role=self.builder.role,
                            pipeline_parameters=mock_get_params.return_value,
                            notebook_root=self.builder.notebook_root
                        )
                        
                        # Verify generate_pipeline was called with the right name
                        self.mock_template.generate_pipeline.assert_called_once_with(
                            "test-pipeline-xgb-train-eval"
                        )
                        
                        # Verify the pipeline was returned
                        self.assertEqual(pipeline, self.mock_template.generate_pipeline.return_value)

    def test_fill_execution_document(self):
        """Test that fill_execution_document updates the execution document correctly."""
        # Set up execution document
        execution_doc = {
            "PIPELINE_STEP_CONFIGS": {
                "cradle_step": {},
                "Registration_us-west-2": {}
            }
        }
        
        # Set up cradle loading requests and registration configs
        self.builder.cradle_loading_requests = {
            "cradle_step": {"key": "value"}
        }
        self.builder.registration_configs = {
            "registration_step": {"key": "value"}
        }
        
        # Call the method
        result = self.builder.fill_execution_document(execution_doc)
        
        # Verify the execution document was updated
        self.assertEqual(
            result["PIPELINE_STEP_CONFIGS"]["cradle_step"]["STEP_CONFIG"],
            {"key": "value"}
        )
        self.assertEqual(
            result["PIPELINE_STEP_CONFIGS"]["Registration_us-west-2"]["STEP_CONFIG"],
            {"key": "value"}
        )

    def test_fill_execution_document_missing_key(self):
        """Test that fill_execution_document raises KeyError for missing key."""
        # Set up execution document without required key
        execution_doc = {}
        
        # Verify KeyError is raised
        with self.assertRaises(KeyError):
            self.builder.fill_execution_document(execution_doc)

    def test_fill_execution_document_missing_steps(self):
        """Test that fill_execution_document handles missing steps gracefully."""
        # Set up execution document without the steps
        execution_doc = {
            "PIPELINE_STEP_CONFIGS": {}
        }
        
        # Set up cradle loading requests and registration configs
        self.builder.cradle_loading_requests = {
            "cradle_step": {"key": "value"}
        }
        self.builder.registration_configs = {
            "registration_step": {"key": "value"}
        }
        
        # Call the method
        result = self.builder.fill_execution_document(execution_doc)
        
        # Verify the execution document was not modified
        self.assertEqual(result["PIPELINE_STEP_CONFIGS"], {})


if __name__ == '__main__':
    unittest.main()
