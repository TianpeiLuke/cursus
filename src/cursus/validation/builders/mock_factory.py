"""
Mock factory system for creating step type-specific mock objects.
"""

from typing import Dict, Any, Optional, Type, List
from types import SimpleNamespace
from unittest.mock import MagicMock
from ...core.base.builder_base import StepBuilderBase


class StepTypeMockFactory:
    """Factory for creating step type-specific mock objects."""
    
    def __init__(self, step_info: Dict[str, Any]):
        """
        Initialize factory with step information.
        
        Args:
            step_info: Step information from StepInfoDetector
        """
        self.step_info = step_info
        self.sagemaker_step_type = step_info.get("sagemaker_step_type")
        self.framework = step_info.get("framework")
        self.test_pattern = step_info.get("test_pattern")
    
    def create_mock_config(self) -> SimpleNamespace:
        """Create appropriate mock config for the step type."""
        # Start with base config
        mock_config = self._create_base_config()
        
        # Add step type-specific configuration
        if self.sagemaker_step_type == "Processing":
            self._add_processing_config(mock_config)
        elif self.sagemaker_step_type == "Training":
            self._add_training_config(mock_config)
        elif self.sagemaker_step_type == "Transform":
            self._add_transform_config(mock_config)
        elif self.sagemaker_step_type == "CreateModel":
            self._add_createmodel_config(mock_config)
        else:
            self._add_generic_config(mock_config)
        
        # Add framework-specific configuration
        self._add_framework_config(mock_config)
        
        return mock_config
    
    def create_step_type_mocks(self) -> Dict[str, Any]:
        """Create step type-specific mock objects."""
        mocks = {}
        
        if self.sagemaker_step_type == "Processing":
            mocks.update(self._create_processing_mocks())
        elif self.sagemaker_step_type == "Training":
            mocks.update(self._create_training_mocks())
        elif self.sagemaker_step_type == "Transform":
            mocks.update(self._create_transform_mocks())
        elif self.sagemaker_step_type == "CreateModel":
            mocks.update(self._create_createmodel_mocks())
        
        return mocks
    
    def get_expected_dependencies(self) -> List[str]:
        """Get expected dependencies based on step type and pattern."""
        if self.sagemaker_step_type == "Processing":
            return self._get_processing_dependencies()
        elif self.sagemaker_step_type == "Training":
            return self._get_training_dependencies()
        elif self.sagemaker_step_type == "Transform":
            return self._get_transform_dependencies()
        elif self.sagemaker_step_type == "CreateModel":
            return self._get_createmodel_dependencies()
        else:
            return ["input"]
    
    def _create_base_config(self) -> SimpleNamespace:
        """Create base configuration common to all step types."""
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'
        mock_config.pipeline_name = 'test-pipeline'
        mock_config.pipeline_s3_loc = 's3://bucket/prefix'
        
        # Add common methods
        mock_config.get_image_uri = lambda: 'mock-image-uri'
        mock_config.get_script_path = lambda: 'mock_script.py'
        mock_config.get_script_contract = lambda: None
        
        return mock_config
    
    def _add_processing_config(self, mock_config: SimpleNamespace) -> None:
        """Add Processing step-specific configuration."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_type_large = 'ml.m5.xlarge'
        mock_config.processing_instance_type_small = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'process.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        mock_config.use_large_processing_instance = False
        
        # Add processing-specific attributes based on builder type
        builder_name = self.step_info.get("builder_class_name", "")
        if "TabularPreprocessing" in builder_name:
            mock_config.job_type = 'training'
            mock_config.label_name = 'target'
            mock_config.train_ratio = 0.7
            mock_config.test_val_ratio = 0.5
            mock_config.categorical_columns = ['category_1', 'category_2']
            mock_config.numerical_columns = ['numeric_1', 'numeric_2']
            mock_config.processing_entry_point = 'tabular_preprocess.py'
            mock_config.processing_framework_version = '1.2-1'
        elif "ModelEval" in builder_name:
            mock_config.id_field = 'id'
            mock_config.label_field = 'label'
            mock_config.processing_entry_point = 'model_evaluation_xgb.py'
        
        # Add processor-specific attributes based on framework
        if self.framework == "sklearn":
            mock_config.framework_version = '1.2-1'
            mock_config.py_version = 'py3'
            mock_config.processing_framework_version = '1.2-1'
        elif self.framework == "xgboost":
            mock_config.framework_version = '1.7-1'
            mock_config.py_version = 'py3'
            mock_config.processing_framework_version = '1.7-1'
    
    def _add_training_config(self, mock_config: SimpleNamespace) -> None:
        """Add Training step-specific configuration."""
        mock_config.training_instance_type = 'ml.m5.xlarge'
        mock_config.training_instance_count = 1
        mock_config.training_volume_size = 30
        mock_config.training_entry_point = 'train.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        
        # Add hyperparameters with enhanced mock
        mock_hp = SimpleNamespace()
        mock_hp.model_dump = lambda: {
            'learning_rate': 0.1, 
            'max_depth': 6,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        # Add derived properties for XGBoost
        mock_hp.is_binary = True
        mock_hp.num_classes = 2
        mock_hp.input_tab_dim = 10
        mock_hp.objective = 'binary:logistic'
        mock_hp.eval_metric = 'auc'
        mock_config.hyperparameters = mock_hp
        
        # Add S3 paths for hyperparameters
        mock_config.hyperparameters_s3_uri = 's3://bucket/config/hyperparameters.json'
        mock_config.bucket = 'test-bucket'
        mock_config.current_date = '2025-08-07'
        
        # Add framework-specific training config
        if self.framework == "xgboost":
            mock_config.framework_version = '1.7-1'
            mock_config.py_version = 'py3'
            mock_config.training_entry_point = 'train_xgb.py'
        elif self.framework == "pytorch":
            mock_config.framework_version = '1.12.0'
            mock_config.py_version = 'py39'
            mock_config.training_entry_point = 'train_pytorch.py'
        elif self.framework == "tensorflow":
            mock_config.framework_version = '2.11.0'
            mock_config.py_version = 'py39'
            mock_config.training_entry_point = 'train_tf.py'
    
    def _add_transform_config(self, mock_config: SimpleNamespace) -> None:
        """Add Transform step-specific configuration."""
        mock_config.transform_instance_type = 'ml.m5.large'
        mock_config.transform_instance_count = 1
        mock_config.transform_max_concurrent_transforms = 1
        mock_config.transform_max_payload = 6
        mock_config.transform_accept = 'text/csv'
        mock_config.transform_content_type = 'text/csv'
        mock_config.transform_strategy = 'MultiRecord'
        mock_config.transform_assemble_with = 'Line'
    
    def _add_createmodel_config(self, mock_config: SimpleNamespace) -> None:
        """Add CreateModel step-specific configuration."""
        mock_config.model_name = 'test-model'
        mock_config.primary_container_image = 'mock-image-uri'
        mock_config.model_data_url = 's3://bucket/model.tar.gz'
        mock_config.execution_role_arn = 'arn:aws:iam::123456789012:role/MockRole'
    
    def _add_generic_config(self, mock_config: SimpleNamespace) -> None:
        """Add generic configuration for unknown step types."""
        mock_config.instance_type = 'ml.m5.large'
        mock_config.instance_count = 1
        mock_config.volume_size = 30
        mock_config.entry_point = 'generic_script.py'
        mock_config.source_dir = 'src/pipeline_scripts'
    
    def _add_framework_config(self, mock_config: SimpleNamespace) -> None:
        """Add framework-specific configuration."""
        if self.framework == "xgboost":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '1.7-1'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py3'
        elif self.framework == "pytorch":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '1.12.0'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py39'
        elif self.framework == "tensorflow":
            if not hasattr(mock_config, 'framework_version'):
                mock_config.framework_version = '2.11.0'
            if not hasattr(mock_config, 'py_version'):
                mock_config.py_version = 'py39'
    
    def _create_processing_mocks(self) -> Dict[str, Any]:
        """Create Processing step-specific mocks."""
        mocks = {}
        
        # Mock ProcessingInput
        mock_processing_input = MagicMock()
        mock_processing_input.source = 's3://bucket/input'
        mock_processing_input.destination = '/opt/ml/processing/input'
        mocks['processing_input'] = mock_processing_input
        
        # Mock ProcessingOutput
        mock_processing_output = MagicMock()
        mock_processing_output.source = '/opt/ml/processing/output'
        mock_processing_output.destination = 's3://bucket/output'
        mocks['processing_output'] = mock_processing_output
        
        # Mock Processor based on framework
        if self.framework == "sklearn":
            from sagemaker.sklearn.processing import SKLearnProcessor
            mocks['processor_class'] = SKLearnProcessor
        elif self.framework == "xgboost":
            from sagemaker.xgboost.processing import XGBoostProcessor
            mocks['processor_class'] = XGBoostProcessor
        else:
            from sagemaker.processing import ScriptProcessor
            mocks['processor_class'] = ScriptProcessor
        
        return mocks
    
    def _create_training_mocks(self) -> Dict[str, Any]:
        """Create Training step-specific mocks."""
        mocks = {}
        
        # Mock TrainingInput
        mock_training_input = MagicMock()
        mock_training_input.config = {
            'DataSource': {
                'S3DataSource': {
                    'S3Uri': 's3://bucket/training-data',
                    'S3DataType': 'S3Prefix'
                }
            }
        }
        mocks['training_input'] = mock_training_input
        
        # Mock Estimator based on framework
        if self.framework == "xgboost":
            from sagemaker.xgboost.estimator import XGBoost
            mocks['estimator_class'] = XGBoost
        elif self.framework == "pytorch":
            from sagemaker.pytorch.estimator import PyTorch
            mocks['estimator_class'] = PyTorch
        elif self.framework == "tensorflow":
            from sagemaker.tensorflow.estimator import TensorFlow
            mocks['estimator_class'] = TensorFlow
        else:
            from sagemaker.estimator import Estimator
            mocks['estimator_class'] = Estimator
        
        return mocks
    
    def _create_transform_mocks(self) -> Dict[str, Any]:
        """Create Transform step-specific mocks."""
        mocks = {}
        
        # Mock TransformInput
        mock_transform_input = MagicMock()
        mock_transform_input.data = 's3://bucket/transform-input'
        mock_transform_input.content_type = 'text/csv'
        mocks['transform_input'] = mock_transform_input
        
        # Mock Transformer
        mock_transformer = MagicMock()
        mock_transformer.model_name = 'test-model'
        mocks['transformer'] = mock_transformer
        
        return mocks
    
    def _create_createmodel_mocks(self) -> Dict[str, Any]:
        """Create CreateModel step-specific mocks."""
        mocks = {}
        
        # Mock Model
        mock_model = MagicMock()
        mock_model.name = 'test-model'
        mock_model.image_uri = 'mock-image-uri'
        mock_model.model_data = 's3://bucket/model.tar.gz'
        mocks['model'] = mock_model
        
        return mocks
    
    def _get_processing_dependencies(self) -> List[str]:
        """Get expected dependencies for Processing steps."""
        builder_name = self.step_info.get("builder_class_name", "")
        
        if "TabularPreprocessing" in builder_name:
            return ["DATA"]
        elif "ModelEval" in builder_name:
            return ["model_input", "eval_data_input"]
        else:
            return ["input_data"]
    
    def _get_training_dependencies(self) -> List[str]:
        """Get expected dependencies for Training steps."""
        return ["input_path"]
    
    def _get_transform_dependencies(self) -> List[str]:
        """Get expected dependencies for Transform steps."""
        return ["model_input", "transform_input"]
    
    def _get_createmodel_dependencies(self) -> List[str]:
        """Get expected dependencies for CreateModel steps."""
        return ["model_artifacts"]
