---
tags:
  - test
  - builders
  - mock
  - factory
  - validation
keywords:
  - mock factory
  - step type mocks
  - test configuration
  - mock objects creation
  - step builder testing
  - configuration factory
  - framework mocks
topics:
  - mock object creation
  - test infrastructure
  - step type configuration
  - validation testing
language: python
date of note: 2025-08-18
---

# Step Type Mock Factory

## Overview

The Step Type Mock Factory provides a consolidated system for creating step type-specific mock objects and configurations for testing step builders. The `StepTypeMockFactory` class generates appropriate mock configurations, dependencies, and objects based on detected step information and framework requirements.

## Architecture

### Core Factory Class: StepTypeMockFactory

The `StepTypeMockFactory` class serves as the central factory for creating step type-specific mock objects:

```python
class StepTypeMockFactory:
    """Consolidated factory for creating step type-specific mock objects with enhanced validation."""
    
    def __init__(self, step_info: Dict[str, Any], test_mode: bool = True):
        """
        Initialize factory with step information.
        
        Args:
            step_info: Step information from StepInfoDetector
            test_mode: Enable test mode for relaxed validation and better error handling
        """
        self.step_info = step_info
        self.sagemaker_step_type = step_info.get("sagemaker_step_type")
        self.framework = step_info.get("framework")
        self.test_pattern = step_info.get("test_pattern")
        self.test_mode = test_mode
```

### Key Components

#### Step Information Integration
- Uses step information from `StepInfoDetector`
- Extracts SageMaker step type, framework, and test patterns
- Enables step type-specific mock creation

#### Test Environment Setup
- Creates test script directories and mock files
- Ensures required script files exist for testing
- Provides proper file system structure for validation

## Key Features

### Mock Configuration Creation

The factory creates appropriate mock configurations based on step type and builder requirements:

```python
def create_mock_config(self) -> Any:
    """Create appropriate mock config for the step type with enhanced validation."""
    builder_name = self.step_info.get("builder_class_name", "")
    
    # Try to create proper config instance for type-strict builders
    proper_config = self._try_create_proper_config_instance(builder_name)
    if proper_config:
        return proper_config
    
    # Fall back to enhanced SimpleNamespace for flexible builders
    mock_config = self._create_base_config()
    
    # Add step type-specific configuration
    if self.sagemaker_step_type == "Processing":
        self._add_processing_config(mock_config)
    elif self.sagemaker_step_type == "Training":
        self._add_training_config(mock_config)
    # ... additional step types
```

### Proper Config Instance Creation

For type-strict builders, the factory attempts to create proper configuration instances:

```python
def _try_create_proper_config_instance(self, builder_name: str) -> Optional[Any]:
    """Try to create proper config instance with enhanced error handling."""
    try:
        # First create base config with all required fields
        base_config = self._create_base_pipeline_config()
        
        # Then create specific config using from_base_config
        if "Payload" in builder_name:
            config_instance = self._create_payload_config_from_base(base_config)
        elif "Package" in builder_name:
            config_instance = self._create_package_config_from_base(base_config)
        elif "TabularPreprocessing" in builder_name:
            config_instance = self._create_tabular_preprocessing_config_from_base(base_config)
        # ... additional builder types
```

### Test Script Directory Management

```python
def _ensure_test_script_directory(self) -> str:
    """Ensure test script directory exists and create necessary script files."""
    test_script_dir = '/tmp/mock_scripts'
    Path(test_script_dir).mkdir(parents=True, exist_ok=True)
    
    # Create common script files that builders expect
    script_files = [
        'tabular_preprocess.py',
        'currency_conversion.py', 
        'risk_table_mapping.py',
        'model_calibration.py',
        'dummy_training.py',
        'model_evaluation_xgb.py',
        'payload.py',
        'package.py',
        'train_xgb.py',
        'train_pytorch.py',
        'inference.py',
        'process.py'
    ]
    
    for script_file in script_files:
        script_path = Path(test_script_dir) / script_file
        if not script_path.exists():
            script_path.write_text(f'# Mock script for testing: {script_file}\nprint("Mock script execution")\n')
```

## Step Type-Specific Configuration

### Processing Step Configuration

```python
def _add_processing_config(self, mock_config: SimpleNamespace) -> None:
    """Add Processing step-specific configuration with enhanced validation."""
    mock_config.processing_instance_type = 'ml.m5.large'
    mock_config.processing_instance_type_large = 'ml.m5.xlarge'
    mock_config.processing_instance_type_small = 'ml.m5.large'
    mock_config.processing_instance_count = 1
    mock_config.processing_volume_size = 30
    mock_config.processing_entry_point = 'process.py'
    mock_config.source_dir = '/tmp/mock_scripts'
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
        # ... additional tabular preprocessing config
```

### Training Step Configuration

```python
def _add_training_config(self, mock_config: SimpleNamespace) -> None:
    """Add Training step-specific configuration with enhanced validation."""
    mock_config.training_instance_type = 'ml.m5.xlarge'
    mock_config.training_instance_count = 1
    mock_config.training_volume_size = 30
    mock_config.training_entry_point = 'train.py'
    mock_config.source_dir = '/tmp/mock_scripts'
    
    # Add enhanced hyperparameters based on framework
    builder_name = self.step_info.get("builder_class_name", "")
    if "XGBoostTraining" in builder_name:
        mock_hp = self._create_enhanced_xgboost_hyperparameters()
    elif "PyTorchTraining" in builder_name:
        mock_hp = self._create_enhanced_pytorch_hyperparameters()
    
    mock_config.hyperparameters = mock_hp
    mock_config.hyperparameters_s3_uri = 's3://test-bucket/config/hyperparameters.json'
```

## Enhanced Hyperparameters Creation

### XGBoost Hyperparameters

```python
def _create_enhanced_xgboost_hyperparameters(self) -> Any:
    """Create enhanced XGBoost hyperparameters that satisfy all validation rules."""
    try:
        from ...steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
        
        # Create comprehensive field lists that satisfy validation
        full_field_list = ['id', 'feature1', 'feature2', 'feature3', 'feature4', 'target']
        cat_field_list = ['feature1', 'feature2']  # Subset of full_field_list
        tab_field_list = ['feature3', 'feature4']  # Subset of full_field_list
        
        return XGBoostModelHyperparameters(
            # Required fields from base ModelHyperparameters
            full_field_list=full_field_list,
            cat_field_list=cat_field_list,
            tab_field_list=tab_field_list,
            id_name='id',
            label_name='target',
            multiclass_categories=['class_0', 'class_1'],
            # Required XGBoost-specific fields
            num_round=100,
            max_depth=6,
            eta=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            # ... additional XGBoost parameters
        )
    except Exception as e:
        # Fallback to SimpleNamespace with all required attributes
        mock_hp = SimpleNamespace()
        mock_hp.full_field_list = ['id', 'feature1', 'feature2', 'feature3', 'feature4', 'target']
        mock_hp.cat_field_list = ['feature1', 'feature2']
        mock_hp.tab_field_list = ['feature3', 'feature4']
        # ... additional fallback attributes
        return mock_hp
```

### PyTorch Hyperparameters

```python
def _create_enhanced_pytorch_hyperparameters(self) -> Any:
    """Create enhanced PyTorch hyperparameters that satisfy all validation rules."""
    try:
        from ...steps.hyperparams.hyperparameters_bsm import BSMModelHyperparameters
        
        # Create comprehensive field lists that satisfy validation
        full_field_list = ['id', 'feature1', 'feature2', 'text_field', 'target']
        cat_field_list = ['feature1']  # Subset of full_field_list
        tab_field_list = ['feature2']  # Subset of full_field_list
        
        return BSMModelHyperparameters(
            # Required fields from base ModelHyperparameters
            full_field_list=full_field_list,
            cat_field_list=cat_field_list,
            tab_field_list=tab_field_list,
            id_name='id',
            label_name='target',
            multiclass_categories=['class_0', 'class_1'],
            # Required BSM-specific fields
            tokenizer='bert-base-uncased',
            text_name='text_field',
            # ... additional BSM parameters
        )
    except Exception as e:
        # Fallback to SimpleNamespace
        mock_hp = SimpleNamespace()
        mock_hp.full_field_list = ['feature1', 'feature2', 'text_field', 'target']
        mock_hp.cat_field_list = ['feature1']
        mock_hp.tab_field_list = ['feature2']
        mock_hp.id_name = 'id'
        mock_hp.label_name = 'target'
        mock_hp.text_name = 'text_field'
        mock_hp.tokenizer = 'bert-base-uncased'
        return mock_hp
```

## Specific Configuration Creators

### Payload Configuration

```python
def _create_payload_config_from_base(self, base_config: Any) -> Any:
    """Create proper PayloadConfig instance using from_base_config."""
    try:
        from ...steps.configs.config_payload_step import PayloadConfig
        return PayloadConfig.from_base_config(
            base_config,
            # Payload-specific fields
            model_owner='test-team',
            model_domain='test-domain',
            model_objective='test-objective',
            source_model_inference_output_variable_list={'prediction': 'NUMERIC'},
            source_model_inference_input_variable_list={'feature1': 'NUMERIC', 'feature2': 'TEXT'},
            expected_tps=100,
            max_latency_in_millisecond=1000,
            framework='xgboost',
            processing_entry_point='payload.py',
            # ... additional payload configuration
        )
    except Exception as e:
        if self.test_mode:
            print(f"INFO: Failed to create PayloadConfig from base: {e}")
        return None
```

### Tabular Preprocessing Configuration

```python
def _create_tabular_preprocessing_config_from_base(self, base_config: Any) -> Any:
    """Create proper TabularPreprocessingConfig instance using from_base_config."""
    try:
        from ...steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
        return TabularPreprocessingConfig.from_base_config(
            base_config,
            job_type='training',
            label_name='target',
            train_ratio=0.7,
            test_val_ratio=0.5,
            categorical_columns=['category_1', 'category_2'],
            numerical_columns=['numeric_1', 'numeric_2'],
            text_columns=['text_1', 'text_2'],
            date_columns=['date_1'],
            processing_entry_point='tabular_preprocess.py',
            # ... additional preprocessing configuration
        )
    except Exception as e:
        if self.test_mode:
            print(f"INFO: Failed to create TabularPreprocessingConfig from base: {e}")
        return None
```

## Step Type-Specific Mock Objects

### Processing Mocks

```python
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
```

### Training Mocks

```python
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
    
    return mocks
```

### CreateModel Mocks

```python
def _create_createmodel_mocks(self) -> Dict[str, Any]:
    """Create CreateModel step-specific mocks that don't interfere with SageMaker validation."""
    mocks = {}
    
    # Mock Model with proper string attributes to avoid MagicMock issues
    mock_model = MagicMock()
    mock_model.name = 'test-model'
    mock_model.image_uri = 'mock-image-uri'
    mock_model.model_data = 's3://bucket/model.tar.gz'
    
    # Ensure model.create() returns proper step arguments without conflicts
    mock_model.create.return_value = {
        'ModelName': 'test-model',
        'PrimaryContainer': {
            'Image': 'mock-image-uri',
            'ModelDataUrl': 's3://bucket/model.tar.gz',
            'Environment': {}
        },
        'ExecutionRoleArn': 'arn:aws:iam::123456789012:role/MockRole'
    }
    
    mocks['model'] = mock_model
    return mocks
```

## Dependency Resolution

### Expected Dependencies by Step Type

```python
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

def _get_processing_dependencies(self) -> List[str]:
    """Get expected dependencies for Processing steps."""
    builder_name = self.step_info.get("builder_class_name", "")
    
    if "TabularPreprocessing" in builder_name:
        return ["DATA"]
    elif "RiskTableMapping" in builder_name:
        return ["risk_tables"]
    elif "XGBoostModelEval" in builder_name:
        return ["model_input"]
    elif "ModelEval" in builder_name:
        return ["model_input", "eval_data_input"]
    else:
        return ["input_data"]
```

## Framework-Specific Configuration

### Framework Detection and Configuration

```python
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
            mock_config.py_version = 'py38'
    elif self.framework == "tensorflow":
        if not hasattr(mock_config, 'framework_version'):
            mock_config.framework_version = '2.11.0'
        if not hasattr(mock_config, 'py_version'):
            mock_config.py_version = 'py39'
```

## Error Handling and Test Mode

### Test Mode Features

The factory includes comprehensive error handling with test mode support:

```python
def __init__(self, step_info: Dict[str, Any], test_mode: bool = True):
    """
    Initialize factory with step information.
    
    Args:
        step_info: Step information from StepInfoDetector
        test_mode: Enable test mode for relaxed validation and better error handling
    """
    self.test_mode = test_mode

# Error handling with test mode
try:
    config_instance = self._create_specific_config(base_config)
    if self.test_mode:
        print(f"INFO: Successfully created {type(config_instance).__name__}")
    return config_instance
except Exception as e:
    if self.test_mode:
        print(f"INFO: Could not create proper config, using fallback: {e}")
    else:
        print(f"Failed to create proper config: {e}")
    return None
```

## Integration Points

### With Step Info Detector
- Uses step information from `StepInfoDetector` for configuration
- Leverages detected step type, framework, and patterns
- Enables intelligent mock creation based on detected characteristics

### With Base Test Framework
- Provides mock configurations for `UniversalStepBuilderTestBase`
- Integrates with test environment setup
- Supports comprehensive mock object creation

### With Validation Framework
- Creates configurations that pass validation requirements
- Provides proper hyperparameters and configuration objects
- Ensures compatibility with step builder validation

## Usage Examples

### Basic Mock Configuration Creation

```python
from cursus.validation.builders.mock_factory import StepTypeMockFactory

# Create factory with step info
step_info = {
    "sagemaker_step_type": "Processing",
    "framework": "xgboost",
    "builder_class_name": "TabularPreprocessingStepBuilder"
}

factory = StepTypeMockFactory(step_info, test_mode=True)

# Create mock configuration
mock_config = factory.create_mock_config()

# Create step type-specific mocks
step_mocks = factory.create_step_type_mocks()

# Get expected dependencies
dependencies = factory.get_expected_dependencies()
```

### Framework-Specific Configuration

```python
# XGBoost Processing configuration
xgboost_step_info = {
    "sagemaker_step_type": "Processing",
    "framework": "xgboost",
    "builder_class_name": "XGBoostModelEvalStepBuilder"
}

xgboost_factory = StepTypeMockFactory(xgboost_step_info)
xgboost_config = xgboost_factory.create_mock_config()

# PyTorch Training configuration
pytorch_step_info = {
    "sagemaker_step_type": "Training",
    "framework": "pytorch",
    "builder_class_name": "PyTorchTrainingStepBuilder"
}

pytorch_factory = StepTypeMockFactory(pytorch_step_info)
pytorch_config = pytorch_factory.create_mock_config()
```

## Best Practices

### Configuration Creation
- Use proper configuration instances when possible
- Fall back to SimpleNamespace for flexibility
- Ensure all required fields are populated

### Error Handling
- Enable test mode for development and debugging
- Provide meaningful fallback configurations
- Log configuration creation attempts and failures

### Framework Support
- Detect framework from step information
- Provide framework-specific configurations
- Support multiple ML frameworks consistently

## Conclusion

The Step Type Mock Factory provides a comprehensive system for creating step type-specific mock objects and configurations. Through intelligent step type detection, framework-specific configuration, and robust error handling, it enables reliable and consistent testing across all step builder implementations.

The factory's integration with the step info detector and base test framework ensures that mock objects are appropriate for the specific step type and framework being tested, while providing fallback mechanisms for robust test execution.
