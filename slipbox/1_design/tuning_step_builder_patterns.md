---
tags:
  - design
  - step_builders
  - tuning_steps
  - patterns
  - sagemaker
  - hyperparameter_tuning
keywords:
  - tuning step patterns
  - HyperparameterTuner
  - TuningStep
  - hyperparameter ranges
  - tuning job configuration
  - model optimization
topics:
  - step builder patterns
  - tuning step implementation
  - SageMaker hyperparameter tuning
  - estimator configuration
language: python
date of note: 2025-10-26
---

# Tuning Step Builder Patterns

## Overview

This document analyzes the design patterns for Tuning step builder implementations in the cursus framework. Tuning steps create **TuningStep** instances using **HyperparameterTuner** for automated hyperparameter optimization. Unlike training steps that produce a single model, tuning steps explore hyperparameter spaces to find optimal configurations and produce the best performing model.

## SageMaker Step Type Classification

All Tuning step builders create **TuningStep** instances using framework-specific estimators wrapped in HyperparameterTuner:
- **XGBoost**: XGBoost estimator with gradient boosting hyperparameter ranges
- **PyTorch**: PyTorch estimator with deep learning hyperparameter ranges  
- **LightGBM**: LightGBM estimator with boosting hyperparameter ranges
- **Framework-agnostic**: Generic tuning patterns for other frameworks

## Key Differences from Training Steps

### 1. Estimator Wrapping Pattern
```python
# Training Step: Direct estimator usage
training_step = TrainingStep(
    name="training-step",
    estimator=estimator,  # Direct estimator
    inputs=training_inputs
)

# Tuning Step: Estimator wrapped in HyperparameterTuner
tuner = HyperparameterTuner(
    estimator=estimator,  # Same estimator, but wrapped
    objective_metric_name="validation:auc",
    hyperparameter_ranges=ranges
)
tuning_step = TuningStep(
    name="tuning-step", 
    tuner=tuner,  # Wrapped estimator
    inputs=training_inputs
)
```

### 2. Script Reuse Pattern
Tuning steps **reuse existing training scripts** with **no modifications needed**:
```python
# Same script used for both training and tuning
entry_point = "xgboost_training.py"  # Reused from training step

# The script automatically works for tuning because:
# 1. SageMaker provides hyperparameters via environment variables
# 2. Script saves model artifacts to /opt/ml/model (same as training)
# 3. Script reports metrics to stdout (picked up by tuning job)
# 4. Best model is automatically selected by SageMaker tuning job
```

**Key Script Requirements for Tuning Compatibility:**
- Must report objective metric to stdout in format: `"metric_name: value"`
- Must save model artifacts to `/opt/ml/model` directory
- Must handle hyperparameters from environment variables (standard SageMaker pattern)
- Should implement early stopping based on validation performance for efficiency

### 3. Output Pattern Differences
```python
# Training Step Outputs
training_outputs = {
    "model_output": "properties.ModelArtifacts.S3ModelArtifacts",
    "evaluation_output": "properties.OutputDataConfig.S3OutputPath"
}

# Tuning Step Outputs  
tuning_outputs = {
    "best_model_output": "properties.BestTrainingJob.ModelArtifacts.S3ModelArtifacts",
    "tuning_job_output": "properties.HyperParameterTuningJobArn",
    "all_training_jobs": "properties.TrainingJobSummaries"
}
```

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Tuning step builders follow this consistent architecture:

```python
class TuningStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load framework-specific tuning specification
        spec = FRAMEWORK_TUNING_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate tuning-specific configuration
        
    def _create_base_estimator(self) -> Estimator:
        # Create framework-specific estimator (reuse training patterns)
        
    def _create_hyperparameter_tuner(self) -> HyperparameterTuner:
        # Wrap estimator in HyperparameterTuner with ranges
        
    def _get_hyperparameter_ranges(self) -> Dict:
        # Define tunable hyperparameter ranges
        
    def _get_static_hyperparameters(self) -> Dict:
        # Define fixed hyperparameters
        
    def _get_inputs(self, inputs) -> Dict[str, TrainingInput]:
        # Create TrainingInput objects (same as training steps)
        
    def _get_outputs(self, outputs) -> str:
        # Return output path for tuning job results
        
    def create_step(self, **kwargs) -> TuningStep:
        # Orchestrate tuning step creation
```

### 2. Framework-Specific Estimator Creation Patterns

#### XGBoost Tuning Estimator Pattern
```python
def _create_base_estimator(self) -> XGBoost:
    """Create base XGBoost estimator for tuning (reuse training pattern)"""
    return XGBoost(
        entry_point=self.config.training_entry_point,  # Same as training
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        base_job_name=self._generate_job_name(),
        hyperparameters=self._get_static_hyperparameters(),  # Fixed params only
        sagemaker_session=self.session,
        output_path=None,  # Set by tuning job
        environment=self._get_environment_variables(),
    )

def _create_hyperparameter_tuner(self) -> HyperparameterTuner:
    """Create HyperparameterTuner with XGBoost-specific ranges"""
    estimator = self._create_base_estimator()
    
    return HyperparameterTuner(
        estimator=estimator,
        objective_metric_name=self.config.objective_metric_name,
        objective_type=self.config.objective_type,
        hyperparameter_ranges=self._get_hyperparameter_ranges(),
        max_jobs=self.config.max_jobs,
        max_parallel_jobs=self.config.max_parallel_jobs,
        strategy=self.config.strategy,
        early_stopping_type=self.config.early_stopping_type,
        base_tuning_job_name=self._generate_tuning_job_name(),
        warm_start_config=self.config.warm_start_config,
        tags=self.config.tags
    )
```

#### PyTorch Tuning Estimator Pattern
```python
def _create_base_estimator(self) -> PyTorch:
    """Create base PyTorch estimator for tuning"""
    return PyTorch(
        entry_point=self.config.training_entry_point,  # Same as training
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        base_job_name=self._generate_job_name(),
        hyperparameters=self._get_static_hyperparameters(),  # Fixed params only
        sagemaker_session=self.session,
        output_path=None,  # Set by tuning job
        environment=self._get_environment_variables(),
    )
```

### 3. Hyperparameter Range Definition Patterns

#### Continuous Parameter Ranges
```python
def _get_hyperparameter_ranges(self) -> Dict:
    """Define hyperparameter ranges for tuning"""
    ranges = {}
    
    # Continuous parameters
    if hasattr(self.config, 'eta_range'):
        ranges['eta'] = ContinuousParameter(
            min_value=self.config.eta_range[0],
            max_value=self.config.eta_range[1],
            scaling_type=self.config.eta_scaling or "Auto"
        )
    
    if hasattr(self.config, 'gamma_range'):
        ranges['gamma'] = ContinuousParameter(
            min_value=self.config.gamma_range[0],
            max_value=self.config.gamma_range[1]
        )
    
    return ranges
```

#### Integer Parameter Ranges
```python
def _get_hyperparameter_ranges(self) -> Dict:
    """Define integer hyperparameter ranges"""
    ranges = {}
    
    # Integer parameters
    if hasattr(self.config, 'max_depth_range'):
        ranges['max_depth'] = IntegerParameter(
            min_value=self.config.max_depth_range[0],
            max_value=self.config.max_depth_range[1]
        )
    
    if hasattr(self.config, 'num_round_range'):
        ranges['num_round'] = IntegerParameter(
            min_value=self.config.num_round_range[0],
            max_value=self.config.num_round_range[1]
        )
    
    return ranges
```

#### Categorical Parameter Ranges
```python
def _get_hyperparameter_ranges(self) -> Dict:
    """Define categorical hyperparameter ranges"""
    ranges = {}
    
    # Categorical parameters
    if hasattr(self.config, 'booster_options'):
        ranges['booster'] = CategoricalParameter(
            values=self.config.booster_options  # ['gbtree', 'gblinear', 'dart']
        )
    
    if hasattr(self.config, 'optimizer_options'):
        ranges['optimizer'] = CategoricalParameter(
            values=self.config.optimizer_options  # ['adam', 'sgd', 'rmsprop']
        )
    
    return ranges
```

### 4. Static Hyperparameter Handling Pattern

```python
def _get_static_hyperparameters(self) -> Dict:
    """Get fixed hyperparameters that won't be tuned"""
    static_params = {}
    
    # Framework-specific fixed parameters
    if hasattr(self.config, 'objective'):
        static_params['objective'] = self.config.objective
    
    if hasattr(self.config, 'eval_metric'):
        static_params['eval_metric'] = self.config.eval_metric
    
    # Data-specific parameters
    if hasattr(self.config, 'num_class'):
        static_params['num_class'] = self.config.num_class
    
    # Add any additional static hyperparameters from config
    if hasattr(self.config, 'static_hyperparameters'):
        static_params.update(self.config.static_hyperparameters)
    
    return static_params
```

### 5. Training Input Patterns (Same as Training Steps)

Tuning steps use the same input patterns as training steps:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
    """Get inputs for tuning step (same as training step pattern)"""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    training_inputs = {}
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        if logical_name == "input_path":
            base_path = inputs[logical_name]
            # Use same data channel creation as training steps
            data_channels = self._create_data_channels_from_source(base_path)
            training_inputs.update(data_channels)
            
    return training_inputs
```

### 6. Output Path Handling Pattern

Tuning steps handle outputs differently to accommodate multiple training jobs:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:
    """
    Get outputs for tuning step.
    
    Tuning jobs create multiple training jobs, so output structure is:
    - {output_path}/{tuning_job_name}/
    - Best model accessible via properties.BestTrainingJob.ModelArtifacts
    """
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    # Check if any output path is explicitly provided
    primary_output_path = None
    output_logical_names = [spec.logical_name for _, spec in self.spec.outputs.items()]
    
    for logical_name in output_logical_names:
        if logical_name in outputs:
            primary_output_path = outputs[logical_name]
            break
            
    # Generate default if not provided
    if primary_output_path is None:
        primary_output_path = f"{self.config.pipeline_s3_loc}/tuning/"
        
    # Remove trailing slash for consistency
    if primary_output_path.endswith('/'):
        primary_output_path = primary_output_path[:-1]
    
    return primary_output_path
```

### 7. Tuning Job Configuration Pattern

```python
def _get_tuning_job_config(self) -> Dict:
    """Get tuning job specific configuration"""
    config = {
        'max_jobs': self.config.max_jobs,
        'max_parallel_jobs': self.config.max_parallel_jobs,
        'strategy': self.config.strategy,  # 'Bayesian', 'Random', 'Hyperband'
        'early_stopping_type': self.config.early_stopping_type,  # 'Auto', 'Off'
    }
    
    # Add warm start configuration if provided
    if hasattr(self.config, 'warm_start_config') and self.config.warm_start_config:
        config['warm_start_config'] = self.config.warm_start_config
    
    # Add completion criteria if provided
    if hasattr(self.config, 'completion_criteria'):
        config['completion_criteria'] = self.config.completion_criteria
    
    return config
```

### 8. Step Creation Pattern

Tuning steps follow this orchestration pattern:

```python
def create_step(self, **kwargs) -> TuningStep:
    # Extract parameters
    inputs_raw = kwargs.get('inputs', {})
    input_path = kwargs.get('input_path')
    output_path = kwargs.get('output_path')
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Add direct parameters if provided
    if input_path is not None:
        inputs["input_path"] = input_path
        
    # Get training inputs and output path
    training_inputs = self._get_inputs(inputs)
    output_path = self._get_outputs(outputs or {})
    
    # Create hyperparameter tuner
    tuner = self._create_hyperparameter_tuner()
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create tuning step
    tuning_step = TuningStep(
        name=step_name,
        tuner=tuner,
        inputs=training_inputs,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(tuning_step, '_spec', self.spec)
    
    return tuning_step
```

## Configuration Validation Patterns

### Standard Tuning Configuration
```python
def validate_configuration(self) -> None:
    # Validate base training configuration (reuse training validation)
    required_training_attrs = [
        'training_instance_type',
        'training_instance_count', 
        'training_volume_size',
        'training_entry_point',
        'source_dir',
        'framework_version',
        'py_version'
    ]
    
    for attr in required_training_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"TuningConfig missing required training attribute: {attr}")
    
    # Validate tuning-specific configuration
    required_tuning_attrs = [
        'max_jobs',
        'max_parallel_jobs',
        'objective_metric_name',
        'objective_type'
    ]
    
    for attr in required_tuning_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"TuningConfig missing required tuning attribute: {attr}")
    
    # Validate hyperparameter ranges are provided
    if not hasattr(self.config, 'hyperparameter_ranges') or not self.config.hyperparameter_ranges:
        raise ValueError("TuningConfig must specify hyperparameter_ranges for tuning")
    
    # Validate objective type
    if self.config.objective_type not in ['Maximize', 'Minimize']:
        raise ValueError(f"Invalid objective_type: {self.config.objective_type}")
    
    # Validate strategy
    valid_strategies = ['Bayesian', 'Random', 'Hyperband', 'Grid']
    if hasattr(self.config, 'strategy') and self.config.strategy not in valid_strategies:
        raise ValueError(f"Invalid strategy: {self.config.strategy}")
```

### Framework-Specific Validation
```python
def validate_configuration(self) -> None:
    # XGBoost-specific validation
    if hasattr(self.config, 'objective') and self.config.objective not in [
        'binary:logistic', 'multi:softmax', 'multi:softprob', 'reg:squarederror'
    ]:
        raise ValueError(f"Invalid XGBoost objective: {self.config.objective}")
        
    # PyTorch-specific validation  
    if hasattr(self.config, 'optimizer_options'):
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
        for opt in self.config.optimizer_options:
            if opt not in valid_optimizers:
                raise ValueError(f"Invalid PyTorch optimizer: {opt}")
```

## Design Components Integration

### 1. Step Specification Pattern

```python
# specs/xgboost_tuning_spec.py
XGBOOST_TUNING_SPEC = StepSpecification(
    step_type=get_spec_step_type("XGBoostTuning"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_xgboost_tuning_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "tune"],
            data_type="S3Uri",
            description="Training dataset S3 location for hyperparameter tuning"
        ),
        DependencySpec(
            logical_name="hyperparameters_config",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=["config", "params", "hyperparameters", "ranges", "tuning"],
            data_type="S3Uri",
            description="Hyperparameter ranges and tuning configuration (optional)"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="best_model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="BestTrainingJob.TrainingJobArn",
            data_type="String",
            description="Best training job ARN from hyperparameter tuning (use DescribeTrainingJob to get ModelArtifacts.S3ModelArtifacts)",
            aliases=["BestTrainingJobArn", "best_training_job_arn", "tuned_model_job"]
        ),
        OutputSpec(
            logical_name="best_model_artifacts_direct",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="BestTrainingJob.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Direct access to best model artifacts S3 location from hyperparameter tuning",
            aliases=["BestModelArtifacts", "best_model_data", "tuned_model_artifacts"]
        ),
        OutputSpec(
            logical_name="tuning_job_arn",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="HyperParameterTuningJobArn",
            data_type="String",
            description="Hyperparameter tuning job ARN",
            aliases=["tuning_job_output", "tuning_arn", "optimization_job_arn"]
        ),
        OutputSpec(
            logical_name="best_training_job_name",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="BestTrainingJob.TrainingJobName",
            data_type="String",
            description="Name of the best training job from hyperparameter tuning",
            aliases=["best_job_name", "optimal_training_job"]
        ),
        OutputSpec(
            logical_name="best_objective_metric",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="BestTrainingJob.FinalHyperParameterTuningJobObjectiveMetric.Value",
            data_type="Number",
            description="Best objective metric value achieved during hyperparameter tuning",
            aliases=["best_metric_value", "optimal_metric", "tuning_best_score"]
        ),
        OutputSpec(
            logical_name="tuning_job_status",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="HyperParameterTuningJobStatus",
            data_type="String",
            description="Status of the hyperparameter tuning job",
            aliases=["tuning_status", "job_status"]
        )
    ]
)
```

### 2. Script Contract Pattern

```python
# contracts/xgboost_tuning_contract.py
XGBOOST_TUNING_CONTRACT = TrainingScriptContract(
    entry_point="xgboost_training.py",  # Reuse existing training script
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "best_model_output": "/opt/ml/model",
        "tuning_job_output": "/opt/ml/output/data"
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables
    ],
    optional_env_vars={
        "SAGEMAKER_HYPERPARAMETER_TUNING_JOB_NAME": "Name of the tuning job",
        "SAGEMAKER_HYPERPARAMETER_TUNING_JOB_CONFIG": "Tuning job configuration"
    },
    framework_requirements={
        "boto3": ">=1.26.0",
        "xgboost": "==1.7.6",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "pyarrow": ">=4.0.0,<6.0.0",
        "numpy": ">=1.19.0"
    },
    description="""
    XGBoost training script used for hyperparameter tuning that:
    1. Loads training, validation, and test datasets from split directories
    2. Applies preprocessing using fitted artifacts from previous steps
    3. Trains XGBoost model with hyperparameters provided by tuning job
    4. Evaluates model performance and reports objective metric
    5. Saves model artifacts and evaluation results
    
    Hyperparameter Tuning Integration:
    - Receives hyperparameters from SageMaker tuning job via environment variables
    - Reports objective metric (e.g., validation AUC) for optimization
    - Supports both binary and multiclass classification tuning
    - Handles early stopping based on validation performance
    
    Input Structure (same as training):
    - /opt/ml/input/data: Root directory with train/val/test subdirectories
    - /opt/ml/input/config/hyperparameters.json: Hyperparameters from tuning job
    
    Output Structure (same as training):
    - /opt/ml/model: Model artifacts directory
    - /opt/ml/output/data: Evaluation results directory
    
    Tuning-Specific Considerations:
    - Script must report objective metric to stdout in format: "metric_name: value"
    - Early stopping should be based on validation performance
    - Model checkpointing for best validation performance
    - Robust error handling for invalid hyperparameter combinations
    """
)
```

### 3. Configuration Class Pattern

```python
# configs/config_xgboost_tuning_step.py
class XGBoostTuningConfig(BasePipelineConfig):
    """Configuration for XGBoost hyperparameter tuning step"""
    
    def __init__(self):
        super().__init__()
        
        # Base training configuration (inherited from training step)
        self.training_instance_type: str = "ml.m5.xlarge"
        self.training_instance_count: int = 1
        self.training_volume_size: int = 30
        self.training_entry_point: str = "xgboost_training.py"
        self.source_dir: str = None
        self.framework_version: str = "1.7.6"
        self.py_version: str = "py39"
        
        # Tuning job configuration
        self.max_jobs: int = 20
        self.max_parallel_jobs: int = 3
        self.objective_metric_name: str = "validation:auc"
        self.objective_type: str = "Maximize"  # or "Minimize"
        
        # Tuning strategy configuration
        self.strategy: str = "Bayesian"  # "Bayesian", "Random", "Hyperband", "Grid"
        self.early_stopping_type: str = "Auto"  # "Auto", "Off"
        
        # Hyperparameter ranges - Separate dictionaries by type (RECOMMENDED APPROACH)
        
        # Continuous parameters: (min_value, max_value) tuples
        self.continuous_hyperparameters: Dict[str, Tuple[float, float]] = {
            'eta': (0.01, 0.3),           # Learning rate range
            'gamma': (0, 5),              # Minimum split loss range
            'subsample': (0.5, 1.0),      # Subsample ratio range
            'colsample_bytree': (0.5, 1.0), # Feature subsample ratio range
            'reg_alpha': (0, 10),         # L1 regularization range
            'reg_lambda': (0, 10),        # L2 regularization range
            'scale_pos_weight': (0.1, 10) # Class balance range (binary classification)
        }
        
        # Integer parameters: (min_value, max_value) tuples
        self.integer_hyperparameters: Dict[str, Tuple[int, int]] = {
            'max_depth': (3, 10),         # Tree depth range
            'num_round': (100, 1000),     # Number of boosting rounds
            'min_child_weight': (1, 10),  # Minimum child weight range
            'max_delta_step': (0, 10)     # Maximum delta step range
        }
        
        # Categorical parameters: List of possible values
        self.categorical_hyperparameters: Dict[str, List[str]] = {
            'booster': ['gbtree', 'gblinear', 'dart'],
            'tree_method': ['auto', 'exact', 'approx', 'hist'],
            'grow_policy': ['depthwise', 'lossguide'],
            'sample_type': ['uniform', 'weighted']
        }
        
        # Static hyperparameters (fixed parameters)
        self.static_hyperparameters: Dict = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 4,
            'verbosity': 1,
            'seed': 42
        }
        
        # Advanced tuning configuration
        self.base_tuning_job_name: str = None
        self.warm_start_config: Dict = None
        self.completion_criteria: Dict = None
        self.tags: List[Dict] = None
        
        # Resource optimization
        self.enable_spot_instances: bool = False
        self.max_wait_time_in_seconds: int = None
    
    def validate_hyperparameter_ranges(self) -> None:
        """Validate hyperparameter ranges for all parameter types"""
        
        # Validate continuous parameters
        if hasattr(self, 'continuous_hyperparameters') and self.continuous_hyperparameters:
            for param_name, param_range in self.continuous_hyperparameters.items():
                if not isinstance(param_range, tuple) or len(param_range) != 2:
                    raise ValueError(f"Continuous parameter '{param_name}' must be a tuple (min, max)")
                
                min_val, max_val = param_range
                if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                    raise ValueError(f"Continuous parameter '{param_name}' values must be numeric")
                
                if min_val >= max_val:
                    raise ValueError(f"Continuous parameter '{param_name}' min value ({min_val}) must be less than max value ({max_val})")
                
                if min_val < 0 and param_name in ['eta', 'subsample', 'colsample_bytree']:
                    raise ValueError(f"Parameter '{param_name}' cannot have negative values")
        
        # Validate integer parameters
        if hasattr(self, 'integer_hyperparameters') and self.integer_hyperparameters:
            for param_name, param_range in self.integer_hyperparameters.items():
                if not isinstance(param_range, tuple) or len(param_range) != 2:
                    raise ValueError(f"Integer parameter '{param_name}' must be a tuple (min, max)")
                
                min_val, max_val = param_range
                if not isinstance(min_val, int) or not isinstance(max_val, int):
                    raise ValueError(f"Integer parameter '{param_name}' values must be integers")
                
                if min_val >= max_val:
                    raise ValueError(f"Integer parameter '{param_name}' min value ({min_val}) must be less than max value ({max_val})")
                
                if min_val < 1 and param_name in ['max_depth', 'num_round', 'min_child_weight']:
                    raise ValueError(f"Parameter '{param_name}' must have positive values")
        
        # Validate categorical parameters
        if hasattr(self, 'categorical_hyperparameters') and self.categorical_hyperparameters:
            for param_name, param_values in self.categorical_hyperparameters.items():
                if not isinstance(param_values, list):
                    raise ValueError(f"Categorical parameter '{param_name}' must be a list of values")
                
                if len(param_values) < 2:
                    raise ValueError(f"Categorical parameter '{param_name}' must have at least 2 values")
                
                if len(set(param_values)) != len(param_values):
                    raise ValueError(f"Categorical parameter '{param_name}' contains duplicate values")
                
                # Validate XGBoost-specific categorical values
                if param_name == 'booster':
                    valid_boosters = ['gbtree', 'gblinear', 'dart']
                    invalid_boosters = [v for v in param_values if v not in valid_boosters]
                    if invalid_boosters:
                        raise ValueError(f"Invalid booster values for XGBoost: {invalid_boosters}. Valid values: {valid_boosters}")
                
                elif param_name == 'tree_method':
                    valid_methods = ['auto', 'exact', 'approx', 'hist', 'gpu_hist']
                    invalid_methods = [v for v in param_values if v not in valid_methods]
                    if invalid_methods:
                        raise ValueError(f"Invalid tree_method values for XGBoost: {invalid_methods}. Valid values: {valid_methods}")
        
        # Ensure at least one parameter type is specified
        has_continuous = hasattr(self, 'continuous_hyperparameters') and self.continuous_hyperparameters
        has_integer = hasattr(self, 'integer_hyperparameters') and self.integer_hyperparameters
        has_categorical = hasattr(self, 'categorical_hyperparameters') and self.categorical_hyperparameters
        
        if not (has_continuous or has_integer or has_categorical):
            raise ValueError("At least one hyperparameter range must be specified for tuning")
    
    def get_all_hyperparameter_ranges_for_sagemaker(self) -> Dict:
        """Convert all hyperparameter ranges to SageMaker format"""
        from sagemaker.tuner import ContinuousParameter, IntegerParameter, CategoricalParameter
        
        ranges = {}
        
        # Convert continuous parameters
        if hasattr(self, 'continuous_hyperparameters') and self.continuous_hyperparameters:
            for param_name, (min_val, max_val) in self.continuous_hyperparameters.items():
                ranges[param_name] = ContinuousParameter(min_val, max_val)
        
        # Convert integer parameters
        if hasattr(self, 'integer_hyperparameters') and self.integer_hyperparameters:
            for param_name, (min_val, max_val) in self.integer_hyperparameters.items():
                ranges[param_name] = IntegerParameter(min_val, max_val)
        
        # Convert categorical parameters
        if hasattr(self, 'categorical_hyperparameters') and self.categorical_hyperparameters:
            for param_name, values in self.categorical_hyperparameters.items():
                ranges[param_name] = CategoricalParameter(values)
        
        return ranges
```

### 4. Hyperparameter Classes Pattern

```python
# hyperparams/hyperparameters_xgboost_tuning.py
class XGBoostTuningHyperparameters:
    """XGBoost hyperparameter ranges for tuning"""
    
    def __init__(self):
        # Default hyperparameter ranges for XGBoost tuning
        self.default_ranges = {
            # Learning parameters
            'eta': (0.01, 0.3),                    # Learning rate
            'gamma': (0, 5),                       # Minimum split loss
            'max_depth': (3, 10),                  # Maximum tree depth
            
            # Regularization parameters  
            'lambda': (0, 10),                     # L2 regularization
            'alpha': (0, 10),                      # L1 regularization
            'subsample': (0.5, 1.0),               # Row sampling
            'colsample_bytree': (0.5, 1.0),        # Column sampling
            
            # Tree construction parameters
            'min_child_weight': (1, 10),           # Minimum child weight
            'max_delta_step': (0, 10),             # Maximum delta step
            
            # Advanced parameters
            'scale_pos_weight': (0.1, 10),         # Class balance (binary)
        }
        
        # Default static hyperparameters
        self.default_static = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'nthread': 4,
            'verbosity': 1,
            'seed': 42
        }
    
    def get_binary_classification_ranges(self) -> Dict:
        """Get hyperparameter ranges optimized for binary classification"""
        ranges = self.default_ranges.copy()
        ranges.update({
            'scale_pos_weight': (0.1, 10),  # Important for imbalanced datasets
        })
        return ranges
    
    def get_multiclass_classification_ranges(self) -> Dict:
        """Get hyperparameter ranges optimized for multiclass classification"""
        ranges = self.default_ranges.copy()
        # Remove binary-specific parameters
        ranges.pop('scale_pos_weight', None)
        return ranges
    
    def get_regression_ranges(self) -> Dict:
        """Get hyperparameter ranges optimized for regression"""
        ranges = self.default_ranges.copy()
        # Remove classification-specific parameters
        ranges.pop('scale_pos_weight', None)
        return ranges
    
    def get_static_hyperparameters(self, task_type: str = "binary_classification") -> Dict:
        """Get static hyperparameters based on task type"""
        static = self.default_static.copy()
        
        if task_type == "binary_classification":
            static.update({
                'objective': 'binary:logistic',
                'eval_metric': 'auc'
            })
        elif task_type == "multiclass_classification":
            static.update({
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss'
            })
        elif task_type == "regression":
            static.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            })
        
        return static


## Key Differences Between Tuning Step Types

### 1. By Framework
- **XGBoost**: Gradient boosting with tree-based hyperparameters (depth, learning rate, regularization)
- **PyTorch**: Deep learning with neural network hyperparameters (learning rate, batch size, architecture)
- **LightGBM**: Gradient boosting with LightGBM-specific parameters (num_leaves, feature_fraction)

### 2. By Hyperparameter Strategy
- **Bayesian Optimization**: Intelligent search using Gaussian processes (default, most efficient)
- **Random Search**: Random sampling from hyperparameter space (good baseline)
- **Hyperband**: Multi-fidelity optimization with early stopping (resource-efficient)
- **Grid Search**: Exhaustive search over discrete parameter grid (thorough but expensive)

### 3. By Objective Optimization
- **Maximize**: AUC, accuracy, F1-score, precision, recall
- **Minimize**: Loss, error rate, RMSE, MAE, log-loss

### 4. By Resource Management
- **Standard Instances**: Fixed compute resources per training job
- **Spot Instances**: Cost-optimized with potential interruptions
- **Multi-instance**: Distributed training for large datasets

## Best Practices Identified

1. **Script Reuse**: Leverage existing training scripts - no new scripts needed for tuning
2. **Specification-Driven Design**: Use specifications for input/output definitions and dependency resolution
3. **Framework-Specific Ranges**: Define appropriate hyperparameter ranges for each ML framework
4. **Static vs Tunable Separation**: Clearly separate fixed parameters from tunable ranges
5. **Objective Metric Reporting**: Ensure training scripts report metrics in correct format
6. **Resource Optimization**: Use appropriate instance types and parallelization for cost efficiency
7. **Early Stopping**: Implement early stopping to avoid wasting resources on poor configurations
8. **Warm Start**: Leverage previous tuning results to accelerate new tuning jobs
9. **Error Handling**: Robust handling of invalid hyperparameter combinations
10. **Output Management**: Proper handling of multiple training job outputs and best model selection

## Testing Implications

Tuning step builders should be tested for:

1. **Estimator Creation**: Correct base estimator type and configuration
2. **HyperparameterTuner Creation**: Proper tuner configuration with ranges and strategy
3. **Hyperparameter Range Validation**: Correct parameter types and value ranges
4. **Static Parameter Handling**: Proper separation of fixed vs tunable parameters
5. **Training Input Creation**: Correct TrainingInput object creation (same as training steps)
6. **Output Path Handling**: Proper output path generation for tuning job results
7. **Objective Metric Configuration**: Correct metric name and optimization direction
8. **Tuning Strategy Configuration**: Proper strategy, early stopping, and resource settings
9. **Specification Compliance**: Adherence to step specifications and contracts
10. **Framework-Specific Features**: Framework-specific validation and hyperparameter handling
11. **Dependency Resolution**: Proper integration with specification-driven dependency resolution
12. **Best Model Access**: Correct property paths for accessing best training job results

## Implementation Examples

### Complete XGBoost Tuning Step Builder

```python
from typing import Dict, Optional, Any
from sagemaker.workflow.steps import TuningStep
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

from ..configs.config_xgboost_tuning_step import XGBoostTuningConfig
from ...core.base.builder_base import StepBuilderBase

# Import XGBoost tuning specification
try:
    from ..specs.xgboost_tuning_spec import XGBOOST_TUNING_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    XGBOOST_TUNING_SPEC = None
    SPEC_AVAILABLE = False


class XGBoostTuningStepBuilder(StepBuilderBase):
    """Builder for XGBoost Hyperparameter Tuning Step"""
    
    def __init__(self, config: XGBoostTuningConfig, sagemaker_session=None, 
                 role: Optional[str] = None, registry_manager=None, 
                 dependency_resolver=None):
        if not isinstance(config, XGBoostTuningConfig):
            raise ValueError("XGBoostTuningStepBuilder requires XGBoostTuningConfig")
            
        if not SPEC_AVAILABLE or XGBOOST_TUNING_SPEC is None:
            raise ValueError("XGBoost tuning specification not available")
            
        super().__init__(
            config=config,
            spec=XGBOOST_TUNING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: XGBoostTuningConfig = config
    
    def validate_configuration(self) -> None:
        """Validate XGBoost tuning configuration"""
        # Validate base training configuration
        required_training_attrs = [
            'training_instance_type', 'training_instance_count', 'training_volume_size',
            'training_entry_point', 'source_dir', 'framework_version'
        ]
        
        for attr in required_training_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostTuningConfig missing required attribute: {attr}")
        
        # Validate tuning-specific configuration
        required_tuning_attrs = [
            'max_jobs', 'max_parallel_jobs', 'objective_metric_name', 'objective_type'
        ]
        
        for attr in required_tuning_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostTuningConfig missing required attribute: {attr}")
        
        # Validate hyperparameter ranges
        if not self.config.hyperparameter_ranges:
            raise ValueError("XGBoostTuningConfig must specify hyperparameter_ranges")
        
        self.log_info("XGBoostTuningConfig validation succeeded")
    
    def _create_base_estimator(self) -> XGBoost:
        """Create base XGBoost estimator for tuning"""
        source_dir = self.config.effective_source_dir
        
        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._generate_job_name(),
            hyperparameters=self._get_static_hyperparameters(),
            sagemaker_session=self.session,
            output_path=None,  # Set by tuning job
            environment=self._get_environment_variables(),
        )
    
    def _get_hyperparameter_ranges(self) -> Dict:
        """Convert config ranges to SageMaker parameter objects using separate dictionaries"""
        from sagemaker.tuner import ContinuousParameter, IntegerParameter, CategoricalParameter
        
        ranges = {}
        
        # Convert continuous parameters
        if hasattr(self.config, 'continuous_hyperparameters') and self.config.continuous_hyperparameters:
            for param_name, (min_val, max_val) in self.config.continuous_hyperparameters.items():
                ranges[param_name] = ContinuousParameter(min_val, max_val)
        
        # Convert integer parameters
        if hasattr(self.config, 'integer_hyperparameters') and self.config.integer_hyperparameters:
            for param_name, (min_val, max_val) in self.config.integer_hyperparameters.items():
                ranges[param_name] = IntegerParameter(min_val, max_val)
        
        # Convert categorical parameters
        if hasattr(self.config, 'categorical_hyperparameters') and self.config.categorical_hyperparameters:
            for param_name, values in self.config.categorical_hyperparameters.items():
                ranges[param_name] = CategoricalParameter(values)
        
        return ranges
    
    def _get_static_hyperparameters(self) -> Dict:
        """Get fixed hyperparameters that won't be tuned"""
        return self.config.static_hyperparameters.copy()
    
    def _create_hyperparameter_tuner(self) -> HyperparameterTuner:
        """Create HyperparameterTuner with XGBoost configuration"""
        estimator = self._create_base_estimator()
        
        return HyperparameterTuner(
            estimator=estimator,
            objective_metric_name=self.config.objective_metric_name,
            objective_type=self.config.objective_type,
            hyperparameter_ranges=self._get_hyperparameter_ranges(),
            max_jobs=self.config.max_jobs,
            max_parallel_jobs=self.config.max_parallel_jobs,
            strategy=self.config.strategy,
            early_stopping_type=self.config.early_stopping_type,
            base_tuning_job_name=self._generate_tuning_job_name(),
            warm_start_config=self.config.warm_start_config,
            tags=self.config.tags
        )
    
    def _generate_tuning_job_name(self) -> str:
        """Generate tuning job name following naming conventions"""
        if self.config.base_tuning_job_name:
            return self.config.base_tuning_job_name
        return f"xgboost-tuning-{self._get_timestamp()}"
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """Get inputs for tuning step (same as training step)"""
        if not self.spec or not self.contract:
            raise ValueError("Step specification and contract are required")
            
        training_inputs = {}
        
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            
            if logical_name == "input_path" and logical_name in inputs:
                base_path = inputs[logical_name]
                # Create XGBoost data channels (train, val, test)
                data_channels = self._create_data_channels_from_source(base_path)
                training_inputs.update(data_channels)
        
        return training_inputs
    
    def _create_data_channels_from_source(self, base_path):
        """Create XGBoost data channels from base path"""
        from sagemaker.workflow.functions import Join
        
        return {
            "train": TrainingInput(s3_data=Join(on="/", values=[base_path, "train/"])),
            "val": TrainingInput(s3_data=Join(on="/", values=[base_path, "val/"])),
            "test": TrainingInput(s3_data=Join(on="/", values=[base_path, "test/"]))
        }
    
    def create_step(self, **kwargs) -> TuningStep:
        """Create XGBoost TuningStep"""
        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        input_path = kwargs.get('input_path')
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs from dependencies
        inputs = {}
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
        
        inputs.update(inputs_raw)
        if input_path is not None:
            inputs["input_path"] = input_path
        
        # Get training inputs
        training_inputs = self._get_inputs(inputs)
        if not training_inputs:
            raise ValueError("No training inputs available for tuning step")
        
        # Create hyperparameter tuner
        tuner = self._create_hyperparameter_tuner()
        
        # Create tuning step
        step_name = self._get_step_name()
        tuning_step = TuningStep(
            name=step_name,
            tuner=tuner,
            inputs=training_inputs,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification
        setattr(tuning_step, '_spec', self.spec)
        
        self.log_info("Created XGBoost TuningStep: %s", step_name)
        return tuning_step
```

## Step Registry Integration

### Registry Pattern for Tuning Steps

Following the cursus framework registry pattern, tuning steps must be registered in `src/cursus/registry/step_names_original.py`:

```python
# Registry entries for tuning steps
"XGBoostTuning": {
    "config_class": "XGBoostTuningConfig",
    "builder_step_name": "XGBoostTuningStepBuilder", 
    "spec_type": "XGBoostTuning",
    "sagemaker_step_type": "Tuning",  # SageMaker step type for tuning
    "description": "XGBoost hyperparameter tuning step",
},
"PyTorchTuning": {
    "config_class": "PyTorchTuningConfig",
    "builder_step_name": "PyTorchTuningStepBuilder",
    "spec_type": "PyTorchTuning", 
    "sagemaker_step_type": "Tuning",  # SageMaker step type for tuning
    "description": "PyTorch hyperparameter tuning step",
},
"LightGBMTuning": {
    "config_class": "LightGBMTuningConfig",
    "builder_step_name": "LightGBMTuningStepBuilder",
    "spec_type": "LightGBMTuning",
    "sagemaker_step_type": "Tuning",  # SageMaker step type for tuning
    "description": "LightGBM hyperparameter tuning step",
},
```

**Key Registry Pattern Notes:**
- **sagemaker_step_type**: Must be `"Tuning"` for all tuning steps (matches SageMaker TuningStep)
- **spec_type**: Framework-specific specification type (e.g., "XGBoostTuning")
- **config_class**: Framework-specific configuration class name
- **builder_step_name**: Framework-specific builder class name

## Framework Extension Patterns

### Adding New Framework Support

To add support for a new ML framework (e.g., LightGBM), follow this pattern:

1. **Add Registry Entry**:
```python
# src/cursus/registry/step_names_original.py
"LightGBMTuning": {
    "config_class": "LightGBMTuningConfig",
    "builder_step_name": "LightGBMTuningStepBuilder",
    "spec_type": "LightGBMTuning",
    "sagemaker_step_type": "Tuning",  # Must be "Tuning" for all tuning steps
    "description": "LightGBM hyperparameter tuning step",
},
```

2. **Create Framework-Specific Specification**:
```python
# specs/lightgbm_tuning_spec.py
LIGHTGBM_TUNING_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMTuning"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_lightgbm_tuning_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "tune"],
            data_type="S3Uri",
            description="Training dataset S3 location for LightGBM hyperparameter tuning"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="best_model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.BestTrainingJob.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Best LightGBM model artifacts from hyperparameter tuning"
        )
    ]
)
```

3. **Create Framework-Specific Contract**:
```python
# contracts/lightgbm_tuning_contract.py
LIGHTGBM_TUNING_CONTRACT = TrainingScriptContract(
    entry_point="lightgbm_training.py",  # Reuse existing training script
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "best_model_output": "/opt/ml/model",
        "tuning_job_output": "/opt/ml/output/data"
    },
    framework_requirements={
        "lightgbm": ">=3.0.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.19.0"
    },
    description="LightGBM training script used for hyperparameter tuning"
)
```

4. **Create Framework-Specific Configuration**:
```python
# configs/config_lightgbm_tuning_step.py
class LightGBMTuningConfig(BasePipelineConfig):
    def __init__(self):
        super().__init__()
        
        # Base training configuration
        self.training_instance_type: str = "ml.m5.xlarge"
        self.training_instance_count: int = 1
        self.training_volume_size: int = 30
        self.training_entry_point: str = "lightgbm_training.py"
        self.source_dir: str = None
        self.framework_version: str = "3.0"
        
        # Tuning job configuration
        self.max_jobs: int = 20
        self.max_parallel_jobs: int = 3
        self.objective_metric_name: str = "validation:auc"
        self.objective_type: str = "Maximize"
        self.strategy: str = "Bayesian"
        self.early_stopping_type: str = "Auto"
        
        # LightGBM-specific hyperparameter ranges
        self.hyperparameter_ranges: Dict = {
            'num_leaves': (10, 300),           # Number of leaves
            'feature_fraction': (0.4, 1.0),   # Feature sampling ratio
            'bagging_fraction': (0.4, 1.0),   # Data sampling ratio
            'learning_rate': (0.01, 0.3),     # Learning rate
            'min_data_in_leaf': (10, 100),    # Minimum data in leaf
            'lambda_l1': (0, 10),             # L1 regularization
            'lambda_l2': (0, 10)              # L2 regularization
        }
        
        # Static hyperparameters
        self.static_hyperparameters: Dict = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_threads': 4,
            'verbosity': -1
        }
```

5. **Create Framework-Specific Builder**:
```python
# builders/builder_lightgbm_tuning_step.py
class LightGBMTuningStepBuilder(StepBuilderBase):
    """Builder for LightGBM Hyperparameter Tuning Step"""
    
    def __init__(self, config: LightGBMTuningConfig, sagemaker_session=None, 
                 role: Optional[str] = None, registry_manager=None, 
                 dependency_resolver=None):
        if not isinstance(config, LightGBMTuningConfig):
            raise ValueError("LightGBMTuningStepBuilder requires LightGBMTuningConfig")
            
        super().__init__(
            config=config,
            spec=LIGHTGBM_TUNING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: LightGBMTuningConfig = config
    
    def _create_base_estimator(self) -> LightGBM:
        """Create base LightGBM estimator for tuning"""
        return LightGBM(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.effective_source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._generate_job_name(),
            hyperparameters=self._get_static_hyperparameters(),
            sagemaker_session=self.session,
            output_path=None,  # Set by tuning job
            environment=self._get_environment_variables(),
        )
    
    def _get_hyperparameter_ranges(self) -> Dict:
        """Convert LightGBM config ranges to SageMaker parameter objects"""
        ranges = {}
        
        for param_name, param_range in self.config.hyperparameter_ranges.items():
            if param_name in ['feature_fraction', 'bagging_fraction', 'learning_rate', 'lambda_l1', 'lambda_l2']:
                # Continuous parameters
                min_val, max_val = param_range
                ranges[param_name] = ContinuousParameter(min_val, max_val)
            elif param_name in ['num_leaves', 'min_data_in_leaf']:
                # Integer parameters
                min_val, max_val = param_range
                ranges[param_name] = IntegerParameter(min_val, max_val)
        
        return ranges
    
    # ... rest of the implementation following XGBoost pattern
```

### Registry Integration Requirements

When adding new tuning step types, ensure:

1. **Registry Entry**: Add to `STEP_NAMES` in `step_names_original.py`
2. **SageMaker Step Type**: Always use `"Tuning"` for `sagemaker_step_type`
3. **Consistent Naming**: Follow `{Framework}Tuning` pattern for step names
4. **Complete Implementation**: Include all four components (spec, contract, config, builder)
5. **Framework Alignment**: Ensure tuning step reuses corresponding training step scripts and patterns

This pattern analysis provides the foundation for creating comprehensive, framework-specific validation in the universal tester framework for Tuning steps, while maintaining consistency with the existing cursus framework architecture.
