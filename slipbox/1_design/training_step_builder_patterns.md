---
tags:
  - design
  - step_builders
  - training_steps
  - patterns
  - sagemaker
keywords:
  - training step patterns
  - PyTorch estimator
  - XGBoost estimator
  - TrainingInput
  - hyperparameters
  - model artifacts
topics:
  - step builder patterns
  - training step implementation
  - SageMaker training
  - estimator configuration
language: python
date of note: 2025-01-08
---

# Training Step Builder Patterns

## Overview

This document analyzes the common patterns found in Training step builder implementations in the cursus framework. Training steps create **TrainingStep** instances using framework-specific estimators for model training. Current implementations include PyTorchTraining and XGBoostTraining steps.

## SageMaker Step Type Classification

All Training step builders create **TrainingStep** instances using framework-specific estimators:
- **PyTorch**: PyTorch estimator for deep learning models
- **XGBoost**: XGBoost estimator for gradient boosting models
- **Framework-agnostic**: Generic estimator patterns for other frameworks

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Training step builders follow this consistent architecture:

```python
@register_builder()
class TrainingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load framework-specific specification
        spec = FRAMEWORK_TRAINING_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required training configuration
        
    def _create_estimator(self) -> Estimator:
        # Create framework-specific estimator
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build environment variables for training job
        
    def _get_inputs(self, inputs) -> Dict[str, TrainingInput]:
        # Create TrainingInput objects using specification
        
    def _get_outputs(self, outputs) -> str:
        # Return output path for model artifacts
        
    def create_step(self, **kwargs) -> TrainingStep:
        # Orchestrate step creation
```

### 2. Framework-Specific Estimator Creation Patterns

#### PyTorch Estimator Pattern
```python
def _create_estimator(self) -> PyTorch:
    # Convert hyperparameters object to dict
    hyperparameters = {}
    if hasattr(self.config, "hyperparameters") and self.config.hyperparameters:
        if hasattr(self.config.hyperparameters, "to_dict"):
            hyperparameters.update(self.config.hyperparameters.to_dict())
        else:
            # Add all non-private attributes
            for key, value in vars(self.config.hyperparameters).items():
                if not key.startswith('_'):
                    hyperparameters[key] = value
    
    return PyTorch(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        base_job_name=self._generate_job_name(),
        hyperparameters=hyperparameters,
        sagemaker_session=self.session,
        output_path=None,  # Set by create_step method
        environment=self._get_environment_variables(),
    )
```

#### XGBoost Estimator Pattern
```python
def _create_estimator(self) -> XGBoost:
    # Handle hyperparameters file upload
    hyperparameters_s3_uri = None
    if hasattr(self.config, "hyperparameters") and self.config.hyperparameters:
        hyperparameters_s3_uri = self._upload_hyperparameters_file()
    
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        base_job_name=self._generate_job_name(),
        hyperparameters={"hyperparameters_s3_uri": hyperparameters_s3_uri} if hyperparameters_s3_uri else {},
        sagemaker_session=self.session,
        output_path=None,  # Set by create_step method
        environment=self._get_environment_variables(),
    )
```

### 3. Hyperparameters Handling Patterns

#### Direct Hyperparameters Pattern (PyTorch)
```python
def _create_estimator(self) -> PyTorch:
    # Hyperparameters passed directly to estimator
    hyperparameters = {}
    if hasattr(self.config, "hyperparameters") and self.config.hyperparameters:
        if hasattr(self.config.hyperparameters, "to_dict"):
            hyperparameters.update(self.config.hyperparameters.to_dict())
        else:
            for key, value in vars(self.config.hyperparameters).items():
                if not key.startswith('_'):
                    hyperparameters[key] = value
    
    return PyTorch(
        # ... other parameters
        hyperparameters=hyperparameters,
    )
```

#### File-Based Hyperparameters Pattern (XGBoost)
```python
def _upload_hyperparameters_file(self) -> str:
    """Upload hyperparameters as JSON file to S3."""
    import json
    import tempfile
    
    # Convert hyperparameters to dict
    hyperparams_dict = {}
    if hasattr(self.config.hyperparameters, "to_dict"):
        hyperparams_dict = self.config.hyperparameters.to_dict()
    else:
        for key, value in vars(self.config.hyperparameters).items():
            if not key.startswith('_'):
                hyperparams_dict[key] = value
    
    # Create temporary file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(hyperparams_dict, f, indent=2)
        temp_file_path = f.name
    
    # Upload to S3
    s3_key = f"hyperparameters/{self._generate_job_name()}/hyperparameters.json"
    s3_uri = f"{self.config.pipeline_s3_loc}/{s3_key}"
    
    self.session.upload_data(
        path=temp_file_path,
        bucket=self.config.bucket,
        key_prefix=s3_key
    )
    
    return s3_uri

def _create_estimator(self) -> XGBoost:
    # Upload hyperparameters file and pass S3 URI
    hyperparameters_s3_uri = self._upload_hyperparameters_file()
    
    return XGBoost(
        # ... other parameters
        hyperparameters={"hyperparameters_s3_uri": hyperparameters_s3_uri},
    )
```

### 4. Training Input Patterns

#### Single Data Channel Pattern (PyTorch)
```python
def _create_data_channel_from_source(self, base_path):
    """
    Create a data channel input from a base path.
    
    For PyTorch, we create a single 'data' channel since the PyTorch script 
    expects train/val/test subdirectories within one main directory.
    """
    return {"data": TrainingInput(s3_data=base_path)}

def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    training_inputs = {}
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        if logical_name == "input_path":
            base_path = inputs[logical_name]
            data_channel = self._create_data_channel_from_source(base_path)
            training_inputs.update(data_channel)
            
    return training_inputs
```

#### Multiple Data Channels Pattern (XGBoost)
```python
def _create_data_channels_from_source(self, base_path):
    """
    Create multiple data channel inputs from a base path.
    
    For XGBoost, we create separate train/validation/test channels.
    """
    return {
        "train": TrainingInput(s3_data=f"{base_path}/train/"),
        "validation": TrainingInput(s3_data=f"{base_path}/validation/"),
        "test": TrainingInput(s3_data=f"{base_path}/test/")
    }

def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    training_inputs = {}
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        if logical_name == "input_path":
            base_path = inputs[logical_name]
            data_channels = self._create_data_channels_from_source(base_path)
            training_inputs.update(data_channels)
            
    return training_inputs
```

### 5. Output Path Handling Pattern

Training steps handle outputs differently from Processing steps:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:
    """
    Get outputs for the step using specification and contract.
    
    For training steps, this returns the output path where model artifacts 
    and evaluation results will be stored. SageMaker uses this single 
    output_path parameter for both:
    - model.tar.gz (from /opt/ml/model/)
    - output.tar.gz (from /opt/ml/output/data/)
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
        primary_output_path = f"{self.config.pipeline_s3_loc}/{framework}_training/"
        
    # Remove trailing slash for consistency
    if primary_output_path.endswith('/'):
        primary_output_path = primary_output_path[:-1]
    
    return primary_output_path
```

### 6. Environment Variables Pattern

Training steps use environment variables for configuration:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Constructs environment variables for the training job.
    These variables control training script behavior without 
    needing to pass them as hyperparameters.
    """
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add environment variables from config if they exist
    if hasattr(self.config, "env") and self.config.env:
        env_vars.update(self.config.env)
        
    return env_vars
```

### 7. Metric Definitions Pattern

Training steps define metrics for monitoring:

```python
def _get_metric_definitions(self) -> List[Dict[str, str]]:
    """
    Defines metrics to be captured from training logs.
    These metrics will be visible in SageMaker console.
    """
    return [
        {"Name": "Train Loss", "Regex": "Train Loss: ([0-9\\.]+)"},
        {"Name": "Validation Loss", "Regex": "Validation Loss: ([0-9\\.]+)"},
        {"Name": "Validation F1 Score", "Regex": "Validation F1 Score: ([0-9\\.]+)"},
        {"Name": "Validation AUC ROC", "Regex": "Validation AUC ROC: ([0-9\\.]+)"}
    ]
```

### 8. Step Creation Pattern

Training steps follow this orchestration pattern:

```python
def create_step(self, **kwargs) -> TrainingStep:
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
    
    # Create estimator
    estimator = self._create_estimator()
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create step
    training_step = TrainingStep(
        name=step_name,
        estimator=estimator,
        inputs=training_inputs,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(training_step, '_spec', self.spec)
    
    return training_step
```

## Configuration Validation Patterns

### Standard Training Configuration
```python
def validate_configuration(self) -> None:
    required_attrs = [
        'training_instance_type',
        'training_instance_count',
        'training_volume_size',
        'training_entry_point',
        'source_dir',
        'framework_version',
        'py_version'
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"TrainingConfig missing required attribute: {attr}")
```

### Framework-Specific Validation
```python
def validate_configuration(self) -> None:
    # PyTorch-specific validation
    if not self.config.framework_version.startswith('1.'):
        raise ValueError("PyTorch framework version must start with '1.'")
        
    # XGBoost-specific validation
    if self.config.framework_version not in ['1.0-1', '1.2-1', '1.3-1']:
        raise ValueError(f"Unsupported XGBoost version: {self.config.framework_version}")
```

## Key Differences Between Training Step Types

### 1. By Framework
- **PyTorch**: Deep learning framework with direct hyperparameter passing
- **XGBoost**: Gradient boosting with file-based hyperparameter handling

### 2. By Data Channel Strategy
- **Single Channel**: PyTorch uses one 'data' channel with subdirectories
- **Multiple Channels**: XGBoost uses separate train/validation/test channels

### 3. By Hyperparameter Handling
- **Direct**: PyTorch passes hyperparameters directly to estimator
- **File-based**: XGBoost uploads hyperparameters file to S3

### 4. By Output Structure
- **Model + Evaluation**: Both frameworks produce model.tar.gz and output.tar.gz
- **Path Organization**: Framework-specific output path organization

## Best Practices Identified

1. **Specification-Driven Design**: Use specifications for input/output definitions
2. **Framework-Specific Estimators**: Use appropriate estimator for each framework
3. **Hyperparameter Flexibility**: Support both object and dict hyperparameters
4. **Data Channel Strategy**: Use appropriate channel strategy for framework
5. **Output Path Management**: Handle output paths consistently
6. **Environment Variables**: Use env vars for script configuration
7. **Metric Definitions**: Define metrics for monitoring and debugging
8. **Error Handling**: Comprehensive validation with clear error messages

## Testing Implications

Training step builders should be tested for:

1. **Estimator Creation**: Correct estimator type and configuration
2. **Hyperparameter Handling**: Proper hyperparameter processing and upload
3. **Training Input Creation**: Correct TrainingInput object creation
4. **Data Channel Strategy**: Appropriate channel creation for framework
5. **Output Path Handling**: Proper output path generation and validation
6. **Environment Variables**: Correct environment variable construction
7. **Metric Definitions**: Proper metric definition format
8. **Specification Compliance**: Adherence to step specifications
9. **Framework-Specific Features**: Framework-specific validation and handling

This pattern analysis provides the foundation for creating comprehensive, framework-specific validation in the universal tester framework for Training steps.
