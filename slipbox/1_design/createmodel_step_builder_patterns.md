---
tags:
  - design
  - step_builders
  - createmodel_steps
  - patterns
  - sagemaker
keywords:
  - createmodel step patterns
  - XGBoostModel
  - PyTorchModel
  - model creation
  - inference configuration
  - container images
topics:
  - step builder patterns
  - createmodel step implementation
  - SageMaker model creation
  - model deployment
language: python
date of note: 2025-01-08
---

# CreateModel Step Builder Patterns

## Overview

This document analyzes the common patterns found in CreateModel step builder implementations in the cursus framework. CreateModel steps create **CreateModelStep** instances using framework-specific model classes for model deployment preparation. Current implementations include XGBoostModel and PyTorchModel steps.

## SageMaker Step Type Classification

All CreateModel step builders create **CreateModelStep** instances using framework-specific model classes:
- **XGBoostModel**: XGBoost model for gradient boosting inference
- **PyTorchModel**: PyTorch model for deep learning inference
- **Framework-agnostic**: Generic model patterns for other frameworks

## Common Implementation Patterns

### 1. Base Architecture Pattern

All CreateModel step builders follow this consistent architecture:

```python
@register_builder()
class ModelStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load framework-specific specification
        spec = FRAMEWORK_MODEL_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required model configuration
        
    def _get_image_uri(self) -> str:
        # Generate appropriate container image URI
        
    def _create_model(self, model_data: str) -> Model:
        # Create framework-specific model
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build environment variables for inference
        
    def _get_inputs(self, inputs) -> Dict[str, Any]:
        # Process model_data input using specification
        
    def _get_outputs(self, outputs) -> str:
        # Return None - CreateModelStep handles outputs automatically
        
    def create_step(self, **kwargs) -> CreateModelStep:
        # Orchestrate step creation
```

### 2. Framework-Specific Model Creation Patterns

#### XGBoost Model Pattern
```python
def _get_image_uri(self) -> str:
    """
    Generate the appropriate SageMaker XGBoost container image URI.
    Forces the region to us-east-1 regardless of configured region.
    """
    region = getattr(self.config, "aws_region", "us-east-1")
    if region != "us-east-1":
        self.log_info(f"Region '{region}' specified, but forcing to 'us-east-1'")
    region = "us-east-1"
    
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version=self.config.framework_version,
        py_version=self.config.py_version,
        instance_type=self.config.instance_type,
        image_scope="inference"
    )
    
    return image_uri

def _create_model(self, model_data: str) -> XGBoostModel:
    """
    Creates and configures the XGBoostModel for inference.
    """
    image_uri = self._get_image_uri()
        
    return XGBoostModel(
        model_data=model_data,
        role=self.role,
        entry_point=self.config.entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        image_uri=image_uri,
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )
```

#### PyTorch Model Pattern
```python
def _get_image_uri(self) -> str:
    """
    Generate the appropriate SageMaker PyTorch container image URI.
    """
    region = getattr(self.config, "aws_region", "us-east-1")
    
    image_uri = image_uris.retrieve(
        framework="pytorch",
        region=region,
        version=self.config.framework_version,
        py_version=self.config.py_version,
        instance_type=self.config.instance_type,
        image_scope="inference"
    )
    
    return image_uri

def _create_model(self, model_data: str) -> PyTorchModel:
    """
    Creates and configures the PyTorchModel for inference.
    """
    image_uri = self._get_image_uri()
        
    return PyTorchModel(
        model_data=model_data,
        role=self.role,
        entry_point=self.config.entry_point,
        source_dir=self.config.source_dir,
        framework_version=self.config.framework_version,
        py_version=self.config.py_version,
        image_uri=image_uri,
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )
```

### 3. Image URI Generation Pattern

CreateModel steps automatically generate appropriate container image URIs:

```python
def _get_image_uri(self) -> str:
    """
    Generate the appropriate SageMaker container image URI.
    Uses the SageMaker SDK's built-in image_uris.retrieve function.
    """
    region = getattr(self.config, "aws_region", "us-east-1")
    
    # Retrieve the image URI using SageMaker SDK
    image_uri = image_uris.retrieve(
        framework=self.framework_name,  # "xgboost", "pytorch", etc.
        region=region,
        version=self.config.framework_version,
        py_version=self.config.py_version,
        instance_type=self.config.instance_type,
        image_scope="inference"
    )
    
    self.log_info(f"Generated {self.framework_name} image URI: {image_uri}")
    return image_uri
```

### 4. Model Data Input Processing Pattern

CreateModel steps process model_data from training step outputs:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use specification dependencies to get model_data.
    
    Args:
        inputs: Dictionary of available inputs
        
    Returns:
        Dictionary containing processed inputs for model creation
    """
    # Spec defines: model_data dependency from XGBoostTraining, ProcessingStep, ModelArtifactsStep
    model_data_key = "model_data"  # From spec.dependencies
    
    if model_data_key not in inputs:
        raise ValueError(f"Required input '{model_data_key}' not found")
        
    return {model_data_key: inputs[model_data_key]}
```

### 5. Output Handling Pattern

CreateModel steps don't require explicit output handling:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:
    """
    Use specification outputs - returns model name.
    
    Args:
        outputs: Dictionary to store outputs (not used for CreateModelStep)
        
    Returns:
        None - CreateModelStep handles outputs automatically
    """
    # Spec defines: model output with property_path="properties.ModelName"
    # For CreateModelStep, we don't need to return specific outputs
    # The step automatically provides ModelName property
    return None
```

### 6. Environment Variables Pattern

CreateModel steps use environment variables for inference configuration:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Constructs environment variables for the model inference.
    These variables control inference script behavior.
    """
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add environment variables from config if they exist
    if hasattr(self.config, "env") and self.config.env:
        env_vars.update(self.config.env)
        
    self.log_info("Model environment variables: %s", env_vars)
    return env_vars
```

### 7. Step Creation Pattern

CreateModel steps follow this orchestration pattern:

```python
def create_step(self, **kwargs) -> CreateModelStep:
    """
    Creates the final, fully configured SageMaker CreateModelStep for the pipeline.
    
    Args:
        **kwargs: Keyword arguments for configuring the step, including:
            - inputs: Dictionary mapping input channel names to their S3 locations
            - model_data: Direct parameter for model artifacts S3 URI (backward compatibility)
            - dependencies: Optional list of steps that this step depends on.
            - enable_caching: Whether to enable caching for this step.
            
    Returns:
        A configured CreateModelStep instance.
    """
    # Extract parameters
    dependencies = self._extract_param(kwargs, 'dependencies', [])
    
    # Use dependency resolver to extract inputs
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
    else:
        # Handle direct parameters for backward compatibility
        extracted_inputs = self._normalize_inputs(kwargs.get('inputs', {}))
        model_data = self._extract_param(kwargs, 'model_data')
        if model_data:
            extracted_inputs['model_data'] = model_data
    
    # Use specification-driven input processing
    model_inputs = self._get_inputs(extracted_inputs)
    model_data_value = model_inputs['model_data']

    # Create the model
    model = self._create_model(model_data_value)

    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create the CreateModelStep
    model_step = CreateModelStep(
        name=step_name,
        step_args=model.create(
            instance_type=self.config.instance_type,
            accelerator_type=getattr(self.config, 'accelerator_type', None),
            tags=getattr(self.config, 'tags', None),
            model_name=self.config.get_model_name() if hasattr(self.config, 'get_model_name') else None
        ),
        depends_on=dependencies or []
    )
    
    # Attach specification for future reference
    setattr(model_step, '_spec', self.spec)
    
    return model_step
```

### 8. Model Creation Arguments Pattern

CreateModel steps use the model.create() method with specific arguments:

```python
def create_step(self, **kwargs) -> CreateModelStep:
    # ... input processing ...
    
    # Create the model
    model = self._create_model(model_data_value)
    
    # Create the CreateModelStep with model.create() arguments
    model_step = CreateModelStep(
        name=step_name,
        step_args=model.create(
            instance_type=self.config.instance_type,
            accelerator_type=getattr(self.config, 'accelerator_type', None),
            tags=getattr(self.config, 'tags', None),
            model_name=self.config.get_model_name() if hasattr(self.config, 'get_model_name') else None
        ),
        depends_on=dependencies or []
    )
    
    return model_step
```

## Configuration Validation Patterns

### Standard Model Configuration
```python
def validate_configuration(self) -> None:
    """
    Validates the provided configuration to ensure all required fields 
    for model creation are present and valid.
    """
    required_attrs = [
        'instance_type',
        'entry_point',
        'source_dir'
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
            raise ValueError(f"ModelStepConfig missing required attribute: {attr}")
```

### Framework-Specific Validation
```python
def validate_configuration(self) -> None:
    # XGBoost-specific validation
    if self.config.framework_version not in ['1.0-1', '1.2-1', '1.3-1']:
        raise ValueError(f"Unsupported XGBoost version: {self.config.framework_version}")
        
    # PyTorch-specific validation
    if not self.config.framework_version.startswith('1.'):
        raise ValueError("PyTorch framework version must start with '1.'")
```

## Key Differences Between CreateModel Step Types

### 1. By Framework
- **XGBoost**: Gradient boosting model with XGBoost-specific container images
- **PyTorch**: Deep learning model with PyTorch-specific container images

### 2. By Image URI Generation
- **Framework-specific**: Each framework uses different image URI patterns
- **Region handling**: Some frameworks have region-specific constraints

### 3. By Model Configuration
- **Entry point**: Framework-specific inference script requirements
- **Environment variables**: Framework-specific inference configuration

### 4. By Deployment Options
- **Instance types**: Framework-specific instance type support
- **Accelerators**: GPU/inference accelerator support varies by framework

## Best Practices Identified

1. **Specification-Driven Design**: Use specifications for input/output definitions
2. **Automatic Image URI Generation**: Use SageMaker SDK for container image URIs
3. **Framework-Specific Models**: Use appropriate model class for each framework
4. **Environment Variable Configuration**: Use env vars for inference script configuration
5. **Model Name Generation**: Support both automatic and explicit model naming
6. **Dependency Resolution**: Support both explicit inputs and dependency extraction
7. **Error Handling**: Comprehensive validation with clear error messages
8. **Specification Attachment**: Attach specs to steps for future reference

## Testing Implications

CreateModel step builders should be tested for:

1. **Model Creation**: Correct model class instantiation and configuration
2. **Image URI Generation**: Proper container image URI generation
3. **Model Data Processing**: Correct handling of model artifacts from training steps
4. **Environment Variables**: Proper environment variable construction
5. **Step Arguments**: Correct model.create() argument handling
6. **Instance Type Configuration**: Proper instance type and accelerator configuration
7. **Model Name Handling**: Both automatic and explicit model naming
8. **Specification Compliance**: Adherence to step specifications
9. **Framework-Specific Features**: Framework-specific validation and handling
10. **Dependency Integration**: Proper integration with training step outputs

## Special Considerations

### 1. Model Data Sources
CreateModel steps can receive model_data from various sources:
- **TrainingStep**: Direct model artifacts from training
- **ProcessingStep**: Processed or converted model artifacts
- **External Sources**: Pre-trained models from S3

### 2. Model Naming Strategy
```python
# Automatic model naming (recommended)
model_step = CreateModelStep(
    name=step_name,
    step_args=model.create(instance_type=self.config.instance_type),
    depends_on=dependencies
)

# Explicit model naming (when needed)
model_step = CreateModelStep(
    name=step_name,
    step_args=model.create(
        instance_type=self.config.instance_type,
        model_name=self.config.get_model_name()
    ),
    depends_on=dependencies
)
```

### 3. Multi-Container Models
For complex deployments, CreateModel steps can support multi-container models:
```python
# Future pattern for multi-container models
def _create_multi_container_model(self, model_data_list: List[str]) -> Model:
    containers = []
    for i, model_data in enumerate(model_data_list):
        containers.append({
            'Image': self._get_image_uri(),
            'ModelDataUrl': model_data,
            'Environment': self._get_environment_variables()
        })
    
    return Model(
        role=self.role,
        containers=containers,
        sagemaker_session=self.session
    )
```

This pattern analysis provides the foundation for creating comprehensive, framework-specific validation in the universal tester framework for CreateModel steps.
