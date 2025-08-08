---
tags:
  - design
  - step_builders
  - transform_steps
  - patterns
  - sagemaker
keywords:
  - transform step patterns
  - batch transform
  - Transformer
  - TransformInput
  - model inference
  - batch processing
topics:
  - step builder patterns
  - transform step implementation
  - SageMaker batch transform
  - batch inference
language: python
date of note: 2025-01-08
---

# Transform Step Builder Patterns

## Overview

This document analyzes the common patterns found in Transform step builder implementations in the cursus framework. Transform steps create **TransformStep** instances using SageMaker Transformer for batch inference operations. Current implementation includes BatchTransformStep with job type variants.

## SageMaker Step Type Classification

All Transform step builders create **TransformStep** instances using SageMaker Transformer:
- **Batch Transform**: Batch inference on datasets using trained models
- **Job Type Variants**: Support for training, validation, testing, calibration job types
- **Model Integration**: Integration with CreateModelStep outputs

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Transform step builders follow this consistent architecture:

```python
@register_builder()
class TransformStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load job type-specific specification
        spec = self._load_specification_by_job_type(config.job_type)
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required transform configuration
        
    def _create_transformer(self, model_name, output_path=None) -> Transformer:
        # Create SageMaker Transformer
        
    def _get_inputs(self, inputs) -> tuple[TransformInput, Union[str, Properties]]:
        # Create TransformInput and extract model_name
        
    def _get_outputs(self, outputs) -> Dict[str, str]:
        # Process outputs (mostly for logging)
        
    def create_step(self, **kwargs) -> TransformStep:
        # Orchestrate step creation
```

### 2. Job Type-Based Specification Loading Pattern

Transform steps support multiple job types similar to Processing steps:

```python
def __init__(self, config, ...):
    job_type = config.job_type.lower()
    
    # Get specification based on job type
    if job_type == "training" and BATCH_TRANSFORM_TRAINING_SPEC is not None:
        spec = BATCH_TRANSFORM_TRAINING_SPEC
    elif job_type == "calibration" and BATCH_TRANSFORM_CALIBRATION_SPEC is not None:
        spec = BATCH_TRANSFORM_CALIBRATION_SPEC
    elif job_type == "validation" and BATCH_TRANSFORM_VALIDATION_SPEC is not None:
        spec = BATCH_TRANSFORM_VALIDATION_SPEC
    elif job_type == "testing" and BATCH_TRANSFORM_TESTING_SPEC is not None:
        spec = BATCH_TRANSFORM_TESTING_SPEC
    else:
        # Try dynamic import
        try:
            module_path = f"..pipeline_step_specs.batch_transform_{job_type}_spec"
            module = importlib.import_module(module_path, package=__package__)
            spec_var_name = f"BATCH_TRANSFORM_{job_type.upper()}_SPEC"
            if hasattr(module, spec_var_name):
                spec = getattr(module, spec_var_name)
        except (ImportError, AttributeError) as e:
            self.log_warning("Could not import specification for job type: %s", job_type)
    
    # Continue even without specification for backward compatibility
    if spec:
        self.log_info("Using specification for batch transform %s", job_type)
    else:
        self.log_info("No specification found, continuing with default behavior")
```

### 3. Transformer Creation Pattern

Transform steps create SageMaker Transformer objects:

```python
def _create_transformer(self, model_name: Union[str, Properties], output_path: Optional[str] = None) -> Transformer:
    """
    Create the SageMaker Transformer object.
    
    Args:
        model_name: Name of the model to transform with
        output_path: Optional output path for transform job results
        
    Returns:
        Configured Transformer object
    """
    return Transformer(
        model_name=model_name,
        instance_type=self.config.transform_instance_type,
        instance_count=self.config.transform_instance_count,
        output_path=output_path,  # Will be determined by SageMaker if None
        accept=self.config.accept,
        assemble_with=self.config.assemble_with,
        sagemaker_session=self.session,
    )
```

### 4. Transform Input Processing Pattern

Transform steps process both model_name and input data:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> tuple[TransformInput, Union[str, Properties]]:
    """
    Create transform input using specification and provided inputs.
    
    Args:
        inputs: Input data sources keyed by logical name
        
    Returns:
        Tuple of (TransformInput object, model_name)
        
    Raises:
        ValueError: If required inputs are missing
    """
    # Process model_name input
    model_name = None
    if 'model_name' in inputs:
        model_name = inputs['model_name']
        self.log_info("Using model_name from dependencies: %s", model_name)
    
    if not model_name:
        raise ValueError("model_name is required but not provided in inputs")
        
    # Process data input (must come from dependencies)
    input_data = None
    
    # Check for processed_data or input_data in the inputs
    if 'processed_data' in inputs:
        input_data = inputs['processed_data']
        self.log_info("Using processed_data from dependencies: %s", input_data)
    elif 'input_data' in inputs:  # backward compatibility
        input_data = inputs['input_data']
        self.log_info("Using input_data from dependencies: %s", input_data)
    
    if not input_data:
        raise ValueError("Input data source (processed_data) is required but not provided in inputs")
        
    # Create the transform input
    transform_input = TransformInput(
        data=input_data,
        content_type=self.config.content_type,
        split_type=self.config.split_type,
        join_source=self.config.join_source,
        input_filter=self.config.input_filter,
        output_filter=self.config.output_filter,
    )
    
    return transform_input, model_name
```

### 5. Transform Input Configuration Pattern

Transform steps configure various input processing options:

```python
# TransformInput configuration options
transform_input = TransformInput(
    data=input_data,                    # S3 path to input data
    content_type=self.config.content_type,      # "text/csv", "application/json", etc.
    split_type=self.config.split_type,          # "Line", "RecordIO", "TFRecord"
    join_source=self.config.join_source,        # "Input", "None"
    input_filter=self.config.input_filter,      # JSONPath for input filtering
    output_filter=self.config.output_filter,    # JSONPath for output filtering
)
```

### 6. Output Handling Pattern

Transform steps handle outputs differently from other step types:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> Dict[str, str]:
    """
    Process outputs based on specification.
    
    For batch transform, this simply returns a dictionary of output information
    for reference, as the TransformStep doesn't take explicit output destinations.
    
    Args:
        outputs: Output destinations keyed by logical name
        
    Returns:
        Dictionary of output information
    """
    # No explicit outputs need to be configured for TransformStep
    # Just log the outputs that will be available
    result = {}
    
    # If we have a specification, include output information
    if self.spec:
        for output_spec in self.spec.outputs.values():
            logical_name = output_spec.logical_name
            if logical_name in outputs:
                # If explicit output path provided
                result[logical_name] = outputs[logical_name]
            else:
                # Default transform output path will be determined by SageMaker
                result[logical_name] = f"Will be available at: {output_spec.property_path}"
    
    self.log_info("Transform step will produce outputs: %s", list(result.keys()))
    return result
```

### 7. Step Creation Pattern

Transform steps follow this orchestration pattern:

```python
def create_step(self, **kwargs) -> TransformStep:
    """
    Create a TransformStep for a batch transform.

    Args:
        **kwargs: Keyword arguments for configuring the step, including:
            - model_name: The name of the SageMaker model (string or Properties) (required)
            - inputs: Input data sources keyed by logical name
            - outputs: Output destinations keyed by logical name
            - dependencies: Optional list of Pipeline Step dependencies
            - enable_caching: Whether to enable caching for this step (default: True)

    Returns:
        TransformStep: configured batch transform step.
    """
    # Extract parameters
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        try:
            extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
            inputs.update(extracted_inputs)
        except Exception as e:
            self.log_warning("Failed to extract inputs from dependencies: %s", e)
            
    # Add explicitly provided inputs (overriding any extracted ones)
    inputs.update(inputs_raw)
    
    # Get transformer inputs and model name
    transform_input, model_name = self._get_inputs(inputs)
    
    # Process outputs (mostly for logging in batch transform case)
    self._get_outputs(outputs)
    
    # Build the transformer
    transformer = self._create_transformer(model_name)

    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create the transform step
    transform_step = TransformStep(
        name=step_name,
        transformer=transformer,
        inputs=transform_input,
        depends_on=dependencies or [],
        cache_config=self._get_cache_config(enable_caching) if enable_caching else None
    )
    
    # Attach specification for future reference
    if hasattr(self, 'spec') and self.spec:
        setattr(transform_step, '_spec', self.spec)
        
    self.log_info("Created TransformStep with name: %s", step_name)
    return transform_step
```

## Configuration Validation Patterns

### Standard Transform Configuration
```python
def validate_configuration(self) -> None:
    """
    Validate that all required transform settings are provided.
    """
    # Validate job type
    if self.config.job_type not in {"training", "testing", "validation", "calibration"}:
        raise ValueError(f"Unsupported job_type: {self.config.job_type}")
    
    # Validate other required fields
    required_attrs = [
        'transform_instance_type', 
        'transform_instance_count'
    ]
    
    for attr in required_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required attribute: {attr}")
            
    self.log_info("BatchTransformStepBuilder configuration for '%s' validated.", self.config.job_type)
```

### Transform-Specific Configuration
```python
def validate_configuration(self) -> None:
    # Validate transform-specific settings
    valid_content_types = ["text/csv", "application/json", "text/plain"]
    if self.config.content_type not in valid_content_types:
        raise ValueError(f"Invalid content_type: {self.config.content_type}")
        
    valid_split_types = ["Line", "RecordIO", "TFRecord", "None"]
    if self.config.split_type not in valid_split_types:
        raise ValueError(f"Invalid split_type: {self.config.split_type}")
        
    valid_assemble_with = ["Line", "None"]
    if self.config.assemble_with not in valid_assemble_with:
        raise ValueError(f"Invalid assemble_with: {self.config.assemble_with}")
```

## Key Differences Between Transform Step Types

### 1. By Job Type
- **Training**: Transform training data for model evaluation
- **Validation**: Transform validation data for model validation
- **Testing**: Transform test data for final model assessment
- **Calibration**: Transform calibration data for model calibration

### 2. By Input Processing
- **Content Type**: Different data formats (CSV, JSON, plain text)
- **Split Type**: Different data splitting strategies
- **Filtering**: Input/output filtering with JSONPath

### 3. By Output Assembly
- **Line Assembly**: Assemble outputs line by line
- **No Assembly**: Keep outputs separate
- **Join Source**: Whether to join with input data

### 4. By Model Integration
- **Model Name**: Integration with CreateModelStep outputs
- **Properties**: Use of SageMaker Pipeline Properties for model references

## Best Practices Identified

1. **Specification-Driven Design**: Use specifications for input/output definitions
2. **Job Type Support**: Support multiple job types with appropriate specifications
3. **Model Integration**: Proper integration with CreateModelStep outputs
4. **Input Processing**: Comprehensive TransformInput configuration
5. **Output Handling**: Appropriate output path and assembly configuration
6. **Dependency Resolution**: Support both explicit inputs and dependency extraction
7. **Error Handling**: Comprehensive validation with clear error messages
8. **Backward Compatibility**: Continue operation even without specifications

## Testing Implications

Transform step builders should be tested for:

1. **Transformer Creation**: Correct Transformer object creation and configuration
2. **Transform Input Processing**: Proper TransformInput object creation
3. **Model Name Handling**: Correct model name extraction from dependencies
4. **Input Data Processing**: Proper input data source handling
5. **Content Type Configuration**: Correct content type and processing options
6. **Split and Assembly**: Proper split type and assembly configuration
7. **Output Path Handling**: Appropriate output path generation and logging
8. **Job Type Variants**: Different behavior for different job types
9. **Specification Compliance**: Adherence to step specifications
10. **Dependency Integration**: Proper integration with model and data dependencies

## Special Considerations

### 1. Model Dependencies
Transform steps require model_name from CreateModelStep:
```python
# Model dependency pattern
dependencies = [create_model_step]
extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
# extracted_inputs should contain 'model_name' from CreateModelStep properties
```

### 2. Data Dependencies
Transform steps require processed data from Processing steps:
```python
# Data dependency pattern
dependencies = [preprocessing_step]
extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
# extracted_inputs should contain 'processed_data' from ProcessingStep outputs
```

### 3. Output Path Management
Transform steps handle output paths automatically:
```python
# SageMaker automatically generates output paths based on:
# - Transform job name
# - Input data structure
# - Assembly configuration

# Output structure example:
# s3://bucket/transform-job-name/input-file-name.out
```

### 4. Filtering and Processing Options
Transform steps support advanced input/output processing:
```python
# Advanced TransformInput configuration
transform_input = TransformInput(
    data=input_data,
    content_type="application/json",
    split_type="Line",
    join_source="Input",  # Join output with input
    input_filter="$.features",  # Extract features field
    output_filter="$.predictions",  # Extract predictions field
)
```

### 5. Multi-Model Transform
Future pattern for multi-model batch transform:
```python
# Future pattern for ensemble models
def _create_multi_model_transformer(self, model_names: List[str]) -> Transformer:
    # This would require SageMaker multi-model endpoint support
    # Currently not implemented but could be added
    pass
```

This pattern analysis provides the foundation for creating comprehensive validation in the universal tester framework for Transform steps, focusing on batch inference operations and model integration patterns.
