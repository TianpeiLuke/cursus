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
updated: 2025-09-21
---

# Transform Step Builder Patterns

## Overview

This document analyzes the unique patterns and characteristics found in Transform step builder implementations in the cursus framework. Transform steps create **TransformStep** instances using SageMaker Transformer for batch inference operations. Current implementation includes BatchTransformStep with job type variants.

**Key Finding**: TransformStep has fundamentally different output handling patterns compared to ProcessingStep and TrainingStep, requiring specialized implementation approaches.

## SageMaker Step Type Classification

All Transform step builders create **TransformStep** instances using SageMaker Transformer:
- **Batch Transform**: Batch inference on datasets using trained models
- **Job Type Variants**: Support for training, validation, testing, calibration job types
- **Model Integration**: Integration with CreateModelStep outputs

## Unique TransformStep Characteristics

### 1. **No Explicit Outputs Parameter**

**Critical Difference**: Unlike ProcessingStep, TransformStep does NOT have an `outputs` parameter in its constructor:

```python
# ProcessingStep (HAS explicit outputs)
ProcessingStep(
    name=step_name,
    processor=processor,
    inputs=proc_inputs,
    outputs=proc_outputs,  # ← Explicit outputs parameter
    # ...
)

# TransformStep (NO explicit outputs)
TransformStep(
    name=step_name,
    transformer=transformer,
    inputs=transform_input,
    # NO outputs parameter!
    # ...
)
```

### 2. **Output Path Configuration via Transformer**

TransformStep outputs are determined by the `Transformer` object's configuration:

```python
def _create_transformer(self, model_name: Union[str, Properties], output_path: Optional[str] = None) -> Transformer:
    """
    Create the SageMaker Transformer object.
    
    CRITICAL: The output_path parameter here becomes accessible via 
    step.properties.TransformOutput.S3OutputPath at runtime.
    """
    return Transformer(
        model_name=model_name,
        instance_type=self.config.transform_instance_type,
        instance_count=self.config.transform_instance_count,
        output_path=output_path,  # ← This is where outputs are configured!
        accept=self.config.accept,
        assemble_with=self.config.assemble_with,
        sagemaker_session=self.session,
    )
```

### 3. **_get_outputs Must Return String, Not Dict**

**Corrected Pattern**: Transform step `_get_outputs` should return `str` (the output path), not `Dict[str, str]`:

```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:  # ← Returns str, not Dict!
    """
    Get the S3 output path for the transform job using specification.
    
    This path will be used by the Transformer and will be accessible via 
    properties.TransformOutput.S3OutputPath at runtime.

    Args:
        outputs: Output destinations keyed by logical name

    Returns:
        str: S3 URI where transform outputs will be stored
    """
    if not self.spec:
        raise ValueError("Step specification is required")

    # Check for explicitly provided output path
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name  # e.g., "transform_output"
        if logical_name in outputs:
            self.log_info(
                "Using provided output path from '%s': %s", 
                logical_name, 
                outputs[logical_name]
            )
            return outputs[logical_name]  # Return the explicit S3 path

    # Generate default output path using specification
    base_output_path = self._get_base_output_path()
    step_type = self.spec.step_type.lower() if hasattr(self.spec, 'step_type') else 'batch_transform'
    
    from sagemaker.workflow.functions import Join
    output_path = Join(on="/", values=[base_output_path, step_type, self.config.job_type])
    
    self.log_info("Generated default output path: %s", output_path)
    return output_path
```

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Transform step builders follow this consistent architecture:

```python
@register_builder()
class TransformStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load job type-specific specification
        spec = self._load_specification_by_job_type(config.job_type)
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required transform configuration
        
    def _create_transformer(self, model_name, output_path=None) -> Transformer:
        # Create SageMaker Transformer with output_path
        
    def _get_inputs(self, inputs) -> tuple[TransformInput, Union[str, Properties]]:
        # Create TransformInput and extract model_name
        
    def _get_outputs(self, outputs) -> str:  # ← Returns str, not Dict!
        # Return the S3 output path for the Transformer
        
    def create_step(self, **kwargs) -> TransformStep:
        # Orchestrate step creation using output path from _get_outputs
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

### 3. Transform Input Processing Pattern

Transform steps use **specification-driven input processing** to handle both model_name and input data dependencies:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> Tuple[TransformInput, Union[str, Properties]]:
    """
    Create transform input using specification and provided inputs.

    This method creates a TransformInput object based on the specification
    dependencies and input data sources.

    Args:
        inputs: Input data sources keyed by logical name

    Returns:
        Tuple of (TransformInput object, model_name)

    Raises:
        ValueError: If required inputs are missing or specification is not available
    """
    if not self.spec:
        raise ValueError("Step specification is required")

    model_name = None
    input_data = None

    # Process each dependency in the specification
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name

        # Skip if optional and not provided
        if not dependency_spec.required and logical_name not in inputs:
            self.log_info(
                "Optional input '%s' not provided, skipping", logical_name
            )
            continue

        # Make sure required inputs are present
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(
                f"Required input '{logical_name}' not provided. "
                f"Expected from compatible sources: {dependency_spec.compatible_sources}"
            )

        # Handle specific dependency types based on logical name
        if logical_name == "model_name":
            model_name = inputs[logical_name]
            self.log_info(
                "Using model_name from dependencies: %s (type: %s)",
                model_name,
                dependency_spec.dependency_type,
            )
        elif logical_name == "processed_data":
            input_data = inputs[logical_name]
            self.log_info(
                "Using processed_data from dependencies: %s (type: %s)",
                input_data,
                dependency_spec.dependency_type,
            )
        else:
            # Log unexpected logical names for debugging
            self.log_warning(
                "Unexpected logical name '%s' in specification dependencies",
                logical_name,
            )

    # Final validation - ensure we got the required inputs
    if not model_name:
        raise ValueError(
            "model_name is required but not provided in inputs. "
            "Check that a model step (PytorchModel, XgboostModel) is properly connected."
        )

    if not input_data:
        raise ValueError(
            "processed_data is required but not provided in inputs. "
            "Check that a preprocessing step (TabularPreprocessing) is properly connected."
        )

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

#### **Key Specification-Driven Features:**

1. **Iterates Through Specification Dependencies**: Uses `self.spec.dependencies.items()` instead of hard-coded logical names
2. **Respects Required vs Optional**: Checks `dependency_spec.required` flag
3. **Validates Compatible Sources**: Error messages include `dependency_spec.compatible_sources` information
4. **Logical Name Consistency**: Only uses logical names defined in the specification
5. **Enhanced Error Messages**: Provides specific guidance on which step types should be connected
6. **Type-Aware Logging**: Logs the dependency type for better debugging

#### **Dependency Resolution Integration:**

The specification-driven approach enables perfect integration with the dependency resolver:

```python
# BatchTransform Calibration Specification Dependencies:
dependencies=[
    DependencySpec(
        logical_name="model_name",                    # ← Handled by _get_inputs
        dependency_type=DependencyType.CUSTOM_PROPERTY,
        required=True,
        compatible_sources=["PytorchModel", "XgboostModel"],  # ← Used in error messages
        data_type="String",
    ),
    DependencySpec(
        logical_name="processed_data",                # ← Handled by _get_inputs
        dependency_type=DependencyType.PROCESSING_OUTPUT,
        required=True,
        compatible_sources=["TabularPreprocessing-Calibration"],  # ← Used in error messages
        data_type="S3Uri",
    ),
]

# Compatible Source Steps:
# XGBoostModel provides: model_name (CUSTOM_PROPERTY, String, properties.ModelName)
# TabularPreprocessing-Calibration provides: processed_data (PROCESSING_OUTPUT, S3Uri)
```

#### **Dependency Resolver Compatibility:**

The dependency resolver will automatically resolve these dependencies with **perfect 1.0 compatibility scores**:

- **model_name**: XGBoostModel.properties.ModelName → BatchTransform.model_name
- **processed_data**: TabularPreprocessing.properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri → BatchTransform.processed_data

This ensures seamless step chaining in pipeline assembly.

### 4. Transform Input Configuration Pattern

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

### 5. **CORRECTED** Step Creation Pattern

**Critical Fix**: The `create_step` method must use the output path from `_get_outputs`:

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
    
    # CRITICAL: Get output path using specification-driven method
    output_path = self._get_outputs(outputs)  # ← Must store the result!
    
    # CRITICAL: Build the transformer with the output path
    transformer = self._create_transformer(model_name, output_path)  # ← Pass output_path!

    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create the transform step (NO outputs parameter needed)
    transform_step = TransformStep(
        name=step_name,
        transformer=transformer,  # ← Transformer has output_path configured
        inputs=transform_input,
        depends_on=dependencies or [],
        cache_config=self._get_cache_config(enable_caching) if enable_caching else None
        # NO outputs parameter - TransformStep doesn't have one!
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

## Critical Differences from Other Step Types

### 1. **Output Handling Comparison**

| Step Type | _get_outputs Return Type | Output Configuration |
|-----------|-------------------------|---------------------|
| **ProcessingStep** | `List[ProcessingOutput]` | Explicit outputs parameter |
| **TrainingStep** | `str` | Single output_path parameter |
| **TransformStep** | `str` | Configured via Transformer.output_path |

### 2. **Step Constructor Comparison**

```python
# ProcessingStep - HAS outputs parameter
ProcessingStep(name, processor, inputs, outputs, ...)

# TrainingStep - Uses estimator.output_path internally  
TrainingStep(name, estimator, inputs, ...)

# TransformStep - NO outputs parameter, uses transformer.output_path
TransformStep(name, transformer, inputs, ...)
```

### 3. **Property Path Resolution**

Transform steps use a single property path pattern:
```python
# In specification
OutputSpec(
    logical_name="transform_output",
    output_type=DependencyType.CUSTOM_PROPERTY,
    property_path="properties.TransformOutput.S3OutputPath",  # ← Always this pattern
    data_type="S3Uri",
    description="S3 location of the batch transform output",
)
```

## Best Practices Identified

1. **Specification-Driven Design**: Use specifications for input/output definitions
2. **Job Type Support**: Support multiple job types with appropriate specifications
3. **Model Integration**: Proper integration with CreateModelStep outputs
4. **Input Processing**: Comprehensive TransformInput configuration
5. **Output Path Management**: Proper output path configuration via Transformer
6. **Dependency Resolution**: Support both explicit inputs and dependency extraction
7. **Error Handling**: Comprehensive validation with clear error messages
8. **Backward Compatibility**: Continue operation even without specifications
9. **Single Source of Truth**: `_get_outputs` as the authoritative source for output paths

## Testing Implications

Transform step builders should be tested for:

1. **Transformer Creation**: Correct Transformer object creation and configuration
2. **Transform Input Processing**: Proper TransformInput object creation
3. **Model Name Handling**: Correct model name extraction from dependencies
4. **Input Data Processing**: Proper input data source handling
5. **Content Type Configuration**: Correct content type and processing options
6. **Split and Assembly**: Proper split type and assembly configuration
7. **Output Path Handling**: Correct output path generation and usage
8. **Job Type Variants**: Different behavior for different job types
9. **Specification Compliance**: Adherence to step specifications
10. **Dependency Integration**: Proper integration with model and data dependencies
11. **Property Path Resolution**: Verify `properties.TransformOutput.S3OutputPath` resolves correctly

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
Transform steps handle output paths through Transformer configuration:
```python
# CORRECT: Output path configured via Transformer
transformer = Transformer(
    model_name=model_name,
    output_path=output_path,  # ← This becomes properties.TransformOutput.S3OutputPath
    # ... other params
)

# SageMaker automatically generates final paths based on:
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

### 5. Specification Output Patterns
Transform step specifications follow a consistent pattern:
```python
# Standard transform output specification
outputs=[
    OutputSpec(
        logical_name="transform_output",  # ← Always this logical name
        output_type=DependencyType.CUSTOM_PROPERTY,
        property_path="properties.TransformOutput.S3OutputPath",  # ← Always this path
        data_type="S3Uri",
        description="S3 location of the batch transform output",
    )
]
```

### 6. Error Patterns to Avoid

**❌ WRONG - Old Pattern:**
```python
def _get_outputs(self, outputs: Dict[str, Any]) -> Dict[str, str]:
    # Returns dict for logging only - BROKEN!
    result = {}
    # ... populate result dict
    return result  # ← Wrong return type

def create_step(self, **kwargs) -> TransformStep:
    self._get_outputs(outputs)  # ← Result ignored!
    transformer = self._create_transformer(model_name)  # ← No output_path!
```

**✅ CORRECT - Fixed Pattern:**
```python
def _get_outputs(self, outputs: Dict[str, Any]) -> str:
    # Returns actual output path - CORRECT!
    # ... determine output_path
    return output_path  # ← Correct return type

def create_step(self, **kwargs) -> TransformStep:
    output_path = self._get_outputs(outputs)  # ← Store result!
    transformer = self._create_transformer(model_name, output_path)  # ← Pass output_path!
```

## Future Considerations

### 1. Multi-Model Transform
Future pattern for multi-model batch transform:
```python
# Future pattern for ensemble models
def _create_multi_model_transformer(self, model_names: List[str]) -> Transformer:
    # This would require SageMaker multi-model endpoint support
    # Currently not implemented but could be added
    pass
```

### 2. Real-time Transform
Future pattern for real-time transform endpoints:
```python
# Future pattern for real-time inference
def _create_endpoint_transformer(self, model_name: str) -> Transformer:
    # This would use SageMaker endpoints instead of batch transform
    # Different from current batch-only implementation
    pass
```

## Summary

Transform step builders have unique characteristics that distinguish them from other step types:

1. **No explicit outputs parameter** in TransformStep constructor
2. **Output path configured via Transformer object**, not step constructor
3. **_get_outputs returns string**, not dictionary or list
4. **Single property path pattern**: `properties.TransformOutput.S3OutputPath`
5. **Requires both model_name and input_data** dependencies
6. **Supports advanced input/output filtering** via JSONPath

Understanding these patterns is crucial for implementing correct Transform step builders and avoiding the common pitfalls that led to the original broken implementation.

This pattern analysis provides the foundation for creating comprehensive validation in the universal tester framework for Transform steps, focusing on batch inference operations and the unique output handling requirements that differentiate them from Processing and Training steps.
