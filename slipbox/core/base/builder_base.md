---
tags:
  - code
  - base
  - builder_base
  - step_builders
  - abstract_base_class
keywords:
  - StepBuilderBase
  - step builders
  - abstract base class
  - pipeline steps
  - dependency resolution
  - specification-driven
  - SageMaker integration
topics:
  - step builders
  - pipeline construction
  - abstract base classes
language: python
date of note: 2024-12-07
---

# Builder Base

Abstract base class for all step builders that provides the foundational framework for creating SageMaker pipeline steps with specification-driven dependency resolution and comprehensive validation capabilities.

## Overview

The `StepBuilderBase` class serves as the abstract foundation for all step builders in the cursus pipeline framework. This class implements sophisticated patterns for step construction including specification-driven development, intelligent dependency resolution, safe logging for SageMaker Pipeline variables, and standardized input/output handling patterns.

The builder base supports advanced features including workspace-aware step registry integration, automatic dependency extraction using UnifiedDependencyResolver, script contract integration for environment variables and arguments, comprehensive validation and error handling, and standardized naming conventions for SageMaker resources.

## Classes and Methods

### Classes
- [`StepBuilderBase`](#stepbuilderbase) - Abstract base class for all step builders with specification-driven capabilities

## API Reference

### StepBuilderBase

_class_ cursus.core.base.builder_base.StepBuilderBase(_config_, _spec=None_, _sagemaker_session=None_, _role=None_, _notebook_root=None_, _registry_manager=None_, _dependency_resolver=None_)

Abstract base class for all step builders. This class provides the foundational framework for creating SageMaker pipeline steps with specification-driven dependency resolution and comprehensive validation capabilities.

**Parameters:**
- **config** (_BasePipelineConfig_) – Model configuration containing step-specific parameters and settings.
- **spec** (_Optional[StepSpecification]_) – Optional step specification for specification-driven implementation.
- **sagemaker_session** (_Optional[PipelineSession]_) – SageMaker session for pipeline execution. Defaults to None.
- **role** (_Optional[str]_) – IAM role for step execution permissions. Defaults to None.
- **notebook_root** (_Optional[Path]_) – Root directory of notebook environment. Defaults to current working directory.
- **registry_manager** (_Optional[RegistryManager]_) – Optional registry manager for dependency injection.
- **dependency_resolver** (_Optional[UnifiedDependencyResolver]_) – Optional dependency resolver for specification-based matching.

```python
from cursus.core.base.builder_base import StepBuilderBase
from cursus.core.base.specification_base import StepSpecification

class ProcessingStepBuilder(StepBuilderBase):
    """Example processing step builder."""
    
    def validate_configuration(self):
        if not hasattr(self.config, 'processing_instance_type'):
            raise ValueError("processing_instance_type is required")
    
    def _get_inputs(self, inputs):
        # Implementation for processing inputs
        pass
    
    def _get_outputs(self, outputs):
        # Implementation for processing outputs
        pass
    
    def create_step(self, **kwargs):
        # Implementation for step creation
        pass

# Create builder with specification
builder = ProcessingStepBuilder(
    config=processing_config,
    spec=processing_spec,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)
```

### Properties

#### STEP_NAMES

_property_ STEP_NAMES

Lazy load step names with workspace context awareness. This property supports workspace-aware step name resolution using hybrid registry manager with fallback to traditional registry.

**Returns:**
- **Dict[str, str]** – Step names mapping for the current workspace context.

```python
# Access step names registry
step_names = builder.STEP_NAMES
print(f"Available step types: {list(step_names.keys())}")
```

### Methods

#### get_property_path

get_property_path(_logical_name_, _format_args=None_)

Get property path for an output using the specification. This method retrieves the property path for an output from the specification with optional template formatting.

**Parameters:**
- **logical_name** (_str_) – Logical name of the output to get property path for.
- **format_args** (_Optional[Dict[str, Any]]_) – Optional dictionary of format arguments for template paths.

**Returns:**
- **Optional[str]** – Property path from specification, formatted with args if provided, or None if not found.

```python
# Get property path for model artifacts
model_path = builder.get_property_path("model_artifacts")
print(f"Model path: {model_path}")

# Get property path with formatting
data_path = builder.get_property_path(
    "processed_data", 
    format_args={"output_descriptor": "train"}
)
```

#### get_all_property_paths

get_all_property_paths()

Get all property paths defined in the specification. This method returns a complete mapping of logical output names to their runtime property paths.

**Returns:**
- **Dict[str, str]** – Mapping from logical output names to runtime property paths.

```python
# Get all property paths
all_paths = builder.get_all_property_paths()
for logical_name, path in all_paths.items():
    print(f"{logical_name}: {path}")
```

#### get_required_dependencies

get_required_dependencies()

Get list of required dependency logical names from specification. This method provides direct access to the required dependencies defined in the step specification.

**Returns:**
- **List[str]** – List of logical names for required dependencies.

```python
# Get required dependencies
required_deps = builder.get_required_dependencies()
print(f"Required dependencies: {required_deps}")
```

#### get_optional_dependencies

get_optional_dependencies()

Get list of optional dependency logical names from specification. This method provides direct access to the optional dependencies defined in the step specification.

**Returns:**
- **List[str]** – List of logical names for optional dependencies.

```python
# Get optional dependencies
optional_deps = builder.get_optional_dependencies()
print(f"Optional dependencies: {optional_deps}")
```

#### get_outputs

get_outputs()

Get output specifications directly from the step specification. This method provides direct access to the outputs defined in the step specification.

**Returns:**
- **Dict[str, Any]** – Dictionary mapping output names to their OutputSpec objects.

```python
# Get output specifications
outputs = builder.get_outputs()
for name, output_spec in outputs.items():
    print(f"Output {name}: {output_spec.property_path}")
```

#### extract_inputs_from_dependencies

extract_inputs_from_dependencies(_dependency_steps_)

Extract inputs from dependency steps using the UnifiedDependencyResolver. This method automatically resolves dependencies and creates appropriate input references.

**Parameters:**
- **dependency_steps** (_List[Step]_) – List of dependency steps to extract inputs from.

**Returns:**
- **Dict[str, Any]** – Dictionary of inputs extracted from dependency steps.

```python
# Extract inputs from dependency steps
dependency_steps = [preprocessing_step, data_loading_step]
inputs = builder.extract_inputs_from_dependencies(dependency_steps)

# Use extracted inputs in step creation
step = builder.create_step(inputs=inputs)
```

### Safe Logging Methods

The builder base provides safe logging methods that handle SageMaker Pipeline variables properly:

#### log_info

log_info(_message_, _*args_, _**kwargs_)

Safely log info messages, handling Pipeline variables. This method converts Pipeline variables to safe string representations before logging.

**Parameters:**
- **message** (_str_) – The log message with format placeholders.
- ***args** – Values to format into the message.
- ****kwargs** – Keyword values to format into the message.

```python
# Safe logging with Pipeline variables
builder.log_info("Processing input: %s", pipeline_variable)
builder.log_info("Step configuration: %s", step_config)
```

#### log_debug, log_warning, log_error

Similar safe logging methods for different log levels:
- `log_debug(message, *args, **kwargs)` - Debug level logging
- `log_warning(message, *args, **kwargs)` - Warning level logging  
- `log_error(message, *args, **kwargs)` - Error level logging

### Utility Methods

#### _generate_job_name

_generate_job_name(_step_type=None_)

Generate a standardized job name for SageMaker processing/training jobs. This method creates unique job names with timestamps and proper sanitization.

**Parameters:**
- **step_type** (_Optional[str]_) – Optional type of step. If not provided, determined automatically.

**Returns:**
- **str** – Sanitized job name suitable for SageMaker.

```python
# Generate job name
job_name = builder._generate_job_name()
print(f"Generated job name: {job_name}")

# Generate with specific step type
custom_job_name = builder._generate_job_name("CustomProcessing")
```

#### _get_environment_variables

_get_environment_variables()

Create environment variables for the processing job based on the script contract. This method extracts required and optional environment variables from the contract and config.

**Returns:**
- **Dict[str, str]** – Environment variables for the processing job.

```python
# Get environment variables from contract
env_vars = builder._get_environment_variables()
print(f"Environment variables: {env_vars}")
```

#### _get_job_arguments

_get_job_arguments()

Constructs command-line arguments for the script based on script contract. This method creates argument lists from contract specifications.

**Returns:**
- **Optional[List[str]]** – List of string arguments to pass to the script, or None if no arguments.

```python
# Get job arguments from contract
args = builder._get_job_arguments()
if args:
    print(f"Job arguments: {args}")
```

### Abstract Methods

The following methods must be implemented by subclasses:

#### validate_configuration

_abstractmethod_ validate_configuration()

Validate configuration requirements. This method should check that all required configuration parameters are present and valid.

```python
def validate_configuration(self):
    """Validate processing configuration."""
    if not hasattr(self.config, 'instance_type'):
        raise ValueError("instance_type is required")
    if not hasattr(self.config, 'instance_count'):
        raise ValueError("instance_count is required")
```

#### _get_inputs

_abstractmethod_ _get_inputs(_inputs_)

Get inputs for the step. This method should return the appropriate input type for the specific step type.

**Parameters:**
- **inputs** (_Dict[str, Any]_) – Dictionary mapping logical names to input sources.

**Returns:**
- **Any** – Appropriate inputs object for the step type.

#### _get_outputs

_abstractmethod_ _get_outputs(_outputs_)

Get outputs for the step. This method should return the appropriate output type for the specific step type.

**Parameters:**
- **outputs** (_Dict[str, Any]_) – Dictionary mapping logical names to output destinations.

**Returns:**
- **Any** – Appropriate outputs object for the step type.

#### create_step

_abstractmethod_ create_step(_**kwargs_)

Create pipeline step. This method should create and return a SageMaker pipeline step instance.

**Parameters:**
- ****kwargs** – Keyword arguments for configuring the step.

**Returns:**
- **Step** – SageMaker pipeline step instance.

## Usage Examples

### Basic Step Builder Implementation
```python
from cursus.core.base.builder_base import StepBuilderBase
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

class DataProcessingStepBuilder(StepBuilderBase):
    """Data processing step builder example."""
    
    def validate_configuration(self):
        """Validate processing configuration."""
        required_fields = ['instance_type', 'instance_count', 'script_path']
        for field in required_fields:
            if not hasattr(self.config, field):
                raise ValueError(f"{field} is required in configuration")
    
    def _get_inputs(self, inputs):
        """Convert input dictionary to ProcessingInput list."""
        processing_inputs = []
        for logical_name, input_source in inputs.items():
            processing_inputs.append(
                ProcessingInput(
                    source=input_source,
                    destination=f"/opt/ml/processing/input/{logical_name}",
                    input_name=logical_name
                )
            )
        return processing_inputs
    
    def _get_outputs(self, outputs):
        """Convert output dictionary to ProcessingOutput list."""
        processing_outputs = []
        for logical_name, output_dest in outputs.items():
            processing_outputs.append(
                ProcessingOutput(
                    source=f"/opt/ml/processing/output/{logical_name}",
                    destination=output_dest,
                    output_name=logical_name
                )
            )
        return processing_outputs
    
    def create_step(self, **kwargs):
        """Create processing step."""
        # Extract parameters
        inputs = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Create processor
        processor = SKLearnProcessor(
            framework_version=self.config.framework_version,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            role=self.role,
            sagemaker_session=self.session
        )
        
        # Generate job name
        job_name = self._generate_job_name("DataProcessing")
        
        # Get environment variables and arguments
        env_vars = self._get_environment_variables()
        job_args = self._get_job_arguments()
        
        # Create step
        step = ProcessingStep(
            name=job_name,
            processor=processor,
            inputs=self._get_inputs(inputs),
            outputs=self._get_outputs(outputs),
            code=self.config.script_path,
            job_arguments=job_args,
            env=env_vars,
            cache_config=self._get_cache_config(enable_caching),
            depends_on=dependencies
        )
        
        # Store specification for dependency resolution
        if self.spec:
            step._spec = self.spec
        
        return step
```

### Specification-Driven Builder
```python
# Create builder with specification
from cursus.core.base.specification_base import StepSpecification, DependencySpec, OutputSpec
from cursus.core.base.enums import NodeType, DependencyType

# Define specification
processing_spec = StepSpecification(
    step_type="DataProcessing",
    node_type=NodeType.INTERNAL,
    dependencies={
        "raw_data": DependencySpec(
            logical_name="raw_data",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
    }
)

# Create builder with specification
builder = DataProcessingStepBuilder(
    config=processing_config,
    spec=processing_spec,
    role="arn:aws:iam::123456789012:role/SageMakerRole"
)

# Use specification-driven features
required_deps = builder.get_required_dependencies()
output_paths = builder.get_all_property_paths()

print(f"Required dependencies: {required_deps}")
print(f"Output paths: {output_paths}")
```

### Dependency Resolution Integration
```python
# Extract inputs from dependency steps automatically
dependency_steps = [data_loading_step, preprocessing_step]

# Automatic dependency resolution
try:
    resolved_inputs = builder.extract_inputs_from_dependencies(dependency_steps)
    builder.log_info("Resolved %d inputs automatically", len(resolved_inputs))
    
    # Create step with resolved inputs
    step = builder.create_step(
        inputs=resolved_inputs,
        outputs={"processed_data": "s3://bucket/processed/"},
        enable_caching=True
    )
    
except ValueError as e:
    builder.log_error("Dependency resolution failed: %s", str(e))
    # Fallback to manual input specification
    manual_inputs = {"raw_data": "s3://bucket/raw/data.csv"}
    step = builder.create_step(inputs=manual_inputs)
```

## Related Documentation

- [Config Base](config_base.md) - Configuration classes used by step builders
- [Specification Base](specification_base.md) - Step specifications for specification-driven builders
- [Contract Base](contract_base.md) - Script contracts for environment variables and arguments
- [Dependency Resolver](../deps/dependency_resolver.md) - Automatic dependency resolution system
- [Registry Manager](../deps/registry_manager.md) - Registry management for workspace-aware operations
