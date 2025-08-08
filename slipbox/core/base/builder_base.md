---
tags:
  - code
  - core
  - base
  - builders
  - step_builders
keywords:
  - step builder base class
  - pipeline variables
  - safe logging
  - dependency resolution
  - SageMaker integration
topics:
  - builder pattern
  - pipeline construction
  - step creation
language: python
date of note: 2025-08-07
---

# Step Builder Base Class

## Overview

The `builder_base.py` module implements the foundational base class for all step builders in the cursus framework. This class provides a comprehensive foundation for creating SageMaker pipeline steps with safe logging, dependency resolution, and specification-driven architecture support.

## Purpose

This module provides:
- **Safe Pipeline Variable Handling**: Logging methods that handle SageMaker Pipeline variables safely
- **Dependency Resolution**: Integration with the UnifiedDependencyResolver for automatic dependency extraction
- **Specification-Driven Architecture**: Support for step specifications and contract validation
- **Standard Patterns**: Consistent patterns for input/output handling across all step types
- **Property Path Management**: Registry for runtime property path resolution

## Core Architecture

### Safe Logging System

The base class provides safe logging methods that handle SageMaker Pipeline variables without causing TypeErrors:

```python
def safe_value_for_logging(value):
    """
    Safely format a value for logging, handling Pipeline variables appropriately.
    
    Args:
        value: Any value that might be a Pipeline variable
        
    Returns:
        A string representation safe for logging
    """
    # Check if it's a Pipeline variable or has the expr attribute
    if hasattr(value, 'expr'):
        return f"[Pipeline Variable: {value.__class__.__name__}]"
    
    # Handle collections containing Pipeline variables
    if isinstance(value, dict):
        return "{...}"  # Avoid iterating through dict values
    if isinstance(value, (list, tuple, set)):
        return f"[{type(value).__name__} with {len(value)} items]" 
    
    # For simple values, return the string representation
    try:
        return str(value)
    except Exception:
        return f"[Object of type: {type(value).__name__}]"
```

#### Safe Logging Methods

```python
def log_info(self, message, *args, **kwargs):
    """Safely log info messages, handling Pipeline variables."""
    try:
        safe_args = [safe_value_for_logging(arg) for arg in args]
        safe_kwargs = {k: safe_value_for_logging(v) for k, v in kwargs.items()}
        logger.info(message, *safe_args, **safe_kwargs)
    except Exception as e:
        logger.info(f"Original logging failed ({e}), logging raw message: {message}")

# Similar methods: log_debug, log_warning, log_error
```

### Standard Input/Output Patterns

The base class defines standard patterns for handling inputs and outputs:

#### Configuration Pattern

```python
# In config classes:
output_names = {"logical_name": "DescriptiveValue"}  # VALUE used as key in outputs dict
input_names = {"logical_name": "ScriptInputName"}    # KEY used as key in inputs dict

# In pipeline code:
# Get output using VALUE from output_names
output_value = step_a.config.output_names["logical_name"]
output_uri = step_a.properties.ProcessingOutputConfig.Outputs[output_value].S3Output.S3Uri

# Set input using KEY from input_names
inputs = {"logical_name": output_uri}
```

#### Helper Methods

The base class provides helper methods to enforce these patterns:

- `_validate_inputs()`: Validates inputs using KEYS from input_names
- `_validate_outputs()`: Validates outputs using VALUES from output_names
- `_get_script_input_name()`: Maps logical name to script input name
- `_get_output_destination_name()`: Maps logical name to output destination name

## Initialization and Configuration

### Constructor

```python
def __init__(
    self,
    config: BasePipelineConfig,
    spec: Optional[StepSpecification] = None,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    notebook_root: Optional[Path] = None,
    registry_manager: Optional[RegistryManager] = None,
    dependency_resolver: Optional[UnifiedDependencyResolver] = None
):
```

#### Key Features

- **Configuration Management**: Stores pipeline configuration
- **Specification Support**: Optional step specification for specification-driven implementation
- **Dependency Injection**: Optional registry manager and dependency resolver
- **Contract Integration**: Automatic contract extraction from specification or config
- **Region Validation**: AWS region mapping and validation

### Region Mapping

```python
REGION_MAPPING: Dict[str, str] = {
    "NA": "us-east-1",
    "EU": "eu-west-1", 
    "FE": "us-west-2"
}
```

The base class automatically maps region codes to AWS regions and validates the configuration.

## Specification-Driven Architecture

### Specification Integration

When a specification is provided, the base class:

1. **Stores the Specification**: Available as `self.spec`
2. **Extracts Contract**: Gets script contract from specification
3. **Validates Alignment**: Ensures specification and contract align
4. **Provides Access Methods**: Methods to query dependencies and outputs

### Contract Validation

```python
# Validate specification-contract alignment if both are provided
if self.spec and self.contract and hasattr(self.spec, 'validate_contract_alignment'):
    result = self.spec.validate_contract_alignment()
    if not result.is_valid:
        raise ValueError(f"Spec-Contract alignment errors: {result.errors}")
```

### Specification Query Methods

```python
def get_required_dependencies(self) -> List[str]:
    """Get list of required dependency logical names from specification."""
    if not self.spec or not hasattr(self.spec, 'dependencies'):
        raise ValueError("Step specification is required for dependency information")
        
    return [d.logical_name for _, d in self.spec.dependencies.items() if d.required]

def get_optional_dependencies(self) -> List[str]:
    """Get list of optional dependency logical names from specification."""
    if not self.spec or not hasattr(self.spec, 'dependencies'):
        raise ValueError("Step specification is required for dependency information")
        
    return [d.logical_name for _, d in self.spec.dependencies.items() if not d.required]

def get_outputs(self) -> Dict[str, Any]:
    """Get output specifications directly from the step specification."""
    if not self.spec or not hasattr(self.spec, 'outputs'):
        raise ValueError("Step specification is required for output information")
        
    return {o.logical_name: o for _, o in self.spec.outputs.items()}
```

## Dependency Resolution

### Automatic Dependency Extraction

The base class provides automatic dependency extraction using the UnifiedDependencyResolver:

```python
def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
    """
    Extract inputs from dependency steps using the UnifiedDependencyResolver.
    
    Args:
        dependency_steps: List of dependency steps
        
    Returns:
        Dictionary of inputs extracted from dependency steps
    """
    if not DEPENDENCY_RESOLVER_AVAILABLE:
        raise ValueError("UnifiedDependencyResolver not available.")
        
    if not self.spec:
        raise ValueError("Step specification is required for dependency extraction.")
        
    # Get step name
    step_name = self.__class__.__name__.replace("Builder", "Step")
    
    # Use the injected resolver or create one
    resolver = self._get_dependency_resolver()
    resolver.register_specification(step_name, self.spec)
    
    # Register dependencies and enhance them with metadata
    available_steps = []
    self._enhance_dependency_steps_with_specs(resolver, dependency_steps, available_steps)
    
    # Resolve dependencies
    resolved = resolver.resolve_step_dependencies(step_name, available_steps)
    
    # Convert results to SageMaker properties
    return {name: prop_ref.to_sagemaker_property() for name, prop_ref in resolved.items()}
```

### Dependency Enhancement

The base class automatically enhances dependency steps with specifications:

```python
def _enhance_dependency_steps_with_specs(self, resolver, dependency_steps, available_steps):
    """
    Enhance dependency steps with specifications and additional metadata.
    
    This method extracts specifications from dependency steps and adds them to the resolver.
    It also creates minimal specifications for steps without explicit specifications.
    """
    for i, dep_step in enumerate(dependency_steps):
        dep_name = getattr(dep_step, 'name', f"Step_{i}")
        available_steps.append(dep_name)
        
        # Try to get specification from step
        dep_spec = None
        if hasattr(dep_step, '_spec'):
            dep_spec = getattr(dep_step, '_spec')
        elif hasattr(dep_step, 'spec'):
            dep_spec = getattr(dep_step, 'spec')
            
        if dep_spec:
            resolver.register_specification(dep_name, dep_spec)
            continue
        
        # Create minimal specification for steps without explicit specs
        # ... (handles model artifacts, processing outputs, etc.)
```

## Property Path Management

### Property Path Registry

The base class provides property path management for bridging definition-time and runtime:

```python
def get_property_path(self, logical_name: str, format_args: Dict[str, Any] = None) -> Optional[str]:
    """
    Get property path for an output using the specification.
    
    Args:
        logical_name: Logical name of the output
        format_args: Optional dictionary of format arguments for template paths
    
    Returns:
        Property path from specification, formatted with args if provided
    """
    property_path = None
    
    # Get property path from specification outputs
    if self.spec and hasattr(self.spec, 'outputs'):
        for _, output_spec in self.spec.outputs.items():
            if output_spec.logical_name == logical_name and output_spec.property_path:
                property_path = output_spec.property_path
                break
    
    if not property_path:
        return None
        
    # Format the path if format args are provided
    if format_args:
        try:
            property_path = property_path.format(**format_args)
        except KeyError as e:
            logger.warning(f"Missing format key {e} for property path template: {property_path}")
        except Exception as e:
            logger.warning(f"Error formatting property path: {e}")
    
    return property_path

def get_all_property_paths(self) -> Dict[str, str]:
    """Get all property paths defined in the specification."""
    paths = {}
    if self.spec and hasattr(self.spec, 'outputs'):
        for _, output_spec in self.spec.outputs.items():
            if output_spec.property_path:
                paths[output_spec.logical_name] = output_spec.property_path
    
    return paths
```

## Utility Methods

### Step Naming

```python
def _get_step_name(self, include_job_type: bool = True) -> str:
    """
    Get standard step name from builder class name, optionally including job_type.
    
    Builder class names follow the pattern: RegistryKey + "StepBuilder"
    (e.g., XGBoostTrainingStepBuilder)
    """
    class_name = self.__class__.__name__
    
    # Extract the registry key by removing the "StepBuilder" suffix
    if class_name.endswith("StepBuilder"):
        canonical_name = class_name[:-11]  # Remove "StepBuilder" suffix
    else:
        canonical_name = class_name
    
    # Add job_type suffix if requested and available
    if include_job_type and hasattr(self.config, 'job_type') and self.config.job_type:
        return f"{canonical_name}-{self.config.job_type.capitalize()}"
    
    return canonical_name
```

### Job Name Generation

```python
def _generate_job_name(self, step_type: str = None) -> str:
    """
    Generate a standardized job name for SageMaker processing/training jobs.
    
    Args:
        step_type: Optional type of step. If not provided, determined automatically.
        
    Returns:
        Sanitized job name suitable for SageMaker
    """
    import time
    
    if step_type is None:
        step_type = self._get_step_name()
    
    timestamp = int(time.time())
    
    if hasattr(self.config, 'job_type') and self.config.job_type:
        job_name = f"{step_type}-{self.config.job_type.capitalize()}-{timestamp}"
    else:
        job_name = f"{step_type}-{timestamp}"
        
    return self._sanitize_name_for_sagemaker(job_name)
```

### Name Sanitization

```python
def _sanitize_name_for_sagemaker(self, name: str, max_length: int = 63) -> str:
    """
    Sanitize a string to be a valid SageMaker resource name component.
    
    Args:
        name: Name to sanitize
        max_length: Maximum length of sanitized name
        
    Returns:
        Sanitized name
    """
    if not name:
        return "default-name"
    sanitized = "".join(c if c.isalnum() else '-' for c in str(name))
    sanitized = '-'.join(filter(None, sanitized.split('-')))
    return sanitized[:max_length].rstrip('-')
```

## Environment and Arguments

### Environment Variable Generation

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Create environment variables for the processing job based on the script contract.
    
    Returns:
        Dict[str, str]: Environment variables for the processing job
    """
    env_vars = {}
    
    if not hasattr(self, 'contract') or self.contract is None:
        self.log_warning("No script contract available for environment variable definition")
        return env_vars
    
    # Process required environment variables
    for env_var in self.contract.required_env_vars:
        config_attr = env_var.lower()
        
        # Try to get from config (direct attribute)
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        # Try to get from config.hyperparameters
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            self.log_warning(f"Required environment variable '{env_var}' not found in config")
    
    # Add optional environment variables with defaults
    for env_var, default_value in self.contract.optional_env_vars.items():
        config_attr = env_var.lower()
        
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            env_vars[env_var] = default_value
    
    return env_vars
```

### Job Arguments Generation

```python
def _get_job_arguments(self) -> Optional[List[str]]:
    """
    Constructs command-line arguments for the script based on script contract.
    
    Returns:
        List of string arguments to pass to the script, or None if no arguments
    """
    if not hasattr(self, 'contract') or not self.contract:
        self.log_warning("No contract available for argument generation")
        return None
        
    if not hasattr(self.contract, 'expected_arguments') or not self.contract.expected_arguments:
        return None
        
    args = []
    
    # Add each expected argument with its value
    for arg_name, arg_value in self.contract.expected_arguments.items():
        args.extend([f"--{arg_name}", arg_value])
    
    if args:
        self.log_info("Generated job arguments from contract: %s", args)
        return args
    
    return None
```

## Abstract Methods

### Required Implementation

All derived classes must implement these abstract methods:

```python
@abstractmethod
def validate_configuration(self) -> None:
    """
    Validate configuration requirements.
    
    Raises:
        ValueError: If configuration is invalid
    """
    pass

@abstractmethod
def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
    """
    Get inputs for the step.
    
    Each derived class returns the appropriate input type for its step:
    - ProcessingInput list for ProcessingStep
    - Training channels dict for TrainingStep
    - Model location for ModelStep
    """
    pass
    
@abstractmethod
def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
    """
    Get outputs for the step.
    
    Each derived class returns the appropriate output type for its step:
    - ProcessingOutput list for ProcessingStep
    - Output path for TrainingStep
    - Model output info for ModelStep
    """
    pass

@abstractmethod
def create_step(self, **kwargs) -> Step:
    """
    Create pipeline step.
    
    Common parameters that all step builders should handle:
    - dependencies: Optional list of steps that this step depends on
    - enable_caching: Whether to enable caching for this step (default: True)
    """
    pass
```

## Usage Patterns

### Basic Usage

```python
class MyStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        if not self.config.required_field:
            raise ValueError("required_field is missing")
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        # Convert logical inputs to ProcessingInput objects
        processing_inputs = []
        for logical_name, input_uri in inputs.items():
            processing_inputs.append(ProcessingInput(
                source=input_uri,
                destination=f"/opt/ml/processing/input/{logical_name}"
            ))
        return processing_inputs
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        # Convert logical outputs to ProcessingOutput objects
        processing_outputs = []
        for logical_name, output_uri in outputs.items():
            processing_outputs.append(ProcessingOutput(
                source=f"/opt/ml/processing/output/{logical_name}",
                destination=output_uri
            ))
        return processing_outputs
    
    def create_step(self, **kwargs) -> ProcessingStep:
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Extract inputs from dependencies if specification is available
        if self.spec and dependencies:
            inputs = self.extract_inputs_from_dependencies(dependencies)
        else:
            inputs = kwargs.get('inputs', {})
        
        outputs = kwargs.get('outputs', {})
        
        return ProcessingStep(
            name=self._get_step_name(),
            processor=self._create_processor(),
            inputs=self._get_inputs(inputs),
            outputs=self._get_outputs(outputs),
            job_arguments=self._get_job_arguments(),
            cache_config=self._get_cache_config(enable_caching),
            depends_on=dependencies
        )
```

### Specification-Driven Usage

```python
# Create builder with specification
spec = StepSpecification(...)
builder = MyStepBuilder(config, spec=spec)

# Create step with automatic dependency resolution
step = builder.create_step(
    dependencies=[data_step, model_step],
    outputs={"result": "s3://bucket/result/"}
)
```

## Integration Points

### With Configuration Classes

Step builders integrate with configuration classes that provide:
- Pipeline parameters
- Hyperparameters
- Script contracts
- Input/output mappings

### With Dependency Resolution

Step builders integrate with the dependency resolution system for:
- Automatic input extraction
- Specification-based matching
- Type-safe dependency resolution

### With Pipeline Assembly

Step builders are used by pipeline assemblers for:
- Step creation
- Dependency management
- Validation and error handling

## Best Practices

### Logging

1. **Use Safe Logging**: Always use the provided safe logging methods
2. **Meaningful Messages**: Provide clear, actionable log messages
3. **Appropriate Levels**: Use correct log levels (debug, info, warning, error)

### Error Handling

1. **Validation**: Implement thorough configuration validation
2. **Clear Messages**: Provide specific error messages with context
3. **Early Failure**: Fail fast with clear error descriptions

### Specification Usage

1. **Contract Alignment**: Always validate specification-contract alignment
2. **Property Paths**: Use specification property paths for output access
3. **Dependency Resolution**: Leverage automatic dependency resolution when available

This comprehensive base class provides a robust foundation for implementing type-safe, specification-driven step builders with automatic dependency resolution and safe pipeline variable handling.
