# Step Builder Registry and Auto-Discovery Guide

**Version**: 1.0  
**Date**: July 30, 2025  
**Author**: MODS Development Team

## Overview

This guide explains how to use the Step Builder Registry system with its auto-discovery and registration capabilities. The registry provides a centralized lookup system that maps pipeline step names to their corresponding builder classes, making it easier to add new steps and maintain consistent naming across the pipeline system.

## Key Features

The Step Builder Registry offers several key features:

1. **Auto-Discovery**: Automatic discovery of step builders in the codebase
2. **Decorator-Based Registration**: Simple `@register_builder` decorator for explicit registration
3. **Single Source of Truth**: Alignment with the central step names registry
4. **Legacy Support**: Backward compatibility through aliasing
5. **Validation**: Built-in validation to catch inconsistencies

## Using the @register_builder Decorator

### Basic Usage

The simplest way to register a new step builder is with the `@register_builder` decorator:

```python
from ..pipeline_registry.builder_registry import register_builder
from .builder_step_base import StepBuilderBase

@register_builder("XGBoostTraining")
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """Builder for XGBoost training step."""
    
    # Implementation here
```

This decorator automatically registers your step builder with the specified name in the registry. When the pipeline system needs to create a step of type "XGBoostTraining", it will find and use this builder.

### Auto-Derived Name

If you don't specify a name, the decorator will:

1. First check if the class name exists in the `STEP_NAMES` registry
2. If found, use the canonical step type from the registry
3. If not found, fall back to deriving it from the class name by removing the "StepBuilder" suffix:

```python
@register_builder()  # No name provided
class TabularPreprocessingStepBuilder(StepBuilderBase):
    """Builder for tabular preprocessing step."""
    
    # Implementation here
```

In this example, the step type will be automatically set to "TabularPreprocessing".

### Registration Timing

The decorator registers the builder class when the module is imported, ensuring that all builders are available when the registry is accessed.

## Auto-Discovery Mechanism

Even without using the decorator, the registry can automatically discover step builder classes:

1. During initialization, the registry scans the `src/pipeline_steps` directory
2. It finds all files with names starting with `builder_`
3. In each file, it looks for classes that inherit from `StepBuilderBase`
4. It registers each class with a name derived from the class name

To enable auto-discovery:

1. Name your file using the pattern `builder_xxx_step.py`
2. Place it in the `src/pipeline_steps` directory
3. Make your class inherit from `StepBuilderBase`

## Integration with Step Names Registry (Single Source of Truth)

The step builder registry is now tightly integrated with the central step names registry (`step_names.py`) which serves as the single source of truth for step naming:

```python
# In src/pipeline_registry/step_names.py
STEP_NAMES = {
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder", 
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    # Other steps...
}
```

When registering builders (either through auto-discovery or decorator), the registry now:

1. First checks the `STEP_NAMES` registry using a reverse mapping from builder class name to step type
2. Looks for a matching entry in `STEP_NAMES` based on the `builder_step_name` field
3. Registers the builder with the canonical name (e.g., "XGBoostTraining")
4. Only if not found in the registry, falls back to string manipulation (removing "StepBuilder" suffix)
5. Logs warnings when builders aren't found in the registry, guiding developers to update the registry

This registry-first approach ensures consistent naming across all pipeline components and makes the system more robust against errors that could occur with string manipulation alone.

## Using the Registry

### Getting a Builder for a Configuration

To get a builder for a specific configuration:

```python
from src.pipeline_registry.builder_registry import get_global_registry

# Get the global registry instance
registry = get_global_registry()

# Get a builder class for a configuration
builder_class = registry.get_builder_for_config(my_config)
builder = builder_class(my_config)
```

### Getting a Builder by Step Type

To get a builder for a specific step type:

```python
# Get a builder class by step type
builder_class = registry.get_builder_for_step_type("XGBoostTraining")
builder = builder_class(my_config)
```

### Listing Supported Step Types

To list all supported step types:

```python
# List all supported step types
step_types = registry.list_supported_step_types()
print(f"Supported step types: {step_types}")
```

### Validating the Registry

To check for inconsistencies in the registry:

```python
# Validate the registry
validation = registry.validate_registry()

# Check for issues
if validation['invalid']:
    print("Invalid registry entries:")
    for entry in validation['invalid']:
        print(f"  - {entry}")

if validation['missing']:
    print("Missing registry entries:")
    for entry in validation['missing']:
        print(f"  - {entry}")
```

## Complete Example

Here's a complete example of implementing and registering a step builder:

```python
# src/pipeline_steps/builder_new_processing_step.py

from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ..pipeline_registry.builder_registry import register_builder
from ..pipeline_deps.base_specifications import StepSpecification
from ..pipeline_script_contracts.base_script_contract import ScriptContract
from .builder_step_base import StepBuilderBase
from .config_new_processing_step import NewProcessingStepConfig
from ..pipeline_step_specs.new_processing_step_spec import NEW_PROCESSING_STEP_SPEC

@register_builder("NewProcessing")
class NewProcessingStepBuilder(StepBuilderBase):
    """Builder for a new processing step."""
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        if not isinstance(config, NewProcessingStepConfig):
            raise ValueError(
                "NewProcessingStepBuilder requires a NewProcessingStepConfig instance."
            )
            
        super().__init__(
            config=config,
            spec=NEW_PROCESSING_STEP_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: NewProcessingStepConfig = config
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_outputs(outputs)
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        # Get standard environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Add custom environment variables
        env_vars.update({
            "CUSTOM_PARAM": self.config.custom_param
        })
        
        return env_vars
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step."""
        # Extract inputs from dependencies
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        inputs = self._get_inputs(extracted_inputs)
        outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Set environment variables
        env_vars = self._get_environment_variables()
        
        # Create and return the step
        step = processor.run(
            inputs=inputs,
            outputs=outputs,
            container_arguments=[],
            container_entrypoint=["python", self.config.get_script_path()],
            job_name=self._generate_job_name(),
            wait=False,
            environment=env_vars
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

## Best Practices

### 1. Always Use the Decorator

For clarity and explicitness, always use the `@register_builder` decorator even though the auto-discovery mechanism might find your builder without it.

### 2. Match Names in Registry

Ensure that the step type name used in `@register_builder` matches the canonical name in `STEP_NAMES`:

```python
# In src/pipeline_registry/step_names.py
STEP_NAMES = {
    "NewProcessing": {  # This key...
        "config_class": "NewProcessingStepConfig",
        "builder_step_name": "NewProcessingStepBuilder", 
        "spec_type": "NewProcessing",
        "description": "New processing step"
    }
}

# In src/pipeline_steps/builder_new_processing_step.py
@register_builder("NewProcessing")  # ...should match this name
class NewProcessingStepBuilder(StepBuilderBase):
    # Implementation...
```

### 3. Follow Naming Conventions

Adhere to naming conventions to ensure consistent discovery and registration:

- Step Builder Class: `PascalCaseStepBuilder` (e.g., `NewProcessingStepBuilder`)
- File Name: `builder_snake_case_step.py` (e.g., `builder_new_processing_step.py`)
- Step Type: `PascalCase` (e.g., `NewProcessing`)

### 4. Validate the Registry

After adding new step builders, validate the registry to catch any inconsistencies:

```python
from src.pipeline_registry.builder_registry import get_global_registry

registry = get_global_registry()
validation = registry.validate_registry()

# Check validation results
print(f"Valid entries: {len(validation['valid'])}")
print(f"Invalid entries: {validation['invalid']}")
print(f"Missing entries: {validation['missing']}")
```

### 5. Keep a Single Source of Truth

Always update the `STEP_NAMES` dictionary in `step_names.py` when adding a new step. This ensures that all parts of the system use consistent naming.

## Common Pitfalls

### 1. Missing Registry Entry

**Problem**: You've added a new step builder, but it's not being registered.

**Solutions**:
- Check that your file follows the naming convention (`builder_xxx_step.py`)
- Ensure your class inherits from `StepBuilderBase`
- Verify that you've added the `@register_builder` decorator
- Check that the step name matches an entry in `STEP_NAMES`

### 2. Inconsistent Naming

**Problem**: Your step is registered with a different name than expected.

**Solutions**:
- Use an explicit name in the decorator: `@register_builder("MyStep")`
- Ensure the class name follows the convention: `MyStepStepBuilder`
- Check that the name matches the entry in `STEP_NAMES`

### 3. Import Errors

**Problem**: The auto-discovery mechanism is failing due to import errors.

**Solutions**:
- Ensure all required imports in your builder file are available
- Check for circular imports
- Use relative imports for internal modules

### 4. Multiple Registration

**Problem**: Your step is being registered multiple times with different names.

**Solutions**:
- Use explicit names in the decorator
- Ensure your class name follows conventions
- Check that you're not manually registering the same class elsewhere

### 5. Missing from Step Names Registry

**Problem**: You're seeing warnings that your step builder isn't found in the `STEP_NAMES` registry.

**Solutions**:
- Add your step to the `STEP_NAMES` registry in `src/pipeline_registry/step_names.py`
- Make sure the `builder_step_name` field matches your class name exactly
- This is critical for ensuring your step is properly discovered and registered

## Related Resources

- [Step Builder Registry Design](../pipeline_design/step_builder_registry_design.md)
- [Step Names Registry](../pipeline_design/step_names_registry_design.md)
- [Adding a New Pipeline Step](./adding_new_pipeline_step.md)
- [Step Creation Process](./creation_process.md)
- [Standardization Rules](./standardization_rules.md)

## Recent Improvements

The step builder registry has been enhanced with the following improvements:

1. **STEP_NAMES Registry Integration**: Uses the central `STEP_NAMES` registry as the single source of truth
2. **Reverse Mapping**: Efficiently maps builder class names to canonical step types
3. **Consistent Logic**: Both decorator and discovery method use identical step type inference logic
4. **Registry-First Approach**: First checks the registry before falling back to string manipulation
5. **Enhanced Logging**: Provides clear guidance when steps are missing from the registry
6. **Backward Compatibility**: Maintains support for classes not yet added to the registry

These improvements make the registration system more reliable, consistent, and easier to maintain.
