---
tags:
  - design
  - implementation
  - registry
  - architecture
keywords:
  - single source of truth
  - registry pattern
  - step names registry
  - hyperparameter registry
  - centralized definition
  - validation
topics:
  - architectural design
  - registry pattern
  - component relationships
  - system consistency
language: python
date of note: 2025-07-31
---

# Registry as Single Source of Truth

## Overview

This document describes the design and implementation of the registry pattern used in the pipeline system, focusing on two critical registries:

1. **Step Names Registry** - The centralized registry for all pipeline step names
2. **Hyperparameter Registry** - The centralized registry for hyperparameter configurations

Both registries implement the **Single Source of Truth** design principle, which is foundational to our architecture as described in [Design Principles](../developer_guide/design_principles.md#1-single-source-of-truth).

For an in-depth exploration of how the Step Names Registry is used specifically for consistent step name generation across all components, see [Registry-Based Step Name Generation](registry_based_step_name_generation.md).

## Single Source of Truth Design Principle

The Single Source of Truth principle states that for any significant element in the system:

- Information should be defined exactly once
- There should be a clear owner for that information
- Other components should reference this authoritative source
- Validation should be centralized at the source

This principle is critical for maintaining consistency, reducing errors, and simplifying maintenance across the pipeline system.

## Step Names Registry

### Purpose

The Step Names Registry serves as the single source of truth for step naming across the entire system. It ensures:

1. Consistent naming between configurations, builders, and specifications
2. Prevention of step name collisions
3. Centralized validation of step name references
4. Clear mapping between related components
5. Support for job type variants (e.g., training, calibration) with a standardized naming convention

> **See Also**: [Registry-Based Step Name Generation](registry_based_step_name_generation.md) for detailed information on how the registry ensures consistent step naming across the entire system.

### Structure

The registry is implemented in `src/pipeline_registry/step_names.py` and has the following key components:

#### Core Registry Definition

```python
STEP_NAMES = {
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",           # For config registry
        "builder_step_name": "PyTorchTrainingStepBuilder", # For builder registry
        "spec_type": "PyTorchTraining",                    # For StepSpecification.step_type
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    # ... other steps
}
```

Each step has a canonical name as its key, with a dictionary of related components.

#### Derived Mappings

From the core `STEP_NAMES` dictionary, several specialized mappings are derived:

- `CONFIG_STEP_REGISTRY`: Maps config class names to step names
- `BUILDER_STEP_NAMES`: Maps builder class names to step names
- `SPEC_STEP_TYPES`: Maps step names to specification types

#### Helper Functions

The registry provides functions to access and validate information:

- `get_config_class_name`: Retrieve config class name for a step
- `get_builder_step_name`: Retrieve builder step name for a step
- `get_spec_step_type`: Get specification type for a step
- `get_spec_step_type_with_job_type`: Get specification with optional job type
- `validate_step_name`: Check if a step name exists
- `validate_spec_type`: Verify specification type validity
- `list_all_step_info`: Get complete information for all steps

### Integration with Other Components

The Step Names Registry integrates with other components in the following ways:

1. **[Type-Aware Configuration Serialization](type_aware_serializer.md)**: Uses the registry to:
   - Find the correct class for deserialization
   - Generate consistent step names for configuration objects
   - Support the three-tier configuration hierarchy
   - Handle job type variants during serialization

   ```python
   # In TypeAwareConfigSerializer
   def generate_step_name(self, config: Any) -> str:
       """Generate a step name for a config, including job type and other attributes."""
       # Look up the step name from the registry (primary source of truth)
       from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
       if class_name in CONFIG_STEP_REGISTRY:
           base_step = CONFIG_STEP_REGISTRY[class_name]
       # ...
   ```

2. **Step Builder Creation**: Maps step names to the appropriate builder classes
3. **Step Specification**: References the registry to validate step types and maintain consistent naming
4. **Dependency Resolution**: Validates step names during dependency resolution
5. **Configuration Merging**: Ensures configuration fields are correctly associated with steps
6. **Pipeline Templates**: Provides consistent step naming for pipeline DAG construction
7. **Script Contracts**: Ensures consistency between script contracts and step specifications
8. **Job Type Handling**: Properly handles job type variants (like training/calibration) with consistent naming
9. **[Step Config Resolver](step_config_resolver.md)**: Uses the registry for:
   - Mapping DAG node names to configuration types
   - Implementing the `_config_class_to_step_type` conversion
   - Handling job type variants in node names
   - Pattern-matching between step types and node names
   - Resolving ambiguities by using canonical step names

   ```python
   # In StepConfigResolver
   def _config_class_to_step_type(self, config_class_name: str) -> str:
       """Convert configuration class name to step type using registry."""
       # Use the same logic as in registry
       from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
       if config_class_name in CONFIG_STEP_REGISTRY:
           return CONFIG_STEP_REGISTRY[config_class_name]
       
       # Fall back to simplified transformation
       step_type = config_class_name.replace('Config', '')
       # Handle special cases defined in registry
       if step_type == "CradleDataLoad":
           return "CradleDataLoading"
       # ...
       return step_type
   ```

For comprehensive details on this integration across all components, refer to the [Registry-Based Step Name Generation](registry_based_step_name_generation.md) document.

### Usage Example

```python
from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY, get_builder_step_name

# In config field manager - map config class to step name
step_name = CONFIG_STEP_REGISTRY.get(config.__class__.__name__)

# In builder creation - map step name to builder class
builder_class_name = get_builder_step_name(step_name)

# Job type handling
# Gets "CradleDataLoading_training" step type for specifications
step_type = get_spec_step_type_with_job_type("CradleDataLoading", "training")
```

## Hyperparameter Registry

### Purpose

The Hyperparameter Registry serves as the single source of truth for hyperparameter configurations across all model types. It ensures:

1. Consistent hyperparameter class naming and location
2. Centralized mapping between model types and their hyperparameters
3. Clear module path resolution for dynamic imports
4. Validated access to hyperparameter classes

### Structure

The registry is implemented in `src/pipeline_registry/hyperparameter_registry.py` and has the following key components:

#### Core Registry Definition

```python
HYPERPARAMETER_REGISTRY = {
    "HyperparameterClassName": {
        "class_name": "HyperparameterClassName",
        "module_path": "src.pipeline_steps.hyperparameters_module",
        "model_type": "model_type_identifier",
        "description": "Description of the hyperparameters"
    },
    # ... other hyperparameter classes
}
```

Each hyperparameter class has a unique name as its key, with a dictionary of metadata.

#### Helper Functions

The registry provides functions to access and validate information:

- `get_all_hyperparameter_classes`: List all registered hyperparameter classes
- `get_hyperparameter_class_by_model_type`: Find hyperparameter class for a specific model type
- `get_module_path`: Get module path for dynamic import
- `get_all_hyperparameter_info`: Get complete information for all hyperparameters
- `validate_hyperparameter_class`: Check if a hyperparameter class exists

### Integration with Other Components

The Hyperparameter Registry integrates with other components in the following ways:

1. **Training Steps**: Locate appropriate hyperparameter classes for model training
2. **Configuration Validation**: Validate hyperparameter references in configurations
3. **Model Registry**: Associate models with their corresponding hyperparameter types
4. **Hyperparameter Tuning**: Ensure consistency between tuning jobs and model types

### Usage Example

```python
from src.pipeline_registry.hyperparameter_registry import get_hyperparameter_class_by_model_type, get_module_path

# Get the appropriate hyperparameter class for a model type
hyperparam_class_name = get_hyperparameter_class_by_model_type("xgboost")

# Dynamically import the hyperparameter class
module_path = get_module_path(hyperparam_class_name)
module = importlib.import_module(module_path)
hyperparam_class = getattr(module, hyperparam_class_name)
```

## Step Builder Registry

### Purpose

The [Step Builder Registry](step_builder_registry_design.md) serves as the single source of truth for mapping between configurations and their corresponding step builder classes. It ensures:

1. Consistent builder selection for each configuration type
2. Support for job type variants with appropriate builder selection
3. Centralized validation of builder availability
4. Seamless integration between configuration objects and pipeline steps
5. Clear mapping between related components in the pipeline system

> **See Also**: [Step Builder Registry Design](step_builder_registry_design.md) for detailed information on how the registry maps configurations to step builders.

### Structure

The registry is implemented in `src/pipeline_registry/builder_registry.py` and has the following key components:

#### Core Registry Definition

```python
class StepBuilderRegistry:
    """Registry mapping step types to their builder classes."""

    def __init__(self):
        """Initialize the registry with builder mappings."""
        self._builders = {}
        self._config_to_step_type = ConfigClassToStepType()
        
        # Register core builders
        self._register_core_builders()
    
    def _register_core_builders(self):
        """Register all core step builder classes."""
        # Registration mappings from step_names.py
        from src.pipeline_registry.step_names import STEP_NAMES
        
        for step_name, info in STEP_NAMES.items():
            # Find and register the appropriate builder
            builder_class_name = info["builder_step_name"]
            self._register_builder(step_name, builder_class_name)
```

The registry uses the Step Names Registry as the authoritative source for builder information.

#### Builder Resolution

The registry provides sophisticated builder resolution capabilities:

```python
def get_builder_for_config(self, config: BasePipelineConfig) -> Type[StepBuilderBase]:
    """Get the builder class for a given configuration object."""
    config_class = type(config).__name__
    step_type = self._config_to_step_type.convert(config_class)
    
    # First try with job type if available
    if hasattr(config, 'job_type') and config.job_type:
        job_type_step = f"{step_type}_{config.job_type.lower()}"
        if job_type_step in self._builders:
            return self._builders[job_type_step]
    
    # Fall back to base step type
    if step_type in self._builders:
        return self._builders[step_type]
    
    raise ValueError(f"No builder registered for step type '{step_type}'")
```

This enables automatic resolution of the correct builder class, including handling job type variants.

#### Validation Functions

The registry includes validation capabilities:

```python
def validate_builders_for_step_types(self, step_types: List[str]) -> Dict[str, bool]:
    """Validate that builders exist for all specified step types."""
    results = {}
    for step_type in step_types:
        results[step_type] = step_type in self._builders
    return results

def check_builder_availability(self, step_type: str) -> bool:
    """Check if a builder is available for a step type."""
    return step_type in self._builders
```

### Integration with Other Components

The Step Builder Registry integrates with other components in the following ways:

1. **Pipeline DAG Compilation**: Resolves builders for each node in the pipeline DAG
2. **Step Configuration Resolution**: Works with [Step Config Resolver](step_config_resolver.md) to map configurations to builders
3. **Pipeline Template**: Uses the registry to create step instances during pipeline assembly
4. **Step Names Registry**: References the Step Names Registry for authoritative builder information
5. **Job Type Handling**: Supports job type variants through specialized builder selection

### Usage Example

```python
from src.pipeline_registry import builder_registry

# Get builder for a configuration object
builder_class = builder_registry.get_builder_for_config(config)
builder = builder_class(config)
step = builder.build()

# Check builder availability
is_available = builder_registry.check_builder_availability("PyTorchTraining")

# Validate multiple step types
validation_results = builder_registry.validate_builders_for_step_types([
    "CradleDataLoading", "TabularPreprocessing", "XGBoostTraining"
])
```

## Best Practices for Registry Usage

### Registry Access Patterns

When working with the registries, follow these access patterns for best results:

```python
# Step Names Registry - Use helper functions when possible
from src.pipeline_registry.step_names import (
    get_builder_step_name, get_spec_step_type, 
    get_spec_step_type_with_job_type
)

# For direct dictionary access
from src.pipeline_registry.step_names import (
    STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES
)

# Hyperparameter Registry - Use helper functions
from src.pipeline_registry.hyperparameter_registry import (
    get_hyperparameter_class_by_model_type, 
    get_module_path
)

# Step Builder Registry - Use registry instance
from src.pipeline_registry import builder_registry
builder_class = builder_registry.get_builder_for_config(config)
```

### Do's

1. **Always use registry functions**: Never hardcode step names or class names
2. **Add new steps to the registry first**: The registry should be the starting point for new components
3. **Validate references**: Use validation functions to check step names and hyperparameter classes
4. **Use canonical names**: Reference the canonical step names defined in the registry
5. **Keep metadata complete**: Ensure all registry entries have complete metadata
6. **Follow naming conventions**: Use consistent PascalCase for step types, XxxConfig for config classes, and XxxStepBuilder for builder classes

### Don'ts

1. **Don't duplicate registry information**: Never redefine mappings that exist in the registry
2. **Don't bypass the registry**: Don't create ad-hoc mappings between components
3. **Don't hardcode derived information**: Use the registry to look up related information
4. **Don't create parallel registries**: Extend the existing registries instead of creating new ones
5. **Don't modify the registry at runtime**: The registry should be treated as immutable during execution

## Integration with Dependency Resolution

The registries play a crucial role in dependency resolution between pipeline steps:

1. **Step Types**: The Step Names Registry defines valid step types for dependency resolution
2. **Job Type Variants**: Support for job type variants (e.g., training, calibration) in step naming
3. **Hyperparameter Compatibility**: The Hyperparameter Registry ensures compatible hyperparameters for models
4. **Cross-Validation**: Enables validation of dependencies across different step types

## Benefits of Registry Pattern

Implementing the registry pattern for step names and hyperparameters provides several benefits:

1. **Reduced Configuration Errors**: Centralized validation prevents misconfiguration
2. **Enhanced Discoverability**: Easy access to available components
3. **Simplified Maintenance**: Changes can be made in one place and propagated
4. **Clearer Dependencies**: Explicit relationships between components
5. **Self-Documenting**: The registry serves as documentation of available components
6. **Stronger Type Safety**: Validation against a defined set of options
7. **Consistent Naming**: Enforces standardized naming conventions across all components

## Conclusion

The Step Names Registry and Hyperparameter Registry embody the Single Source of Truth design principle, providing a centralized point of definition for critical pipeline components. By maintaining these registries and following best practices for their use, we ensure consistency, clarity, and correctness across the pipeline system.

This approach aligns with our broader architectural principles:

- **Single Source of Truth**: Centralized definition of key components
- **Explicit Over Implicit**: Clear, explicit relationships between components
- **Type-Safe Specifications**: Validated references to well-defined components
- **Declarative Over Imperative**: Component relationships defined declaratively

By using these registries consistently throughout the system, we create a more maintainable, robust pipeline architecture that can adapt to changing requirements while maintaining internal consistency.

## Related Documents

- [Registry-Based Step Name Generation](registry_based_step_name_generation.md): Details the implementation of consistent step naming using the registry
- [Type-Aware Serializer](type_aware_serializer.md): Serializer that uses the registry for type-aware configuration serialization
- [Step Config Resolver](step_config_resolver.md): Resolves DAG nodes to configurations using the registry
- [Step Builder Registry](step_builder_registry_design.md): Registry for mapping configurations to step builders
- [Pipeline Registry](pipeline_registry.md): Describes the broader registry pattern used across the pipeline system
- [Config Types Format](config_types_format.md): Explains the format used for configuration type definitions in the registry
- [Job Type Variant Handling](job_type_variant_handling.md): Covers how job type variants are handled consistently using the registry
