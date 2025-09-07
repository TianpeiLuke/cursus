---
tags:
  - code
  - registry
  - builder_registry
  - step_builders
  - auto_discovery
keywords:
  - StepBuilderRegistry
  - builder registry
  - step builders
  - auto-discovery
  - pipeline construction
topics:
  - builder registry
  - step builder management
  - auto-discovery
language: python
date of note: 2024-12-07
---

# Builder Registry

Step Builder Registry for the Pipeline API that provides a centralized registry mapping configuration types to step builder classes, enabling automatic resolution during pipeline construction.

## Overview

The builder registry module provides a sophisticated system for managing step builder classes in the pipeline system. The `StepBuilderRegistry` class serves as a centralized registry that maps step types to their corresponding builder classes, enabling automatic resolution during pipeline construction. The registry uses the step_names registry as the single source of truth for step naming and supports auto-discovery of step builders.

The system includes advanced features such as automatic builder discovery through module scanning, intelligent configuration-to-builder mapping with job type support, legacy alias handling for backward compatibility, validation framework integration for step registration, and comprehensive registry validation and statistics.

## Classes and Methods

### Classes
- [`StepBuilderRegistry`](#stepbuilderregistry) - Centralized registry mapping step types to builder classes
- [`register_builder`](#register_builder) - Decorator for auto-registering step builders

### Functions
- [`get_global_registry`](#get_global_registry) - Get the global step builder registry instance
- [`register_global_builder`](#register_global_builder) - Register a builder in the global registry
- [`list_global_step_types`](#list_global_step_types) - List all step types in the global registry

## API Reference

### StepBuilderRegistry

_class_ cursus.registry.builder_registry.StepBuilderRegistry()

Centralized registry mapping step types to builder classes. This registry maintains the mapping between step types and their corresponding step builder classes, enabling automatic resolution during pipeline construction. It uses the step_names registry as the single source of truth for step naming and has been enhanced to handle job type variants.

```python
from cursus.registry.builder_registry import StepBuilderRegistry

# Create registry instance
registry = StepBuilderRegistry()

# Get builder for configuration
config = XGBoostTrainingConfig()
builder_class = registry.get_builder_for_config(config)

# List all supported step types
supported_types = registry.list_supported_step_types()
print(f"Registry supports {len(supported_types)} step types")
```

#### get_builder_for_config

get_builder_for_config(_config_, _node_name=None_)

Get step builder class for a specific configuration with intelligent resolution that considers job types and node names.

**Parameters:**
- **config** (_BasePipelineConfig_) – Configuration instance to find builder for.
- **node_name** (_Optional[str]_) – Original DAG node name for enhanced resolution. Defaults to None.

**Returns:**
- **Type[StepBuilderBase]** – Step builder class for the configuration.

**Raises:**
- **RegistryError** – If no builder found for config type with detailed error information.

```python
from cursus.registry.builder_registry import StepBuilderRegistry

registry = StepBuilderRegistry()

# Basic configuration resolution
config = XGBoostTrainingConfig()
builder_class = registry.get_builder_for_config(config)
print(f"Builder: {builder_class.__name__}")

# Enhanced resolution with node name
training_config = CradleDataLoadingConfig(job_type="training")
builder_with_context = registry.get_builder_for_config(
    training_config, 
    node_name="cradle_data_loading_training"
)
print(f"Context-aware builder: {builder_with_context.__name__}")
```

#### get_builder_for_step_type

get_builder_for_step_type(_step_type_)

Get step builder class for a specific step type with legacy alias support.

**Parameters:**
- **step_type** (_str_) – Step type name to find builder for.

**Returns:**
- **Type[StepBuilderBase]** – Step builder class for the step type.

**Raises:**
- **RegistryError** – If no builder found for step type.

```python
# Get builder by step type
builder_class = registry.get_builder_for_step_type("XGBoostTraining")
print(f"XGBoost builder: {builder_class.__name__}")

# Legacy alias support
legacy_builder = registry.get_builder_for_step_type("MIMSPackaging")  # Maps to "Package"
print(f"Legacy builder: {legacy_builder.__name__}")
```

#### register_builder

register_builder(_step_type_, _builder_class_, _validation_mode="warn"_)

Register a new step builder with comprehensive validation support.

**Parameters:**
- **step_type** (_str_) – Step type name for the builder.
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to register.
- **validation_mode** (_str_) – Validation mode ("warn", "strict", "auto_correct"). Defaults to "warn".

**Returns:**
- **List[str]** – List of validation warnings/messages from the registration process.

**Raises:**
- **ValueError** – If validation fails in strict mode or builder class is invalid.

```python
from cursus.core.base.builder_base import StepBuilderBase

# Define custom step builder
class CustomProcessingStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        return self.create_processing_step(config, context)

# Register with validation
registry = StepBuilderRegistry()
warnings = registry.register_builder(
    "CustomProcessing", 
    CustomProcessingStepBuilder, 
    validation_mode="warn"
)

for warning in warnings:
    print(f"Registration warning: {warning}")
```

#### discover_builders

_classmethod_ discover_builders()

Automatically discover and register step builders through module scanning.

**Returns:**
- **Dict[str, Type[StepBuilderBase]]** – Dictionary of discovered builders mapped by step type.

```python
# Discover all available builders
discovered_builders = StepBuilderRegistry.discover_builders()
print(f"Discovered {len(discovered_builders)} builders:")

for step_type, builder_class in discovered_builders.items():
    print(f"  {step_type} -> {builder_class.__name__}")
```

#### list_supported_step_types

list_supported_step_types()

List all supported step types including legacy aliases.

**Returns:**
- **List[str]** – Sorted list of all supported step type names.

```python
# Get all supported step types
supported_types = registry.list_supported_step_types()
print(f"Supported step types ({len(supported_types)}):")
for step_type in supported_types[:10]:  # Show first 10
    print(f"  - {step_type}")
```

#### is_step_type_supported

is_step_type_supported(_step_type_)

Check if a step type is supported by the registry.

**Parameters:**
- **step_type** (_str_) – Step type name to check.

**Returns:**
- **bool** – True if step type is supported, False otherwise.

```python
# Check step type support
supported_types = ["XGBoostTraining", "InvalidStep", "MIMSPackaging"]

for step_type in supported_types:
    is_supported = registry.is_step_type_supported(step_type)
    status = "✓" if is_supported else "✗"
    print(f"{status} {step_type}")
```

#### validate_registry

validate_registry()

Validate the registry for consistency and completeness.

**Returns:**
- **Dict[str, List[str]]** – Dictionary with validation results containing 'valid', 'invalid', and 'missing' mappings.

```python
# Validate registry consistency
validation_results = registry.validate_registry()

print(f"Valid mappings: {len(validation_results['valid'])}")
print(f"Invalid mappings: {len(validation_results['invalid'])}")
print(f"Missing builders: {len(validation_results['missing'])}")

# Show invalid mappings
for invalid in validation_results['invalid']:
    print(f"Invalid: {invalid}")

# Show missing builders
for missing in validation_results['missing']:
    print(f"Missing: {missing}")
```

#### get_registry_stats

get_registry_stats()

Get comprehensive statistics about the registry.

**Returns:**
- **Dict[str, int]** – Dictionary with registry statistics including counts of various registry components.

```python
# Get registry statistics
stats = registry.get_registry_stats()
print("Registry Statistics:")
print(f"  Total builders: {stats['total_builders']}")
print(f"  Default builders: {stats['default_builders']}")
print(f"  Custom builders: {stats['custom_builders']}")
print(f"  Legacy aliases: {stats['legacy_aliases']}")
print(f"  Step registry names: {stats['step_registry_names']}")
```

### register_builder

register_builder(_step_type=None_)

Decorator for automatically registering step builder classes with intelligent step type detection.

**Parameters:**
- **step_type** (_Optional[str]_) – Optional step type name. If not provided, will be derived from the class name using the STEP_NAMES registry. Defaults to None.

**Returns:**
- **Callable** – Decorator function that registers the class and returns it unchanged.

**Raises:**
- **TypeError** – If decorator is used on non-StepBuilderBase subclasses.

```python
from cursus.registry.builder_registry import register_builder
from cursus.core.base.builder_base import StepBuilderBase

# Auto-register with explicit step type
@register_builder("CustomProcessing")
class CustomProcessingStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        return self.create_processing_step(config, context)

# Auto-register with automatic step type detection
@register_builder()  # Will derive step type from class name
class XGBoostTrainingStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        return self.create_training_step(config, context)
```

### get_global_registry

get_global_registry()

Get the global step builder registry instance with lazy initialization and auto-discovery.

**Returns:**
- **StepBuilderRegistry** – Global StepBuilderRegistry instance.

```python
from cursus.registry.builder_registry import get_global_registry

# Get global registry (creates if doesn't exist)
global_registry = get_global_registry()

# Use global registry operations
config = XGBoostTrainingConfig()
builder_class = global_registry.get_builder_for_config(config)

# Get registry information
stats = global_registry.get_registry_stats()
print(f"Global registry has {stats['total_builders']} builders")
```

### register_global_builder

register_global_builder(_step_type_, _builder_class_)

Register a builder in the global registry for system-wide availability.

**Parameters:**
- **step_type** (_str_) – Step type name for the builder.
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to register globally.

```python
from cursus.registry.builder_registry import register_global_builder
from cursus.core.base.builder_base import StepBuilderBase

# Define custom builder
class GlobalCustomStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        # Custom implementation
        return self.create_processing_step(config, context)

# Register in global registry
register_global_builder("GlobalCustom", GlobalCustomStepBuilder)

# Verify registration
global_registry = get_global_registry()
is_supported = global_registry.is_step_type_supported("GlobalCustom")
print(f"GlobalCustom registered: {is_supported}")
```

### list_global_step_types

list_global_step_types()

List all step types available in the global registry.

**Returns:**
- **List[str]** – List of supported step type names from the global registry.

```python
from cursus.registry.builder_registry import list_global_step_types

# Get all global step types
global_types = list_global_step_types()
print(f"Global registry supports {len(global_types)} step types:")

# Show first 10 types
for step_type in sorted(global_types)[:10]:
    print(f"  - {step_type}")
```

## Usage Examples

### Complete Builder Registry Workflow

```python
from cursus.registry.builder_registry import (
    StepBuilderRegistry,
    register_builder,
    get_global_registry
)
from cursus.core.base.builder_base import StepBuilderBase

# Create and configure registry
registry = StepBuilderRegistry()

# Validate existing registry
validation_results = registry.validate_registry()
print(f"Registry validation:")
print(f"  Valid: {len(validation_results['valid'])}")
print(f"  Invalid: {len(validation_results['invalid'])}")
print(f"  Missing: {len(validation_results['missing'])}")

# Define and register custom builder
@register_builder("CustomAnalysis")
class CustomAnalysisStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        return self.create_processing_step(
            config=config,
            context=context,
            processor_class="CustomAnalysisProcessor"
        )

# Get builder for configuration
config = CustomAnalysisConfig()
builder_class = registry.get_builder_for_config(config)
print(f"Builder for CustomAnalysis: {builder_class.__name__}")

# Get registry statistics
stats = registry.get_registry_stats()
print(f"Registry now has {stats['total_builders']} total builders")
```

### Advanced Configuration Resolution

```python
from cursus.registry.builder_registry import StepBuilderRegistry

registry = StepBuilderRegistry()

# Test different configuration scenarios
test_configs = [
    (XGBoostTrainingConfig(), "xgb_training"),
    (CradleDataLoadingConfig(job_type="training"), "cradle_data_loading_training"),
    (TabularPreprocessingConfig(job_type="validation"), "tabular_preprocessing_validation")
]

for config, node_name in test_configs:
    try:
        builder_class = registry.get_builder_for_config(config, node_name)
        print(f"✓ {config.__class__.__name__} -> {builder_class.__name__}")
    except Exception as e:
        print(f"✗ {config.__class__.__name__}: {e}")

# Test step type resolution
step_types_to_test = [
    "XGBoostTraining",
    "MIMSPackaging",  # Legacy alias
    "InvalidStepType"
]

for step_type in step_types_to_test:
    try:
        builder_class = registry.get_builder_for_step_type(step_type)
        print(f"✓ {step_type} -> {builder_class.__name__}")
    except Exception as e:
        print(f"✗ {step_type}: {e}")
```

### Builder Discovery and Registration

```python
from cursus.registry.builder_registry import StepBuilderRegistry
from cursus.core.base.builder_base import StepBuilderBase

# Discover existing builders
discovered = StepBuilderRegistry.discover_builders()
print(f"Auto-discovered {len(discovered)} builders")

# Create custom registry with discovered builders
registry = StepBuilderRegistry()

# Register additional custom builders
class AdvancedAnalysisStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        return self.create_processing_step(
            config=config,
            context=context,
            processor_class="AdvancedAnalysisProcessor"
        )

# Register with different validation modes
warnings = registry.register_builder(
    "AdvancedAnalysis", 
    AdvancedAnalysisStepBuilder, 
    validation_mode="auto_correct"
)

print(f"Registration completed with {len(warnings)} warnings")
for warning in warnings:
    print(f"  - {warning}")

# Verify all registrations
all_types = registry.list_supported_step_types()
print(f"Registry now supports {len(all_types)} step types")
```

### Registry Validation and Maintenance

```python
from cursus.registry.builder_registry import get_global_registry

# Get global registry for validation
registry = get_global_registry()

# Perform comprehensive validation
validation_results = registry.validate_registry()

print("Registry Validation Results:")
print("=" * 40)

# Show valid mappings
print(f"✓ Valid mappings ({len(validation_results['valid'])}):")
for valid in validation_results['valid'][:5]:  # Show first 5
    print(f"    {valid}")

# Show invalid mappings
if validation_results['invalid']:
    print(f"✗ Invalid mappings ({len(validation_results['invalid'])}):")
    for invalid in validation_results['invalid']:
        print(f"    {invalid}")

# Show missing builders
if validation_results['missing']:
    print(f"⚠ Missing builders ({len(validation_results['missing'])}):")
    for missing in validation_results['missing']:
        print(f"    {missing}")

# Get detailed statistics
stats = registry.get_registry_stats()
print("\nRegistry Statistics:")
print("=" * 20)
for key, value in stats.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

# Check specific step type support
critical_steps = [
    "XGBoostTraining",
    "CradleDataLoading", 
    "TabularPreprocessing",
    "Package",
    "Registration"
]

print("\nCritical Step Support:")
print("=" * 22)
for step in critical_steps:
    supported = registry.is_step_type_supported(step)
    status = "✓" if supported else "✗"
    print(f"{status} {step}")
```

### Error Handling and Troubleshooting

```python
from cursus.registry.builder_registry import StepBuilderRegistry, RegistryError

registry = StepBuilderRegistry()

# Test error handling scenarios
test_scenarios = [
    ("ValidConfig", XGBoostTrainingConfig()),
    ("InvalidConfig", None),
    ("UnknownConfig", type('UnknownConfig', (), {})())
]

for scenario_name, config in test_scenarios:
    try:
        if config is None:
            print(f"✗ {scenario_name}: Skipping None config")
            continue
            
        builder_class = registry.get_builder_for_config(config)
        print(f"✓ {scenario_name}: {builder_class.__name__}")
        
    except RegistryError as e:
        print(f"✗ {scenario_name}: Registry error")
        print(f"    Message: {e}")
        if hasattr(e, 'available_builders') and e.available_builders:
            print(f"    Available builders: {len(e.available_builders)}")
            
    except Exception as e:
        print(f"✗ {scenario_name}: Unexpected error - {e}")

# Test step type resolution with error handling
problematic_step_types = [
    "XGBoostTraining",      # Should work
    "NonExistentStep",      # Should fail
    "MIMSPackaging",        # Legacy alias, should work
    "",                     # Empty string, should fail
    None                    # None value, should fail
]

print("\nStep Type Resolution Test:")
print("=" * 28)
for step_type in problematic_step_types:
    try:
        if step_type is None:
            print(f"✗ None: Cannot resolve None step type")
            continue
            
        builder_class = registry.get_builder_for_step_type(step_type)
        print(f"✓ '{step_type}': {builder_class.__name__}")
        
    except RegistryError as e:
        print(f"✗ '{step_type}': {e}")
        
    except Exception as e:
        print(f"✗ '{step_type}': Unexpected error - {e}")
```

## Performance Considerations

The builder registry is designed for efficient operation with several performance optimizations:

### Lazy Initialization
- Registry instances are created only when first accessed
- Builder discovery runs once during initialization
- Global registry uses singleton pattern for efficiency

### Caching Strategy
- Step type mappings are cached after first resolution
- Builder class references are stored for fast lookup
- Validation results can be cached for repeated checks

### Memory Management
- Registry maintains references to classes, not instances
- Automatic cleanup of unused custom builders
- Efficient storage of legacy alias mappings

## Best Practices

### Builder Registration
1. **Use Decorators**: Prefer `@register_builder` decorator for automatic registration
2. **Explicit Step Types**: Provide explicit step types when class names don't match conventions
3. **Validation Modes**: Use "strict" mode in production, "warn" during development
4. **Error Handling**: Always handle `RegistryError` exceptions in production code

### Registry Management
1. **Global Registry**: Use global registry for system-wide builder access
2. **Validation**: Regularly validate registry consistency in CI/CD pipelines
3. **Statistics**: Monitor registry statistics for performance insights
4. **Legacy Support**: Maintain legacy aliases for backward compatibility

### Custom Builders
1. **Inheritance**: Always extend `StepBuilderBase` for custom builders
2. **Naming**: Follow naming conventions (e.g., `CustomStepBuilder`)
3. **Documentation**: Document custom builders thoroughly
4. **Testing**: Test custom builders with various configuration scenarios

## Related Components

- **[Registry Module](__init__.md)** - Main registry module initialization and exports
- **[Step Names](step_names.md)** - Enhanced step names registry with workspace awareness
- **[Exceptions](exceptions.md)** - Registry-specific exception classes
- **[Core Builder Base](../core/base/builder_base.md)** - Base class for all step builders
- **[Hybrid Manager](hybrid/manager.md)** - Unified registry manager implementation
