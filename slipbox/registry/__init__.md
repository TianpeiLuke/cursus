---
tags:
  - code
  - registry
  - __init__
  - hybrid_registry
  - workspace_management
keywords:
  - registry module
  - hybrid registry
  - workspace context
  - step registry
  - builder registry
topics:
  - registry initialization
  - workspace management
  - hybrid registry support
language: python
date of note: 2024-12-07
---

# Registry Module Initialization

Enhanced Pipeline Registry Module with Hybrid Registry Support that provides comprehensive registry functionality for tracking step types, specifications, hyperparameters, and workspace-aware configurations.

## Overview

The registry module initialization provides a unified interface to all registry components including step names, builder registry, hyperparameter registry, and hybrid registry backend. It maintains backward compatibility while adding workspace-aware step resolution, context management for multi-developer workflows, and hybrid registry backend support.

The module exports all essential registry functions and classes, providing convenient access to registry operations, workspace context management, and hybrid registry components. It serves as the main entry point for all registry-related functionality in the cursus pipeline system.

## Classes and Methods

### Core Registry Functions
- [`get_config_class_name`](#get_config_class_name) - Get configuration class name for a step
- [`get_builder_step_name`](#get_builder_step_name) - Get builder class name for a step
- [`validate_step_name`](#validate_step_name) - Validate step name exists
- [`get_all_step_names`](#get_all_step_names) - Get all canonical step names

### Builder Registry
- [`StepBuilderRegistry`](#stepbuilderregistry) - Centralized registry for step builder classes
- [`get_global_registry`](#get_global_registry) - Get global step builder registry instance
- [`register_global_builder`](#register_global_builder) - Register builder in global registry
- [`list_global_step_types`](#list_global_step_types) - List all step types in global registry

### Workspace Management
- [`set_workspace_context`](#set_workspace_context) - Set current workspace context
- [`get_workspace_context`](#get_workspace_context) - Get current workspace context
- [`clear_workspace_context`](#clear_workspace_context) - Clear workspace context
- [`workspace_context`](#workspace_context) - Context manager for temporary workspace

### Convenience Functions
- [`switch_to_workspace`](#switch_to_workspace) - Switch to specific workspace context
- [`switch_to_core`](#switch_to_core) - Switch back to core registry
- [`get_registry_info`](#get_registry_info) - Get comprehensive registry information

## API Reference

### get_config_class_name

get_config_class_name(_step_name_, _workspace_id=None_)

Get configuration class name for a step with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Configuration class name for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry import get_config_class_name

# Get config class for core step
config_class = get_config_class_name("XGBoostTraining")
print(f"Config class: {config_class}")  # XGBoostTrainingConfig

# Get config class with workspace context
workspace_config = get_config_class_name("CustomStep", workspace_id="developer_1")
print(f"Workspace config: {workspace_config}")
```

### get_builder_step_name

get_builder_step_name(_step_name_, _workspace_id=None_)

Get builder step class name for a step with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Builder class name for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry import get_builder_step_name

# Get builder class name
builder_name = get_builder_step_name("XGBoostTraining")
print(f"Builder class: {builder_name}")  # XGBoostTrainingStepBuilder

# Get builder with workspace context
workspace_builder = get_builder_step_name("CustomStep", workspace_id="developer_1")
print(f"Workspace builder: {workspace_builder}")
```

### validate_step_name

validate_step_name(_step_name_, _workspace_id=None_)

Validate that a step name exists in the registry with workspace context support.

**Parameters:**
- **step_name** (_str_) – Step name to validate.
- **workspace_id** (_Optional[str]_) – Optional workspace context for validation. Defaults to None.

**Returns:**
- **bool** – True if step name exists, False otherwise.

```python
from cursus.registry import validate_step_name

# Validate core step
is_valid = validate_step_name("XGBoostTraining")
print(f"Valid step: {is_valid}")  # True

# Validate with workspace context
workspace_valid = validate_step_name("CustomStep", workspace_id="developer_1")
print(f"Workspace step valid: {workspace_valid}")
```

### get_all_step_names

get_all_step_names(_workspace_id=None_)

Get all canonical step names with workspace context support.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context for step resolution. Defaults to None.

**Returns:**
- **List[str]** – List of all canonical step names available in the context.

```python
from cursus.registry import get_all_step_names

# Get all core step names
core_steps = get_all_step_names()
print(f"Core steps: {len(core_steps)}")

# Get all steps in workspace context
workspace_steps = get_all_step_names(workspace_id="developer_1")
print(f"Workspace steps: {len(workspace_steps)}")
```

### StepBuilderRegistry

_class_ cursus.registry.builder_registry.StepBuilderRegistry()

Centralized registry mapping step types to builder classes. This registry maintains the mapping between step types and their corresponding step builder classes, enabling automatic resolution during pipeline construction.

```python
from cursus.registry import StepBuilderRegistry

# Create registry instance
registry = StepBuilderRegistry()

# Get builder for configuration
config = XGBoostTrainingConfig()
builder_class = registry.get_builder_for_config(config)

# List supported step types
supported_types = registry.list_supported_step_types()
print(f"Supported types: {len(supported_types)}")
```

#### get_builder_for_config

get_builder_for_config(_config_, _node_name=None_)

Get step builder class for a specific configuration with intelligent resolution.

**Parameters:**
- **config** (_BasePipelineConfig_) – Configuration instance to find builder for.
- **node_name** (_Optional[str]_) – Original DAG node name for enhanced resolution. Defaults to None.

**Returns:**
- **Type[StepBuilderBase]** – Step builder class for the configuration.

**Raises:**
- **RegistryError** – If no builder found for config type.

```python
# Get builder for configuration
config = XGBoostTrainingConfig()
builder_class = registry.get_builder_for_config(config)

# Get builder with node name context
builder_with_context = registry.get_builder_for_config(config, node_name="xgb_training")
```

#### register_builder

register_builder(_step_type_, _builder_class_, _validation_mode="warn"_)

Register a new step builder with validation support.

**Parameters:**
- **step_type** (_str_) – Step type name for the builder.
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to register.
- **validation_mode** (_str_) – Validation mode ("warn", "strict", "auto_correct"). Defaults to "warn".

**Returns:**
- **List[str]** – List of validation warnings/messages.

**Raises:**
- **ValueError** – If validation fails in strict mode or builder class is invalid.

```python
from cursus.core.base.builder_base import StepBuilderBase

# Define custom builder
class CustomStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        # Implementation
        pass

# Register with validation
warnings = registry.register_builder("CustomStep", CustomStepBuilder, "warn")
for warning in warnings:
    print(f"Warning: {warning}")
```

### get_global_registry

get_global_registry()

Get the global step builder registry instance with lazy initialization.

**Returns:**
- **StepBuilderRegistry** – Global StepBuilderRegistry instance.

```python
from cursus.registry import get_global_registry

# Get global registry
registry = get_global_registry()

# Use registry operations
supported_types = registry.list_supported_step_types()
validation_results = registry.validate_registry()
```

### register_global_builder

register_global_builder(_step_type_, _builder_class_)

Register a builder in the global registry for system-wide availability.

**Parameters:**
- **step_type** (_str_) – Step type name for the builder.
- **builder_class** (_Type[StepBuilderBase]_) – Step builder class to register globally.

```python
from cursus.registry import register_global_builder
from cursus.core.base.builder_base import StepBuilderBase

# Define and register global builder
class GlobalCustomBuilder(StepBuilderBase):
    def build_step(self, config, context):
        # Implementation
        pass

register_global_builder("GlobalCustom", GlobalCustomBuilder)
```

### set_workspace_context

set_workspace_context(_workspace_id_)

Set current workspace context for registry resolution with automatic variable refresh.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to set as current context.

```python
from cursus.registry import set_workspace_context, get_workspace_context

# Set workspace context
set_workspace_context("developer_1")
current = get_workspace_context()
print(f"Current workspace: {current}")  # developer_1

# Registry operations now use workspace context
step_names = get_all_step_names()  # Uses developer_1 context
```

### get_workspace_context

get_workspace_context()

Get current workspace context identifier.

**Returns:**
- **Optional[str]** – Current workspace identifier or None if no context set.

```python
from cursus.registry import get_workspace_context, set_workspace_context

# Check current context
current = get_workspace_context()
print(f"Current context: {current}")  # None initially

# Set and check context
set_workspace_context("developer_1")
current = get_workspace_context()
print(f"New context: {current}")  # developer_1
```

### clear_workspace_context

clear_workspace_context()

Clear current workspace context and return to core registry.

```python
from cursus.registry import clear_workspace_context, get_workspace_context

# Clear workspace context
clear_workspace_context()
current = get_workspace_context()
print(f"Context after clear: {current}")  # None

# Registry operations now use core registry
step_names = get_all_step_names()  # Uses core registry
```

### workspace_context

workspace_context(_workspace_id_)

Context manager for temporary workspace context with automatic restoration.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier for temporary context.

**Returns:**
- **ContextManager[None]** – Context manager for workspace operations.

```python
from cursus.registry import workspace_context, get_workspace_context

# Use temporary workspace context
with workspace_context("developer_1"):
    # Operations use developer_1 context
    workspace_steps = get_all_step_names()
    print(f"Workspace steps: {len(workspace_steps)}")

# Context automatically restored
current = get_workspace_context()
print(f"Context after with block: {current}")  # Original context
```

### switch_to_workspace

switch_to_workspace(_workspace_id_)

Convenience function to switch to a specific workspace context.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to switch to.

```python
from cursus.registry import switch_to_workspace, STEP_NAMES

# Switch to workspace
switch_to_workspace("developer_1")

# Registry variables now use workspace context
step_names = STEP_NAMES  # Uses developer_1 context
print(f"Steps in workspace: {len(step_names)}")
```

### switch_to_core

switch_to_core()

Convenience function to switch back to core registry (no workspace context).

```python
from cursus.registry import switch_to_core, STEP_NAMES

# Switch back to core
switch_to_core()

# Registry variables now use core registry
step_names = STEP_NAMES  # Uses core registry only
print(f"Core steps: {len(step_names)}")
```

### get_registry_info

get_registry_info(_workspace_id=None_)

Get comprehensive registry information for a workspace or core registry.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace identifier for specific workspace info. Defaults to None.

**Returns:**
- **Dict[str, Any]** – Dictionary with comprehensive registry information including step counts, available steps, and workspace details.

```python
from cursus.registry import get_registry_info

# Get core registry info
core_info = get_registry_info()
print(f"Core registry: {core_info['step_count']} steps")

# Get workspace-specific info
workspace_info = get_registry_info("developer_1")
print(f"Workspace info: {workspace_info}")
```

## Usage Examples

### Complete Workspace Management Workflow

```python
from cursus.registry import (
    set_workspace_context,
    get_workspace_context,
    workspace_context,
    get_all_step_names,
    get_registry_info,
    switch_to_workspace,
    switch_to_core
)

# Check initial state
print(f"Initial context: {get_workspace_context()}")  # None
core_steps = get_all_step_names()
print(f"Core steps: {len(core_steps)}")

# Switch to workspace
switch_to_workspace("developer_1")
workspace_steps = get_all_step_names()
print(f"Workspace steps: {len(workspace_steps)}")

# Get comprehensive info
info = get_registry_info()
print(f"Registry info: {info}")

# Use temporary context
with workspace_context("developer_2"):
    temp_steps = get_all_step_names()
    print(f"Temporary context steps: {len(temp_steps)}")

# Context restored automatically
print(f"Context after with: {get_workspace_context()}")

# Switch back to core
switch_to_core()
print(f"Back to core: {get_workspace_context()}")  # None
```

### Builder Registry Integration

```python
from cursus.registry import (
    get_global_registry,
    register_global_builder,
    get_config_class_name,
    get_builder_step_name
)
from cursus.core.base.builder_base import StepBuilderBase

# Get global registry
registry = get_global_registry()

# Validate existing mappings
validation_results = registry.validate_registry()
print(f"Valid mappings: {len(validation_results['valid'])}")
print(f"Invalid mappings: {len(validation_results['invalid'])}")
print(f"Missing builders: {len(validation_results['missing'])}")

# Register custom builder
class MyCustomStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        # Custom implementation
        return self.create_processing_step(config, context)

register_global_builder("MyCustomStep", MyCustomStepBuilder)

# Verify registration
supported_types = registry.list_supported_step_types()
print(f"MyCustomStep supported: {'MyCustomStep' in supported_types}")

# Get registry statistics
stats = registry.get_registry_stats()
print(f"Registry stats: {stats}")
```

### Hyperparameter Registry Integration

```python
from cursus.registry import (
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    validate_hyperparameter_class,
    get_all_hyperparameter_info
)

# Get all hyperparameter classes
all_classes = get_all_hyperparameter_classes()
print(f"Available hyperparameter classes: {all_classes}")

# Get class for specific model type
xgboost_class = get_hyperparameter_class_by_model_type("xgboost")
print(f"XGBoost hyperparameters: {xgboost_class}")

# Validate hyperparameter class
is_valid = validate_hyperparameter_class("XGBoostHyperparameters")
print(f"Valid hyperparameter class: {is_valid}")

# Get complete hyperparameter info
all_info = get_all_hyperparameter_info()
for class_name, info in all_info.items():
    print(f"{class_name}: {info['description']}")
```

### Error Handling and Validation

```python
from cursus.registry import (
    validate_step_name,
    get_config_class_name,
    RegistryError
)

# Validate step names before use
step_names_to_check = ["XGBoostTraining", "InvalidStep", "CustomStep"]

for step_name in step_names_to_check:
    if validate_step_name(step_name):
        try:
            config_class = get_config_class_name(step_name)
            print(f"✓ {step_name} -> {config_class}")
        except ValueError as e:
            print(f"✗ {step_name}: {e}")
    else:
        print(f"✗ {step_name}: Not found in registry")

# Handle registry errors gracefully
try:
    registry = get_global_registry()
    config = UnknownConfig()  # This would cause an error
    builder_class = registry.get_builder_for_config(config)
except RegistryError as e:
    print(f"Registry error: {e}")
    print(f"Available builders: {e.available_builders}")
```

## Related Components

- **[Builder Registry](builder_registry.md)** - Detailed builder registry functionality
- **[Step Names](step_names.md)** - Enhanced step names registry with workspace awareness
- **[Hyperparameter Registry](hyperparameter_registry.md)** - Central hyperparameter class registry
- **[Exceptions](exceptions.md)** - Registry-specific exception classes
- **[Hybrid Manager](hybrid/manager.md)** - Unified registry manager implementation
