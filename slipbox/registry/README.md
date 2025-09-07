---
tags:
  - entry_point
  - code
  - registry
  - step_registry
  - builder_registry
  - hyperparameter_registry
keywords:
  - step registry
  - builder registry
  - hyperparameter registry
  - workspace management
  - hybrid registry
topics:
  - registry system
  - step management
  - workspace awareness
language: python
date of note: 2024-12-07
---

# Registry Module

Enhanced Pipeline Registry Module with Hybrid Registry Support for tracking step types, specifications, hyperparameters, and other metadata used in the pipeline system.

## Overview

The registry module provides a comprehensive system for managing pipeline components including step types, builder classes, hyperparameters, and workspace-aware configurations. It serves as the single source of truth for step naming and configuration, ensuring consistency across the entire pipeline system.

The module has been enhanced with workspace-aware step resolution, hybrid registry backend support, context management for multi-developer workflows, and backward compatibility with existing code. The registry system supports both core pipeline steps and workspace-specific customizations, enabling collaborative development while maintaining system integrity.

## Module Structure

### Core Registry Components
- [`__init__.py`](__init__.md) - Main registry module with enhanced hybrid registry support
- [`builder_registry.py`](builder_registry.md) - Step builder registry for pipeline API with auto-discovery
- [`exceptions.py`](exceptions.md) - Custom exception classes for registry operations
- [`step_names.py`](step_names.md) - Enhanced step names registry with workspace awareness
- [`hyperparameter_registry.py`](hyperparameter_registry.md) - Central registry for hyperparameter classes
- [`validation_utils.py`](validation_utils.md) - Validation utilities for step registration

### Hybrid Registry System
- [`hybrid/`](hybrid/) - Hybrid registry backend components
  - [`manager.py`](hybrid/manager.md) - Unified registry manager implementation
  - [`models.py`](hybrid/models.md) - Data models for hybrid registry system
  - [`setup.py`](hybrid/setup.md) - Setup utilities for hybrid registry
  - [`utils.py`](hybrid/utils.md) - Utility functions for registry operations

## Key Features

### Workspace-Aware Registry
- **Context Management**: Automatic workspace context switching and isolation
- **Multi-Developer Support**: Collaborative development with conflict resolution
- **Backward Compatibility**: Seamless integration with existing pipeline code
- **Dynamic Resolution**: Runtime step resolution based on workspace context

### Builder Registry
- **Auto-Discovery**: Automatic detection and registration of step builders
- **Type Safety**: Strong typing and validation for builder classes
- **Legacy Support**: Backward compatibility with existing builder patterns
- **Extensibility**: Easy registration of custom step builders

### Hyperparameter Management
- **Centralized Registry**: Single source of truth for hyperparameter classes
- **Model Type Mapping**: Automatic hyperparameter class resolution by model type
- **Validation**: Comprehensive validation of hyperparameter configurations
- **Extensibility**: Support for custom hyperparameter classes

### Hybrid Registry Backend
- **Unified Management**: Single manager for all registry operations
- **Performance Optimization**: Caching and efficient data structures
- **Conflict Resolution**: Intelligent handling of step name conflicts
- **Scalability**: Support for large-scale multi-workspace environments

## Usage Examples

### Basic Registry Operations

```python
from cursus.registry import (
    get_config_class_name,
    get_builder_step_name,
    validate_step_name,
    get_global_registry
)

# Get configuration class name for a step
config_class = get_config_class_name("XGBoostTraining")
print(f"Config class: {config_class}")  # XGBoostTrainingConfig

# Get builder class name for a step
builder_name = get_builder_step_name("XGBoostTraining")
print(f"Builder class: {builder_name}")  # XGBoostTrainingStepBuilder

# Validate step name
is_valid = validate_step_name("XGBoostTraining")
print(f"Valid step: {is_valid}")  # True

# Get global builder registry
registry = get_global_registry()
supported_types = registry.list_supported_step_types()
print(f"Supported types: {len(supported_types)}")
```

### Workspace-Aware Operations

```python
from cursus.registry import (
    set_workspace_context,
    get_workspace_context,
    workspace_context,
    get_step_names
)

# Set workspace context
set_workspace_context("developer_1")
current_workspace = get_workspace_context()
print(f"Current workspace: {current_workspace}")

# Get workspace-specific step names
workspace_steps = get_step_names("developer_1")
print(f"Steps in workspace: {len(workspace_steps)}")

# Use context manager for temporary workspace
with workspace_context("developer_2"):
    temp_steps = get_step_names()
    print(f"Temporary workspace steps: {len(temp_steps)}")

# Context automatically restored after with block
print(f"Back to workspace: {get_workspace_context()}")
```

### Builder Registry Usage

```python
from cursus.registry import get_global_registry, register_global_builder
from cursus.core.base.builder_base import StepBuilderBase

# Get builder for configuration
registry = get_global_registry()
config = XGBoostTrainingConfig()
builder_class = registry.get_builder_for_config(config)

# Register custom builder
class CustomStepBuilder(StepBuilderBase):
    def build_step(self, config, context):
        # Custom implementation
        pass

register_global_builder("CustomStep", CustomStepBuilder)

# Validate registry
validation_results = registry.validate_registry()
print(f"Valid mappings: {len(validation_results['valid'])}")
print(f"Invalid mappings: {len(validation_results['invalid'])}")
```

### Hyperparameter Registry Usage

```python
from cursus.registry import (
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    validate_hyperparameter_class
)

# Get all hyperparameter classes
all_classes = get_all_hyperparameter_classes()
print(f"Available hyperparameter classes: {all_classes}")

# Get hyperparameter class for specific model type
xgboost_class = get_hyperparameter_class_by_model_type("xgboost")
print(f"XGBoost hyperparameters: {xgboost_class}")

# Validate hyperparameter class
is_valid = validate_hyperparameter_class("XGBoostHyperparameters")
print(f"Valid hyperparameter class: {is_valid}")
```

## Related Components

- **[Core Base](../core/base/)** - Base classes that work with registry components
- **[Steps](../steps/)** - Pipeline step implementations registered in the system
- **[Validation](../validation/)** - Validation framework that uses registry information
- **[Workspace](../workspace/)** - Workspace management system integration
