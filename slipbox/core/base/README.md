---
tags:
  - entry_point
  - code
  - core
  - base
  - module_init
keywords:
  - module initialization
  - lazy imports
  - circular dependency handling
  - base classes
  - cursus framework
topics:
  - module organization
  - import management
  - circular dependency resolution
language: python
date of note: 2025-08-07
---

# Core Base Module Initialization

## Overview

The `__init__.py` file serves as the entry point for the `cursus.core.base` module, providing foundational base classes for the cursus framework. This module implements a sophisticated lazy import system to handle circular dependencies while maintaining clean public APIs.

## Purpose

This module provides the foundational base classes used throughout the cursus framework for:
- Configuration management (`BasePipelineConfig`)
- Contract definitions (`ScriptContract`, `ValidationResult`, `ScriptAnalyzer`)
- Specifications (`DependencySpec`, `OutputSpec`, `StepSpecification`)
- Builder patterns (`StepBuilderBase`)
- Hyperparameter management (`ModelHyperparameters`)
- Enumeration types (`DependencyType`, `NodeType`)

## Architecture

### Import Strategy

The module uses a three-tier import strategy to handle circular dependencies:

1. **Direct Imports**: Classes with no circular dependencies are imported directly
   - Enums (`DependencyType`, `NodeType`)
   - Contract classes (`ScriptContract`, `ValidationResult`, `ScriptAnalyzer`)
   - Hyperparameters (`ModelHyperparameters`)

2. **TYPE_CHECKING Imports**: Classes that might have circular dependencies are imported only for type checking
   - Configuration classes (`BasePipelineConfig`)
   - Specification classes (`DependencySpec`, `OutputSpec`, `StepSpecification`)
   - Builder classes (`StepBuilderBase`)

3. **Lazy Loading Functions**: Runtime access to circular-dependent classes through factory functions
   - `get_base_pipeline_config()`
   - `get_dependency_spec()`
   - `get_output_spec()`
   - `get_step_specification()`
   - `get_step_builder_base()`

### Backward Compatibility

The module provides backward compatibility through a custom `__getattr__` function that enables lazy loading of classes when accessed as module attributes:

```python
from cursus.core.base import BasePipelineConfig  # Works via __getattr__
```

## Key Components

### Enums (Always Available)
- `DependencyType`: Types of dependencies in the pipeline
- `NodeType`: Types of nodes based on dependency/output characteristics

### Contract Classes (Always Available)
- `ScriptContract`: Script execution contract with I/O and environment requirements
- `ValidationResult`: Result of script contract validation
- `ScriptAnalyzer`: Analyzes Python scripts to extract I/O patterns

### Hyperparameters (Always Available)
- `ModelHyperparameters`: Base model hyperparameters for training tasks

### Lazy-Loaded Classes
- `BasePipelineConfig`: Base configuration with shared pipeline attributes
- `DependencySpec`: Declarative specification for step dependencies
- `OutputSpec`: Declarative specification for step outputs
- `StepSpecification`: Complete specification for step dependencies and outputs
- `StepBuilderBase`: Base class for all step builders

## Usage Patterns

### Direct Usage
```python
from cursus.core.base import DependencyType, ScriptContract, ModelHyperparameters

# These are immediately available
dep_type = DependencyType.MODEL_ARTIFACTS
contract = ScriptContract(...)
hyperparams = ModelHyperparameters(...)
```

### Lazy Loading
```python
from cursus.core.base import BasePipelineConfig  # Lazy loaded via __getattr__

# Or explicit lazy loading
from cursus.core.base import get_base_pipeline_config
BasePipelineConfig = get_base_pipeline_config()
```

### Type Hints
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cursus.core.base import StepSpecification

def process_spec(spec: 'StepSpecification') -> None:
    # Type checking works, runtime loading is lazy
    pass
```

## Design Benefits

1. **Circular Dependency Resolution**: Prevents import cycles while maintaining usability
2. **Performance**: Only loads classes when actually needed
3. **Type Safety**: Full type checking support through TYPE_CHECKING imports
4. **Backward Compatibility**: Existing code continues to work unchanged
5. **Clean API**: Simple import statements hide complexity

## Implementation Details

### Lazy Loading Mechanism
Each lazy loading function follows the pattern:
```python
def get_class_name():
    """Lazy import for ClassName to avoid circular imports."""
    from .module_name import ClassName
    return ClassName
```

### __getattr__ Implementation
The `__getattr__` function maps class names to their lazy loading functions:
```python
def __getattr__(name):
    """Provide lazy loading for backward compatibility."""
    if name == 'ClassName':
        return get_class_name()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
```

## Best Practices

1. **Import Order**: Always import enums first as they have no dependencies
2. **Lazy Loading**: Use lazy loading for classes that might have circular dependencies
3. **Type Checking**: Use TYPE_CHECKING imports for type hints
4. **Factory Functions**: Provide explicit factory functions for programmatic access
5. **Documentation**: Clearly document which classes are lazy-loaded

## Error Handling

The module includes proper error handling for:
- Missing attributes in `__getattr__`
- Import failures in lazy loading functions
- Circular import detection and prevention

This design ensures robust operation even when dealing with complex dependency graphs in large codebases.
