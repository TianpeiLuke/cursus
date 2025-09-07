---
tags:
  - code
  - deps
  - factory
  - component_creation
  - dependency_injection
keywords:
  - create_pipeline_components
  - dependency_resolution_context
  - get_thread_components
  - factory functions
  - component wiring
  - thread safety
  - context management
topics:
  - factory pattern
  - dependency injection
  - component management
language: python
date of note: 2024-12-07
---

# Factory

Factory functions for creating pipeline dependency components with proper wiring and lifecycle management.

## Overview

The factory module provides convenience functions for instantiating the core components of the dependency resolution system with proper dependency wiring. This module implements the factory pattern to ensure consistent component creation and configuration across the pipeline system.

The factory functions handle the complex wiring between components including semantic matchers, specification registries, registry managers, and dependency resolvers. This approach ensures that all components are properly initialized with their required dependencies and configured for optimal performance.

The module supports advanced features including thread-local component instances for multi-threaded environments, scoped dependency resolution contexts with automatic cleanup, centralized component creation with consistent configuration, and proper lifecycle management for resource cleanup.

## Classes and Methods

### Functions
- [`create_pipeline_components`](#create_pipeline_components) - Create all necessary pipeline components with proper dependencies
- [`get_thread_components`](#get_thread_components) - Get thread-specific component instances for thread safety
- [`dependency_resolution_context`](#dependency_resolution_context) - Create scoped dependency resolution context with cleanup

## API Reference

### create_pipeline_components

create_pipeline_components(_context_name=None_)

Create all necessary pipeline components with proper dependencies. This function instantiates and wires together all core components of the dependency resolution system including semantic matcher, registry manager, specification registry, and dependency resolver.

**Parameters:**
- **context_name** (_Optional[str]_) – Optional context name for registry isolation. If None, uses "default" context.

**Returns:**
- **Dict[str, Any]** – Dictionary containing all created components with keys: "semantic_matcher", "registry_manager", "registry", "resolver".

```python
from cursus.core.deps.factory import create_pipeline_components

# Create components with default context
components = create_pipeline_components()
resolver = components["resolver"]
registry = components["registry"]

# Create components with specific context
components = create_pipeline_components(context_name="experiment-1")
resolver = components["resolver"]
```

### get_thread_components

get_thread_components()

Get thread-specific component instances. This function returns component instances that are isolated per thread, ensuring thread safety in multi-threaded environments. Components are created lazily on first access per thread.

**Returns:**
- **Dict[str, Any]** – Dictionary containing thread-local component instances with same structure as create_pipeline_components.

```python
import threading
from cursus.core.deps.factory import get_thread_components

def process_pipeline_in_thread():
    # Each thread gets its own component instances
    components = get_thread_components()
    resolver = components["resolver"]
    
    # Use resolver safely in this thread
    resolved = resolver.resolve_all_dependencies(available_steps)
    return resolved

# Create multiple threads
threads = []
for i in range(3):
    thread = threading.Thread(target=process_pipeline_in_thread)
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

### dependency_resolution_context

dependency_resolution_context(_clear_on_exit=True_)

Create a scoped dependency resolution context. This context manager creates a fresh set of components and ensures proper cleanup when the context exits, preventing resource leaks and ensuring clean state.

**Parameters:**
- **clear_on_exit** (_bool_) – Whether to clear caches and contexts on exit. Defaults to True.

**Returns:**
- **ContextManager[Dict[str, Any]]** – Context manager yielding dictionary of components with automatic cleanup.

```python
from cursus.core.deps.factory import dependency_resolution_context

# Use with automatic cleanup
with dependency_resolution_context() as components:
    resolver = components["resolver"]
    registry = components["registry"]
    
    # Register specifications and resolve dependencies
    resolver.register_specification("step1", spec1)
    resolved = resolver.resolve_all_dependencies(["step1", "step2"])
    
    # Components are automatically cleaned up on exit

# Use without cleanup (for debugging)
with dependency_resolution_context(clear_on_exit=False) as components:
    resolver = components["resolver"]
    # Caches and contexts persist after exit
```

## Usage Patterns

### Basic Component Creation

```python
from cursus.core.deps.factory import create_pipeline_components

# Create components for single-threaded use
components = create_pipeline_components()
resolver = components["resolver"]
registry_manager = components["registry_manager"]

# Register specifications
resolver.register_specification("preprocessing", preprocessing_spec)
resolver.register_specification("training", training_spec)

# Resolve dependencies
resolved = resolver.resolve_all_dependencies(["preprocessing", "training"])
```

### Thread-Safe Multi-Threading

```python
import threading
from cursus.core.deps.factory import get_thread_components

def worker_function(step_specs, available_steps):
    # Each thread gets isolated components
    components = get_thread_components()
    resolver = components["resolver"]
    
    # Register specifications for this thread
    for step_name, spec in step_specs.items():
        resolver.register_specification(step_name, spec)
    
    # Resolve dependencies safely
    return resolver.resolve_all_dependencies(available_steps)

# Create worker threads
threads = []
for i in range(5):
    thread = threading.Thread(
        target=worker_function,
        args=(thread_specs[i], available_steps[i])
    )
    threads.append(thread)
    thread.start()

# Wait for completion
results = []
for thread in threads:
    thread.join()
```

### Scoped Context Management

```python
from cursus.core.deps.factory import dependency_resolution_context

def process_pipeline_batch(pipeline_configs):
    results = []
    
    for config in pipeline_configs:
        # Each pipeline gets fresh components
        with dependency_resolution_context() as components:
            resolver = components["resolver"]
            
            # Load specifications from config
            for step_name, spec in config.specifications.items():
                resolver.register_specification(step_name, spec)
            
            # Resolve and store results
            resolved = resolver.resolve_all_dependencies(config.available_steps)
            results.append(resolved)
            
            # Automatic cleanup ensures no interference between pipelines
    
    return results
```

## Related Documentation

- [Dependency Resolver](dependency_resolver.md) - Core dependency resolution engine created by factory
- [Registry Manager](registry_manager.md) - Registry management component created by factory
- [Specification Registry](specification_registry.md) - Specification storage component created by factory
- [Semantic Matcher](semantic_matcher.md) - Semantic matching component created by factory
- [Pipeline Assembler](../assembler/pipeline_assembler.md) - Uses factory-created components for pipeline assembly
