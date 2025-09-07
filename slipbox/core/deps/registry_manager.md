---
tags:
  - code
  - deps
  - registry_manager
  - context_isolation
  - workspace_awareness
keywords:
  - RegistryManager
  - context isolation
  - workspace awareness
  - registry coordination
  - hybrid integration
  - specification management
  - multi-context support
topics:
  - registry management
  - context isolation
  - workspace integration
language: python
date of note: 2024-12-07
---

# Registry Manager

Registry manager for coordinating multiple isolated specification registries with complete context isolation and workspace awareness.

## Overview

The `RegistryManager` class provides centralized management of multiple registry instances, ensuring complete isolation between different contexts such as pipelines, environments, and workspaces. This system enables safe concurrent operation of multiple pipeline contexts without interference or specification conflicts.

The manager supports workspace-aware registry resolution that combines workspace context with local context names, integration with hybrid registry systems for automatic specification population, backward compatibility with existing code patterns, and thread-local workspace context management for multi-threaded environments.

The system provides advanced features including automatic registry creation with hybrid population, comprehensive context statistics and monitoring, workspace-aware context naming for isolation, and decorator-based integration with pipeline builder classes.

## Classes and Methods

### Classes
- [`RegistryManager`](#registrymanager) - Manager for context-scoped registries with workspace awareness

### Functions
- [`get_registry`](#get_registry) - Get registry for specific context
- [`list_contexts`](#list_contexts) - Get list of all registered context names
- [`clear_context`](#clear_context) - Clear registry for specific context
- [`get_context_stats`](#get_context_stats) - Get statistics for all contexts
- [`integrate_with_pipeline_builder`](#integrate_with_pipeline_builder) - Decorator for pipeline builder integration

## API Reference

### RegistryManager

_class_ cursus.core.deps.registry_manager.RegistryManager(_workspace_context=None_)

Manager for context-scoped registries with complete isolation and workspace awareness. This class coordinates multiple registry instances, ensuring that different contexts operate independently without specification conflicts.

**Parameters:**
- **workspace_context** (_Optional[str]_) – Optional workspace context for registry isolation. When provided, all context names are prefixed with this workspace identifier.

```python
from cursus.core.deps.registry_manager import RegistryManager

# Create manager with default workspace
manager = RegistryManager()

# Create manager with specific workspace context
workspace_manager = RegistryManager(workspace_context="experiment-1")

# Get registry for specific context
registry = manager.get_registry("pipeline-a")
```

#### get_registry

get_registry(_context_name="default"_, _create_if_missing=True_)

Get the registry for a specific context with workspace awareness. This method supports workspace-aware context naming, integration with hybrid registry systems, automatic registry population from hybrid sources, and backward compatibility with existing code.

**Parameters:**
- **context_name** (_str_) – Name of the context (e.g., pipeline name, environment). Defaults to "default".
- **create_if_missing** (_bool_) – Whether to create a new registry if one doesn't exist. Defaults to True.

**Returns:**
- **Optional[SpecificationRegistry]** – Context-specific registry or None if not found and create_if_missing is False.

```python
# Get or create registry for specific pipeline
pipeline_registry = manager.get_registry("ml-pipeline-v1")

# Get registry without creating if missing
existing_registry = manager.get_registry("test-pipeline", create_if_missing=False)

# Workspace-aware context (if workspace_context was set)
# Context "pipeline-a" becomes "experiment-1::pipeline-a"
workspace_registry = workspace_manager.get_registry("pipeline-a")
```

#### list_contexts

list_contexts()

Get list of all registered context names. This method returns all context names that currently have associated registries.

**Returns:**
- **List[str]** – List of context names with registries, including workspace-aware names if applicable.

```python
contexts = manager.list_contexts()
print(f"Active contexts: {contexts}")
# Output: ['default', 'pipeline-a', 'experiment-1::pipeline-b']
```

#### clear_context

clear_context(_context_name_)

Clear the registry for a specific context. This method removes the registry and all its specifications for the specified context.

**Parameters:**
- **context_name** (_str_) – Name of the context to clear.

**Returns:**
- **bool** – True if the registry was cleared, False if it didn't exist.

```python
# Clear specific context
success = manager.clear_context("old-pipeline")
if success:
    print("Registry cleared successfully")
else:
    print("Registry not found")
```

#### clear_all_contexts

clear_all_contexts()

Clear all registries. This method removes all context registries and their specifications, resetting the manager to an empty state.

```python
# Clear all registries
manager.clear_all_contexts()
print("All registries cleared")
```

#### get_context_stats

get_context_stats()

Get statistics for all contexts. This method provides comprehensive statistics about each context including step counts and step type counts.

**Returns:**
- **Dict[str, Dict[str, int]]** – Dictionary mapping context names to their statistics with keys "step_count" and "step_type_count".

```python
stats = manager.get_context_stats()
for context, context_stats in stats.items():
    print(f"Context {context}: {context_stats['step_count']} steps, "
          f"{context_stats['step_type_count']} step types")

# Output:
# Context default: 5 steps, 3 step types
# Context pipeline-a: 8 steps, 4 step types
```

### get_registry

get_registry(_manager_, _context_name="default"_)

Get the registry for a specific context. This standalone function provides a functional interface to registry access.

**Parameters:**
- **manager** (_RegistryManager_) – Registry manager instance.
- **context_name** (_str_) – Name of the context. Defaults to "default".

**Returns:**
- **SpecificationRegistry** – Context-specific registry.

```python
from cursus.core.deps.registry_manager import get_registry

registry = get_registry(manager, "my-pipeline")
```

### list_contexts

list_contexts(_manager_)

Get list of all registered context names. This standalone function provides a functional interface to context listing.

**Parameters:**
- **manager** (_RegistryManager_) – Registry manager instance.

**Returns:**
- **List[str]** – List of context names with registries.

```python
from cursus.core.deps.registry_manager import list_contexts

contexts = list_contexts(manager)
```

### clear_context

clear_context(_manager_, _context_name_)

Clear the registry for a specific context. This standalone function provides a functional interface to context clearing.

**Parameters:**
- **manager** (_RegistryManager_) – Registry manager instance.
- **context_name** (_str_) – Name of the context to clear.

**Returns:**
- **bool** – True if the registry was cleared, False if it didn't exist.

```python
from cursus.core.deps.registry_manager import clear_context

success = clear_context(manager, "old-context")
```

### get_context_stats

get_context_stats(_manager_)

Get statistics for all contexts. This standalone function provides a functional interface to statistics retrieval.

**Parameters:**
- **manager** (_RegistryManager_) – Registry manager instance.

**Returns:**
- **Dict[str, Dict[str, int]]** – Dictionary mapping context names to their statistics.

```python
from cursus.core.deps.registry_manager import get_context_stats

stats = get_context_stats(manager)
```

### integrate_with_pipeline_builder

integrate_with_pipeline_builder(_pipeline_builder_cls_, _manager=None_)

Decorator to integrate context-scoped registries with a pipeline builder class. This decorator modifies a pipeline builder class to use context-scoped registries automatically.

**Parameters:**
- **pipeline_builder_cls** (_Type_) – Pipeline builder class to modify.
- **manager** (_Optional[RegistryManager]_) – Registry manager instance. If None, a new instance will be created.

**Returns:**
- **Type** – Modified pipeline builder class with integrated registry management.

```python
from cursus.core.deps.registry_manager import integrate_with_pipeline_builder

@integrate_with_pipeline_builder
class MyPipelineBuilder:
    def __init__(self, config_path):
        self.config_path = config_path
        # registry_manager and registry are automatically added
    
    def build_pipeline(self):
        # Use self.registry for specifications
        pass

# Create builder instance
builder = MyPipelineBuilder("config.json")
# builder.registry_manager and builder.registry are available
```

## Usage Patterns

### Basic Context Management
```python
from cursus.core.deps.registry_manager import RegistryManager

# Create manager
manager = RegistryManager()

# Create separate registries for different pipelines
training_registry = manager.get_registry("training-pipeline")
inference_registry = manager.get_registry("inference-pipeline")

# Register specifications in isolated contexts
training_registry.register("preprocessing", training_preprocessing_spec)
inference_registry.register("preprocessing", inference_preprocessing_spec)

# Contexts are completely isolated
assert training_registry != inference_registry
```

### Workspace-Aware Management
```python
# Create workspace-specific manager
workspace_manager = RegistryManager(workspace_context="project-alpha")

# All contexts are prefixed with workspace
pipeline_registry = workspace_manager.get_registry("main-pipeline")
# Actual context name: "project-alpha::main-pipeline"

# Different workspace
other_workspace = RegistryManager(workspace_context="project-beta")
other_registry = other_workspace.get_registry("main-pipeline")
# Actual context name: "project-beta::main-pipeline"

# Complete isolation between workspaces
assert pipeline_registry != other_registry
```

### Statistics and Monitoring
```python
# Monitor registry usage
stats = manager.get_context_stats()
for context, info in stats.items():
    print(f"Context: {context}")
    print(f"  Steps: {info['step_count']}")
    print(f"  Step Types: {info['step_type_count']}")

# List active contexts
active_contexts = manager.list_contexts()
print(f"Active contexts: {len(active_contexts)}")

# Clean up unused contexts
for context in active_contexts:
    if context.startswith("temp-"):
        manager.clear_context(context)
```

## Related Documentation

- [Specification Registry](specification_registry.md) - Individual registry instances managed by this class
- [Factory](factory.md) - Factory functions that create registry managers
- [Dependency Resolver](dependency_resolver.md) - Uses registries managed by this class
- [Pipeline Assembler](../assembler/pipeline_assembler.md) - Integrates with registry manager for component isolation
- [Hybrid Registry Integration](../../registry/hybrid/manager.md) - Hybrid registry system integration
