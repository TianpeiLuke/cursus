---
tags:
  - entry_point
  - code
  - registry
  - hybrid
  - unified_manager
  - workspace_management
keywords:
  - hybrid registry
  - unified manager
  - workspace management
  - registry backend
  - performance optimization
topics:
  - hybrid registry system
  - unified management
  - workspace support
language: python
date of note: 2024-12-07
---

# Hybrid Registry System

Hybrid registry backend components that provide unified registry management with workspace awareness, performance optimization, and conflict resolution capabilities.

## Overview

The hybrid registry system provides a sophisticated backend for managing pipeline step registries with support for multiple workspaces, performance optimization through caching, and intelligent conflict resolution. It consolidates multiple registry managers into a single, efficient system that eliminates redundancy while maintaining all functionality.

The system includes a unified registry manager that handles all registry operations, data models for step definitions and resolution contexts, setup utilities for registry initialization, and utility functions for registry operations and conversions.

## Module Structure

### Core Components
- [`manager.py`](manager.md) - Unified registry manager implementation
- [`models.py`](models.md) - Data models for hybrid registry system
- [`setup.py`](setup.md) - Setup utilities for hybrid registry initialization
- [`utils.py`](utils.md) - Utility functions for registry operations

## Key Features

### Unified Management
- **Single Manager**: Consolidates CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager
- **Efficient Operations**: Eliminates redundancy while maintaining all functionality
- **Thread Safety**: Thread-safe operations with proper locking mechanisms
- **Performance Optimization**: Caching infrastructure for fast lookups

### Workspace Support
- **Multi-Workspace**: Support for multiple developer workspaces
- **Context Isolation**: Proper isolation between workspace contexts
- **Conflict Resolution**: Intelligent handling of step name conflicts
- **Dynamic Loading**: Auto-discovery and loading of workspace registries

### Performance Features
- **Caching Strategy**: LRU caches for definitions, legacy dictionaries, and step lists
- **Lazy Loading**: Components loaded only when needed
- **Memory Management**: Efficient storage and cleanup of registry data
- **Cache Invalidation**: Smart cache invalidation when registries change

### Data Models
- **Step Definitions**: Comprehensive step definition with metadata
- **Resolution Context**: Context for step resolution with workspace information
- **Validation Results**: Detailed validation results with conflict analysis
- **Registry Types**: Enumeration of registry types and resolution strategies

## Usage Examples

### Basic Unified Manager Usage

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Create unified manager
manager = UnifiedRegistryManager()

# Get step definition
step_def = manager.get_step_definition("XGBoostTraining")
if step_def:
    print(f"Found step: {step_def.step_name}")
    print(f"Config class: {step_def.config_class}")
    print(f"Builder class: {step_def.builder_step_name}")

# List all steps
all_steps = manager.list_steps()
print(f"Total steps: {len(all_steps)}")

# Get registry status
status = manager.get_registry_status()
print(f"Core loaded: {status['core']['loaded']}")
print(f"Workspaces: {list(status.keys())}")
```

### Workspace Management

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

manager = UnifiedRegistryManager()

# Add workspace registry
manager.add_workspace_registry("developer_1", "/path/to/workspace")

# Set workspace context
manager.set_workspace_context("developer_1")

# Get workspace-specific steps
workspace_steps = manager.list_steps("developer_1")
print(f"Workspace steps: {len(workspace_steps)}")

# Check for conflicts
conflicts = manager.get_step_conflicts()
if conflicts:
    print("Step conflicts detected:")
    for step_name, definitions in conflicts.items():
        print(f"  {step_name}: {len(definitions)} definitions")
```

### Performance Optimization

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

manager = UnifiedRegistryManager()

# Cached operations
legacy_dict = manager.create_legacy_step_names_dict()  # Cached
all_definitions = manager.get_all_step_definitions()   # Cached

# Cache management
manager.clear_component_cache()  # Clear component cache
manager.clear_builder_class_cache()  # Clear builder cache

# Performance monitoring
status = manager.get_registry_status()
for registry_id, info in status.items():
    print(f"{registry_id}: {info['step_count']} steps")
```

## Related Components

- **[Registry Module](../__init__.md)** - Main registry module that uses hybrid backend
- **[Step Names](../step_names.md)** - Step names registry with hybrid backend support
- **[Builder Registry](../builder_registry.md)** - Builder registry integration
- **[Validation Utils](../validation_utils.md)** - Validation utilities integration
