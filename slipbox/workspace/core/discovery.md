---
tags:
  - code
  - workspace
  - discovery
  - components
  - dependencies
keywords:
  - WorkspaceDiscoveryManager
  - ComponentInventory
  - DependencyGraph
  - component discovery
  - cross-workspace dependencies
topics:
  - workspace management
  - component discovery
  - dependency resolution
language: python
date of note: 2024-12-07
---

# Workspace Discovery Manager

Cross-workspace component discovery and resolution system for managing workspace components and dependencies.

## Overview

The Workspace Discovery Manager provides comprehensive component discovery across multiple workspaces, dependency resolution, and component compatibility analysis. This module enables cross-workspace component inventory, dependency graph construction, and component metadata caching for efficient workspace management.

The discovery system supports developer workspace isolation while enabling cross-workspace collaboration through component sharing and dependency resolution. It integrates with the workspace validation system to provide file resolvers and module loaders for component access.

Key features include component inventory management, dependency graph analysis, cross-workspace dependency resolution, component caching, and workspace structure discovery.

## Classes and Methods

### Classes
- [`ComponentInventory`](#componentinventory) - Inventory of discovered workspace components
- [`DependencyGraph`](#dependencygraph) - Component dependency relationships representation
- [`WorkspaceDiscoveryManager`](#workspacediscoverymanager) - Cross-workspace component discovery and resolution

### Methods
- [`discover_workspaces`](#discover_workspaces) - Discover and analyze workspace structure
- [`discover_components`](#discover_components) - Discover components across workspaces
- [`resolve_cross_workspace_dependencies`](#resolve_cross_workspace_dependencies) - Resolve cross-workspace dependencies
- [`get_file_resolver`](#get_file_resolver) - Get workspace-aware file resolver
- [`get_module_loader`](#get_module_loader) - Get workspace-aware module loader
- [`list_available_developers`](#list_available_developers) - Get list of available developer IDs
- [`get_workspace_info`](#get_workspace_info) - Get workspace information
- [`refresh_cache`](#refresh_cache) - Refresh component discovery cache

## API Reference

### ComponentInventory

_class_ cursus.workspace.core.discovery.ComponentInventory()

Inventory of discovered workspace components with categorized storage and summary tracking.

```python
from cursus.workspace.core.discovery import ComponentInventory

# Create component inventory
inventory = ComponentInventory()

# Add components
inventory.add_component("builders", "alice:data_prep", {
    "developer_id": "alice",
    "step_name": "data_prep",
    "step_type": "ProcessingStep"
})

# Convert to dictionary
inventory_dict = inventory.to_dict()
print("Total components:", inventory_dict['summary']['total_components'])
```

#### add_component

add_component(_component_type_, _component_id_, _component_info_)

Add component to inventory with categorization and summary updates.

**Parameters:**
- **component_type** (_str_) – Type of component ('builders', 'configs', 'contracts', 'specs', 'scripts').
- **component_id** (_str_) – Unique identifier for the component.
- **component_info** (_Dict[str, Any]_) – Component information dictionary.

```python
# Add different types of components
inventory.add_component("builders", "alice:training", {
    "developer_id": "alice",
    "step_name": "training",
    "step_type": "TrainingStep",
    "class_name": "XGBoostTrainingBuilder"
})

inventory.add_component("scripts", "bob:preprocessing", {
    "developer_id": "bob",
    "step_name": "preprocessing",
    "file_path": "/path/to/preprocessing.py"
})
```

#### to_dict

to_dict()

Convert inventory to dictionary representation.

**Returns:**
- **Dict[str, Any]** – Dictionary containing all components organized by type and summary information.

```python
# Get complete inventory as dictionary
inventory_data = inventory.to_dict()

print("Builders:", len(inventory_data['builders']))
print("Scripts:", len(inventory_data['scripts']))
print("Developers:", inventory_data['summary']['developers'])
```

### DependencyGraph

_class_ cursus.workspace.core.discovery.DependencyGraph()

Represents component dependency relationships with cycle detection capabilities.

```python
from cursus.workspace.core.discovery import DependencyGraph

# Create dependency graph
dep_graph = DependencyGraph()

# Add components and dependencies
dep_graph.add_component("alice:data_prep", {"step_type": "ProcessingStep"})
dep_graph.add_component("alice:training", {"step_type": "TrainingStep"})
dep_graph.add_dependency("alice:training", "alice:data_prep")

# Check for circular dependencies
has_cycles = dep_graph.has_circular_dependencies()
print("Has circular dependencies:", has_cycles)
```

#### add_component

add_component(_component_id_, _metadata=None_)

Add component to dependency graph.

**Parameters:**
- **component_id** (_str_) – Unique identifier for the component.
- **metadata** (_Optional[Dict[str, Any]]_) – Optional metadata dictionary for the component.

```python
# Add components with metadata
dep_graph.add_component("alice:model_eval", {
    "step_type": "EvaluationStep",
    "developer_id": "alice"
})
```

#### add_dependency

add_dependency(_from_component_, _to_component_)

Add dependency relationship between components.

**Parameters:**
- **from_component** (_str_) – Component that depends on another.
- **to_component** (_str_) – Component that is depended upon.

```python
# Add dependency relationships
dep_graph.add_dependency("alice:model_eval", "alice:training")
dep_graph.add_dependency("alice:training", "alice:data_prep")
```

#### get_dependencies

get_dependencies(_component_id_)

Get dependencies for a specific component.

**Parameters:**
- **component_id** (_str_) – Component identifier to get dependencies for.

**Returns:**
- **List[str]** – List of component IDs that this component depends on.

```python
# Get component dependencies
deps = dep_graph.get_dependencies("alice:training")
print("Training depends on:", deps)
```

#### get_dependents

get_dependents(_component_id_)

Get components that depend on the specified component.

**Parameters:**
- **component_id** (_str_) – Component identifier to get dependents for.

**Returns:**
- **List[str]** – List of component IDs that depend on this component.

```python
# Get components that depend on data_prep
dependents = dep_graph.get_dependents("alice:data_prep")
print("Components depending on data_prep:", dependents)
```

#### has_circular_dependencies

has_circular_dependencies()

Check for circular dependencies in the graph using depth-first search.

**Returns:**
- **bool** – True if circular dependencies exist, False otherwise.

```python
# Check for cycles before processing
if not dep_graph.has_circular_dependencies():
    print("Dependency graph is valid")
else:
    print("Circular dependencies detected!")
```

### WorkspaceDiscoveryManager

_class_ cursus.workspace.core.discovery.WorkspaceDiscoveryManager(_workspace_manager_)

Cross-workspace component discovery and resolution with caching and dependency analysis.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent WorkspaceManager instance for integration.

```python
from cursus.workspace.core.discovery import WorkspaceDiscoveryManager
from cursus.workspace.core.manager import WorkspaceManager

# Create discovery manager
workspace_manager = WorkspaceManager("/path/to/workspace")
discovery_manager = WorkspaceDiscoveryManager(workspace_manager)

# Discover workspaces
workspaces = discovery_manager.discover_workspaces(Path("/path/to/workspace"))
print("Found workspaces:", workspaces['summary']['total_workspaces'])
```

#### discover_workspaces

discover_workspaces(_workspace_root_)

Discover and analyze workspace structure including developer and shared workspaces.

**Parameters:**
- **workspace_root** (_Path_) – Root directory to discover workspaces in.

**Returns:**
- **Dict[str, Any]** – Workspace discovery information with structure analysis and summary.

```python
from pathlib import Path

# Discover all workspaces
discovery_result = discovery_manager.discover_workspaces(Path("/path/to/workspace"))

print("Total workspaces:", discovery_result['summary']['total_workspaces'])
print("Total developers:", discovery_result['summary']['total_developers'])
print("Total components:", discovery_result['summary']['total_components'])

for workspace in discovery_result['workspaces']:
    print(f"Workspace {workspace['workspace_id']}: {workspace['component_count']} components")
```

#### discover_components

discover_components(_workspace_ids=None_, _developer_id=None_)

Discover components across workspaces with caching support.

**Parameters:**
- **workspace_ids** (_Optional[List[str]]_) – Optional list of workspace IDs to search.
- **developer_id** (_Optional[str]_) – Optional specific developer ID to search.

**Returns:**
- **Dict[str, Any]** – Dictionary containing discovered components organized by type.

```python
# Discover all components
all_components = discovery_manager.discover_components()
print("Total components found:", all_components['summary']['total_components'])

# Discover components for specific developer
alice_components = discovery_manager.discover_components(developer_id="alice")
print("Alice's components:", alice_components['summary']['total_components'])

# Discover components in specific workspaces
specific_components = discovery_manager.discover_components(
    workspace_ids=["alice", "bob"]
)
```

#### resolve_cross_workspace_dependencies

resolve_cross_workspace_dependencies(_pipeline_definition_)

Resolve dependencies across workspace boundaries with validation.

**Parameters:**
- **pipeline_definition** (_Dict[str, Any]_) – Pipeline definition with cross-workspace dependencies.

**Returns:**
- **Dict[str, Any]** – Resolved dependency information with validation results and dependency graph.

```python
# Define pipeline with cross-workspace dependencies
pipeline_def = {
    "pipeline_name": "cross_workspace_pipeline",
    "steps": [
        {
            "step_name": "data_prep",
            "developer_id": "alice",
            "dependencies": []
        },
        {
            "step_name": "training",
            "developer_id": "bob",
            "dependencies": ["alice:data_prep"]
        }
    ]
}

# Resolve dependencies
resolution = discovery_manager.resolve_cross_workspace_dependencies(pipeline_def)

if not resolution['issues']:
    print("All dependencies resolved successfully")
    print("Dependency graph:", resolution['dependency_graph'])
else:
    print("Resolution issues:", resolution['issues'])
```

#### get_file_resolver

get_file_resolver(_developer_id=None_, _**kwargs_)

Get workspace-aware file resolver for component access.

**Parameters:**
- **developer_id** (_Optional[str]_) – Developer to target, uses config default if None.
- ****kwargs** – Additional arguments for file resolver configuration.

**Returns:**
- **DeveloperWorkspaceFileResolver** – Configured file resolver instance.

```python
# Get file resolver for specific developer
alice_resolver = discovery_manager.get_file_resolver("alice")

# Get file resolver with shared fallback disabled
isolated_resolver = discovery_manager.get_file_resolver(
    "bob",
    enable_shared_fallback=False
)
```

#### get_module_loader

get_module_loader(_developer_id=None_, _**kwargs_)

Get workspace-aware module loader for component loading.

**Parameters:**
- **developer_id** (_Optional[str]_) – Developer to target, uses config default if None.
- ****kwargs** – Additional arguments for module loader configuration.

**Returns:**
- **WorkspaceModuleLoader** – Configured module loader instance.

```python
# Get module loader for specific developer
alice_loader = discovery_manager.get_module_loader("alice")

# Get module loader with caching disabled
no_cache_loader = discovery_manager.get_module_loader(
    "bob",
    cache_modules=False
)

# Load builder class using module loader
builder_class = alice_loader.load_builder_class("data_preprocessing")
```

#### list_available_developers

list_available_developers()

Get list of available developer IDs in the workspace.

**Returns:**
- **List[str]** – Sorted list of developer IDs found in the workspace.

```python
# Get all available developers
developers = discovery_manager.list_available_developers()
print("Available developers:", developers)

# Check if specific developer exists
if "alice" in developers:
    print("Alice's workspace is available")
```

#### get_workspace_info

get_workspace_info(_workspace_id=None_, _developer_id=None_)

Get workspace information for specific workspace or all workspaces.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace ID to get info for.
- **developer_id** (_Optional[str]_) – Optional developer ID to get info for.

**Returns:**
- **Dict[str, Any]** – Workspace information dictionary with structure and component details.

```python
# Get info for specific workspace
alice_info = discovery_manager.get_workspace_info(developer_id="alice")
print("Alice's workspace:", alice_info['component_count'], "components")

# Get info for shared workspace
shared_info = discovery_manager.get_workspace_info(workspace_id="shared")

# Get info for all workspaces
all_info = discovery_manager.get_workspace_info()
print("Total workspaces:", all_info['summary']['total_workspaces'])
```

#### refresh_cache

refresh_cache()

Refresh component discovery cache to force re-discovery.

```python
# Refresh cache after workspace changes
discovery_manager.refresh_cache()

# Re-discover components with fresh cache
fresh_components = discovery_manager.discover_components()
```

#### get_discovery_summary

get_discovery_summary()

Get summary of discovery activities and cache status.

**Returns:**
- **Dict[str, Any]** – Summary of cached discoveries and available developers.

```python
# Get discovery activity summary
summary = discovery_manager.get_discovery_summary()

print("Cached discoveries:", summary['cached_discoveries'])
print("Available developers:", summary['available_developers'])
print("Last discovery:", summary['last_discovery'])
```

#### get_statistics

get_statistics()

Get comprehensive discovery management statistics.

**Returns:**
- **Dict[str, Any]** – Statistics including cache performance and component summaries.

```python
# Get comprehensive statistics
stats = discovery_manager.get_statistics()

print("Cache hit ratio:", stats['discovery_operations']['cache_hit_ratio'])
print("Component summary:", stats['component_summary'])
print("Discovery operations:", stats['discovery_operations'])
```

## Related Documentation

- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Validation](../validation/README.md) - File resolver and module loader components
- [Workspace Configuration](config.md) - Pipeline and step configuration models
- [Workspace Pipeline Assembler](assembler.md) - Pipeline assembly using discovered components
- [Workspace API](../api.md) - High-level workspace API interface
