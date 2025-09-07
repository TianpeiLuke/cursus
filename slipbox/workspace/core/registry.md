---
tags:
  - code
  - workspace
  - registry
  - components
  - discovery
keywords:
  - WorkspaceComponentRegistry
  - component discovery
  - builder registry
  - workspace components
  - unified registry
topics:
  - workspace management
  - component registry
  - component discovery
language: python
date of note: 2024-12-07
---

# Workspace Component Registry

Workspace component registry for discovering and managing workspace components with unified registry integration.

## Overview

The Workspace Component Registry provides registry functionality for workspace component discovery, caching, and management, integrating with the unified registry system to eliminate redundancy. This module serves as the workspace-aware interface to the component registry system, providing efficient component discovery and caching.

The registry system supports component discovery across multiple workspaces, builder and configuration class resolution, component availability validation, and integration with the UnifiedRegistryManager for consistent caching. It provides backward compatibility with legacy discovery mechanisms while leveraging modern unified registry capabilities.

Key features include workspace component discovery and management, integration with UnifiedRegistryManager, builder and configuration class resolution, component availability validation, and comprehensive caching with expiration.

## Classes and Methods

### Classes
- [`WorkspaceComponentRegistry`](#workspacecomponentregistry) - Registry for workspace component discovery and management

### Methods
- [`discover_components`](#discover_components) - Discover components in workspace(s) with unified caching
- [`find_builder_class`](#find_builder_class) - Find builder class for a step using unified registry
- [`find_config_class`](#find_config_class) - Find config class for a step
- [`get_workspace_summary`](#get_workspace_summary) - Get summary of workspace components
- [`validate_component_availability`](#validate_component_availability) - Validate component availability for pipeline assembly
- [`clear_cache`](#clear_cache) - Clear all caches

## API Reference

### WorkspaceComponentRegistry

_class_ cursus.workspace.core.registry.WorkspaceComponentRegistry(_workspace_root_, _discovery_manager=None_)

Registry for workspace component discovery and management with UnifiedRegistryManager integration.

**Parameters:**
- **workspace_root** (_str_) – Root path of the workspace directory.
- **discovery_manager** (_Optional[WorkspaceDiscoveryManager]_) – Optional consolidated WorkspaceDiscoveryManager instance for backward compatibility.

```python
from cursus.workspace.core.registry import WorkspaceComponentRegistry

# Create workspace component registry
registry = WorkspaceComponentRegistry(
    workspace_root="/path/to/workspace"
)

# Create with discovery manager
from cursus.workspace.core.discovery import WorkspaceDiscoveryManager
discovery_manager = WorkspaceDiscoveryManager(workspace_manager)
registry = WorkspaceComponentRegistry(
    workspace_root="/path/to/workspace",
    discovery_manager=discovery_manager
)

print("Registry initialized with UnifiedRegistryManager:", registry._unified_available)
```

#### discover_components

discover_components(_developer_id=None_)

Discover components in workspace(s) using unified caching when available.

**Parameters:**
- **developer_id** (_Optional[str]_) – Optional specific developer ID to discover components for.

**Returns:**
- **Dict[str, Any]** – Dictionary containing discovered components organized by type with summary information.

```python
# Discover all components
all_components = registry.discover_components()

print("Total components:", all_components['summary']['total_components'])
print("Developers:", all_components['summary']['developers'])
print("Step types:", all_components['summary']['step_types'])

# Component counts by type
print("Builders:", len(all_components['builders']))
print("Configs:", len(all_components['configs']))
print("Scripts:", len(all_components['scripts']))

# Discover components for specific developer
alice_components = registry.discover_components(developer_id="alice")
print("Alice's components:", alice_components['summary']['total_components'])

# Access specific components
for component_id, component_info in all_components['builders'].items():
    print(f"Builder: {component_id}")
    print(f"  Developer: {component_info['developer_id']}")
    print(f"  Step type: {component_info['step_type']}")
    print(f"  Class: {component_info['class_name']}")
```

#### find_builder_class

find_builder_class(_step_name_, _developer_id=None_)

Find builder class for a step using UnifiedRegistryManager when available.

**Parameters:**
- **step_name** (_str_) – Name of the step to find builder for.
- **developer_id** (_Optional[str]_) – Optional developer ID to search in.

**Returns:**
- **Optional[Type[StepBuilderBase]]** – Builder class if found, None otherwise.

```python
from cursus.core.base import StepBuilderBase

# Find builder class for specific step
builder_class = registry.find_builder_class("data_preprocessing")

if builder_class:
    print(f"Found builder: {builder_class.__name__}")
    print(f"Step type: {getattr(builder_class, 'step_type', 'Unknown')}")
    
    # Create builder instance
    builder_instance = builder_class()
    print(f"Builder created: {type(builder_instance)}")
else:
    print("Builder not found")

# Find builder in specific developer workspace
alice_builder = registry.find_builder_class("custom_processor", developer_id="alice")

if alice_builder:
    print(f"Found Alice's builder: {alice_builder.__name__}")
else:
    print("Builder not found in Alice's workspace")

# Search across all workspaces
any_builder = registry.find_builder_class("common_step")
print(f"Builder found: {any_builder is not None}")
```

#### find_config_class

find_config_class(_step_name_, _developer_id=None_)

Find config class for a step with workspace-aware resolution.

**Parameters:**
- **step_name** (_str_) – Name of the step to find config for.
- **developer_id** (_Optional[str]_) – Optional developer ID to search in.

**Returns:**
- **Optional[Type[BasePipelineConfig]]** – Config class if found, None otherwise.

```python
from cursus.core.base import BasePipelineConfig

# Find config class for step
config_class = registry.find_config_class("data_preprocessing")

if config_class:
    print(f"Found config: {config_class.__name__}")
    
    # Create config instance
    config_instance = config_class()
    print(f"Config created: {type(config_instance)}")
else:
    print("Config not found")

# Find config in specific developer workspace
bob_config = registry.find_config_class("model_training", developer_id="bob")

if bob_config:
    print(f"Found Bob's config: {bob_config.__name__}")
    
    # Inspect config fields
    if hasattr(bob_config, '__fields__'):
        print("Config fields:", list(bob_config.__fields__.keys()))
else:
    print("Config not found in Bob's workspace")
```

#### get_workspace_summary

get_workspace_summary()

Get summary of workspace components with counts and metadata.

**Returns:**
- **Dict[str, Any]** – Summary dictionary containing workspace information, component counts, and developer list.

```python
# Get comprehensive workspace summary
summary = registry.get_workspace_summary()

print("Workspace Summary:")
print(f"  Root: {summary['workspace_root']}")
print(f"  Total components: {summary['total_components']}")
print(f"  Developers: {summary['developers']}")
print(f"  Step types: {summary['step_types']}")

# Component breakdown
component_counts = summary['component_counts']
print("Component Counts:")
for component_type, count in component_counts.items():
    print(f"  {component_type}: {count}")

# Check for errors
if 'error' in summary:
    print(f"Summary error: {summary['error']}")
```

#### validate_component_availability

validate_component_availability(_workspace_config_)

Validate component availability for pipeline assembly with detailed results.

**Parameters:**
- **workspace_config** (_WorkspacePipelineDefinition_) – WorkspacePipelineDefinition instance to validate.

**Returns:**
- **Dict[str, Any]** – Validation result with availability status, missing components, and warnings.

```python
from cursus.workspace.core.config import WorkspacePipelineDefinition, WorkspaceStepDefinition

# Create workspace pipeline configuration
pipeline_config = WorkspacePipelineDefinition(
    pipeline_name="test_pipeline",
    workspace_root="/path/to/workspace",
    steps=[
        WorkspaceStepDefinition(
            step_name="data_prep",
            developer_id="alice",
            step_type="ProcessingStep",
            config_data={},
            workspace_root="/path/to/workspace"
        ),
        WorkspaceStepDefinition(
            step_name="training",
            developer_id="bob",
            step_type="TrainingStep",
            config_data={},
            workspace_root="/path/to/workspace"
        )
    ]
)

# Validate component availability
validation = registry.validate_component_availability(pipeline_config)

print("Validation Results:")
print(f"  Valid: {validation['valid']}")
print(f"  Available components: {len(validation['available_components'])}")
print(f"  Missing components: {len(validation['missing_components'])}")
print(f"  Warnings: {len(validation['warnings'])}")

# Review available components
for component in validation['available_components']:
    print(f"  ✅ {component['step_name']} ({component['component_type']}) - {component['class_name']}")

# Review missing components
for component in validation['missing_components']:
    print(f"  ❌ {component['step_name']} ({component['component_type']}) - Missing")

# Review warnings
for warning in validation['warnings']:
    print(f"  ⚠️  {warning}")
```

#### clear_cache

clear_cache()

Clear all caches including component, builder, and config caches.

```python
# Clear all registry caches
registry.clear_cache()
print("All caches cleared")

# Force fresh component discovery
fresh_components = registry.discover_components()
print("Fresh discovery completed:", fresh_components['summary']['total_components'])
```

## Registry Integration

### UnifiedRegistryManager Integration

The registry integrates with the UnifiedRegistryManager when available:

```python
# Check if unified registry is available
if registry._unified_available:
    print("Using UnifiedRegistryManager for enhanced caching")
    
    # Unified caching is automatically used
    components = registry.discover_components()
    
    # Step resolution uses unified registry
    builder = registry.find_builder_class("my_step")
else:
    print("Using fallback registry implementation")
    
    # Legacy caching and discovery
    components = registry.discover_components()
```

### Component Discovery Workflow

```python
# Complete component discovery workflow
def discover_workspace_components(registry, developer_id=None):
    print(f"Discovering components for: {developer_id or 'all developers'}")
    
    # 1. Discover components
    components = registry.discover_components(developer_id)
    
    if 'error' in components['summary']:
        print(f"Discovery error: {components['summary']['error']}")
        return None
    
    # 2. Analyze discovered components
    summary = components['summary']
    print(f"Found {summary['total_components']} components")
    print(f"Developers: {summary['developers']}")
    print(f"Step types: {summary['step_types']}")
    
    # 3. Validate key components
    validation_results = {}
    
    for component_id, component_info in components['builders'].items():
        step_name = component_info['step_name']
        dev_id = component_info['developer_id']
        
        # Try to load builder class
        builder_class = registry.find_builder_class(step_name, dev_id)
        validation_results[component_id] = {
            'builder_loadable': builder_class is not None,
            'builder_class': builder_class.__name__ if builder_class else None
        }
        
        # Try to load config class
        config_class = registry.find_config_class(step_name, dev_id)
        validation_results[component_id]['config_loadable'] = config_class is not None
        validation_results[component_id]['config_class'] = config_class.__name__ if config_class else None
    
    return {
        'components': components,
        'validation': validation_results
    }
```

### Registry Performance Monitoring

```python
# Monitor registry performance
def monitor_registry_performance(registry):
    import time
    
    # Test component discovery performance
    start_time = time.time()
    components = registry.discover_components()
    discovery_time = time.time() - start_time
    
    print(f"Component discovery took: {discovery_time:.3f}s")
    print(f"Components found: {components['summary']['total_components']}")
    
    # Test builder resolution performance
    builder_times = []
    for component_id, component_info in list(components['builders'].items())[:5]:  # Test first 5
        start_time = time.time()
        builder = registry.find_builder_class(
            component_info['step_name'], 
            component_info['developer_id']
        )
        builder_time = time.time() - start_time
        builder_times.append(builder_time)
        
        print(f"Builder resolution for {component_info['step_name']}: {builder_time:.3f}s")
    
    if builder_times:
        avg_builder_time = sum(builder_times) / len(builder_times)
        print(f"Average builder resolution time: {avg_builder_time:.3f}s")
    
    # Test cache effectiveness
    print("\nTesting cache effectiveness...")
    
    # First discovery (cache miss)
    start_time = time.time()
    registry.discover_components()
    first_time = time.time() - start_time
    
    # Second discovery (cache hit)
    start_time = time.time()
    registry.discover_components()
    second_time = time.time() - start_time
    
    print(f"First discovery (cache miss): {first_time:.3f}s")
    print(f"Second discovery (cache hit): {second_time:.3f}s")
    print(f"Cache speedup: {first_time / second_time:.1f}x")
```

### Error Handling and Fallbacks

```python
# Robust registry operations with fallbacks
def safe_registry_operations(registry, step_name, developer_id=None):
    results = {
        'discovery': None,
        'builder': None,
        'config': None,
        'validation': None,
        'errors': []
    }
    
    try:
        # Safe component discovery
        components = registry.discover_components(developer_id)
        if 'error' not in components['summary']:
            results['discovery'] = components
        else:
            results['errors'].append(f"Discovery error: {components['summary']['error']}")
    except Exception as e:
        results['errors'].append(f"Discovery exception: {e}")
    
    try:
        # Safe builder resolution
        builder_class = registry.find_builder_class(step_name, developer_id)
        results['builder'] = {
            'found': builder_class is not None,
            'class_name': builder_class.__name__ if builder_class else None
        }
    except Exception as e:
        results['errors'].append(f"Builder resolution error: {e}")
    
    try:
        # Safe config resolution
        config_class = registry.find_config_class(step_name, developer_id)
        results['config'] = {
            'found': config_class is not None,
            'class_name': config_class.__name__ if config_class else None
        }
    except Exception as e:
        results['errors'].append(f"Config resolution error: {e}")
    
    # Overall success assessment
    results['success'] = len(results['errors']) == 0
    
    return results
```

## Related Documentation

- [Workspace Discovery Manager](discovery.md) - Component discovery and cross-workspace resolution
- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Step Builder Registry](../../registry/builder_registry.md) - Core step builder registry
- [Unified Registry Manager](../../registry/hybrid/manager.md) - Unified registry management system
- [Workspace Configuration](config.md) - Pipeline and step configuration models
