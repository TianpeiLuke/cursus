---
tags:
  - code
  - registry
  - hybrid
  - manager
  - unified_registry
keywords:
  - UnifiedRegistryManager
  - hybrid registry
  - workspace management
  - registry backend
  - performance optimization
topics:
  - unified registry manager
  - hybrid registry backend
  - workspace support
language: python
date of note: 2024-12-07
---

# Unified Registry Manager

Unified Registry Manager Implementation that consolidates all registry operations into a single, efficient manager eliminating redundancy while maintaining all functionality.

## Overview

The UnifiedRegistryManager replaces CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager with a single, efficient manager that handles all registry operations. It provides workspace-aware step resolution, performance optimization through caching, thread-safe operations, and intelligent conflict resolution.

The manager supports core registry data loading, workspace registry management, performance optimization with LRU caches, thread safety with proper locking, and comprehensive registry status reporting.

## Classes and Methods

### Classes
- [`UnifiedRegistryManager`](#unifiedregistrymanager) - Unified registry manager for all registry operations

## API Reference

### UnifiedRegistryManager

_class_ cursus.registry.hybrid.manager.UnifiedRegistryManager(_core_registry_path=None_, _workspaces_root=None_)

Unified registry manager that consolidates all registry operations. Replaces CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager with a single, efficient manager that eliminates redundancy while maintaining all functionality.

**Parameters:**
- **core_registry_path** (_Optional[str]_) – Path to core registry file. Defaults to "src/cursus/registry/step_names.py".
- **workspaces_root** (_Optional[str]_) – Root directory for workspace registries. Defaults to "developer_workspaces/developers".

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Create manager with default paths
manager = UnifiedRegistryManager()

# Create manager with custom paths
custom_manager = UnifiedRegistryManager(
    core_registry_path="custom/path/to/registry.py",
    workspaces_root="custom/workspaces"
)

# Get basic information
status = manager.get_registry_status()
print(f"Core loaded: {status['core']['loaded']}")
print(f"Core steps: {status['core']['step_count']}")
```

#### get_step_definition

get_step_definition(_step_name_, _workspace_id=None_)

Get a step definition by name with optional workspace context.

**Parameters:**
- **step_name** (_str_) – Name of the step to retrieve.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **Optional[StepDefinition]** – StepDefinition if found, None otherwise.

```python
# Get step from core registry
step_def = manager.get_step_definition("XGBoostTraining")
if step_def:
    print(f"Step: {step_def.step_name}")
    print(f"Config: {step_def.config_class}")
    print(f"Builder: {step_def.builder_step_name}")
    print(f"SageMaker type: {step_def.sagemaker_step_type}")

# Get step with workspace context
workspace_step = manager.get_step_definition("CustomStep", workspace_id="developer_1")
if workspace_step:
    print(f"Workspace step: {workspace_step.step_name}")
    print(f"Source: {workspace_step.registry_type}")
```

#### get_all_step_definitions

get_all_step_definitions(_workspace_id=None_)

Get all step definitions with caching for performance optimization.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context. Defaults to None.

**Returns:**
- **Dict[str, StepDefinition]** – Dictionary mapping step names to their definitions.

```python
# Get all core step definitions
all_steps = manager.get_all_step_definitions()
print(f"Total core steps: {len(all_steps)}")

# Get all steps in workspace context
workspace_steps = manager.get_all_step_definitions("developer_1")
print(f"Workspace steps: {len(workspace_steps)}")

# Iterate through definitions
for step_name, step_def in all_steps.items():
    print(f"{step_name}: {step_def.description}")
```

#### list_steps

list_steps(_workspace_id=None_)

List all available step names for a workspace or core registry.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context. Defaults to None.

**Returns:**
- **List[str]** – Sorted list of step names.

```python
# List core steps
core_steps = manager.list_steps()
print(f"Core steps ({len(core_steps)}): {core_steps[:5]}...")

# List workspace steps
workspace_steps = manager.list_steps("developer_1")
print(f"Workspace steps ({len(workspace_steps)}): {workspace_steps[:5]}...")

# Compare step counts
print(f"Additional steps in workspace: {len(workspace_steps) - len(core_steps)}")
```

#### has_step

has_step(_step_name_, _workspace_id=None_)

Check if a step exists in the registry.

**Parameters:**
- **step_name** (_str_) – Name of the step to check.
- **workspace_id** (_Optional[str]_) – Optional workspace context. Defaults to None.

**Returns:**
- **bool** – True if step exists, False otherwise.

```python
# Check core steps
test_steps = ["XGBoostTraining", "CustomStep", "NonExistentStep"]

for step_name in test_steps:
    exists_core = manager.has_step(step_name)
    exists_workspace = manager.has_step(step_name, "developer_1")
    
    print(f"{step_name}:")
    print(f"  Core: {exists_core}")
    print(f"  Workspace: {exists_workspace}")
```

#### get_step_count

get_step_count(_workspace_id=None_)

Get the total number of steps in the registry.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context. Defaults to None.

**Returns:**
- **int** – Total number of steps.

```python
# Get step counts
core_count = manager.get_step_count()
workspace_count = manager.get_step_count("developer_1")

print(f"Core registry: {core_count} steps")
print(f"Workspace registry: {workspace_count} steps")
print(f"Additional workspace steps: {workspace_count - core_count}")
```

#### add_workspace_registry

add_workspace_registry(_workspace_id_, _workspace_path_)

Add a new workspace registry with automatic discovery and loading.

**Parameters:**
- **workspace_id** (_str_) – Identifier for the workspace.
- **workspace_path** (_str_) – Path to the workspace directory.

```python
# Add workspace registry
try:
    manager.add_workspace_registry("developer_2", "/path/to/developer_2/workspace")
    print("✓ Workspace registry added successfully")
    
    # Verify addition
    status = manager.get_registry_status()
    if "developer_2" in status:
        workspace_info = status["developer_2"]
        print(f"  Local steps: {workspace_info['local_step_count']}")
        print(f"  Overrides: {workspace_info['override_count']}")
        print(f"  Total steps: {workspace_info['total_step_count']}")
    
except Exception as e:
    print(f"✗ Failed to add workspace: {e}")
```

#### remove_workspace_registry

remove_workspace_registry(_workspace_id_)

Remove a workspace registry and clean up associated data.

**Parameters:**
- **workspace_id** (_str_) – Identifier for the workspace to remove.

**Returns:**
- **bool** – True if workspace was removed, False if it didn't exist.

```python
# Remove workspace registry
removed = manager.remove_workspace_registry("developer_2")
if removed:
    print("✓ Workspace registry removed")
    
    # Verify removal
    status = manager.get_registry_status()
    remaining_workspaces = [k for k in status.keys() if k != "core"]
    print(f"Remaining workspaces: {remaining_workspaces}")
else:
    print("✗ Workspace not found")
```

#### get_step_conflicts

get_step_conflicts()

Identify steps defined in multiple registries (conflict detection).

**Returns:**
- **Dict[str, List[StepDefinition]]** – Dictionary mapping conflicting step names to their definitions.

```python
# Check for step conflicts
conflicts = manager.get_step_conflicts()

if conflicts:
    print(f"Found {len(conflicts)} step conflicts:")
    for step_name, definitions in conflicts.items():
        print(f"\n{step_name} ({len(definitions)} definitions):")
        for step_def in definitions:
            print(f"  - {step_def.registry_type}: {step_def.workspace_id or 'core'}")
            print(f"    Config: {step_def.config_class}")
            print(f"    Builder: {step_def.builder_step_name}")
else:
    print("✓ No step conflicts detected")
```

#### get_registry_status

get_registry_status()

Get comprehensive status information for all registries.

**Returns:**
- **Dict[str, Dict[str, Any]]** – Dictionary with status information for each registry.

```python
# Get comprehensive registry status
status = manager.get_registry_status()

print("Registry Status Report:")
print("=" * 30)

# Core registry status
core_status = status["core"]
print(f"Core Registry:")
print(f"  Loaded: {core_status['loaded']}")
print(f"  Steps: {core_status['step_count']}")
print(f"  Path: {core_status['registry_path']}")

# Workspace registry status
workspace_ids = [k for k in status.keys() if k != "core"]
if workspace_ids:
    print(f"\nWorkspace Registries ({len(workspace_ids)}):")
    for workspace_id in workspace_ids:
        ws_status = status[workspace_id]
        print(f"  {workspace_id}:")
        print(f"    Local steps: {ws_status['local_step_count']}")
        print(f"    Overrides: {ws_status['override_count']}")
        print(f"    Total steps: {ws_status['total_step_count']}")
        
        # Show metadata if available
        metadata = ws_status.get('metadata', {})
        if metadata:
            print(f"    Developer: {metadata.get('developer_id', 'Unknown')}")
            print(f"    Description: {metadata.get('description', 'No description')}")
else:
    print("\nNo workspace registries loaded")
```

#### create_legacy_step_names_dict

create_legacy_step_names_dict(_workspace_id=None_)

Create legacy STEP_NAMES dictionary for backward compatibility with caching.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context. Defaults to None.

**Returns:**
- **Dict[str, Dict[str, Any]]** – Legacy format step names dictionary.

```python
# Create legacy format dictionary
legacy_dict = manager.create_legacy_step_names_dict()
print(f"Legacy dictionary contains {len(legacy_dict)} steps")

# Access step information in legacy format
if "XGBoostTraining" in legacy_dict:
    xgb_info = legacy_dict["XGBoostTraining"]
    print(f"XGBoost Training:")
    print(f"  Config class: {xgb_info['config_class']}")
    print(f"  Builder: {xgb_info['builder_step_name']}")
    print(f"  SageMaker type: {xgb_info['sagemaker_step_type']}")
    print(f"  Description: {xgb_info['description']}")

# Create workspace-specific legacy dictionary
workspace_legacy = manager.create_legacy_step_names_dict("developer_1")
print(f"Workspace legacy dictionary: {len(workspace_legacy)} steps")
```

#### set_workspace_context

set_workspace_context(_workspace_id_)

Set current workspace context with cache invalidation.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to set as current context.

```python
# Set workspace context
manager.set_workspace_context("developer_1")
print("Workspace context set to developer_1")

# Operations now use workspace context by default
current_context = manager.get_workspace_context()
print(f"Current context: {current_context}")

# Get steps in current context
context_steps = manager.list_steps()
print(f"Steps in current context: {len(context_steps)}")
```

#### get_workspace_context

get_workspace_context()

Get current workspace context identifier.

**Returns:**
- **Optional[str]** – Current workspace identifier or None if no context set.

```python
# Check current workspace context
current_context = manager.get_workspace_context()
if current_context:
    print(f"Current workspace context: {current_context}")
    
    # Get context-specific information
    context_steps = manager.list_steps()
    context_count = len(context_steps)
    print(f"Steps in context: {context_count}")
else:
    print("No workspace context set (using core registry)")
```

#### clear_workspace_context

clear_workspace_context()

Clear current workspace context and return to core registry.

```python
# Clear workspace context
manager.clear_workspace_context()
print("Workspace context cleared")

# Verify context is cleared
current_context = manager.get_workspace_context()
print(f"Current context: {current_context}")  # Should be None

# Operations now use core registry
core_steps = manager.list_steps()
print(f"Core registry steps: {len(core_steps)}")
```

## Usage Examples

### Complete Registry Management Workflow

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Initialize manager
manager = UnifiedRegistryManager()

print("Registry Management Workflow")
print("=" * 35)

# 1. Check initial status
print("1. Initial Status:")
status = manager.get_registry_status()
print(f"   Core loaded: {status['core']['loaded']}")
print(f"   Core steps: {status['core']['step_count']}")
print(f"   Workspaces: {len([k for k in status.keys() if k != 'core'])}")

# 2. Add workspace registries
print("\n2. Adding Workspace Registries:")
workspaces = [
    ("developer_1", "/path/to/dev1/workspace"),
    ("developer_2", "/path/to/dev2/workspace")
]

for workspace_id, path in workspaces:
    try:
        manager.add_workspace_registry(workspace_id, path)
        print(f"   ✓ Added {workspace_id}")
    except Exception as e:
        print(f"   ✗ Failed to add {workspace_id}: {e}")

# 3. Check for conflicts
print("\n3. Conflict Detection:")
conflicts = manager.get_step_conflicts()
if conflicts:
    print(f"   Found {len(conflicts)} conflicts:")
    for step_name, definitions in list(conflicts.items())[:3]:  # Show first 3
        print(f"     {step_name}: {len(definitions)} definitions")
else:
    print("   ✓ No conflicts detected")

# 4. Workspace context management
print("\n4. Workspace Context Management:")
available_workspaces = [k for k in manager.get_registry_status().keys() if k != "core"]

if available_workspaces:
    workspace_id = available_workspaces[0]
    manager.set_workspace_context(workspace_id)
    print(f"   Set context to: {workspace_id}")
    
    context_steps = manager.list_steps()
    core_steps = manager.list_steps(None)  # Explicitly get core steps
    print(f"   Context steps: {len(context_steps)}")
    print(f"   Core steps: {len(core_steps)}")
    print(f"   Additional: {len(context_steps) - len(core_steps)}")
    
    # Clear context
    manager.clear_workspace_context()
    print("   ✓ Context cleared")

# 5. Performance and caching
print("\n5. Performance Information:")
final_status = manager.get_registry_status()
total_steps = sum(info.get('step_count', info.get('total_step_count', 0)) 
                 for info in final_status.values())
print(f"   Total steps across all registries: {total_steps}")
print(f"   Registries loaded: {len(final_status)}")
```

### Advanced Step Resolution

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.registry.hybrid.models import ResolutionContext

# Create manager and resolution context
manager = UnifiedRegistryManager()
context = ResolutionContext(workspace_id="developer_1")

print("Advanced Step Resolution")
print("=" * 28)

# Test step resolution with different scenarios
test_steps = [
    "XGBoostTraining",      # Core step
    "CustomAnalysis",       # Workspace-specific step
    "NonExistentStep"       # Should fail
]

for step_name in test_steps:
    print(f"\nResolving: {step_name}")
    
    # Use the get_step method for advanced resolution
    result = manager.get_step(step_name, context)
    
    print(f"  Resolved: {result.resolved}")
    print(f"  Source: {result.source_registry}")
    print(f"  Strategy: {result.resolution_strategy}")
    
    if result.resolved and result.selected_definition:
        step_def = result.selected_definition
        print(f"  Config: {step_def.config_class}")
        print(f"  Builder: {step_def.builder_step_name}")
        print(f"  Type: {step_def.sagemaker_step_type}")
    
    if result.errors:
        print(f"  Errors: {result.errors}")
    
    if result.warnings:
        print(f"  Warnings: {result.warnings}")
    
    if result.conflict_detected:
        print(f"  Conflict detected: {result.conflict_analysis}")
```

### Registry Maintenance and Monitoring

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

manager = UnifiedRegistryManager()

print("Registry Maintenance and Monitoring")
print("=" * 40)

# 1. Registry health check
print("1. Registry Health Check:")
status = manager.get_registry_status()

for registry_id, info in status.items():
    print(f"   {registry_id}:")
    print(f"     Loaded: {info.get('loaded', 'Unknown')}")
    
    if registry_id == "core":
        print(f"     Steps: {info['step_count']}")
        print(f"     Path: {info['registry_path']}")
    else:
        print(f"     Local steps: {info['local_step_count']}")
        print(f"     Overrides: {info['override_count']}")
        print(f"     Total steps: {info['total_step_count']}")

# 2. Conflict analysis
print("\n2. Conflict Analysis:")
conflicts = manager.get_step_conflicts()
if conflicts:
    print(f"   Total conflicts: {len(conflicts)}")
    
    # Analyze conflict types
    conflict_types = {}
    for step_name, definitions in conflicts.items():
        for step_def in definitions:
            reg_type = step_def.registry_type
            conflict_types[reg_type] = conflict_types.get(reg_type, 0) + 1
    
    print("   Conflict breakdown:")
    for reg_type, count in conflict_types.items():
        print(f"     {reg_type}: {count}")
else:
    print("   ✓ No conflicts detected")

# 3. Cache management
print("\n3. Cache Management:")
# Clear caches for fresh data
manager._invalidate_all_caches()
print("   ✓ All caches invalidated")

# Warm up caches with common operations
legacy_dict = manager.create_legacy_step_names_dict()
all_definitions = manager.get_all_step_definitions()
print(f"   ✓ Caches warmed up ({len(legacy_dict)} steps)")

# 4. Workspace management
print("\n4. Workspace Management:")
workspace_ids = [k for k in status.keys() if k != "core"]

if workspace_ids:
    print(f"   Active workspaces: {len(workspace_ids)}")
    for workspace_id in workspace_ids:
        step_count = manager.get_step_count(workspace_id)
        print(f"     {workspace_id}: {step_count} steps")
    
    # Test workspace removal and re-addition
    if len(workspace_ids) > 1:
        test_workspace = workspace_ids[-1]
        print(f"   Testing removal of {test_workspace}:")
        
        removed = manager.remove_workspace_registry(test_workspace)
        print(f"     Removed: {removed}")
        
        # Check status after removal
        new_status = manager.get_registry_status()
        remaining = [k for k in new_status.keys() if k != "core"]
        print(f"     Remaining workspaces: {len(remaining)}")
else:
    print("   No workspaces currently loaded")
```

## Related Components

- **[Hybrid Models](models.md)** - Data models used by the unified manager
- **[Hybrid Utils](utils.md)** - Utility functions for registry operations
- **[Registry Module](../__init__.md)** - Main registry module that uses the manager
- **[Step Names](../step_names.md)** - Step names registry with manager integration
