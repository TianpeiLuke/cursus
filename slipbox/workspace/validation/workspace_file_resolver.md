---
tags:
  - code
  - workspace
  - validation
  - file-resolver
  - multi-developer
keywords:
  - DeveloperWorkspaceFileResolver
  - workspace file resolution
  - multi-developer workspace
  - file discovery
  - workspace isolation
topics:
  - workspace validation
  - file resolution
  - developer workspaces
language: python
date of note: 2024-12-07
---

# Developer Workspace File Resolver

Workspace-aware file resolver that extends FlexibleFileResolver to support multi-developer workspace structures with isolation and fallback capabilities.

## Overview

The Developer Workspace File Resolver extends the existing FlexibleFileResolver capabilities to support multi-developer workspace directory structures. This module provides workspace-aware file discovery for contracts, specs, builders, scripts, and configurations while maintaining backward compatibility with single workspace mode.

The resolver supports developer workspace discovery and validation, workspace-specific file resolution with fallback to shared resources, path isolation between developer workspaces, and comprehensive component discovery across workspace boundaries. It integrates Phase 4 consolidation features for enhanced workspace discovery and statistics gathering.

Key features include workspace-aware file resolution, developer workspace isolation, shared workspace fallback, component discovery and statistics, and backward compatibility with single workspace mode.

## Classes and Methods

### Classes
- [`DeveloperWorkspaceFileResolver`](#developerworkspacefileresolver) - Workspace-aware file resolver with multi-developer support

### Methods
- [`find_contract_file`](#find_contract_file) - Find contract file with workspace-aware search
- [`find_spec_file`](#find_spec_file) - Find spec file with workspace-aware search
- [`find_builder_file`](#find_builder_file) - Find builder file with workspace-aware search
- [`find_script_file`](#find_script_file) - Find script file with workspace-aware search
- [`find_config_file`](#find_config_file) - Find config file with workspace-aware search
- [`get_workspace_info`](#get_workspace_info) - Get workspace configuration information
- [`discover_workspace_components`](#discover_workspace_components) - Discover components across workspaces
- [`discover_components_by_type`](#discover_components_by_type) - Discover components by type
- [`resolve_component_path`](#resolve_component_path) - Resolve component path with workspace awareness
- [`get_component_statistics`](#get_component_statistics) - Get component statistics across workspaces
- [`list_available_developers`](#list_available_developers) - List all available developer workspaces
- [`switch_developer`](#switch_developer) - Switch to different developer workspace

## API Reference

### DeveloperWorkspaceFileResolver

_class_ cursus.workspace.validation.workspace_file_resolver.DeveloperWorkspaceFileResolver(_workspace_root=None_, _developer_id=None_, _enable_shared_fallback=True_, _**kwargs_)

Workspace-aware file resolver that extends FlexibleFileResolver to support multi-developer workspace structures.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory containing developer workspaces.
- **developer_id** (_Optional[str]_) – Specific developer workspace to target.
- **enable_shared_fallback** (_bool_) – Whether to fallback to shared workspace, defaults to True.
- ****kwargs** – Additional arguments passed to FlexibleFileResolver.

```python
from cursus.workspace.validation.workspace_file_resolver import DeveloperWorkspaceFileResolver
from pathlib import Path

# Create workspace-aware file resolver
resolver = DeveloperWorkspaceFileResolver(
    workspace_root="/path/to/workspace",
    developer_id="alice",
    enable_shared_fallback=True
)

# Create resolver without workspace mode (single workspace compatibility)
single_resolver = DeveloperWorkspaceFileResolver()

print("Workspace mode:", resolver.workspace_mode)
print("Developer ID:", resolver.developer_id)
```

#### find_contract_file

find_contract_file(_step_name_)

Find contract file with workspace-aware search including developer and shared workspace fallback.

**Parameters:**
- **step_name** (_str_) – Name of the step to find contract for.

**Returns:**
- **Optional[str]** – Path to contract file if found, None otherwise.

```python
# Find contract file with workspace-aware search
contract_path = resolver.find_contract_file("data_preprocessing")

if contract_path:
    print(f"Found contract: {contract_path}")
    
    # Check if it's from developer or shared workspace
    if "developers/alice" in contract_path:
        print("Found in Alice's workspace")
    elif "shared" in contract_path:
        print("Found in shared workspace")
else:
    print("Contract not found")

# Search order:
# 1. Developer workspace contracts
# 2. Shared workspace contracts (if enabled)
# 3. Parent class fallback behavior
```

#### find_spec_file

find_spec_file(_step_name_)

Find spec file with workspace-aware search and fallback mechanisms.

**Parameters:**
- **step_name** (_str_) – Name of the step to find spec for.

**Returns:**
- **Optional[str]** – Path to spec file if found, None otherwise.

```python
# Find spec file across workspaces
spec_path = resolver.find_spec_file("model_training")

if spec_path:
    print(f"Found spec: {spec_path}")
    
    # Determine file format
    if spec_path.endswith('.json'):
        print("JSON spec file")
    elif spec_path.endswith(('.yaml', '.yml')):
        print("YAML spec file")
else:
    print("Spec not found")
```

#### find_builder_file

find_builder_file(_step_name_)

Find builder file with workspace-aware search and developer isolation.

**Parameters:**
- **step_name** (_str_) – Name of the step to find builder for.

**Returns:**
- **Optional[str]** – Path to builder file if found, None otherwise.

```python
# Find builder file with workspace awareness
builder_path = resolver.find_builder_file("custom_processor")

if builder_path:
    print(f"Found builder: {builder_path}")
    
    # Load and inspect builder
    import importlib.util
    spec = importlib.util.spec_from_file_location("builder", builder_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find builder class
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if hasattr(attr, '__bases__') and 'StepBuilderBase' in [b.__name__ for b in attr.__bases__]:
            print(f"Found builder class: {attr_name}")
else:
    print("Builder not found")
```

#### find_script_file

find_script_file(_step_name_, _script_name=None_)

Find script file with workspace-aware search and optional script name specification.

**Parameters:**
- **step_name** (_str_) – Name of the step to find script for.
- **script_name** (_Optional[str]_) – Optional specific script name to search for.

**Returns:**
- **Optional[str]** – Path to script file if found, None otherwise.

```python
# Find script file by step name
script_path = resolver.find_script_file("data_processing")

if script_path:
    print(f"Found script: {script_path}")
else:
    print("Script not found")

# Find specific script file
specific_script = resolver.find_script_file("training", "train_model")

if specific_script:
    print(f"Found specific script: {specific_script}")
```

#### find_config_file

find_config_file(_step_name_, _config_name=None_)

Find config file with workspace-aware search supporting multiple formats.

**Parameters:**
- **step_name** (_str_) – Name of the step to find config for.
- **config_name** (_Optional[str]_) – Optional specific config name to search for.

**Returns:**
- **Optional[str]** – Path to config file if found, None otherwise.

```python
# Find config file
config_path = resolver.find_config_file("hyperparameter_tuning")

if config_path:
    print(f"Found config: {config_path}")
    
    # Load config based on format
    if config_path.endswith('.json'):
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    elif config_path.endswith(('.yaml', '.yml')):
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    
    print("Config loaded:", list(config_data.keys()))
```

#### get_workspace_info

get_workspace_info()

Get comprehensive information about current workspace configuration and status.

**Returns:**
- **Dict[str, Any]** – Dictionary containing workspace configuration details and status information.

```python
# Get workspace configuration info
workspace_info = resolver.get_workspace_info()

print("Workspace Configuration:")
print(f"  Mode: {workspace_info['workspace_mode']}")
print(f"  Root: {workspace_info['workspace_root']}")
print(f"  Developer: {workspace_info['developer_id']}")
print(f"  Shared fallback: {workspace_info['enable_shared_fallback']}")
print(f"  Developer workspace exists: {workspace_info['developer_workspace_exists']}")
print(f"  Shared workspace exists: {workspace_info['shared_workspace_exists']}")

# Use info for conditional logic
if workspace_info['workspace_mode']:
    print("Operating in multi-developer workspace mode")
else:
    print("Operating in single workspace mode")
```

#### discover_workspace_components

discover_workspace_components()

Enhanced method with consolidated discovery logic for comprehensive workspace component discovery.

**Returns:**
- **Dict[str, Any]** – Discovery result with workspace information, component counts, and summary statistics.

```python
# Discover all workspace components
discovery_result = resolver.discover_workspace_components()

if 'error' in discovery_result:
    print(f"Discovery error: {discovery_result['error']}")
else:
    print("Workspace Discovery Results:")
    print(f"  Root: {discovery_result['workspace_root']}")
    print(f"  Total workspaces: {discovery_result['summary']['total_workspaces']}")
    print(f"  Total developers: {discovery_result['summary']['total_developers']}")
    print(f"  Total components: {discovery_result['summary']['total_components']}")
    
    # Review individual workspaces
    for workspace in discovery_result['workspaces']:
        print(f"  Workspace: {workspace['workspace_id']}")
        print(f"    Type: {workspace['workspace_type']}")
        print(f"    Components: {workspace['component_count']}")
        print(f"    Module types: {workspace['module_types']}")
```

#### discover_components_by_type

discover_components_by_type(_component_type_)

Enhanced method to discover components by type across all workspaces with consolidated logic.

**Parameters:**
- **component_type** (_str_) – Type of component ('builders', 'contracts', 'specs', 'scripts', 'configs').

**Returns:**
- **Dict[str, List[str]]** – Dictionary mapping workspace_id to list of component names.

```python
# Discover builders across all workspaces
builders = resolver.discover_components_by_type("builders")

print("Builders by workspace:")
for workspace_id, builder_list in builders.items():
    print(f"  {workspace_id}: {builder_list}")

# Discover contracts
contracts = resolver.discover_components_by_type("contracts")

# Discover scripts
scripts = resolver.discover_components_by_type("scripts")

# Get comprehensive view
all_component_types = ["builders", "contracts", "specs", "scripts", "configs"]
for comp_type in all_component_types:
    components = resolver.discover_components_by_type(comp_type)
    total_components = sum(len(comp_list) for comp_list in components.values())
    print(f"{comp_type}: {total_components} total across {len(components)} workspaces")
```

#### resolve_component_path

resolve_component_path(_component_type_, _component_name_, _workspace_id=None_)

Enhanced method to resolve component path with workspace-aware search and consolidated logic.

**Parameters:**
- **component_type** (_str_) – Type of component ('builders', 'contracts', 'specs', 'scripts', 'configs').
- **component_name** (_str_) – Name of the component to resolve.
- **workspace_id** (_Optional[str]_) – Optional workspace ID to search in, uses current if None.

**Returns:**
- **Optional[str]** – Full path to component file if found, None otherwise.

```python
# Resolve component path in current workspace
builder_path = resolver.resolve_component_path("builders", "data_processor")

if builder_path:
    print(f"Found builder at: {builder_path}")

# Resolve component in specific workspace
alice_contract = resolver.resolve_component_path(
    "contracts", 
    "model_validation", 
    workspace_id="alice"
)

# Resolve with fallback to shared
shared_script = resolver.resolve_component_path("scripts", "common_utils")

# Try multiple workspaces
component_name = "feature_engineering"
for workspace_id in resolver.list_available_developers():
    path = resolver.resolve_component_path("builders", component_name, workspace_id)
    if path:
        print(f"Found {component_name} in {workspace_id}: {path}")
        break
```

#### get_component_statistics

get_component_statistics()

Enhanced method to get comprehensive component statistics across all workspaces.

**Returns:**
- **Dict[str, Any]** – Statistics including component counts, workspace information, and detailed breakdowns.

```python
# Get comprehensive component statistics
stats = resolver.get_component_statistics()

if 'error' in stats:
    print(f"Statistics error: {stats['error']}")
else:
    print("Component Statistics:")
    print(f"  Workspace root: {stats['workspace_root']}")
    print(f"  Total workspaces: {stats['total_workspaces']}")
    print(f"  Total components: {stats['total_components']}")
    
    # Component type breakdown
    print("Component types:")
    for comp_type, count in stats['component_types'].items():
        print(f"  {comp_type}: {count}")
    
    # Per-workspace breakdown
    print("Per-workspace statistics:")
    for workspace_id, workspace_stats in stats['workspaces'].items():
        print(f"  {workspace_id}:")
        print(f"    Total: {workspace_stats['total_components']}")
        for comp_type, count in workspace_stats['component_types'].items():
            if count > 0:
                print(f"    {comp_type}: {count}")
```

#### list_available_developers

list_available_developers()

List all available developer workspaces with proper structure validation.

**Returns:**
- **List[str]** – Sorted list of developer IDs with valid workspace structures.

```python
# Get list of available developers
developers = resolver.list_available_developers()

print("Available developers:")
for developer_id in developers:
    print(f"  - {developer_id}")

# Check if specific developer exists
if "alice" in developers:
    print("Alice's workspace is available")

# Get developer count
print(f"Total developers: {len(developers)}")

# Use for iteration
for developer_id in developers:
    # Switch to each developer and get their components
    temp_resolver = DeveloperWorkspaceFileResolver(
        workspace_root=resolver.workspace_root,
        developer_id=developer_id
    )
    components = temp_resolver.discover_components_by_type("builders")
    print(f"{developer_id} has {len(components.get(developer_id, []))} builders")
```

#### switch_developer

switch_developer(_developer_id_)

Switch to a different developer workspace and update all internal paths.

**Parameters:**
- **developer_id** (_str_) – Developer identifier to switch to.

**Raises:**
- **ValueError** – If not in workspace mode or developer workspace not found.

```python
# Switch to different developer workspace
try:
    resolver.switch_developer("bob")
    print(f"Switched to Bob's workspace")
    
    # Verify switch
    workspace_info = resolver.get_workspace_info()
    print(f"Current developer: {workspace_info['developer_id']}")
    
    # Find components in new workspace
    builder_path = resolver.find_builder_file("bob_special_processor")
    if builder_path:
        print(f"Found Bob's builder: {builder_path}")
    
except ValueError as e:
    print(f"Switch failed: {e}")

# Switch back to original developer
resolver.switch_developer("alice")
print("Switched back to Alice's workspace")
```

## Workspace Structure

The resolver supports the following workspace structure:

```
development/
├── developers/
│   ├── alice/
│   │   └── src/cursus_dev/steps/
│   │       ├── builders/
│   │       ├── contracts/
│   │       ├── scripts/
│   │       ├── specs/
│   │       └── configs/
│   └── bob/
│       └── src/cursus_dev/steps/
│           ├── builders/
│           ├── contracts/
│           ├── scripts/
│           ├── specs/
│           └── configs/
└── shared/
    └── src/cursus_dev/steps/
        ├── builders/
        ├── contracts/
        ├── scripts/
        ├── specs/
        └── configs/
```

## Usage Patterns

### Multi-Developer Workflow

```python
# Complete multi-developer workflow
def analyze_multi_developer_workspace(workspace_root):
    # Create resolver for workspace analysis
    resolver = DeveloperWorkspaceFileResolver(workspace_root=workspace_root)
    
    # Get workspace overview
    discovery = resolver.discover_workspace_components()
    print(f"Found {discovery['summary']['total_workspaces']} workspaces")
    
    # Analyze each developer
    developers = resolver.list_available_developers()
    
    for developer_id in developers:
        print(f"\nAnalyzing {developer_id}'s workspace:")
        
        # Switch to developer
        resolver.switch_developer(developer_id)
        
        # Find their components
        for comp_type in ["builders", "contracts", "specs", "scripts"]:
            components = resolver.discover_components_by_type(comp_type)
            dev_components = components.get(developer_id, [])
            print(f"  {comp_type}: {len(dev_components)} components")
            
            # Show first few components
            for comp_name in dev_components[:3]:
                path = resolver.resolve_component_path(comp_type, comp_name)
                print(f"    - {comp_name}: {path}")
    
    return discovery
```

### Component Resolution with Fallback

```python
# Component resolution with fallback strategy
def find_component_with_fallback(resolver, component_type, component_name):
    # Try current developer first
    path = resolver.resolve_component_path(component_type, component_name)
    if path:
        return path, "current_developer"
    
    # Try all other developers
    for developer_id in resolver.list_available_developers():
        if developer_id != resolver.developer_id:
            path = resolver.resolve_component_path(
                component_type, component_name, developer_id
            )
            if path:
                return path, developer_id
    
    # Try shared workspace
    path = resolver.resolve_component_path(
        component_type, component_name, "shared"
    )
    if path:
        return path, "shared"
    
    return None, None

# Usage
component_path, found_in = find_component_with_fallback(
    resolver, "builders", "advanced_processor"
)

if component_path:
    print(f"Found component in {found_in}: {component_path}")
else:
    print("Component not found anywhere")
```

### Statistics and Monitoring

```python
# Comprehensive workspace monitoring
def monitor_workspace_health(resolver):
    stats = resolver.get_component_statistics()
    
    print("Workspace Health Report:")
    print(f"  Total workspaces: {stats['total_workspaces']}")
    print(f"  Total components: {stats['total_components']}")
    
    # Check for empty workspaces
    empty_workspaces = []
    for workspace_id, workspace_stats in stats['workspaces'].items():
        if workspace_stats['total_components'] == 0:
            empty_workspaces.append(workspace_id)
    
    if empty_workspaces:
        print(f"  Empty workspaces: {empty_workspaces}")
    
    # Check component distribution
    print("Component distribution:")
    for comp_type, total_count in stats['component_types'].items():
        if total_count > 0:
            print(f"  {comp_type}: {total_count}")
            
            # Show per-workspace breakdown
            for workspace_id, workspace_stats in stats['workspaces'].items():
                count = workspace_stats['component_types'].get(comp_type, 0)
                if count > 0:
                    print(f"    {workspace_id}: {count}")
    
    return stats
```

## Related Documentation

- [Flexible File Resolver](../../validation/alignment/file_resolver.md) - Base file resolver functionality
- [Workspace Module Loader](workspace_module_loader.md) - Module loading for workspace components
- [Workspace Manager](workspace_manager.md) - Workspace management and coordination
- [Cross Workspace Validator](cross_workspace_validator.md) - Cross-workspace validation capabilities
- [Workspace Discovery Manager](../core/discovery.md) - Component discovery and inventory management
