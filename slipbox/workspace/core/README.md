---
tags:
  - code
  - workspace
  - core
  - management
  - architecture
keywords:
  - WorkspaceManager
  - WorkspaceContext
  - WorkspaceConfig
  - workspace lifecycle
  - workspace isolation
  - workspace discovery
  - workspace integration
topics:
  - workspace core
  - workspace management
  - workspace architecture
  - functional separation
language: python
date of note: 2024-12-07
---

# Workspace Core

Core workspace management functionality with centralized coordination through specialized functional managers, providing the foundation for all workspace operations in the Cursus system.

## Overview

The Workspace Core module implements a consolidated workspace management architecture with functional separation through specialized managers. It provides centralized coordination of workspace operations while maintaining clear separation of concerns through dedicated managers for lifecycle, isolation, discovery, and integration operations.

The module consolidates functionality previously distributed across multiple systems, providing a unified interface for workspace management while maintaining backward compatibility. It supports comprehensive workspace lifecycle management, cross-workspace component discovery, workspace isolation enforcement, and integration staging coordination.

## Classes and Methods

### Core Management Classes
- [`WorkspaceManager`](#workspacemanager) - Centralized workspace management with functional separation
- [`WorkspaceContext`](#workspacecontext) - Context information for workspace tracking
- [`WorkspaceConfig`](#workspaceconfig) - Configuration for workspace management operations

### Specialized Manager Classes
- [`WorkspaceLifecycleManager`](#workspacelifecyclemanager) - Workspace creation, setup, and teardown operations
- [`WorkspaceIsolationManager`](#workspaceisolationmanager) - Workspace isolation and sandboxing utilities
- [`WorkspaceDiscoveryManager`](#workspacediscoverymanager) - Cross-workspace component discovery and resolution
- [`WorkspaceIntegrationManager`](#workspaceintegrationmanager) - Integration staging coordination and management

## API Reference

### WorkspaceManager

_class_ cursus.workspace.core.manager.WorkspaceManager(_workspace_root=None_, _config_file=None_, _auto_discover=True_)

Centralized workspace management with functional separation through specialized managers, providing unified interface for workspace lifecycle, isolation, discovery, and integration operations.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory for workspaces
- **config_file** (_Optional[Union[str, Path]]_) – Path to workspace configuration file
- **auto_discover** (_bool_) – Whether to automatically discover workspaces on initialization

```python
from cursus.workspace.core.manager import WorkspaceManager

# Initialize with default settings
manager = WorkspaceManager()

# Initialize with custom workspace root
manager = WorkspaceManager("/custom/workspace/root")

# Initialize with configuration file
manager = WorkspaceManager(
    workspace_root="/workspaces",
    config_file="/config/workspace.yaml",
    auto_discover=True
)
```

#### create_workspace

create_workspace(_developer_id_, _workspace_type="developer"_, _template=None_, _**kwargs_)

Create a new workspace with specified type and optional template using the lifecycle manager.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for the workspace
- **workspace_type** (_str_) – Type of workspace ("developer", "shared", "test")
- **template** (_str_) – Optional template to use for workspace creation
- **kwargs** – Additional arguments passed to lifecycle manager

**Returns:**
- **WorkspaceContext** – Context information for the created workspace

```python
# Create basic developer workspace
context = manager.create_workspace("alice")

# Create ML workspace with template
context = manager.create_workspace(
    "bob", 
    workspace_type="developer",
    template="ml_pipeline",
    create_structure=True
)

print(f"Created workspace: {context.workspace_id}")
print(f"Location: {context.workspace_path}")
```

#### configure_workspace

configure_workspace(_workspace_id_, _config_)

Configure an existing workspace with new settings and update workspace context.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to configure
- **config** (_Dict[str, Any]_) – Configuration dictionary with new settings

**Returns:**
- **WorkspaceContext** – Updated workspace context with new configuration

```python
# Configure workspace settings
config = {
    "enable_gpu": True,
    "memory_limit": "16GB",
    "python_version": "3.9"
}

updated_context = manager.configure_workspace("alice_workspace", config)
print(f"Updated workspace: {updated_context.workspace_id}")
```

#### discover_components

discover_components(_workspace_ids=None_, _developer_id=None_)

Discover components across workspaces using the discovery manager with optional filtering.

**Parameters:**
- **workspace_ids** (_Optional[List[str]]_) – Optional list of workspace IDs to search
- **developer_id** (_Optional[str]_) – Optional specific developer ID to search

**Returns:**
- **Dict[str, Any]** – Dictionary containing discovered components organized by type

```python
# Discover all components
all_components = manager.discover_components()
print(f"Found component types: {list(all_components.keys())}")

# Discover components for specific developer
alice_components = manager.discover_components(developer_id="alice")
print(f"Alice's components: {alice_components}")

# Discover components in specific workspaces
workspace_components = manager.discover_components(
    workspace_ids=["alice_workspace", "bob_workspace"]
)
```

#### resolve_cross_workspace_dependencies

resolve_cross_workspace_dependencies(_pipeline_definition_)

Resolve dependencies across workspace boundaries using the discovery manager.

**Parameters:**
- **pipeline_definition** (_Dict[str, Any]_) – Pipeline definition with cross-workspace dependencies

**Returns:**
- **Dict[str, Any]** – Resolved dependency information with workspace mappings

```python
# Define pipeline with cross-workspace dependencies
pipeline_def = {
    "steps": [
        {
            "name": "data_preprocessing",
            "workspace": "data_team",
            "outputs": ["processed_data"]
        },
        {
            "name": "model_training", 
            "workspace": "ml_team",
            "inputs": ["processed_data"],
            "depends_on": ["data_preprocessing"]
        }
    ]
}

# Resolve dependencies
resolved = manager.resolve_cross_workspace_dependencies(pipeline_def)
print(f"Resolved dependencies: {resolved}")
```

#### validate_workspace_structure

validate_workspace_structure(_workspace_root=None_, _strict=False_)

Validate workspace structure using the isolation manager for compliance checking.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory to validate
- **strict** (_bool_) – Whether to apply strict validation rules

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, list_of_issues)

```python
# Basic validation
is_valid, issues = manager.validate_workspace_structure()

if not is_valid:
    print("Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")

# Strict validation
is_valid, issues = manager.validate_workspace_structure(strict=True)
print(f"Strict validation: {'PASSED' if is_valid else 'FAILED'}")
```

#### stage_for_integration

stage_for_integration(_component_id_, _source_workspace_, _target_stage="integration"_)

Stage component for integration using the integration manager.

**Parameters:**
- **component_id** (_str_) – Component identifier to stage
- **source_workspace** (_str_) – Source workspace identifier
- **target_stage** (_str_) – Target staging area (default "integration")

**Returns:**
- **Dict[str, Any]** – Staging result information with status and details

```python
# Stage component for integration
result = manager.stage_for_integration(
    component_id="ml_model_v2",
    source_workspace="alice_workspace",
    target_stage="staging"
)

if result.get("success"):
    print(f"Component staged successfully: {result}")
else:
    print(f"Staging failed: {result.get('error')}")
```

#### get_workspace_summary

get_workspace_summary()

Get comprehensive summary of workspace information including all manager statistics.

**Returns:**
- **Dict[str, Any]** – Comprehensive workspace summary with manager details

```python
summary = manager.get_workspace_summary()

print(f"Workspace root: {summary['workspace_root']}")
print(f"Active workspaces: {summary['active_workspaces']}")
print(f"Discovery summary: {summary['discovery_summary']}")
print(f"Validation summary: {summary['validation_summary']}")
print(f"Integration summary: {summary['integration_summary']}")
```

### WorkspaceContext

_class_ cursus.workspace.core.manager.WorkspaceContext(_workspace_id_, _workspace_path_, _developer_id=None_, _workspace_type="developer"_, _created_at=datetime.now()_, _last_accessed=datetime.now()_, _status="active"_, _metadata={}_ )

Context information for workspace tracking with validation and metadata management.

**Parameters:**
- **workspace_id** (_str_) – Unique workspace identifier
- **workspace_path** (_str_) – Path to the workspace directory
- **developer_id** (_Optional[str]_) – Developer identifier (optional)
- **workspace_type** (_str_) – Type of workspace (default "developer")
- **created_at** (_datetime_) – Workspace creation timestamp
- **last_accessed** (_datetime_) – Last access timestamp
- **status** (_str_) – Workspace status (default "active")
- **metadata** (_Dict[str, Any]_) – Additional workspace metadata

```python
from cursus.workspace.core.manager import WorkspaceContext
from datetime import datetime

# Create workspace context
context = WorkspaceContext(
    workspace_id="alice_ml_workspace",
    workspace_path="/workspaces/alice",
    developer_id="alice",
    workspace_type="developer",
    metadata={"template": "ml_pipeline", "gpu_enabled": True}
)

print(f"Workspace: {context.workspace_id}")
print(f"Developer: {context.developer_id}")
print(f"Type: {context.workspace_type}")
print(f"Status: {context.status}")
```

### WorkspaceConfig

_class_ cursus.workspace.core.manager.WorkspaceConfig(_workspace_root_, _developer_id=None_, _enable_shared_fallback=True_, _cache_modules=True_, _auto_create_structure=False_, _validation_settings={}_, _isolation_settings={}_, _integration_settings={}_ )

Configuration for workspace management with validation and settings for specialized managers.

**Parameters:**
- **workspace_root** (_str_) – Root directory for workspace operations
- **developer_id** (_Optional[str]_) – Default developer identifier
- **enable_shared_fallback** (_bool_) – Whether to enable shared workspace fallback
- **cache_modules** (_bool_) – Whether to cache loaded modules
- **auto_create_structure** (_bool_) – Whether to automatically create directory structure
- **validation_settings** (_Dict[str, Any]_) – Settings for validation manager
- **isolation_settings** (_Dict[str, Any]_) – Settings for isolation manager
- **integration_settings** (_Dict[str, Any]_) – Settings for integration manager

```python
from cursus.workspace.core.manager import WorkspaceConfig

# Create workspace configuration
config = WorkspaceConfig(
    workspace_root="/workspaces",
    developer_id="default_dev",
    enable_shared_fallback=True,
    cache_modules=True,
    validation_settings={
        "strict_mode": False,
        "check_isolation": True
    },
    isolation_settings={
        "enforce_boundaries": True,
        "allow_shared_imports": True
    },
    integration_settings={
        "staging_area": "/staging",
        "auto_promote": False
    }
)

# Use with workspace manager
manager = WorkspaceManager()
manager.config = config
```

### WorkspaceLifecycleManager

_class_ cursus.workspace.core.lifecycle.WorkspaceLifecycleManager(_workspace_manager_)

Specialized manager for workspace creation, setup, configuration, and teardown operations.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent workspace manager instance

```python
# Access through workspace manager
lifecycle_manager = manager.lifecycle_manager

# Create workspace through lifecycle manager
context = lifecycle_manager.create_workspace(
    developer_id="charlie",
    workspace_type="test",
    template="basic"
)
```

#### create_workspace

create_workspace(_developer_id_, _workspace_type="developer"_, _template=None_, _**kwargs_)

Create a new workspace with directory structure and initial configuration.

**Parameters:**
- **developer_id** (_str_) – Developer identifier
- **workspace_type** (_str_) – Type of workspace to create
- **template** (_Optional[str]_) – Template to apply during creation
- **kwargs** – Additional creation parameters

**Returns:**
- **WorkspaceContext** – Context for the created workspace

```python
# Create workspace with template
context = lifecycle_manager.create_workspace(
    developer_id="data_scientist",
    workspace_type="developer", 
    template="ml_pipeline",
    create_structure=True,
    enable_gpu=True
)
```

#### cleanup_inactive_workspaces

cleanup_inactive_workspaces(_inactive_threshold=timedelta(days=30)_)

Clean up workspaces that have been inactive beyond the threshold.

**Parameters:**
- **inactive_threshold** (_timedelta_) – Threshold for considering workspace inactive

**Returns:**
- **Dict[str, Any]** – Cleanup results with cleaned workspace list

```python
from datetime import timedelta

# Clean up workspaces inactive for 60 days
cleanup_result = lifecycle_manager.cleanup_inactive_workspaces(
    inactive_threshold=timedelta(days=60)
)

print(f"Cleaned workspaces: {cleanup_result['cleaned_up']}")
```

### WorkspaceIsolationManager

_class_ cursus.workspace.core.isolation.WorkspaceIsolationManager(_workspace_manager_)

Specialized manager for workspace isolation enforcement and sandboxing utilities.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent workspace manager instance

```python
# Access through workspace manager
isolation_manager = manager.isolation_manager

# Validate workspace isolation
is_valid, issues = isolation_manager.validate_workspace_structure()
```

#### validate_workspace_structure

validate_workspace_structure(_workspace_root=None_, _strict=False_)

Validate workspace structure for isolation compliance and best practices.

**Parameters:**
- **workspace_root** (_Optional[Union[str, Path]]_) – Root directory to validate
- **strict** (_bool_) – Whether to apply strict validation rules

**Returns:**
- **Tuple[bool, List[str]]** – Validation result and list of issues

```python
# Validate with strict rules
is_valid, issues = isolation_manager.validate_workspace_structure(
    workspace_root=Path("/workspaces"),
    strict=True
)

if not is_valid:
    print("Isolation violations found:")
    for issue in issues:
        print(f"  - {issue}")
```

#### get_workspace_health

get_workspace_health(_workspace_id_)

Get health information for a specific workspace including isolation status.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to check

**Returns:**
- **Dict[str, Any]** – Health information with isolation details

```python
health = isolation_manager.get_workspace_health("alice_workspace")
print(f"Workspace health: {health['healthy']}")
print(f"Isolation status: {health['isolation_status']}")
```

### WorkspaceDiscoveryManager

_class_ cursus.workspace.core.discovery.WorkspaceDiscoveryManager(_workspace_manager_)

Specialized manager for cross-workspace component discovery and dependency resolution.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent workspace manager instance

```python
# Access through workspace manager
discovery_manager = manager.discovery_manager

# Discover components
components = discovery_manager.discover_components()
```

#### discover_workspaces

discover_workspaces(_workspace_root_)

Discover and analyze workspace structure in the specified root directory.

**Parameters:**
- **workspace_root** (_Union[str, Path]_) – Root directory to discover workspaces in

**Returns:**
- **Dict[str, Any]** – Discovery results with workspace information

```python
# Discover all workspaces
discovery_result = discovery_manager.discover_workspaces("/workspaces")

print(f"Found {len(discovery_result['workspaces'])} workspaces")
for workspace in discovery_result['workspaces']:
    print(f"  - {workspace['workspace_id']}: {workspace['developer_id']}")
```

#### list_available_developers

list_available_developers()

Get list of available developer IDs from discovered workspaces.

**Returns:**
- **List[str]** – List of developer IDs found in workspaces

```python
developers = discovery_manager.list_available_developers()
print(f"Available developers: {', '.join(developers)}")
```

### WorkspaceIntegrationManager

_class_ cursus.workspace.core.integration.WorkspaceIntegrationManager(_workspace_manager_)

Specialized manager for integration staging coordination and artifact promotion.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent workspace manager instance

```python
# Access through workspace manager
integration_manager = manager.integration_manager

# Stage component for integration
result = integration_manager.stage_for_integration("component_id", "workspace_id")
```

#### promote_artifacts

promote_artifacts(_workspace_path_, _target_environment_)

Promote artifacts from workspace to target environment.

**Parameters:**
- **workspace_path** (_str_) – Path to source workspace
- **target_environment** (_str_) – Target environment for promotion

**Returns:**
- **List[str]** – List of promoted artifact names

```python
# Promote artifacts to staging
artifacts = integration_manager.promote_artifacts(
    "/workspaces/alice",
    "staging"
)

print(f"Promoted artifacts: {artifacts}")
```

#### validate_integration_readiness

validate_integration_readiness(_staged_components_)

Validate integration readiness for staged components.

**Parameters:**
- **staged_components** (_List[str]_) – List of staged component identifiers

**Returns:**
- **Dict[str, Any]** – Integration readiness validation results

```python
# Validate readiness for integration
readiness = integration_manager.validate_integration_readiness([
    "component_1", "component_2"
])

if readiness['ready']:
    print("Components ready for integration")
else:
    print(f"Integration issues: {readiness['issues']}")
```

## Usage Examples

### Complete Workspace Management Workflow

```python
from cursus.workspace.core.manager import WorkspaceManager, WorkspaceConfig
from pathlib import Path

# Initialize workspace manager with configuration
config = WorkspaceConfig(
    workspace_root="/development/workspaces",
    enable_shared_fallback=True,
    cache_modules=True,
    validation_settings={"strict_mode": True},
    isolation_settings={"enforce_boundaries": True}
)

manager = WorkspaceManager(
    workspace_root="/development/workspaces",
    auto_discover=True
)
manager.config = config

# Create new developer workspace
print("Creating workspace...")
context = manager.create_workspace(
    developer_id="new_developer",
    workspace_type="developer",
    template="ml_pipeline"
)

print(f"✓ Created workspace: {context.workspace_id}")
print(f"  Location: {context.workspace_path}")
print(f"  Type: {context.workspace_type}")

# Validate workspace structure
print("\nValidating workspace...")
is_valid, issues = manager.validate_workspace_structure(strict=True)

if is_valid:
    print("✓ Workspace validation passed")
else:
    print("⚠ Validation issues found:")
    for issue in issues:
        print(f"  - {issue}")

# Discover components
print("\nDiscovering components...")
components = manager.discover_components(developer_id="new_developer")
print(f"Found component types: {list(components.keys())}")

# Get workspace summary
print("\nWorkspace summary:")
summary = manager.get_workspace_summary()
print(f"  Active workspaces: {summary['active_workspaces']}")
print(f"  Workspace root: {summary['workspace_root']}")
```

### Cross-Workspace Dependency Resolution

```python
# Define complex pipeline with cross-workspace dependencies
pipeline_definition = {
    "name": "ml_training_pipeline",
    "steps": [
        {
            "name": "data_ingestion",
            "workspace": "data_team",
            "outputs": ["raw_data", "metadata"]
        },
        {
            "name": "data_preprocessing", 
            "workspace": "data_team",
            "inputs": ["raw_data"],
            "outputs": ["processed_data", "feature_stats"],
            "depends_on": ["data_ingestion"]
        },
        {
            "name": "feature_engineering",
            "workspace": "ml_team", 
            "inputs": ["processed_data", "feature_stats"],
            "outputs": ["features", "feature_importance"],
            "depends_on": ["data_preprocessing"]
        },
        {
            "name": "model_training",
            "workspace": "ml_team",
            "inputs": ["features"],
            "outputs": ["trained_model", "metrics"],
            "depends_on": ["feature_engineering"]
        },
        {
            "name": "model_evaluation",
            "workspace": "validation_team",
            "inputs": ["trained_model", "features"],
            "outputs": ["evaluation_report"],
            "depends_on": ["model_training"]
        }
    ]
}

# Resolve cross-workspace dependencies
print("Resolving cross-workspace dependencies...")
resolved = manager.resolve_cross_workspace_dependencies(pipeline_definition)

print("Dependency resolution results:")
for step_name, resolution in resolved.items():
    print(f"  {step_name}:")
    print(f"    Workspace: {resolution['workspace']}")
    print(f"    Dependencies resolved: {resolution['dependencies_resolved']}")
    if resolution.get('issues'):
        print(f"    Issues: {resolution['issues']}")
```

### Integration Staging Workflow

```python
# Stage components for integration
components_to_stage = [
    {"id": "data_processor_v2", "workspace": "data_team"},
    {"id": "ml_model_v3", "workspace": "ml_team"},
    {"id": "validator_v1", "workspace": "validation_team"}
]

staged_components = []

print("Staging components for integration...")
for component in components_to_stage:
    result = manager.stage_for_integration(
        component_id=component["id"],
        source_workspace=component["workspace"],
        target_stage="integration"
    )
    
    if result.get("success"):
        print(f"✓ Staged {component['id']} from {component['workspace']}")
        staged_components.append(component["id"])
    else:
        print(f"✗ Failed to stage {component['id']}: {result.get('error')}")

# Validate integration readiness
if staged_components:
    print(f"\nValidating integration readiness for {len(staged_components)} components...")
    readiness = manager.integration_manager.validate_integration_readiness(staged_components)
    
    if readiness.get("ready"):
        print("✓ All components ready for integration")
        
        # Promote to staging environment
        for component_id in staged_components:
            # Find source workspace for component
            source_workspace = next(
                c["workspace"] for c in components_to_stage 
                if c["id"] == component_id
            )
            
            artifacts = manager.integration_manager.promote_artifacts(
                f"/workspaces/{source_workspace}",
                "staging"
            )
            print(f"✓ Promoted {len(artifacts)} artifacts from {source_workspace}")
    else:
        print("⚠ Integration readiness issues:")
        for issue in readiness.get("issues", []):
            print(f"  - {issue}")
```

### Workspace Health Monitoring

```python
# Get comprehensive workspace health information
print("Checking workspace health...")

# Get overall system summary
summary = manager.get_workspace_summary()
print(f"System overview:")
print(f"  Total workspaces: {summary['active_workspaces']}")

# Check individual workspace health
for workspace_id in manager.active_workspaces:
    health = manager.isolation_manager.get_workspace_health(workspace_id)
    
    status_icon = "✓" if health.get("healthy") else "⚠"
    print(f"  {status_icon} {workspace_id}: {health.get('status', 'unknown')}")
    
    if not health.get("healthy"):
        issues = health.get("issues", [])
        for issue in issues[:3]:  # Show first 3 issues
            print(f"    - {issue}")

# Cleanup inactive workspaces
print("\nChecking for inactive workspaces...")
from datetime import timedelta

cleanup_result = manager.lifecycle_manager.cleanup_inactive_workspaces(
    inactive_threshold=timedelta(days=30)
)

if cleanup_result.get("cleaned_up"):
    print(f"Cleaned up {len(cleanup_result['cleaned_up'])} inactive workspaces")
else:
    print("No inactive workspaces found")
```

## Architecture and Design

### Functional Separation

The workspace core implements functional separation through specialized managers:

- **Lifecycle Manager**: Handles workspace creation, configuration, and cleanup
- **Isolation Manager**: Enforces workspace boundaries and validates isolation
- **Discovery Manager**: Discovers components and resolves cross-workspace dependencies  
- **Integration Manager**: Manages staging and promotion of workspace artifacts

### Backward Compatibility

The consolidated architecture maintains full backward compatibility with existing workspace manager APIs while providing enhanced functionality through the new specialized managers.

### Configuration Management

Comprehensive configuration system supports settings for all specialized managers, enabling fine-tuned control over workspace behavior and policies.

## Integration Points

### High-Level API Integration
The core managers integrate with the high-level WorkspaceAPI to provide simplified access to complex workspace operations.

### Template System Integration
Lifecycle manager integrates with the template system for automated workspace creation with pre-configured structures.

### Validation Framework Integration
Isolation manager integrates with the validation framework for comprehensive workspace compliance checking.

## Related Documentation

- [Workspace API](../api.md) - High-level workspace API built on core managers
- [Workspace Templates](../templates.md) - Template system used by lifecycle manager
- [Workspace Validation](../validation/README.md) - Validation systems integrated with isolation manager
- [Main Workspace Documentation](../README.md) - Overview of complete workspace system
