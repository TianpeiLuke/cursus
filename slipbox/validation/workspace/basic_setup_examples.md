---
tags:
  - validation
  - workspace
  - examples
  - setup
  - basic
keywords:
  - workspace setup
  - basic configuration
  - workspace structure
  - initialization
topics:
  - basic workspace setup
  - workspace initialization
  - directory structure
language: python
date of note: 2025-08-28
---

# Basic Workspace Setup Examples

This document provides basic setup examples for the Cursus workspace validation system, covering workspace structure, initialization, and basic configuration.

## Workspace Directory Structure

### Expected Structure

Based on the real Cursus codebase, here's the expected workspace structure:

```
developer_workspaces/
├── developers/
│   ├── developer_1/
│   │   └── src/cursus_dev/steps/
│   │       ├── __init__.py
│   │       ├── builders/
│   │       │   ├── __init__.py
│   │       │   ├── builder_custom_feature_engineering_step.py
│   │       │   ├── builder_neural_network_training_step.py
│   │       │   ├── builder_data_validation_step.py
│   │       │   └── s3_utils.py
│   │       ├── configs/
│   │       │   ├── __init__.py
│   │       │   ├── config_custom_feature_engineering_step.py
│   │       │   ├── config_neural_network_training_step.py
│   │       │   ├── config_data_validation_step.py
│   │       │   └── utils.py
│   │       ├── contracts/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering_contract.py
│   │       │   ├── neural_network_training_contract.py
│   │       │   ├── data_validation_contract.py
│   │       │   └── contract_validator.py
│   │       ├── scripts/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering.py
│   │       │   ├── neural_network_training.py
│   │       │   └── data_validation.py
│   │       ├── specs/
│   │       │   ├── __init__.py
│   │       │   ├── custom_feature_engineering_spec.py
│   │       │   ├── neural_network_training_spec.py
│   │       │   ├── data_validation_training_spec.py
│   │       │   └── data_validation_testing_spec.py
│   │       ├── hyperparams/
│   │       │   ├── __init__.py
│   │       │   ├── hyperparameters_neural_network.py
│   │       │   └── hyperparameters_custom_model.py
│   │       └── registry/
│   │           ├── __init__.py
│   │           ├── builder_registry.py
│   │           ├── exceptions.py
│   │           ├── step_names.py
│   │           └── step_type_test_variants.py
│   └── developer_2/
│       └── src/cursus_dev/steps/
│           ├── __init__.py
│           ├── builders/
│           ├── configs/
│           ├── contracts/
│           ├── scripts/
│           ├── specs/
│           ├── hyperparams/
│           └── registry/
└── shared/
    └── src/cursus_dev/steps/
        ├── __init__.py
        ├── builders/
        ├── configs/
        ├── contracts/
        ├── scripts/
        ├── specs/
        ├── hyperparams/
        └── registry/
```

### Key Structure Notes

- **builders/**: Contains step builder classes following the pattern `builder_[step_name]_step.py`
- **configs/**: Contains configuration classes following the pattern `config_[step_name]_step.py`
- **contracts/**: Contains script contracts following the pattern `[step_name]_contract.py`
- **scripts/**: Contains actual execution scripts following the pattern `[step_name].py`
- **specs/**: Contains step specifications with variant support
- **hyperparams/**: Contains hyperparameter classes following the pattern `hyperparameters_[model_type].py`
- **registry/**: Contains registration and metadata utilities
- Each directory includes `__init__.py` for proper Python module structure
- Utility files provide shared functionality
- Specs support multiple variants per step type (training, testing, validation, calibration)

## Basic Import and Setup

### Simple Initialization

```python
from cursus.validation.workspace import (
    WorkspaceManager,
    DeveloperWorkspaceFileResolver,
    WorkspaceModuleLoader,
    WorkspaceConfig
)

# Initialize workspace manager
workspace_root = "/path/to/developer_workspaces"
manager = WorkspaceManager(workspace_root)

# Discover available workspaces
workspace_info = manager.discover_workspaces()
print(f"Found {workspace_info.total_developers} developer workspaces")
print(f"Available developers: {[dev.developer_id for dev in workspace_info.developers]}")
```

### Workspace Discovery

```python
# Create workspace manager
manager = WorkspaceManager("/path/to/developer_workspaces")

# Discover workspaces
workspace_info = manager.discover_workspaces()
print(f"Workspace root: {workspace_info.workspace_root}")
print(f"Has shared workspace: {workspace_info.has_shared}")
print(f"Total developers: {workspace_info.total_developers}")
print(f"Total modules: {workspace_info.total_modules}")

# List developer details
for dev in workspace_info.developers:
    print(f"Developer: {dev.developer_id}")
    print(f"  Module count: {dev.module_count}")
    print(f"  Has builders: {dev.has_builders}")
    print(f"  Has contracts: {dev.has_contracts}")
    print(f"  Has specs: {dev.has_specs}")
    print(f"  Has scripts: {dev.has_scripts}")
    print(f"  Has configs: {dev.has_configs}")
```

### Workspace Validation

```python
# Validate workspace structure
manager = WorkspaceManager("/path/to/developer_workspaces")

# Basic validation
is_valid, issues = manager.validate_workspace_structure()
if is_valid:
    print("Workspace structure is valid")
else:
    print("Workspace structure issues:")
    for issue in issues:
        print(f"  - {issue}")

# Strict validation
is_valid_strict, issues_strict = manager.validate_workspace_structure(strict=True)
if not is_valid_strict:
    print("Strict validation issues:")
    for issue in issues_strict:
        print(f"  - {issue}")
```

## Creating New Workspaces

### Create Developer Workspace

```python
# Create new developer workspace
manager = WorkspaceManager()

# Create workspace with full directory structure
new_workspace = manager.create_developer_workspace(
    "new_developer",
    workspace_root="/path/to/developer_workspaces",
    create_structure=True
)
print(f"Created workspace at: {new_workspace}")
```

### Create Shared Workspace

```python
# Create shared workspace
shared_workspace = manager.create_shared_workspace(
    workspace_root="/path/to/developer_workspaces",
    create_structure=True
)
print(f"Created shared workspace at: {shared_workspace}")
```

### Batch Workspace Creation

```python
# Create multiple developer workspaces
developers = ["team_a_dev1", "team_a_dev2", "team_b_dev1"]
workspace_root = "/path/to/developer_workspaces"

manager = WorkspaceManager()

for developer_id in developers:
    try:
        workspace_path = manager.create_developer_workspace(
            developer_id,
            workspace_root=workspace_root,
            create_structure=True
        )
        print(f"✓ Created workspace for {developer_id}: {workspace_path}")
    except ValueError as e:
        print(f"✗ Failed to create workspace for {developer_id}: {e}")
```

## Basic Configuration

### Simple Configuration

```python
# Create basic workspace configuration
config = WorkspaceConfig(
    workspace_root="/path/to/developer_workspaces",
    developer_id="developer_1",
    enable_shared_fallback=True,
    cache_modules=True
)

# Initialize manager with configuration
manager = WorkspaceManager()
manager.config = config

print(f"Configured for developer: {config.developer_id}")
print(f"Workspace root: {config.workspace_root}")
print(f"Shared fallback enabled: {config.enable_shared_fallback}")
```

### Configuration Templates

```python
from cursus.validation.workspace import get_config_template

# Get basic configuration template
basic_config = get_config_template("basic")
basic_config["workspace_root"] = "/my/workspace/path"
basic_config["developer_id"] = "my_developer"

print("Basic config template:")
for key, value in basic_config.items():
    print(f"  {key}: {value}")

# Get multi-developer configuration template
multi_dev_config = get_config_template("multi_developer")
multi_dev_config["workspace_root"] = "/my/workspace/path"

print("\nMulti-developer config template:")
for key, value in multi_dev_config.items():
    print(f"  {key}: {value}")
```

## Workspace Summary

### Get Comprehensive Summary

```python
# Get comprehensive workspace summary
manager = WorkspaceManager("/path/to/developer_workspaces")
summary = manager.get_workspace_summary()

print("Workspace Summary:")
print("=" * 30)
print(f"Root: {summary['workspace_root']}")
print(f"Has shared: {summary['has_shared']}")
print(f"Total developers: {summary['total_developers']}")
print(f"Total modules: {summary['total_modules']}")

print("\nDeveloper Details:")
for dev in summary['developers']:
    print(f"  {dev['developer_id']}:")
    print(f"    Modules: {dev['module_count']}")
    print(f"    Has builders: {dev['has_builders']}")
    print(f"    Has contracts: {dev['has_contracts']}")
    print(f"    Has specs: {dev['has_specs']}")
    print(f"    Has scripts: {dev['has_scripts']}")
    print(f"    Has configs: {dev['has_configs']}")
```

### Quick Health Check

```python
def quick_workspace_health_check(workspace_root):
    """Perform a quick health check on workspace."""
    
    try:
        manager = WorkspaceManager(workspace_root)
        
        # Basic validation
        is_valid, issues = manager.validate_workspace_structure()
        
        # Discovery
        workspace_info = manager.discover_workspaces()
        
        print(f"Workspace Health Check:")
        print(f"  Structure valid: {'✓' if is_valid else '✗'}")
        print(f"  Developers found: {workspace_info.total_developers}")
        print(f"  Has shared workspace: {'✓' if workspace_info.has_shared else '✗'}")
        print(f"  Total modules: {workspace_info.total_modules}")
        
        if issues:
            print(f"  Issues found: {len(issues)}")
            for issue in issues[:3]:  # Show first 3 issues
                print(f"    - {issue}")
            if len(issues) > 3:
                print(f"    ... and {len(issues) - 3} more")
        
        return is_valid and workspace_info.total_developers > 0
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

# Usage
is_healthy = quick_workspace_health_check("/path/to/developer_workspaces")
print(f"Overall health: {'✓ HEALTHY' if is_healthy else '✗ NEEDS ATTENTION'}")
```

## Best Practices for Setup

### 1. Workspace Organization

```python
# Good: Organize workspaces by developer/team
workspace_structure = {
    "developers": {
        "team_a_dev1": "Individual developer workspace",
        "team_a_dev2": "Individual developer workspace", 
        "team_b_dev1": "Individual developer workspace"
    },
    "shared": "Common components and utilities"
}

# Good: Use consistent naming conventions
step_naming = {
    "builder": "my_custom_step_builder.py",
    "contract": "my_custom_step_contract.py",
    "spec": "my_custom_step_spec.json",
    "script": "my_custom_step_script.py"
}
```

### 2. Error Handling During Setup

```python
def safe_workspace_setup(workspace_root, developers):
    """Safely set up workspace with error handling."""
    
    try:
        manager = WorkspaceManager()
        
        # Create shared workspace first
        try:
            shared_path = manager.create_shared_workspace(
                workspace_root=workspace_root,
                create_structure=True
            )
            print(f"✓ Created shared workspace: {shared_path}")
        except ValueError:
            print("ℹ Shared workspace already exists")
        
        # Create developer workspaces
        created_count = 0
        for developer_id in developers:
            try:
                dev_path = manager.create_developer_workspace(
                    developer_id,
                    workspace_root=workspace_root,
                    create_structure=True
                )
                print(f"✓ Created workspace for {developer_id}")
                created_count += 1
            except ValueError:
                print(f"ℹ Workspace for {developer_id} already exists")
            except Exception as e:
                print(f"✗ Failed to create workspace for {developer_id}: {e}")
        
        print(f"Setup complete: {created_count}/{len(developers)} new workspaces created")
        return True
        
    except Exception as e:
        print(f"Setup failed: {e}")
        return False

# Usage
developers = ["developer_1", "developer_2", "developer_3"]
success = safe_workspace_setup("/path/to/workspaces", developers)
```

### 3. Configuration Management

```python
# Good: Use configuration files for workspace settings
def create_standard_config(workspace_root, developer_id):
    """Create standard workspace configuration."""
    
    config = WorkspaceConfig(
        workspace_root=workspace_root,
        developer_id=developer_id,
        enable_shared_fallback=True,
        cache_modules=True,
        auto_create_structure=False,
        validation_settings={
            "strict_validation": False,
            "require_all_module_types": False,
            "validate_imports": True
        }
    )
    
    return config

# Save configuration for reuse
config = create_standard_config("/path/to/workspaces", "developer_1")
manager = WorkspaceManager()
manager.save_config("workspace.json", config)
print("Standard configuration saved")
```

This guide covers the essential setup procedures for the Cursus workspace validation system, providing a solid foundation for multi-developer workspace environments.
