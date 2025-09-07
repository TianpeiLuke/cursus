---
tags:
  - code
  - workspace
  - config
  - pydantic
  - models
keywords:
  - WorkspaceStepDefinition
  - WorkspacePipelineDefinition
  - Pydantic models
  - workspace configuration
  - pipeline definition
topics:
  - workspace management
  - configuration models
  - data validation
language: python
date of note: 2024-12-07
---

# Workspace Configuration Models

Workspace configuration models using Pydantic V2 for workspace step definitions and pipeline configurations.

## Overview

This module provides Pydantic V2 models for workspace step definitions and pipeline configurations, enabling workspace-aware pipeline assembly with comprehensive validation and dependency management. The models support Phase 2 optimization with consolidated workspace manager integration.

The configuration models provide structured data validation, dependency resolution, cross-workspace integration, and comprehensive validation using the consolidated workspace management system. They support JSON and YAML serialization for configuration persistence and sharing.

Key features include step definition validation, pipeline dependency management, circular dependency detection, cross-workspace dependency resolution, and integration with specialized workspace managers.

## Classes and Methods

### Classes
- [`WorkspaceStepDefinition`](#workspacestepdefinition) - Pydantic V2 model for workspace step definitions
- [`WorkspacePipelineDefinition`](#workspacepipelinedefinition) - Pydantic V2 model for workspace pipeline definition

### Methods
- [`to_config_instance`](#to_config_instance) - Convert step to BasePipelineConfig instance
- [`get_workspace_path`](#get_workspace_path) - Get path relative to workspace root
- [`validate_with_workspace_manager`](#validate_with_workspace_manager) - Enhanced validation using workspace manager
- [`resolve_dependencies`](#resolve_dependencies) - Enhanced dependency resolution
- [`validate_workspace_dependencies`](#validate_workspace_dependencies) - Validate workspace dependencies
- [`to_pipeline_config`](#to_pipeline_config) - Convert to standard pipeline configuration
- [`get_developers`](#get_developers) - Get list of unique developers
- [`get_steps_by_developer`](#get_steps_by_developer) - Get steps for specific developer
- [`get_step_by_name`](#get_step_by_name) - Get step by name

### Class Methods
- [`from_json_file`](#from_json_file) - Load configuration from JSON file
- [`from_yaml_file`](#from_yaml_file) - Load configuration from YAML file

## API Reference

### WorkspaceStepDefinition

_class_ cursus.workspace.core.config.WorkspaceStepDefinition(_step_name_, _developer_id_, _step_type_, _config_data_, _workspace_root_, _dependencies=[]_)

Pydantic V2 model for workspace step definitions with comprehensive validation.

**Parameters:**
- **step_name** (_str_) – Name of the step, must be non-empty string.
- **developer_id** (_str_) – Developer workspace identifier, must be non-empty string.
- **step_type** (_str_) – Type of the step (e.g., 'XGBoostTraining'), must be non-empty string.
- **config_data** (_Dict[str, Any]_) – Step configuration data dictionary.
- **workspace_root** (_str_) – Root path of the workspace, must be non-empty string.
- **dependencies** (_List[str]_) – List of step dependencies, defaults to empty list.

```python
from cursus.workspace.core.config import WorkspaceStepDefinition

# Create workspace step definition
step_def = WorkspaceStepDefinition(
    step_name="data_preprocessing",
    developer_id="alice",
    step_type="ProcessingStep",
    config_data={
        "instance_type": "ml.m5.large",
        "script_path": "preprocess.py"
    },
    workspace_root="/path/to/workspace",
    dependencies=["data_ingestion"]
)
```

#### to_config_instance

to_config_instance()

Convert workspace step definition to a BasePipelineConfig instance.

**Returns:**
- **BasePipelineConfig** – Configuration instance for the step.

```python
# Convert to config instance
config_instance = step_def.to_config_instance()
```

#### get_workspace_path

get_workspace_path(_relative_path=""_)

Get a path relative to the workspace root directory.

**Parameters:**
- **relative_path** (_str_) – Relative path to append to workspace root, defaults to empty string.

**Returns:**
- **str** – Full path combining workspace root and relative path.

```python
# Get workspace-relative paths
workspace_root = step_def.get_workspace_path()
script_path = step_def.get_workspace_path("scripts/preprocess.py")
```

#### validate_with_workspace_manager

validate_with_workspace_manager(_workspace_manager_)

Enhanced validation using consolidated workspace manager for Phase 2 optimization.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Consolidated WorkspaceManager instance.

**Returns:**
- **Dict[str, Any]** – Validation result dictionary with validation status, errors, warnings, and detailed validations.

```python
from cursus.workspace.core.manager import WorkspaceManager

# Validate with workspace manager
workspace_manager = WorkspaceManager("/path/to/workspace")
validation_result = step_def.validate_with_workspace_manager(workspace_manager)

if validation_result['valid']:
    print("Step definition is valid")
else:
    print("Validation errors:", validation_result['errors'])
```

#### resolve_dependencies

resolve_dependencies(_workspace_manager_)

Enhanced dependency resolution using discovery manager for Phase 2 optimization.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Consolidated WorkspaceManager instance.

**Returns:**
- **Dict[str, Any]** – Dependency resolution result with validation status and resolved dependencies.

```python
# Resolve step dependencies
resolution_result = step_def.resolve_dependencies(workspace_manager)

if resolution_result['valid']:
    print("Dependencies resolved successfully")
else:
    print("Resolution error:", resolution_result['error'])
```

### WorkspacePipelineDefinition

_class_ cursus.workspace.core.config.WorkspacePipelineDefinition(_pipeline_name_, _workspace_root_, _steps_, _global_config={}_ )

Pydantic V2 model for workspace pipeline definition with comprehensive validation and dependency management.

**Parameters:**
- **pipeline_name** (_str_) – Name of the pipeline, must be non-empty string.
- **workspace_root** (_str_) – Root path of the workspace, must be non-empty string.
- **steps** (_List[WorkspaceStepDefinition]_) – List of pipeline steps, cannot be empty.
- **global_config** (_Dict[str, Any]_) – Global pipeline configuration dictionary, defaults to empty dict.

```python
from cursus.workspace.core.config import WorkspacePipelineDefinition, WorkspaceStepDefinition

# Create pipeline definition
pipeline_def = WorkspacePipelineDefinition(
    pipeline_name="ml_training_pipeline",
    workspace_root="/path/to/workspace",
    steps=[
        WorkspaceStepDefinition(
            step_name="data_prep",
            developer_id="alice",
            step_type="ProcessingStep",
            config_data={...},
            workspace_root="/path/to/workspace"
        ),
        WorkspaceStepDefinition(
            step_name="training",
            developer_id="bob",
            step_type="TrainingStep",
            config_data={...},
            workspace_root="/path/to/workspace",
            dependencies=["data_prep"]
        )
    ],
    global_config={"region": "us-west-2"}
)
```

#### validate_workspace_dependencies

validate_workspace_dependencies()

Validate workspace dependencies and references with circular dependency detection.

**Returns:**
- **Dict[str, Any]** – Validation result with dependency graph, errors, and warnings.

```python
# Validate pipeline dependencies
validation = pipeline_def.validate_workspace_dependencies()

if validation['valid']:
    print("All dependencies are valid")
    print("Dependency graph:", validation['dependency_graph'])
else:
    print("Dependency errors:", validation['errors'])
```

#### to_pipeline_config

to_pipeline_config()

Convert workspace pipeline definition to standard pipeline configuration format.

**Returns:**
- **Dict[str, Any]** – Standard pipeline configuration dictionary.

```python
# Convert to standard config format
standard_config = pipeline_def.to_pipeline_config()
print("Pipeline config:", standard_config)
```

#### get_developers

get_developers()

Get list of unique developers involved in the pipeline.

**Returns:**
- **List[str]** – List of unique developer IDs.

```python
# Get all developers in pipeline
developers = pipeline_def.get_developers()
print("Pipeline developers:", developers)
```

#### get_steps_by_developer

get_steps_by_developer(_developer_id_)

Get all steps assigned to a specific developer.

**Parameters:**
- **developer_id** (_str_) – Developer identifier to filter steps.

**Returns:**
- **List[WorkspaceStepDefinition]** – List of steps for the specified developer.

```python
# Get steps for specific developer
alice_steps = pipeline_def.get_steps_by_developer("alice")
print(f"Alice has {len(alice_steps)} steps")
```

#### get_step_by_name

get_step_by_name(_step_name_)

Get a step definition by its name.

**Parameters:**
- **step_name** (_str_) – Name of the step to retrieve.

**Returns:**
- **Optional[WorkspaceStepDefinition]** – Step definition if found, None otherwise.

```python
# Get specific step
data_prep_step = pipeline_def.get_step_by_name("data_prep")
if data_prep_step:
    print("Found step:", data_prep_step.step_name)
```

#### validate_with_consolidated_managers

validate_with_consolidated_managers(_workspace_manager_)

Comprehensive validation using all consolidated managers for Phase 2 optimization.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Consolidated WorkspaceManager instance.

**Returns:**
- **Dict[str, Any]** – Comprehensive validation result with overall status, individual validations, and summary.

```python
# Comprehensive validation with all managers
validation = pipeline_def.validate_with_consolidated_managers(workspace_manager)

print("Overall valid:", validation['overall_valid'])
print("Validation summary:", validation['summary'])

for validation_type, result in validation['validations'].items():
    print(f"{validation_type}: {'PASS' if result.get('valid', True) else 'FAIL'}")
```

#### resolve_cross_workspace_dependencies

resolve_cross_workspace_dependencies(_workspace_manager_)

Enhanced cross-workspace dependency resolution for Phase 2 optimization.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Consolidated WorkspaceManager instance.

**Returns:**
- **Dict[str, Any]** – Cross-workspace dependency resolution result.

```python
# Resolve cross-workspace dependencies
resolution = pipeline_def.resolve_cross_workspace_dependencies(workspace_manager)

if resolution['valid']:
    print("Cross-workspace dependencies resolved")
else:
    print("Resolution error:", resolution['error'])
```

#### prepare_for_integration

prepare_for_integration(_workspace_manager_)

Prepare pipeline for integration staging using Phase 2 optimization.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Consolidated WorkspaceManager instance.

**Returns:**
- **Dict[str, Any]** – Integration preparation result with readiness status.

```python
# Prepare for integration
preparation = pipeline_def.prepare_for_integration(workspace_manager)

if preparation['ready']:
    print("Pipeline ready for integration")
else:
    print("Integration preparation error:", preparation['error'])
```

#### to_json_file

to_json_file(_file_path_, _indent=2_)

Save configuration to JSON file.

**Parameters:**
- **file_path** (_str_) – Path to save JSON file.
- **indent** (_int_) – JSON indentation level, defaults to 2.

```python
# Save to JSON file
pipeline_def.to_json_file("pipeline_config.json", indent=4)
```

#### to_yaml_file

to_yaml_file(_file_path_)

Save configuration to YAML file.

**Parameters:**
- **file_path** (_str_) – Path to save YAML file.

```python
# Save to YAML file
pipeline_def.to_yaml_file("pipeline_config.yaml")
```

### from_json_file

from_json_file(_file_path_)

Load workspace pipeline configuration from JSON file.

**Parameters:**
- **file_path** (_str_) – Path to JSON configuration file.

**Returns:**
- **WorkspacePipelineDefinition** – Loaded pipeline definition instance.

```python
# Load from JSON file
pipeline_def = WorkspacePipelineDefinition.from_json_file("pipeline_config.json")
print("Loaded pipeline:", pipeline_def.pipeline_name)
```

### from_yaml_file

from_yaml_file(_file_path_)

Load workspace pipeline configuration from YAML file.

**Parameters:**
- **file_path** (_str_) – Path to YAML configuration file.

**Returns:**
- **WorkspacePipelineDefinition** – Loaded pipeline definition instance.

```python
# Load from YAML file
pipeline_def = WorkspacePipelineDefinition.from_yaml_file("pipeline_config.yaml")
print("Loaded pipeline:", pipeline_def.pipeline_name)
```

## Related Documentation

- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Pipeline Assembler](assembler.md) - Pipeline assembly from workspace components
- [Base Pipeline Config](../../core/base.md) - Base configuration models
- [Workspace API](../api.md) - High-level workspace API interface
- [Workspace Utils](../utils.md) - Workspace utility functions and classes
