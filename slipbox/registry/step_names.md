---
tags:
  - code
  - registry
  - step_names
  - workspace_awareness
  - hybrid_registry
keywords:
  - step names registry
  - workspace context
  - step resolution
  - canonical names
  - hybrid backend
topics:
  - step names registry
  - workspace management
  - step resolution
language: python
date of note: 2024-12-07
---

# Step Names Registry

Enhanced step names registry with hybrid backend support that maintains 100% backward compatibility while adding workspace awareness and context management capabilities.

## Overview

The step names registry module provides a drop-in replacement for the original step_names.py that uses the hybrid registry backend transparently while maintaining all existing functions and variables. It serves as the single source of truth for step naming and configuration, ensuring consistency across the entire pipeline system.

The module has been enhanced with workspace-aware step resolution, hybrid registry backend support, context management for multi-developer workflows, and seamless workspace-aware step resolution. It provides both the traditional module-level variables (STEP_NAMES, CONFIG_STEP_REGISTRY, etc.) and new workspace-aware functions for advanced use cases.

## Classes and Methods

### Core Registry Variables
- [`STEP_NAMES`](#step_names) - Dynamic step names dictionary that respects workspace context
- [`CONFIG_STEP_REGISTRY`](#config_step_registry) - Mapping from config classes to step names
- [`BUILDER_STEP_NAMES`](#builder_step_names) - Mapping from step names to builder classes
- [`SPEC_STEP_TYPES`](#spec_step_types) - Mapping from step names to specification types

### Workspace Context Management
- [`set_workspace_context`](#set_workspace_context) - Set current workspace context for registry resolution
- [`get_workspace_context`](#get_workspace_context) - Get current workspace context
- [`clear_workspace_context`](#clear_workspace_context) - Clear current workspace context
- [`workspace_context`](#workspace_context) - Context manager for temporary workspace context

### Core Helper Functions
- [`get_config_class_name`](#get_config_class_name) - Get config class name with workspace context
- [`get_builder_step_name`](#get_builder_step_name) - Get builder step class name with workspace context
- [`get_spec_step_type`](#get_spec_step_type) - Get step_type value for StepSpecification with workspace context
- [`get_all_step_names`](#get_all_step_names) - Get all canonical step names with workspace context

### SageMaker Integration Functions
- [`get_sagemaker_step_type`](#get_sagemaker_step_type) - Get SageMaker step type with workspace context
- [`get_steps_by_sagemaker_type`](#get_steps_by_sagemaker_type) - Get steps by SageMaker type with workspace context
- [`get_canonical_name_from_file_name`](#get_canonical_name_from_file_name) - Enhanced file name resolution with workspace context

### Workspace Management Functions
- [`list_available_workspaces`](#list_available_workspaces) - List all available workspace contexts
- [`get_workspace_step_count`](#get_workspace_step_count) - Get number of steps available in a workspace
- [`has_workspace_conflicts`](#has_workspace_conflicts) - Check if there are step name conflicts between workspaces

## API Reference

### STEP_NAMES

_property_ STEP_NAMES

Dynamic STEP_NAMES dictionary that respects workspace context. This property returns the appropriate step names based on the current workspace context, providing seamless workspace-aware behavior.

**Returns:**
- **Dict[str, Dict[str, str]]** – Step names dictionary in original format, context-aware.

```python
from cursus.registry.step_names import STEP_NAMES, set_workspace_context

# Use core registry
print(f"Core steps: {len(STEP_NAMES)}")

# Switch to workspace context
set_workspace_context("developer_1")
print(f"Workspace steps: {len(STEP_NAMES)}")  # Now uses developer_1 context

# Access step information
xgb_info = STEP_NAMES["XGBoostTraining"]
print(f"XGBoost config: {xgb_info['config_class']}")
```

### set_workspace_context

set_workspace_context(_workspace_id_)

Set current workspace context for registry resolution with automatic variable refresh.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to set as current context.

```python
from cursus.registry.step_names import set_workspace_context, STEP_NAMES

# Set workspace context
set_workspace_context("developer_1")

# All module variables now use workspace context
step_names = STEP_NAMES  # Uses developer_1 context
print(f"Steps in workspace: {len(step_names)}")
```

### get_workspace_context

get_workspace_context()

Get current workspace context identifier.

**Returns:**
- **Optional[str]** – Current workspace identifier or None if no context set.

```python
from cursus.registry.step_names import get_workspace_context, set_workspace_context

# Check current context
current = get_workspace_context()
print(f"Current context: {current}")  # None initially

# Set and verify context
set_workspace_context("developer_1")
current = get_workspace_context()
print(f"New context: {current}")  # developer_1
```

### workspace_context

workspace_context(_workspace_id_)

Context manager for temporary workspace context with automatic restoration.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier for temporary context.

**Returns:**
- **ContextManager[None]** – Context manager for workspace operations.

```python
from cursus.registry.step_names import workspace_context, get_all_step_names

# Use temporary workspace context
with workspace_context("developer_1"):
    # Operations use developer_1 context
    workspace_steps = get_all_step_names()
    print(f"Workspace steps: {len(workspace_steps)}")

# Context automatically restored
print(f"Back to original context")
```

### get_config_class_name

get_config_class_name(_step_name_, _workspace_id=None_)

Get config class name with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Configuration class name for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry.step_names import get_config_class_name

# Get config class for core step
config_class = get_config_class_name("XGBoostTraining")
print(f"Config class: {config_class}")  # XGBoostTrainingConfig

# Get config class with workspace context
workspace_config = get_config_class_name("CustomStep", workspace_id="developer_1")
print(f"Workspace config: {workspace_config}")
```

### get_builder_step_name

get_builder_step_name(_step_name_, _workspace_id=None_)

Get builder step class name with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Builder class name for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry.step_names import get_builder_step_name

# Get builder class name
builder_name = get_builder_step_name("XGBoostTraining")
print(f"Builder class: {builder_name}")  # XGBoostTrainingStepBuilder

# Get builder with workspace context
workspace_builder = get_builder_step_name("CustomStep", workspace_id="developer_1")
print(f"Workspace builder: {workspace_builder}")
```

### get_spec_step_type

get_spec_step_type(_step_name_, _workspace_id=None_)

Get step_type value for StepSpecification with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Specification step type for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry.step_names import get_spec_step_type

# Get spec type for step
spec_type = get_spec_step_type("XGBoostTraining")
print(f"Spec type: {spec_type}")  # XGBoostTraining

# Get spec type with job type suffix
spec_with_job = get_spec_step_type_with_job_type("CradleDataLoading", "training")
print(f"Spec with job type: {spec_with_job}")  # CradleDataLoading_Training
```

### get_all_step_names

get_all_step_names(_workspace_id=None_)

Get all canonical step names with workspace context support.

**Parameters:**
- **workspace_id** (_Optional[str]_) – Optional workspace context for step resolution. Defaults to None.

**Returns:**
- **List[str]** – List of all canonical step names available in the context.

```python
from cursus.registry.step_names import get_all_step_names

# Get all core step names
core_steps = get_all_step_names()
print(f"Core steps: {len(core_steps)}")

# Get all steps in workspace context
workspace_steps = get_all_step_names(workspace_id="developer_1")
print(f"Workspace steps: {len(workspace_steps)}")
```

### get_sagemaker_step_type

get_sagemaker_step_type(_step_name_, _workspace_id=None_)

Get SageMaker step type with workspace context support.

**Parameters:**
- **step_name** (_str_) – Canonical step name to look up.
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – SageMaker step type for the step.

**Raises:**
- **ValueError** – If step name is not found in the registry.

```python
from cursus.registry.step_names import get_sagemaker_step_type

# Get SageMaker step type
sm_type = get_sagemaker_step_type("XGBoostTraining")
print(f"SageMaker type: {sm_type}")  # Training

# Get SageMaker type with workspace context
workspace_sm_type = get_sagemaker_step_type("CustomStep", workspace_id="developer_1")
print(f"Workspace SageMaker type: {workspace_sm_type}")
```

### get_canonical_name_from_file_name

get_canonical_name_from_file_name(_file_name_, _workspace_id=None_)

Enhanced file name resolution with workspace context awareness and intelligent pattern matching.

**Parameters:**
- **file_name** (_str_) – File-based name (e.g., "model_evaluation_xgb", "dummy_training").
- **workspace_id** (_Optional[str]_) – Optional workspace context for resolution. Defaults to None.

**Returns:**
- **str** – Canonical step name (e.g., "XGBoostModelEval", "DummyTraining").

**Raises:**
- **ValueError** – If file name cannot be mapped to a canonical name with detailed suggestions.

```python
from cursus.registry.step_names import get_canonical_name_from_file_name

# Convert file names to canonical names
test_files = [
    "model_evaluation_xgb",
    "dummy_training", 
    "cradle_data_loading_training",
    "tabular_preprocessing_validation"
]

for file_name in test_files:
    try:
        canonical = get_canonical_name_from_file_name(file_name)
        print(f"✓ {file_name} → {canonical}")
    except ValueError as e:
        print(f"✗ {file_name}: {e}")

# With workspace context
workspace_canonical = get_canonical_name_from_file_name(
    "custom_analysis_step", 
    workspace_id="developer_1"
)
print(f"Workspace canonical: {workspace_canonical}")
```

### list_available_workspaces

list_available_workspaces()

List all available workspace contexts discovered by the registry system.

**Returns:**
- **List[str]** – List of available workspace identifiers.

```python
from cursus.registry.step_names import list_available_workspaces

# Get all available workspaces
workspaces = list_available_workspaces()
print(f"Available workspaces: {workspaces}")

# Iterate through workspaces
for workspace_id in workspaces:
    step_count = get_workspace_step_count(workspace_id)
    print(f"  {workspace_id}: {step_count} steps")
```

### get_workspace_step_count

get_workspace_step_count(_workspace_id_)

Get number of steps available in a specific workspace.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to count steps for.

**Returns:**
- **int** – Number of steps available in the workspace.

```python
from cursus.registry.step_names import get_workspace_step_count

# Get step count for workspace
count = get_workspace_step_count("developer_1")
print(f"Developer 1 has {count} steps")

# Compare with core registry
core_count = len(get_all_step_names())
print(f"Core registry has {core_count} steps")
```

### has_workspace_conflicts

has_workspace_conflicts()

Check if there are any step name conflicts between workspaces.

**Returns:**
- **bool** – True if conflicts exist, False otherwise.

```python
from cursus.registry.step_names import has_workspace_conflicts

# Check for conflicts
conflicts_exist = has_workspace_conflicts()
if conflicts_exist:
    print("⚠ Step name conflicts detected between workspaces")
    # Handle conflicts appropriately
else:
    print("✓ No step name conflicts detected")
```

## Usage Examples

### Complete Workspace Management Workflow

```python
from cursus.registry.step_names import (
    set_workspace_context,
    get_workspace_context,
    workspace_context,
    get_all_step_names,
    list_available_workspaces,
    STEP_NAMES
)

# Check initial state
print(f"Initial context: {get_workspace_context()}")  # None
print(f"Core steps: {len(STEP_NAMES)}")

# List available workspaces
workspaces = list_available_workspaces()
print(f"Available workspaces: {workspaces}")

# Switch to workspace
if workspaces:
    workspace_id = workspaces[0]
    set_workspace_context(workspace_id)
    print(f"Switched to workspace: {workspace_id}")
    print(f"Workspace steps: {len(STEP_NAMES)}")

# Use temporary context
with workspace_context("developer_2"):
    temp_steps = get_all_step_names()
    print(f"Temporary context steps: {len(temp_steps)}")

# Context restored automatically
print(f"Context after with: {get_workspace_context()}")
```

### File Name Resolution with Error Handling

```python
from cursus.registry.step_names import get_canonical_name_from_file_name

# Test various file name patterns
test_files = [
    "model_evaluation_xgb",      # Should resolve to XGBoostModelEval
    "dummy_training",            # Should resolve to DummyTraining
    "invalid_file_name",         # Should fail with suggestions
    "cradle_data_loading_train", # Should resolve to CradleDataLoading
    "tabular_preprocess_val"     # Should resolve to TabularPreprocessing
]

for file_name in test_files:
    try:
        canonical = get_canonical_name_from_file_name(file_name)
        print(f"✓ {file_name} → {canonical}")
        
        # Verify the canonical name exists
        config_class = get_config_class_name(canonical)
        print(f"    Config: {config_class}")
        
    except ValueError as e:
        print(f"✗ {file_name}: {e}")
        # Error includes suggestions for resolution
```

### SageMaker Integration

```python
from cursus.registry.step_names import (
    get_sagemaker_step_type,
    get_steps_by_sagemaker_type,
    get_all_sagemaker_step_types
)

# Get all SageMaker step types
sm_types = get_all_sagemaker_step_types()
print(f"SageMaker step types: {sm_types}")

# Get steps by SageMaker type
for sm_type in sm_types:
    steps = get_steps_by_sagemaker_type(sm_type)
    print(f"{sm_type} steps: {steps}")

# Get SageMaker type for specific steps
test_steps = ["XGBoostTraining", "CradleDataLoading", "Package"]
for step in test_steps:
    try:
        sm_type = get_sagemaker_step_type(step)
        print(f"{step} → {sm_type}")
    except ValueError as e:
        print(f"✗ {step}: {e}")
```

### Step Validation and Registry Analysis

```python
from cursus.registry.step_names import (
    validate_step_name,
    validate_spec_type,
    get_step_description,
    list_all_step_info
)

# Validate step names
test_steps = ["XGBoostTraining", "InvalidStep", "CradleDataLoading"]
for step in test_steps:
    is_valid = validate_step_name(step)
    status = "✓" if is_valid else "✗"
    print(f"{status} {step}")
    
    if is_valid:
        description = get_step_description(step)
        print(f"    Description: {description}")

# Get complete step information
all_info = list_all_step_info()
print(f"\nComplete registry information:")
for step_name, info in list(all_info.items())[:3]:  # Show first 3
    print(f"{step_name}:")
    for key, value in info.items():
        print(f"  {key}: {value}")
```

## Related Components

- **[Registry Module](__init__.md)** - Main registry module initialization
- **[Builder Registry](builder_registry.md)** - Step builder registry that uses step names
- **[Hyperparameter Registry](hyperparameter_registry.md)** - Hyperparameter registry integration
- **[Hybrid Manager](hybrid/manager.md)** - Unified registry manager backend
- **[Validation Utils](validation_utils.md)** - Step validation utilities
