---
tags:
  - code
  - registry
  - hybrid
  - setup
  - workspace
  - initialization
keywords:
  - create_workspace_registry
  - create_workspace_structure
  - create_workspace_documentation
  - create_example_implementations
  - validate_workspace_setup
  - copy_registry_from_developer
  - workspace initialization
  - registry templates
topics:
  - workspace management
  - registry setup
  - developer tools
  - workspace initialization
language: python
date of note: 2024-12-07
---

# Registry Hybrid Setup

Workspace registry initialization utilities with simplified implementation following redundancy evaluation guide principles.

## Overview

The hybrid setup module provides utilities for creating and managing developer workspace registries. It offers simplified workspace initialization with essential functionality for custom step development, template-based registry creation with standard and minimal options, comprehensive workspace structure generation with proper directory organization, and validation utilities to ensure correct workspace setup.

This module enables developers to quickly bootstrap workspace environments with proper registry structures, supporting both new workspace creation and copying configurations from existing developer workspaces. The implementation follows redundancy evaluation principles by focusing on essential functionality without theoretical complexity.

## Classes and Methods

### Functions
- [`create_workspace_registry`](#create_workspace_registry) - Create simple workspace registry structure for a developer
- [`create_workspace_structure`](#create_workspace_structure) - Create complete workspace directory structure
- [`create_workspace_documentation`](#create_workspace_documentation) - Create comprehensive workspace documentation
- [`create_example_implementations`](#create_example_implementations) - Create example step implementations for reference
- [`validate_workspace_setup`](#validate_workspace_setup) - Validate that workspace setup is correct
- [`copy_registry_from_developer`](#copy_registry_from_developer) - Copy registry configuration from existing developer workspace

### Private Functions
- [`_get_registry_template`](#_get_registry_template) - Get simplified registry template content
- [`_get_standard_template`](#_get_standard_template) - Get standard registry template with essential fields only
- [`_get_minimal_template`](#_get_minimal_template) - Get minimal registry template with bare essentials

## API Reference

### create_workspace_registry

create_workspace_registry(_workspace_path_, _developer_id_, _template="standard"_)

Creates simple workspace registry structure for a developer with template-based initialization.

**Parameters:**
- **workspace_path** (_str_) – Path to the developer workspace directory.
- **developer_id** (_str_) – Unique identifier for the developer (alphanumeric with underscores/hyphens).
- **template** (_str_) – Registry template type, either "standard" or "minimal" (default: "standard").

**Returns:**
- **str** – Path to the created registry file.

**Raises:**
- **ValueError** – If developer_id is invalid or workspace creation fails.

```python
from cursus.registry.hybrid.setup import create_workspace_registry

# Create standard workspace registry
registry_path = create_workspace_registry(
    workspace_path="/path/to/workspace",
    developer_id="john_doe",
    template="standard"
)
print(f"Registry created at: {registry_path}")
```

### create_workspace_structure

create_workspace_structure(_workspace_path_)

Creates complete workspace directory structure with all necessary directories and initialization files.

**Parameters:**
- **workspace_path** (_str_) – Path to the workspace directory to create.

```python
from cursus.registry.hybrid.setup import create_workspace_structure

# Create complete workspace structure
create_workspace_structure("/path/to/new/workspace")
```

### create_workspace_documentation

create_workspace_documentation(_workspace_dir_, _developer_id_, _registry_file_)

Creates comprehensive workspace documentation including README and usage examples.

**Parameters:**
- **workspace_dir** (_Path_) – Path object for the workspace directory.
- **developer_id** (_str_) – Developer identifier for documentation customization.
- **registry_file** (_str_) – Path to the registry file for documentation references.

**Returns:**
- **Path** – Path to the created README.md file.

```python
from pathlib import Path
from cursus.registry.hybrid.setup import create_workspace_documentation

# Create workspace documentation
workspace_dir = Path("/path/to/workspace")
readme_path = create_workspace_documentation(
    workspace_dir=workspace_dir,
    developer_id="john_doe",
    registry_file="src/cursus_dev/registry/workspace_registry.py"
)
```

### create_example_implementations

create_example_implementations(_workspace_dir_, _developer_id_)

Creates example step implementations for reference and learning purposes.

**Parameters:**
- **workspace_dir** (_Path_) – Path object for the workspace directory.
- **developer_id** (_str_) – Developer identifier for example customization.

```python
from pathlib import Path
from cursus.registry.hybrid.setup import create_example_implementations

# Create example implementations
workspace_dir = Path("/path/to/workspace")
create_example_implementations(workspace_dir, "john_doe")
```

### validate_workspace_setup

validate_workspace_setup(_workspace_path_, _developer_id_)

Validates that workspace setup is correct and all required components are present.

**Parameters:**
- **workspace_path** (_str_) – Path to the workspace directory to validate.
- **developer_id** (_str_) – Developer identifier for validation context.

**Raises:**
- **ValueError** – If workspace setup is invalid or missing required components.

```python
from cursus.registry.hybrid.setup import validate_workspace_setup

# Validate workspace setup
try:
    validate_workspace_setup("/path/to/workspace", "john_doe")
    print("Workspace setup is valid")
except ValueError as e:
    print(f"Validation failed: {e}")
```

### copy_registry_from_developer

copy_registry_from_developer(_workspace_path_, _developer_id_, _source_developer_)

Copies registry configuration from an existing developer workspace with ID replacement.

**Parameters:**
- **workspace_path** (_str_) – Path to the target workspace directory.
- **developer_id** (_str_) – New developer identifier for the copied registry.
- **source_developer** (_str_) – Source developer identifier to copy from.

**Returns:**
- **str** – Path to the created registry file.

**Raises:**
- **ValueError** – If source developer registry doesn't exist or copy operation fails.

```python
from cursus.registry.hybrid.setup import copy_registry_from_developer

# Copy registry from existing developer
registry_path = copy_registry_from_developer(
    workspace_path="/path/to/new/workspace",
    developer_id="jane_smith",
    source_developer="john_doe"
)
```

### _get_registry_template

_get_registry_template(_developer_id_, _template_)

Gets simplified registry template content following redundancy evaluation guide principles.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for template customization.
- **template** (_str_) – Template type ("standard" or "minimal").

**Returns:**
- **str** – Registry template content as string.

```python
# Internal function - typically not called directly
template_content = _get_registry_template("john_doe", "standard")
```

### _get_standard_template

_get_standard_template(_developer_id_)

Gets standard registry template with essential fields only, including LOCAL_STEPS, STEP_OVERRIDES, and WORKSPACE_METADATA sections.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for template customization.

**Returns:**
- **str** – Standard template content with examples and documentation.

```python
# Internal function - generates standard template
standard_template = _get_standard_template("john_doe")
```

### _get_minimal_template

_get_minimal_template(_developer_id_)

Gets minimal registry template with bare essentials for lightweight workspace setup.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for template customization.

**Returns:**
- **str** – Minimal template content with basic structure only.

```python
# Internal function - generates minimal template
minimal_template = _get_minimal_template("john_doe")
```

## Usage Examples

### Complete Workspace Setup

```python
from pathlib import Path
from cursus.registry.hybrid.setup import (
    create_workspace_registry,
    create_workspace_structure,
    create_workspace_documentation,
    create_example_implementations,
    validate_workspace_setup
)

# Complete workspace initialization
workspace_path = "/path/to/new/workspace"
developer_id = "john_doe"

# 1. Create directory structure
create_workspace_structure(workspace_path)

# 2. Create registry with standard template
registry_path = create_workspace_registry(
    workspace_path=workspace_path,
    developer_id=developer_id,
    template="standard"
)

# 3. Create documentation
workspace_dir = Path(workspace_path)
create_workspace_documentation(
    workspace_dir=workspace_dir,
    developer_id=developer_id,
    registry_file=registry_path
)

# 4. Create example implementations
create_example_implementations(workspace_dir, developer_id)

# 5. Validate setup
validate_workspace_setup(workspace_path, developer_id)

print(f"Workspace setup complete for {developer_id}")
```

### Minimal Workspace Setup

```python
from cursus.registry.hybrid.setup import create_workspace_registry, create_workspace_structure

# Minimal workspace for quick prototyping
workspace_path = "/path/to/minimal/workspace"
developer_id = "prototype_dev"

# Create basic structure and minimal registry
create_workspace_structure(workspace_path)
registry_path = create_workspace_registry(
    workspace_path=workspace_path,
    developer_id=developer_id,
    template="minimal"
)

print(f"Minimal workspace created: {registry_path}")
```

### Copy Existing Configuration

```python
from cursus.registry.hybrid.setup import copy_registry_from_developer, create_workspace_structure

# Copy configuration from experienced developer
workspace_path = "/path/to/new/workspace"
new_developer_id = "jane_smith"
source_developer_id = "john_doe"

# Create structure and copy registry
create_workspace_structure(workspace_path)
registry_path = copy_registry_from_developer(
    workspace_path=workspace_path,
    developer_id=new_developer_id,
    source_developer=source_developer_id
)

print(f"Registry copied from {source_developer_id} to {new_developer_id}")
```

## Template Structure

### Standard Template Features

- **LOCAL_STEPS**: Section for custom step definitions with examples
- **STEP_OVERRIDES**: Section for overriding core steps with examples
- **WORKSPACE_METADATA**: Essential workspace information
- **Documentation**: Inline comments and usage examples
- **Examples**: Commented example step definitions

### Minimal Template Features

- **LOCAL_STEPS**: Empty section for custom steps
- **STEP_OVERRIDES**: Empty section for overrides
- **WORKSPACE_METADATA**: Basic workspace information only
- **Minimal Documentation**: Essential comments only

## Directory Structure Created

```
project/
├── src/cursus_dev/           # Custom step implementations
│   ├── steps/
│   │   ├── builders/         # Step builder classes
│   │   ├── configs/          # Configuration classes
│   │   ├── contracts/        # Script contracts
│   │   ├── scripts/          # Processing scripts
│   │   └── specs/            # Step specifications
│   └── registry/             # Local registry
│       └── workspace_registry.py
├── test/                     # Unit and integration tests
│   ├── unit/
│   └── integration/
├── validation_reports/       # Validation results
├── examples/                 # Usage examples
├── docs/                     # Additional documentation
└── README.md                 # Workspace documentation
```

## Validation Checks

The `validate_workspace_setup` function performs these checks:

1. **Required Directories**: Verifies all essential directories exist
2. **Registry File**: Confirms registry file was created successfully
3. **Registry Content**: Validates registry file contains required sections
4. **Python Packages**: Ensures `__init__.py` files are present
5. **File Permissions**: Checks files are readable and writable

## Error Handling

Common error scenarios and their handling:

- **Invalid Developer ID**: Validates alphanumeric characters with underscores/hyphens
- **Missing Directories**: Creates parent directories as needed
- **File Permissions**: Handles permission errors during file creation
- **Template Errors**: Validates template content before writing
- **Source Registry Missing**: Checks source exists before copying

## Related Documentation

- [Registry Hybrid Manager](manager.md) - UnifiedRegistryManager for workspace integration
- [Registry Hybrid Models](models.md) - Data models used in workspace registries
- [Registry Hybrid Utils](utils.md) - Utility functions for registry operations
- [Registry Builder Registry](../builder_registry.md) - Step builder registry integration
- [Workspace Developer Guide](../../../slipbox/0_developer_guide/workspace_setup_guide.md) - Complete workspace setup guide
