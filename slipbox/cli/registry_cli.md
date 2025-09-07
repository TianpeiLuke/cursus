---
tags:
  - code
  - cli
  - registry
  - workspace
  - management
  - hybrid-registry
keywords:
  - registry
  - init-workspace
  - list-steps
  - validate-registry
  - resolve-step
  - validate-step-definition
  - validation-status
  - reset-validation-metrics
  - workspace management
  - step resolution
topics:
  - registry management
  - workspace initialization
  - step validation
  - CLI tools
language: python
date of note: 2024-12-07
---

# Registry CLI

Command-line interface for hybrid registry management providing comprehensive tools for workspace initialization, registry validation, and step resolution.

## Overview

The Registry CLI provides comprehensive management tools for the cursus hybrid registry system, supporting developer workspace initialization with template-based setup, registry validation and conflict detection, step resolution with intelligent conflict handling, and performance monitoring with detailed metrics. The CLI offers both basic registry operations and advanced hybrid registry features for complex development scenarios.

The module supports multiple workspace templates including minimal, standard, and advanced configurations, comprehensive validation with auto-correction capabilities, intelligent step resolution with framework preferences, and detailed performance monitoring with cache management. All commands provide detailed help documentation and examples for effective usage.

## Classes and Methods

### Commands
- [`registry`](#registry) - Main command group for registry management tools
- [`init-workspace`](#init-workspace) - Initialize a new developer workspace with hybrid registry support
- [`list-steps`](#list-steps) - List available steps in registry with optional workspace context
- [`validate-registry`](#validate-registry) - Validate registry configuration and check for issues
- [`resolve-step`](#resolve-step) - Resolve a specific step name and show resolution details
- [`validate-step-definition`](#validate-step-definition) - Validate a step definition against standardization rules
- [`validation-status`](#validation-status) - Show validation system status and performance metrics
- [`reset-validation-metrics`](#reset-validation-metrics) - Reset validation performance metrics and cache

### Helper Functions
- [`_create_workspace_structure`](#_create_workspace_structure) - Create complete workspace directory structure
- [`_create_workspace_registry`](#_create_workspace_registry) - Create workspace registry configuration file
- [`_create_workspace_documentation`](#_create_workspace_documentation) - Create comprehensive workspace documentation
- [`_create_example_implementations`](#_create_example_implementations) - Create example step implementations for reference
- [`main`](#main) - Main entry point for registry CLI

## API Reference

### registry

@click.group(name='registry')

Main command group for registry management tools with comprehensive workspace and validation features.

```bash
# Access registry commands
python -m cursus.cli registry --help
```

### init-workspace

init-workspace(_workspace_id_, _--workspace-path_, _--template_, _--force_)

Initializes a new developer workspace with hybrid registry support and complete directory structure.

**Parameters:**
- **workspace_id** (_str_) – Unique identifier for the developer workspace (alphanumeric with hyphens/underscores).
- **--workspace-path** (_str_) – Custom workspace path (default: developer_workspaces/developers/{workspace_id}).
- **--template** (_Choice_) – Registry template to use: minimal, standard, or advanced (default: standard).
- **--force** (_Flag_) – Overwrite existing workspace if it exists.

```bash
# Create standard workspace
python -m cursus.cli registry init-workspace john_doe

# Create advanced workspace with custom path
python -m cursus.cli registry init-workspace ml_team --workspace-path ./workspaces/ml_team --template advanced

# Force overwrite existing workspace
python -m cursus.cli registry init-workspace john_doe --force

# Create minimal workspace for quick prototyping
python -m cursus.cli registry init-workspace prototype --template minimal
```

### list-steps

list-steps(_--workspace_, _--conflicts-only_, _--include-source_)

Lists available steps in registry with optional workspace context and conflict detection.

**Parameters:**
- **--workspace** (_str_) – Workspace ID to list steps for (uses current context if not specified).
- **--conflicts-only** (_Flag_) – Show only conflicting steps between registries.
- **--include-source** (_Flag_) – Include source registry information for each step.

```bash
# List all steps in current context
python -m cursus.cli registry list-steps

# List steps for specific workspace
python -m cursus.cli registry list-steps --workspace john_doe

# Show only conflicting steps
python -m cursus.cli registry list-steps --conflicts-only

# List steps with source information
python -m cursus.cli registry list-steps --include-source --workspace ml_team
```

### validate-registry

validate-registry(_--workspace_, _--check-conflicts_)

Validates registry configuration and checks for issues with comprehensive reporting.

**Parameters:**
- **--workspace** (_str_) – Workspace ID to validate (uses current context if not specified).
- **--check-conflicts** (_Flag_) – Check for step name conflicts between registries.

```bash
# Validate current registry
python -m cursus.cli registry validate-registry

# Validate specific workspace
python -m cursus.cli registry validate-registry --workspace john_doe

# Validate with conflict checking
python -m cursus.cli registry validate-registry --workspace ml_team --check-conflicts

# Comprehensive validation
python -m cursus.cli registry validate-registry --check-conflicts
```

### resolve-step

resolve-step(_step_name_, _--workspace_, _--framework_)

Resolves a specific step name and shows detailed resolution information with conflict analysis.

**Parameters:**
- **step_name** (_str_) – Name of the step to resolve.
- **--workspace** (_str_) – Workspace context for resolution (uses current context if not specified).
- **--framework** (_str_) – Preferred framework for intelligent resolution.

```bash
# Resolve step in current context
python -m cursus.cli registry resolve-step XGBoostTraining

# Resolve with workspace context
python -m cursus.cli registry resolve-step CustomProcessing --workspace john_doe

# Resolve with framework preference
python -m cursus.cli registry resolve-step ModelTraining --framework pytorch --workspace ml_team

# Detailed resolution analysis
python -m cursus.cli registry resolve-step ConflictingStep --workspace john_doe --framework xgboost
```

### validate-step-definition

validate-step-definition(_--name_, _--config-class_, _--builder-name_, _--sagemaker-type_, _--auto-correct_, _--performance_)

Validates a step definition against standardization rules with auto-correction capabilities.

**Parameters:**
- **--name** (_str_) – Step name to validate (required).
- **--config-class** (_str_) – Config class name (optional).
- **--builder-name** (_str_) – Builder class name (optional).
- **--sagemaker-type** (_str_) – SageMaker step type (optional).
- **--auto-correct** (_Flag_) – Apply auto-correction to naming violations.
- **--performance** (_Flag_) – Show performance metrics for validation.

```bash
# Basic step validation
python -m cursus.cli registry validate-step-definition --name MyCustomStep

# Comprehensive validation
python -m cursus.cli registry validate-step-definition --name MyCustomStep --config-class MyCustomStepConfig --builder-name MyCustomStepBuilder --sagemaker-type Processing

# Validation with auto-correction
python -m cursus.cli registry validate-step-definition --name my_custom_step --auto-correct

# Performance analysis
python -m cursus.cli registry validate-step-definition --name MyCustomStep --performance
```

### validation-status

validation-status()

Shows validation system status and performance metrics with detailed analysis.

```bash
# Show validation system status
python -m cursus.cli registry validation-status
```

### reset-validation-metrics

reset-validation-metrics()

Resets validation performance metrics and cache with confirmation prompt.

```bash
# Reset validation metrics (with confirmation)
python -m cursus.cli registry reset-validation-metrics
```

### _create_workspace_structure

_create_workspace_structure(_workspace_dir_)

Creates complete workspace directory structure with proper Python package initialization.

**Parameters:**
- **workspace_dir** (_Path_) – Path to the workspace directory to create.

```python
from pathlib import Path
from cursus.cli.registry_cli import _create_workspace_structure

# Create workspace structure
workspace_dir = Path("./my_workspace")
_create_workspace_structure(workspace_dir)
```

### _create_workspace_registry

_create_workspace_registry(_workspace_dir_, _workspace_id_, _template_)

Creates workspace registry configuration file with template-based content.

**Parameters:**
- **workspace_dir** (_Path_) – Path to the workspace directory.
- **workspace_id** (_str_) – Developer workspace identifier.
- **template** (_str_) – Registry template type (minimal, standard, advanced).

**Returns:**
- **str** – Path to the created registry file.

```python
from pathlib import Path
from cursus.cli.registry_cli import _create_workspace_registry

# Create registry file
workspace_dir = Path("./my_workspace")
registry_path = _create_workspace_registry(workspace_dir, "john_doe", "standard")
```

### _create_workspace_documentation

_create_workspace_documentation(_workspace_dir_, _workspace_id_, _registry_file_)

Creates comprehensive workspace documentation including README and usage examples.

**Parameters:**
- **workspace_dir** (_Path_) – Path to the workspace directory.
- **workspace_id** (_str_) – Developer workspace identifier.
- **registry_file** (_str_) – Path to the registry file for documentation references.

**Returns:**
- **str** – Path to the created README file.

```python
from pathlib import Path
from cursus.cli.registry_cli import _create_workspace_documentation

# Create documentation
workspace_dir = Path("./my_workspace")
readme_path = _create_workspace_documentation(workspace_dir, "john_doe", "registry.py")
```

### _create_example_implementations

_create_example_implementations(_workspace_dir_, _workspace_id_, _template_)

Creates example step implementations for reference and learning purposes.

**Parameters:**
- **workspace_dir** (_Path_) – Path to the workspace directory.
- **workspace_id** (_str_) – Developer workspace identifier.
- **template** (_str_) – Registry template type for example customization.

```python
from pathlib import Path
from cursus.cli.registry_cli import _create_example_implementations

# Create example implementations
workspace_dir = Path("./my_workspace")
_create_example_implementations(workspace_dir, "john_doe", "standard")
```

### main

main()

Main entry point for registry CLI with command group initialization.

```python
from cursus.cli.registry_cli import main

# Run registry CLI
main()
```

## Workspace Templates

The registry CLI supports three workspace templates for different development needs:

### Minimal Template
- **Purpose**: Quick prototyping and simple customizations
- **Features**: Basic registry structure with essential sections
- **Use Cases**: Simple step overrides, proof-of-concept development
- **Structure**: Minimal metadata, empty step sections with examples

### Standard Template
- **Purpose**: Regular development with moderate customization
- **Features**: Complete registry structure with examples and documentation
- **Use Cases**: Custom step development, framework integration
- **Structure**: Full metadata, example step definitions, comprehensive comments

### Advanced Template
- **Purpose**: Complex development with extensive customization
- **Features**: Full registry structure with advanced conflict resolution
- **Use Cases**: Multi-framework projects, complex workspace hierarchies
- **Structure**: Rich metadata, priority settings, conflict resolution strategies

## Workspace Structure

The CLI creates a comprehensive workspace structure:

```
workspace_id/
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

## Registry Validation

### Basic Validation
- **Step Count**: Verifies registry contains valid steps
- **Structure Check**: Ensures proper registry file structure
- **Metadata Validation**: Validates workspace metadata

### Conflict Detection
- **Name Conflicts**: Identifies steps with conflicting names
- **Resolution Analysis**: Shows available resolution strategies
- **Priority Assessment**: Analyzes step priority settings

### Performance Monitoring
- **Validation Metrics**: Tracks validation performance
- **Cache Statistics**: Monitors cache hit rates and efficiency
- **Error Tracking**: Records validation errors and patterns

## Step Resolution

### Resolution Context
- **Workspace Priority**: Prioritizes workspace-specific definitions
- **Framework Matching**: Matches steps by framework compatibility
- **Environment Tags**: Considers environment-specific requirements

### Resolution Strategies
- **Automatic**: Uses configured resolution strategies
- **Interactive**: Prompts for conflict resolution choices
- **Strict**: Fails on any conflicts without resolution

### Resolution Results
- **Success Information**: Shows selected definition and source
- **Conflict Analysis**: Details conflicting definitions found
- **Strategy Used**: Reports resolution strategy applied

## Usage Patterns

### Development Workflow
```bash
# 1. Initialize workspace
python -m cursus.cli registry init-workspace john_doe --template standard

# 2. Set workspace context
export CURSUS_WORKSPACE_ID=john_doe

# 3. Validate initial setup
python -m cursus.cli registry validate-registry --workspace john_doe

# 4. Add custom steps (edit registry file)
# Edit: developer_workspaces/developers/john_doe/src/cursus_dev/registry/workspace_registry.py

# 5. Validate custom steps
python -m cursus.cli registry validate-step-definition --name MyCustomStep --auto-correct

# 6. Check for conflicts
python -m cursus.cli registry validate-registry --workspace john_doe --check-conflicts

# 7. Test step resolution
python -m cursus.cli registry resolve-step MyCustomStep --workspace john_doe
```

### Team Collaboration
```bash
# Create team workspace
python -m cursus.cli registry init-workspace ml_team --template advanced

# List all available steps
python -m cursus.cli registry list-steps --include-source

# Check for conflicts across workspaces
python -m cursus.cli registry list-steps --conflicts-only

# Validate team registry
python -m cursus.cli registry validate-registry --workspace ml_team --check-conflicts
```

### Performance Monitoring
```bash
# Check validation system status
python -m cursus.cli registry validation-status

# Reset metrics for fresh monitoring
python -m cursus.cli registry reset-validation-metrics

# Monitor validation performance
python -m cursus.cli registry validate-step-definition --name TestStep --performance
```

## Error Handling

The registry CLI provides comprehensive error handling:

- **Workspace Creation**: Validates workspace IDs and handles creation failures
- **Registry Validation**: Provides detailed error messages for validation issues
- **Step Resolution**: Reports resolution failures with suggested solutions
- **File Operations**: Handles file system errors with cleanup on failure
- **Import Errors**: Graceful fallback when hybrid registry features unavailable

## Integration Points

- **Hybrid Registry System**: Full integration with UnifiedRegistryManager
- **Validation Framework**: Uses cursus validation utilities for step validation
- **Workspace Management**: Integrates with workspace context management
- **Performance Monitoring**: Comprehensive metrics and cache management
- **Template System**: Flexible template-based workspace initialization

## Related Documentation

- [CLI Module](__init__.md) - Main CLI dispatcher and command routing
- [Registry Hybrid Manager](../registry/hybrid/manager.md) - UnifiedRegistryManager implementation
- [Registry Validation Utils](../registry/validation_utils.md) - Validation utilities and performance monitoring
- [Registry Hybrid Setup](../registry/hybrid/setup.md) - Workspace setup utilities
- [Workspace CLI](workspace_cli.md) - Developer workspace management tools
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
