# Workspace CLI Reference for Isolated Projects

**Version**: 1.0.0  
**Date**: September 5, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides comprehensive reference documentation for the workspace-aware CLI commands available for isolated project development. These commands enable efficient management of project workspaces, component registration, validation, and testing in the hybrid registry system.

## CLI Architecture

The workspace-aware CLI extends the main cursus CLI with project-specific commands:

```
cursus
├── Workspace Management
│   ├── init-workspace          # Initialize new project workspace
│   ├── list-workspaces         # List available workspaces
│   ├── activate-workspace      # Activate specific workspace
│   ├── current-workspace       # Show current workspace
│   └── deactivate-workspace    # Deactivate current workspace
├── Component Management
│   ├── list-steps              # List step builders (workspace-aware)
│   ├── list-components         # List all components (workspace-aware)
│   ├── register-component      # Register project-specific component
│   └── unregister-component    # Unregister component
├── Validation & Testing
│   ├── validate-workspace      # Validate workspace setup
│   ├── validate-registry       # Validate registry integration
│   ├── validate-alignment      # Validate component alignment
│   ├── test                    # Run tests (workspace-aware)
│   └── check-conflicts         # Check for component conflicts
└── Development Tools
    ├── generate-step           # Generate step template
    ├── create-project-template # Create project template
    └── workspace-info          # Show workspace information
```

## Workspace Management Commands

### init-workspace

Initialize a new isolated project workspace.

**Syntax:**
```bash
cursus init-workspace --project PROJECT_NAME --type isolated [OPTIONS]
```

**Options:**
- `--project PROJECT_NAME` (required): Name of the project workspace
- `--type TYPE` (required): Workspace type (`isolated`, `shared`, `hybrid`)
- `--template TEMPLATE`: Template to use for initialization
- `--existing`: Initialize workspace in existing directory
- `--force`: Force initialization even if directory exists
- `--description DESC`: Project description
- `--maintainer EMAIL`: Project maintainer email

**Examples:**
```bash
# Create new isolated project
cursus init-workspace --project my_ml_project --type isolated

# Create from template
cursus init-workspace --project data_pipeline --type isolated --template standard

# Initialize existing directory
cursus init-workspace --project existing_project --type isolated --existing

# Create with metadata
cursus init-workspace --project analytics --type isolated \
  --description "Analytics pipeline project" \
  --maintainer "developer@company.com"
```

**Output:**
```
✓ Created project directory: development/projects/my_ml_project
✓ Initialized workspace structure
✓ Created workspace_config.yaml
✓ Set up project registry
✓ Created test templates
✓ Project workspace 'my_ml_project' ready for development
```

### list-workspaces

List all available workspaces in the system.

**Syntax:**
```bash
cursus list-workspaces [OPTIONS]
```

**Options:**
- `--type TYPE`: Filter by workspace type (`isolated`, `shared`, `hybrid`)
- `--active-only`: Show only active workspaces
- `--detailed`: Show detailed workspace information
- `--format FORMAT`: Output format (`table`, `json`, `yaml`)

**Examples:**
```bash
# List all workspaces
cursus list-workspaces

# List only isolated workspaces
cursus list-workspaces --type isolated

# Detailed information
cursus list-workspaces --detailed

# JSON output
cursus list-workspaces --format json
```

**Output:**
```
Available Workspaces:
┌─────────────────┬──────────┬────────────┬─────────────────────┐
│ Name            │ Type     │ Status     │ Last Modified       │
├─────────────────┼──────────┼────────────┼─────────────────────┤
│ main            │ shared   │ active     │ 2025-09-05 10:30:00 │
│ my_ml_project   │ isolated │ inactive   │ 2025-09-05 14:15:00 │
│ data_pipeline   │ isolated │ active     │ 2025-09-05 16:45:00 │
│ project_alpha   │ isolated │ inactive   │ 2025-09-04 09:20:00 │
└─────────────────┴──────────┴────────────┴─────────────────────┘
```

### activate-workspace

Activate a specific workspace for CLI operations.

**Syntax:**
```bash
cursus activate-workspace WORKSPACE_NAME [OPTIONS]
```

**Options:**
- `--persist`: Persist activation across terminal sessions
- `--validate`: Validate workspace before activation
- `--verbose`: Show detailed activation process

**Examples:**
```bash
# Activate workspace
cursus activate-workspace my_ml_project

# Activate with validation
cursus activate-workspace my_ml_project --validate

# Persistent activation
cursus activate-workspace my_ml_project --persist
```

**Output:**
```
✓ Activated workspace: my_ml_project
✓ Registry context set to: my_ml_project
✓ CLI commands will use workspace context
Current workspace: my_ml_project
```

### current-workspace

Show the currently active workspace.

**Syntax:**
```bash
cursus current-workspace [OPTIONS]
```

**Options:**
- `--detailed`: Show detailed workspace information
- `--registry-info`: Include registry information
- `--components`: List available components

**Examples:**
```bash
# Show current workspace
cursus current-workspace

# Detailed information
cursus current-workspace --detailed

# Include components
cursus current-workspace --components
```

**Output:**
```
Current Workspace: my_ml_project
Type: isolated
Status: active
Registry Context: my_ml_project
Components: 5 project-specific, 23 shared accessible
```

### deactivate-workspace

Deactivate the current workspace.

**Syntax:**
```bash
cursus deactivate-workspace [OPTIONS]
```

**Options:**
- `--reset-to-main`: Reset to main workspace instead of no workspace
- `--clear-cache`: Clear workspace-specific cache

**Examples:**
```bash
# Deactivate current workspace
cursus deactivate-workspace

# Deactivate and reset to main
cursus deactivate-workspace --reset-to-main
```

## Component Management Commands

### list-steps

List step builders available in the current or specified workspace.

**Syntax:**
```bash
cursus list-steps [OPTIONS]
```

**Options:**
- `--workspace WORKSPACE`: Specify workspace (overrides current)
- `--all`: Include components from all accessible workspaces
- `--project-only`: Show only project-specific components
- `--shared-only`: Show only shared components
- `--detailed`: Show detailed component information
- `--format FORMAT`: Output format (`table`, `json`, `yaml`, `tree`)

**Examples:**
```bash
# List steps in current workspace
cursus list-steps

# List steps in specific workspace
cursus list-steps --workspace my_ml_project

# Show all accessible steps
cursus list-steps --all

# Project-specific only
cursus list-steps --project-only

# Detailed information
cursus list-steps --detailed
```

**Output:**
```
Step Builders in workspace 'my_ml_project':
┌─────────────────────┬─────────────┬──────────────────────┬─────────────┐
│ Name                │ Source      │ Class                │ Module      │
├─────────────────────┼─────────────┼──────────────────────┼─────────────┤
│ custom_data_loader  │ project     │ CustomDataLoader     │ cursus_dev  │
│ feature_engineer    │ project     │ FeatureEngineer      │ cursus_dev  │
│ preprocessing_step  │ shared      │ PreprocessingStep    │ cursus.core │
│ training_step       │ shared      │ TrainingStep         │ cursus.ml   │
│ evaluation_step     │ shared      │ EvaluationStep       │ cursus.ml   │
└─────────────────────┴─────────────┴──────────────────────┴─────────────┘
```

### list-components

List all components (not just step builders) in the workspace.

**Syntax:**
```bash
cursus list-components [OPTIONS]
```

**Options:**
- `--workspace WORKSPACE`: Specify workspace
- `--type TYPE`: Filter by component type (`step_builder`, `validator`, `config`, `utility`)
- `--detailed`: Show detailed information
- `--tree`: Show hierarchical view

**Examples:**
```bash
# List all components
cursus list-components

# Filter by type
cursus list-components --type step_builder

# Tree view
cursus list-components --tree
```

### register-component

Register a project-specific component in the current workspace.

**Syntax:**
```bash
cursus register-component --name NAME --class CLASS --module MODULE [OPTIONS]
```

**Options:**
- `--type TYPE`: Component type (`step_builder`, `validator`, `config`)
- `--description DESC`: Component description
- `--version VERSION`: Component version
- `--force`: Force registration even if component exists

**Examples:**
```bash
# Register step builder
cursus register-component --name custom_processor \
  --class CustomProcessor --module cursus_dev.steps.builders

# Register with metadata
cursus register-component --name data_validator \
  --class DataValidator --module cursus_dev.validators \
  --type validator --description "Custom data validation" --version "1.0.0"
```

## Validation & Testing Commands

### validate-workspace

Validate workspace setup and configuration.

**Syntax:**
```bash
cursus validate-workspace [WORKSPACE] [OPTIONS]
```

**Options:**
- `--comprehensive`: Run comprehensive validation
- `--check-structure`: Validate directory structure
- `--check-registry`: Validate registry integration
- `--check-dependencies`: Validate dependencies
- `--check-config`: Validate configuration files
- `--fix`: Attempt to fix found issues
- `--report-file FILE`: Save validation report to file

**Examples:**
```bash
# Basic validation
cursus validate-workspace

# Comprehensive validation
cursus validate-workspace my_ml_project --comprehensive

# Validate specific aspects
cursus validate-workspace --check-structure --check-registry

# Fix issues
cursus validate-workspace --fix
```

**Output:**
```
Validating workspace: my_ml_project
✓ Directory structure valid
✓ Configuration files present and valid
✓ Registry integration working
✓ Dependencies satisfied
✓ Python path configured correctly
⚠ Warning: 2 unused test files found
✗ Error: Missing __init__.py in cursus_dev/utils

Validation Summary:
- Passed: 5 checks
- Warnings: 1
- Errors: 1
- Overall Status: NEEDS_ATTENTION
```

### validate-registry

Validate registry integration and component resolution.

**Syntax:**
```bash
cursus validate-registry [OPTIONS]
```

**Options:**
- `--workspace WORKSPACE`: Validate specific workspace
- `--test-fallback`: Test fallback to shared components
- `--test-resolution`: Test component resolution priority
- `--performance`: Include performance tests
- `--report-file FILE`: Save report to file

**Examples:**
```bash
# Validate current workspace registry
cursus validate-registry

# Test fallback functionality
cursus validate-registry --test-fallback

# Performance testing
cursus validate-registry --performance
```

### validate-alignment

Validate component alignment across the 4-tier validation system.

**Syntax:**
```bash
cursus validate-alignment [OPTIONS]
```

**Options:**
- `--step STEP_TYPE`: Validate specific step type
- `--workspace WORKSPACE`: Validate specific workspace
- `--tier TIER`: Validate specific tier (1-4)
- `--check-shared`: Include shared component alignment
- `--detailed`: Show detailed alignment information

**Examples:**
```bash
# Validate all components
cursus validate-alignment

# Validate specific step
cursus validate-alignment --step custom_data_loader

# Validate with shared components
cursus validate-alignment --check-shared
```

### test

Run tests with workspace awareness.

**Syntax:**
```bash
cursus test [OPTIONS] [TEST_PATTERN]
```

**Options:**
- `--workspace WORKSPACE`: Run tests for specific workspace
- `--integration`: Include integration tests
- `--shared`: Include shared component tests
- `--coverage`: Generate coverage report
- `--parallel`: Run tests in parallel
- `--verbose`: Verbose test output

**Examples:**
```bash
# Run all project tests
cursus test

# Run integration tests
cursus test --integration

# Run with coverage
cursus test --coverage

# Run specific test pattern
cursus test test_custom_steps.py
```

### check-conflicts

Check for component conflicts between project and shared code.

**Syntax:**
```bash
cursus check-conflicts [OPTIONS]
```

**Options:**
- `--workspace WORKSPACE`: Check specific workspace
- `--resolve`: Suggest conflict resolution
- `--detailed`: Show detailed conflict information

**Examples:**
```bash
# Check for conflicts
cursus check-conflicts

# Get resolution suggestions
cursus check-conflicts --resolve
```

## Development Tools

### generate-step

Generate step template files for new pipeline steps.

**Syntax:**
```bash
cursus generate-step --name STEP_NAME --type STEP_TYPE [OPTIONS]
```

**Options:**
- `--type TYPE`: Step type (`processing`, `training`, `transform`)
- `--template TEMPLATE`: Template to use
- `--output-dir DIR`: Output directory (default: current workspace)
- `--include-tests`: Generate test files
- `--include-config`: Generate configuration class

**Examples:**
```bash
# Generate processing step
cursus generate-step --name data_cleaner --type processing

# Generate with tests and config
cursus generate-step --name model_trainer --type training \
  --include-tests --include-config
```

### workspace-info

Show detailed information about workspace.

**Syntax:**
```bash
cursus workspace-info [WORKSPACE] [OPTIONS]
```

**Options:**
- `--components`: Include component information
- `--dependencies`: Include dependency information
- `--performance`: Include performance metrics
- `--format FORMAT`: Output format (`table`, `json`, `yaml`)

**Examples:**
```bash
# Show workspace info
cursus workspace-info

# Detailed information
cursus workspace-info my_ml_project --components --dependencies
```

## Global Options

All commands support these global options:

- `--help, -h`: Show help message
- `--verbose, -v`: Verbose output
- `--quiet, -q`: Quiet output (errors only)
- `--config CONFIG_FILE`: Use specific configuration file
- `--log-level LEVEL`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--no-color`: Disable colored output

## Configuration

### CLI Configuration File

The CLI can be configured using a configuration file:

```yaml
# ~/.cursus/cli_config.yaml
default_workspace: "my_ml_project"
auto_activate: true
output_format: "table"
log_level: "INFO"
enable_colors: true

workspaces:
  my_ml_project:
    auto_validate: true
    default_test_pattern: "test_*.py"
    
validation:
  comprehensive_by_default: false
  auto_fix: false
  
performance:
  enable_caching: true
  cache_timeout: 3600
```

### Environment Variables

Configure CLI behavior using environment variables:

```bash
# Workspace settings
export CURSUS_DEFAULT_WORKSPACE="my_ml_project"
export CURSUS_AUTO_ACTIVATE="true"

# Output settings
export CURSUS_OUTPUT_FORMAT="json"
export CURSUS_ENABLE_COLORS="false"

# Performance settings
export CURSUS_ENABLE_CACHE="true"
export CURSUS_CACHE_TIMEOUT="7200"

# Validation settings
export CURSUS_AUTO_VALIDATE="true"
export CURSUS_COMPREHENSIVE_VALIDATION="false"
```

## Scripting and Automation

### Bash Completion

Enable bash completion for the CLI:

```bash
# Add to ~/.bashrc or ~/.bash_profile
eval "$(_CURSUS_COMPLETE=source_bash cursus)"
```

### Automation Scripts

Example automation scripts:

```bash
#!/bin/bash
# setup_project.sh - Set up new project workspace

PROJECT_NAME=$1
if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: $0 <project_name>"
    exit 1
fi

echo "Setting up project workspace: $PROJECT_NAME"

# Initialize workspace
cursus init-workspace --project "$PROJECT_NAME" --type isolated

# Activate workspace
cursus activate-workspace "$PROJECT_NAME"

# Validate setup
cursus validate-workspace --comprehensive

# Generate initial step template
cursus generate-step --name "data_processor" --type processing --include-tests

echo "Project workspace '$PROJECT_NAME' ready for development"
```

```bash
#!/bin/bash
# validate_all_workspaces.sh - Validate all project workspaces

echo "Validating all project workspaces..."

# Get list of isolated workspaces
WORKSPACES=$(cursus list-workspaces --type isolated --format json | jq -r '.[].name')

for workspace in $WORKSPACES; do
    echo "Validating workspace: $workspace"
    cursus validate-workspace "$workspace" --comprehensive
    cursus validate-registry --workspace "$workspace"
    echo "---"
done

echo "Validation complete"
```

## Troubleshooting CLI Issues

### Common Issues and Solutions

#### Issue 1: Command Not Found

```bash
# Error: cursus: command not found
# Solution: Ensure CLI is properly installed
pip install -e .  # From cursus root directory

# Or add to PATH
export PATH="$PATH:/path/to/cursus/bin"
```

#### Issue 2: Workspace Not Found

```bash
# Error: Workspace 'my_project' not found
# Solution: List available workspaces and verify name
cursus list-workspaces
cursus activate-workspace correct_workspace_name
```

#### Issue 3: Permission Errors

```bash
# Error: Permission denied when creating workspace
# Solution: Check directory permissions
ls -la development/projects/
chmod 755 development/projects/

# Or create in user directory
cursus init-workspace --project my_project --type isolated --output-dir ~/projects
```

#### Issue 4: Registry Context Issues

```bash
# Error: No registry context set
# Solution: Activate workspace or set context explicitly
cursus activate-workspace my_project

# Or check current context
cursus current-workspace
```

### Debug Mode

Enable debug mode for troubleshooting:

```bash
# Enable debug logging
cursus --log-level DEBUG command

# Or set environment variable
export CURSUS_LOG_LEVEL=DEBUG
cursus command
```

### Getting Help

```bash
# General help
cursus --help

# Command-specific help
cursus init-workspace --help
cursus validate-workspace --help

# List all available commands
cursus --help | grep "Commands:"
```

## Performance Tips

### 1. Use Workspace Activation

```bash
# ✅ Good: Activate workspace once
cursus activate-workspace my_project
cursus list-steps
cursus validate-registry
cursus test

# ❌ Bad: Specify workspace for each command
cursus list-steps --workspace my_project
cursus validate-registry --workspace my_project
cursus test --workspace my_project
```

### 2. Enable Caching

```bash
# Enable caching for better performance
export CURSUS_ENABLE_CACHE=true
export CURSUS_CACHE_TIMEOUT=3600
```

### 3. Use Batch Operations

```bash
# ✅ Good: Comprehensive validation in one command
cursus validate-workspace --comprehensive

# ❌ Bad: Multiple separate validations
cursus validate-workspace --check-structure
cursus validate-workspace --check-registry
cursus validate-workspace --check-dependencies
```

## Related Documentation

- [Workspace Setup Guide](ws_workspace_setup_guide.md) - Initial project setup procedures
- [Hybrid Registry Integration](ws_hybrid_registry_integration.md) - Registry system usage
- [Testing in Isolated Projects](ws_testing_in_isolated_projects.md) - Testing strategies and commands
- [Troubleshooting Workspace Issues](ws_troubleshooting_workspace_issues.md) - Common problems and solutions

### Main Developer Guide References
- [Adding a New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Main workspace development
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md) - Validation system overview
