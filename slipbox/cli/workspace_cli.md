---
tags:
  - code
  - cli
  - workspace
  - lifecycle-management
  - developer-tools
keywords:
  - workspace CLI
  - developer workspace
  - workspace lifecycle
  - isolation boundaries
  - cross-workspace operations
  - workspace validation
  - component discovery
  - runtime testing
topics:
  - workspace management
  - developer tools
  - isolation boundaries
  - cross-workspace operations
language: python
date of note: 2024-12-07
---

# Workspace CLI

Command-line interface for comprehensive workspace lifecycle management, providing tools for creating, managing, validating, and coordinating developer workspaces with isolation boundaries and cross-workspace operations.

## Overview

The Workspace CLI (`cursus workspace`) provides a complete set of commands for managing developer workspaces in the Cursus framework. It supports workspace creation, validation, component discovery, runtime testing, and cross-workspace operations while maintaining proper isolation boundaries.

The CLI integrates with the Phase 4 WorkspaceAPI for lifecycle management, WorkspaceComponentRegistry for cross-workspace component discovery, UnifiedValidationCore for multi-level alignment validation, and runtime testing framework for workspace-aware script testing.

## Commands and Functions

### create

cursus workspace create _developer_name_ [_options_]

Create a new developer workspace with optional templates and configuration.

**Parameters:**
- **developer_name** (_str_) – Name/ID of the developer workspace to create

**Options:**
- **--template** (_Optional[str]_) – Workspace template to use (basic, ml_pipeline, data_processing, custom)
- **--from-existing** (_Optional[str]_) – Clone from existing workspace
- **--interactive** (_bool_) – Interactive setup mode
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--config** (_Optional[File]_) – JSON configuration file for workspace setup
- **--output** (_str_) – Output format (json, text)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Create basic workspace
cursus workspace create alice

# Create ML pipeline workspace with template
cursus workspace create bob --template ml_pipeline

# Interactive workspace creation
cursus workspace create charlie --interactive

# Create workspace with custom configuration
cursus workspace create diana --config workspace_config.json

# Clone from existing workspace
cursus workspace create eve --from-existing alice
```

### list

cursus workspace list [_options_]

List available developer workspaces with status and component information.

**Options:**
- **--active** (_bool_) – Show only active workspaces (modified within 30 days)
- **--format** (_str_) – Output format (table, json)
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--show-components** (_bool_) – Show component counts for each workspace

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# List all workspaces
cursus workspace list

# List only active workspaces
cursus workspace list --active

# List with component information
cursus workspace list --show-components

# JSON output for programmatic use
cursus workspace list --format json
```

### validate

cursus workspace validate [_options_]

Validate workspace isolation and compliance with framework standards.

**Options:**
- **--workspace-path** (_Optional[Path]_) – Specific workspace path to validate
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--report** (_Optional[str]_) – Output report path
- **--format** (_str_) – Output format (text, json)
- **--strict** (_bool_) – Enable strict validation mode

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Validate all workspaces
cursus workspace validate

# Validate specific workspace
cursus workspace validate --workspace-path ./development/developers/alice

# Generate validation report
cursus workspace validate --report validation_report.json --format json

# Strict validation mode
cursus workspace validate --strict
```

### info

cursus workspace info _developer_name_ [_options_]

Show detailed information about a specific workspace.

**Parameters:**
- **developer_name** (_str_) – Name/ID of the developer workspace

**Options:**
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--show-components** (_bool_) – Show detailed component information
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Basic workspace information
cursus workspace info alice

# Detailed component information
cursus workspace info alice --show-components

# JSON output
cursus workspace info alice --format json
```

### health-check

cursus workspace health-check [_options_]

Perform comprehensive health check on workspace(s) with optional issue fixing.

**Options:**
- **--workspace** (_Optional[str]_) – Specific workspace to check
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--fix-issues** (_bool_) – Attempt to fix detected issues automatically
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Check all workspaces
cursus workspace health-check

# Check specific workspace
cursus workspace health-check --workspace alice

# Check and fix issues
cursus workspace health-check --fix-issues

# JSON output for automation
cursus workspace health-check --format json
```

### remove

cursus workspace remove _developer_name_ [_options_]

Remove a developer workspace with optional backup.

**Parameters:**
- **developer_name** (_str_) – Name/ID of the developer workspace to remove

**Options:**
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--backup** (_bool_) – Create backup before removal
- **--yes** (_bool_) – Skip confirmation prompt

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Remove workspace with confirmation
cursus workspace remove alice

# Remove with backup
cursus workspace remove alice --backup

# Remove without confirmation
cursus workspace remove alice --yes
```

### promote

cursus workspace promote _workspace_path_ [_options_]

Promote artifacts from workspace to target environment.

**Parameters:**
- **workspace_path** (_Path_) – Path to the workspace to promote from

**Options:**
- **--target** (_str_) – Target environment (default: staging)
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--output** (_str_) – Output format (json, text)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Promote to staging
cursus workspace promote ./development/developers/alice

# Promote to production
cursus workspace promote ./development/developers/alice --target production

# JSON output
cursus workspace promote ./development/developers/alice --output json
```

### health

cursus workspace health [_options_]

Get overall system health report for all workspaces.

**Options:**
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--output** (_str_) – Output format (json, text)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# System health report
cursus workspace health

# JSON output for monitoring
cursus workspace health --output json
```

### cleanup

cursus workspace cleanup [_options_]

Clean up inactive workspaces based on activity threshold.

**Options:**
- **--inactive-days** (_int_) – Days of inactivity before cleanup (default: 30)
- **--dry-run/--no-dry-run** (_bool_) – Show what would be cleaned without doing it (default: --dry-run)
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--output** (_str_) – Output format (json, text)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Dry run cleanup (default)
cursus workspace cleanup

# Cleanup workspaces inactive for 60 days
cursus workspace cleanup --inactive-days 60

# Actually perform cleanup
cursus workspace cleanup --no-dry-run

# JSON output
cursus workspace cleanup --output json
```

### discover

cursus workspace discover _component_type_ [_options_]

Discover components across workspaces with filtering and detailed information.

**Parameters:**
- **component_type** (_str_) – Type of components to discover (components, pipelines, scripts, builders, configs, contracts, specs)

**Options:**
- **--workspace** (_Optional[str]_) – Target workspace (if not specified, searches all)
- **--type-filter** (_Optional[str]_) – Component type filter
- **--format** (_str_) – Output format (table, json)
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--show-details** (_bool_) – Show detailed component information

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Discover all components
cursus workspace discover components

# Discover builders in specific workspace
cursus workspace discover builders --workspace alice

# Discover with details
cursus workspace discover scripts --show-details

# JSON output
cursus workspace discover configs --format json
```

### build

cursus workspace build _pipeline_name_ [_options_]

Build pipeline using cross-workspace components with validation.

**Parameters:**
- **pipeline_name** (_str_) – Name of the pipeline to build

**Options:**
- **--workspace** (_Optional[str]_) – Primary workspace for the pipeline
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--cross-workspace** (_bool_) – Enable cross-workspace component usage
- **--output-path** (_Optional[str]_) – Output path for generated pipeline
- **--dry-run** (_bool_) – Show what would be built without executing
- **--validate/--no-validate** (_bool_) – Validate pipeline before building (default: --validate)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Build pipeline from specific workspace
cursus workspace build my_pipeline --workspace alice

# Build with cross-workspace components
cursus workspace build my_pipeline --workspace alice --cross-workspace

# Dry run to see build plan
cursus workspace build my_pipeline --workspace alice --dry-run

# Build with custom output path
cursus workspace build my_pipeline --workspace alice --output-path ./pipelines/
```

### test-compatibility

cursus workspace test-compatibility [_options_]

Test compatibility between workspace components.

**Options:**
- **--source-workspace** (_str_) – Source workspace (required)
- **--target-workspace** (_str_) – Target workspace (required)
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--component-type** (_Optional[str]_) – Specific component type to test
- **--format** (_str_) – Output format (text, json)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Test overall compatibility
cursus workspace test-compatibility --source-workspace alice --target-workspace bob

# Test specific component type
cursus workspace test-compatibility --source-workspace alice --target-workspace bob --component-type builders

# JSON output
cursus workspace test-compatibility --source-workspace alice --target-workspace bob --format json
```

### merge

cursus workspace merge _source_workspace_ _target_workspace_ [_options_]

Merge components between workspaces with different strategies.

**Parameters:**
- **source_workspace** (_str_) – Source workspace to merge from
- **target_workspace** (_str_) – Target workspace to merge into

**Options:**
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--component-type** (_Optional[str]_) – Specific component type to merge
- **--component-name** (_Optional[str]_) – Specific component to merge
- **--strategy** (_str_) – Merge strategy (copy, link, reference)
- **--dry-run** (_bool_) – Show what would be merged without executing
- **--yes** (_bool_) – Skip confirmation prompt

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Merge all components
cursus workspace merge alice bob

# Merge specific component type
cursus workspace merge alice bob --component-type builders

# Merge with link strategy
cursus workspace merge alice bob --strategy link

# Dry run merge
cursus workspace merge alice bob --dry-run
```

### test-runtime

cursus workspace test-runtime _test_type_ [_options_]

Run workspace-aware runtime tests with isolation and cross-workspace support.

**Parameters:**
- **test_type** (_str_) – Type of runtime test to execute (script, pipeline, component, integration)

**Options:**
- **--workspace** (_Optional[str]_) – Target workspace for testing
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--test-name** (_Optional[str]_) – Specific test or component name
- **--cross-workspace** (_bool_) – Enable cross-workspace testing
- **--isolation-mode** (_str_) – Workspace isolation mode (strict, permissive)
- **--output-path** (_Optional[str]_) – Output path for test results
- **--format** (_str_) – Test result format (text, json, junit)

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Test workspace script
cursus workspace test-runtime script --workspace alice --test-name my_script

# Test pipeline with cross-workspace components
cursus workspace test-runtime pipeline --workspace alice --test-name my_pipeline --cross-workspace

# Integration testing
cursus workspace test-runtime integration --workspace alice

# JUnit output for CI/CD
cursus workspace test-runtime script --workspace alice --format junit --output-path test_results.xml
```

### validate-alignment

cursus workspace validate-alignment [_options_]

Validate workspace component alignment across multiple levels.

**Options:**
- **--workspace** (_Optional[str]_) – Target workspace for validation
- **--workspace-root** (_str_) – Root directory for workspaces (default: ./development)
- **--validation-level** (_str_) – Validation level (script_contract, contract_spec, spec_dependency, builder_config, all)
- **--cross-workspace** (_bool_) – Include cross-workspace validation
- **--strict-mode** (_bool_) – Enable strict validation mode
- **--output-path** (_Optional[str]_) – Output path for validation report
- **--format** (_str_) – Validation report format (text, json, html)
- **--fix-issues** (_bool_) – Attempt to fix detected alignment issues

**Returns:**
- **int** – Exit code (0 for success, 1 for error)

```bash
# Validate all alignment levels
cursus workspace validate-alignment --workspace alice

# Validate specific level
cursus workspace validate-alignment --workspace alice --validation-level script_contract

# Cross-workspace validation
cursus workspace validate-alignment --workspace alice --cross-workspace

# Generate HTML report
cursus workspace validate-alignment --workspace alice --format html --output-path report.html

# Validate and fix issues
cursus workspace validate-alignment --workspace alice --fix-issues
```

## Integration Points

### WorkspaceAPI Integration
The CLI integrates with the Phase 4 WorkspaceAPI for workspace lifecycle management, status monitoring and health checks, artifact promotion workflows, and system-wide health reporting.

### Component Registry Integration
Leverages WorkspaceComponentRegistry for cross-workspace component discovery, component compatibility testing, and merge operation planning and execution.

### Validation Framework Integration
Integrates with UnifiedValidationCore for multi-level alignment validation, cross-workspace compatibility testing, and automated issue detection and fixing.

### Runtime Testing Integration
Connects with runtime testing framework for workspace-aware script testing, pipeline execution validation, integration testing scenarios, and isolation boundary verification.

## Configuration

### Workspace Root Configuration
The workspace root directory can be configured via command-line option `--workspace-root PATH`, environment variable `CURSUS_WORKSPACE_ROOT`, or defaults to `./development`.

### Output Formats
Supported output formats include text (human-readable console output), json (machine-readable JSON format), table (formatted table output), html (HTML reports for validation commands), and junit (JUnit XML format for testing commands).

## Error Handling

### Exit Codes
- **0** – Success
- **1** – General error or validation failure
- **2** – Configuration error
- **3** – Workspace not found
- **4** – Permission error

### Error Categories
- **Workspace Errors** – Missing or invalid workspace structure
- **Validation Errors** – Component alignment or compatibility issues
- **Runtime Errors** – Test execution or pipeline build failures
- **System Errors** – File system or permission issues

## Related Documentation

- [Workspace API](../workspace/api.md) - Phase 4 WorkspaceAPI for lifecycle management
- [Workspace Core](../workspace/core.md) - Core workspace management components
- [Workspace Validation](../workspace/validation.md) - Workspace validation framework
- [Registry CLI](registry_cli.md) - Registry management tools
- [Runtime Testing CLI](runtime_testing_cli.md) - Runtime testing framework
