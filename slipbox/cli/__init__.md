---
tags:
  - code
  - cli
  - command-line
  - interface
  - entry-point
keywords:
  - main
  - CLI dispatcher
  - command routing
  - argument parsing
  - alignment
  - builder-test
  - catalog
  - registry
  - runtime-testing
  - validation
  - workspace
topics:
  - command-line interface
  - CLI tools
  - pipeline development
  - validation tools
language: python
date of note: 2024-12-07
---

# CLI Module

Command-line interfaces for the Cursus package providing comprehensive pipeline development and validation tools.

## Overview

The CLI module serves as the main entry point for all Cursus command-line tools, providing a unified interface for pipeline development, validation, and management tasks. It features a dispatcher architecture that routes commands to specialized CLI modules, comprehensive argument parsing with help documentation, and consistent error handling across all CLI tools.

The module supports seven main command categories: alignment validation tools, step builder testing tools, pipeline catalog management, registry management tools, runtime testing for pipeline scripts, naming and interface validation, and workspace management tools. Each command provides detailed help documentation and examples for effective usage.

## Functions

### main

main()

Main CLI entry point that dispatches commands to appropriate CLI modules with comprehensive argument parsing and error handling.

**Returns:**
- **int** – Exit code (0 for success, 1 for error).

```python
from cursus.cli import main

# Run CLI dispatcher
exit_code = main()
```

## CLI Commands

The CLI provides the following command groups, each documented as a function-like interface:

### alignment

cursus alignment _command_ [_options_]

Alignment validation tools for comprehensive script validation across all four levels.

**Commands:**
- **validate** (_str_) – Validate specific script alignment
- **validate-all** (_Optional[str]_) – Validate all scripts with optional output directory

**Options:**
- **--verbose** (_bool_) – Enable verbose output mode
- **--output-dir** (_Optional[str]_) – Directory for validation reports
- **--format** (_str_) – Output format (text, json, both)

```bash
# Validate specific script
cursus alignment validate my_script --verbose

# Validate all scripts with reports
cursus alignment validate-all --output-dir ./reports --format both
```

### builder-test

cursus builder-test _command_ [_options_]

Step builder testing tools for validating step builder implementations and configurations.

**Commands:**
- **validate** (_str_) – Validate specific step builder
- **validate-all** (_Optional[str]_) – Validate all step builders

**Options:**
- **--config-dir** (_Optional[str]_) – Directory containing custom configurations
- **--verbose** (_bool_) – Enable detailed output

```bash
# Test specific builder
cursus builder-test validate MyStepBuilder

# Test all builders with custom configs
cursus builder-test validate-all --config-dir ./custom_configs
```

### catalog

cursus catalog _command_ [_options_]

Pipeline catalog management tools for discovering, managing, and organizing pipeline templates.

**Commands:**
- **find** (_Optional[str]_) – Search pipelines by criteria
- **list** (_Optional[str]_) – List available pipelines
- **show** (_str_) – Show pipeline details

**Options:**
- **--tags** (_List[str]_) – Filter by tags (comma-separated)
- **--framework** (_Optional[str]_) – Filter by ML framework
- **--format** (_str_) – Output format (table, json)

```bash
# Search by tags
cursus catalog find --tags training,xgboost

# List with framework filter
cursus catalog list --framework pytorch --format json
```

### registry

cursus registry _command_ [_options_]

Registry management tools for step registration, conflict resolution, and workspace management.

**Commands:**
- **list-steps** (_Optional[str]_) – List registered steps
- **validate-registry** (_Optional[str]_) – Validate registry consistency

**Options:**
- **--workspace** (_Optional[str]_) – Target workspace context
- **--check-conflicts** (_bool_) – Enable conflict detection
- **--format** (_str_) – Output format (table, json)

```bash
# List steps in workspace
cursus registry list-steps --workspace my_project

# Validate with conflict checking
cursus registry validate-registry --workspace my_project --check-conflicts
```

### runtime-testing

cursus runtime-testing _command_ _script_path_ [_options_]

Runtime testing tools for pipeline scripts with execution validation and performance analysis.

**Commands:**
- **test_script** (_str_) – Test individual script execution
- **benchmark** (_str_) – Benchmark script performance

**Parameters:**
- **script_path** (_str_) – Path to the script file to test

**Options:**
- **--iterations** (_int_) – Number of benchmark iterations (default: 1)
- **--timeout** (_Optional[int]_) – Execution timeout in seconds
- **--verbose** (_bool_) – Enable detailed output

```bash
# Test script execution
cursus runtime-testing test_script my_script.py

# Benchmark with multiple iterations
cursus runtime-testing benchmark my_script.py --iterations 10
```

### validation

cursus validation _command_ [_target_] [_options_]

Naming and interface validation tools for ensuring consistency across pipeline components.

**Commands:**
- **registry** (_None_) – Validate all registry entries
- **file** (_str_, _str_) – Validate specific file naming
- **step** (_str_) – Validate canonical step names
- **logical** (_str_) – Validate logical names
- **interface** (_str_) – Validate interface compliance

**Parameters:**
- **target** (_Optional[str]_) – Target file, step, or class path to validate
- **file_type** (_Optional[str]_) – File type for file validation (builder, config, spec, contract)

**Options:**
- **--verbose** (_bool_) – Enable detailed violation reporting

```bash
# Validate registry
cursus validation registry

# Validate file naming
cursus validation file builder_xgboost_training_step.py builder

# Validate interface compliance
cursus validation interface src.cursus.steps.builders.MyBuilder --verbose
```

### workspace

cursus workspace _command_ [_options_]

Workspace management tools for developer environment setup and configuration.

**Commands:**
- **setup** (_Optional[str]_) – Setup new workspace
- **create** (_str_) – Create developer workspace
- **list** (_None_) – List available workspaces

**Options:**
- **--project** (_Optional[str]_) – Project name for workspace
- **--template** (_Optional[str]_) – Workspace template to use
- **--developer-id** (_Optional[str]_) – Developer identifier

```bash
# Setup workspace with template
cursus workspace setup --project my_project

# Create developer workspace
cursus workspace create --template standard --developer-id john_doe
```

## Error Handling

The CLI module provides consistent error handling across all commands:

### Exit Codes

- **0** – Success
- **1** – General error or validation failure
- **2** – Invalid arguments or usage
- **3** – File or resource not found
- **4** – Permission or access error

### Error Categories

- **Command Routing** – Validates command names and provides suggestions for typos
- **Argument Parsing** – Comprehensive argument validation with helpful error messages
- **Exception Handling** – Graceful handling of unexpected errors with debugging information
- **Help Integration** – Automatic help display for invalid usage patterns

```bash
# Get help for main CLI
cursus --help

# Get help for specific command
cursus alignment --help
cursus builder-test --help
```

## Integration Points

### Validation Framework
Integrates with cursus validation system for comprehensive validation capabilities.

### Registry System
Uses cursus registry for step management and workspace operations.

### Pipeline Catalog
Connects to pipeline catalog for template management and discovery.

### Workspace System
Integrates with workspace management tools for developer environment setup.

## Related Documentation

- [Alignment CLI](alignment_cli.md) - Comprehensive alignment validation tools
- [Builder Test CLI](builder_test_cli.md) - Step builder testing and validation
- [Catalog CLI](catalog_cli.md) - Pipeline catalog management tools
- [Registry CLI](registry_cli.md) - Registry management and workspace tools
- [Runtime Testing CLI](runtime_testing_cli.md) - Script runtime testing and benchmarking
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
- [Workspace CLI](workspace_cli.md) - Developer workspace management tools
