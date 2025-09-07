---
tags:
  - entry_point
  - cli
  - command-line
  - interface
  - documentation
keywords:
  - CLI tools
  - command-line interface
  - cursus CLI
  - alignment validation
  - builder testing
  - catalog management
  - registry management
  - runtime testing
  - validation tools
  - workspace management
topics:
  - command-line interface
  - CLI documentation
  - pipeline development
  - validation tools
  - developer tools
language: python
date of note: 2024-12-07
---

# CLI Documentation

Comprehensive documentation for all Cursus command-line interface tools, providing unified access to pipeline development, validation, and management capabilities.

## Overview

The CLI documentation directory contains complete reference documentation for all Cursus command-line tools. These tools provide a comprehensive interface for pipeline development, validation, and management tasks, featuring a unified dispatcher architecture, consistent argument parsing, and robust error handling across all CLI modules.

The CLI system supports multiple command categories including alignment validation, step builder testing, pipeline catalog management, registry management, runtime testing, naming and interface validation, and workspace management. All documentation follows the API reference documentation style guide with function-like command documentation, proper parameter formatting, and comprehensive code examples.

## Documentation Files

### Core CLI Components

- **[CLI Module](__init__.md)** - Main CLI dispatcher and command routing system
- **[CLI Main Entry Point](__main__.md)** - Entry point for module execution (`python -m cursus.cli`)

### Command-Specific Documentation

- **[Alignment CLI](alignment_cli.md)** - Comprehensive alignment validation across four levels
- **[Builder Test CLI](builder_test_cli.md)** - Step builder testing and validation tools
- **[Catalog CLI](catalog_cli.md)** - Pipeline catalog management with dual-compiler support
- **[Registry CLI](registry_cli.md)** - Registry management and workspace tools
- **[Runtime Testing CLI](runtime_testing_cli.md)** - Script runtime testing and benchmarking
- **[Validation CLI](validation_cli.md)** - Naming and interface validation tools
- **[Workspace CLI](workspace_cli.md)** - Comprehensive workspace lifecycle management

## Quick Start

### Installation and Basic Usage

```bash
# Access main CLI help
cursus --help

# Or use module execution
python -m cursus.cli --help
```

### Common Command Patterns

```bash
# Alignment validation
cursus alignment validate my_script --verbose
cursus alignment validate-all --output-dir ./reports

# Builder testing
cursus builder-test validate MyStepBuilder
cursus builder-test validate-all --scoring

# Catalog management
cursus catalog list --format json
cursus catalog search --framework xgboost

# Registry operations
cursus registry list-steps --workspace my_project
cursus registry validate-registry --check-conflicts

# Runtime testing
cursus runtime-testing test_script my_script.py
cursus runtime-testing benchmark my_script.py --iterations 10

# Validation tools
cursus validation registry --verbose
cursus validation interface src.cursus.steps.builders.MyBuilder

# Workspace management
cursus workspace create alice --template ml_pipeline
cursus workspace list --show-components
```

## Documentation Standards

### API Reference Format

All CLI documentation follows the API reference documentation style guide:

- **YAML Frontmatter** - Proper tags, keywords, topics, language, and date fields
- **Overview Sections** - Comprehensive module descriptions with context
- **Function-like Commands** - CLI commands documented as functions with signatures
- **Parameter Documentation** - Consistent `**--option** (_Type_) – Description` format
- **Code Examples** - Practical bash usage examples for all commands
- **Integration Points** - Clear documentation of system integrations
- **Error Handling** - Standardized exit codes and error categories

### Command Documentation Structure

Each CLI command is documented with:

1. **Command Signature** - `cursus command subcommand _parameter_ [_options_]`
2. **Parameters Section** - Required parameters with types and descriptions
3. **Options Section** - Optional flags and parameters with defaults
4. **Returns Section** - Exit codes and their meanings
5. **Code Examples** - Practical usage scenarios
6. **Integration Notes** - How the command integrates with the framework

## CLI Architecture

### Dispatcher Pattern

The CLI system uses a dispatcher pattern for command routing:

1. **Main Dispatcher** - Routes commands to appropriate CLI modules
2. **Argument Parsing** - Comprehensive argument validation with help
3. **Command Forwarding** - Forwards arguments to selected CLI modules
4. **Exit Code Handling** - Preserves exit codes from sub-commands
5. **Error Management** - Consistent error handling across all commands

### Integration Points

- **Validation Framework** - Comprehensive validation capabilities
- **Registry System** - Step management and workspace operations
- **Pipeline Catalog** - Template management and discovery
- **Workspace System** - Developer environment setup and management
- **Builder Framework** - Step builder testing and validation

## Error Handling

### Standardized Exit Codes

- **0** - Success
- **1** - General error or validation failure
- **2** - Invalid arguments or usage
- **3** - File or resource not found
- **4** - Permission or access error

### Error Categories

- **Command Routing** - Invalid command names with suggestions
- **Argument Parsing** - Comprehensive argument validation
- **Exception Handling** - Graceful handling with debugging information
- **Help Integration** - Automatic help display for invalid usage

## Development Workflow Integration

### Pre-commit Validation

```bash
# Validate alignment before commit
cursus alignment validate-all --continue-on-error

# Validate naming standards
cursus validation registry

# Test builders
cursus builder-test validate-all
```

### Continuous Integration

```bash
# CI pipeline validation
cursus alignment validate-all --format json --output-dir ./ci-reports
cursus validation registry || exit 1
cursus builder-test validate-all --scoring || exit 1
```

### Development Environment Setup

```bash
# Setup workspace
cursus workspace create developer_name --template ml_pipeline

# Validate workspace
cursus workspace validate --strict

# List available tools
cursus workspace discover components
```

## Advanced Usage

### Batch Operations

```bash
# Generate multiple pipeline reports
cursus alignment validate-all --output-dir ./reports --format both

# Test all builders with custom configurations
cursus builder-test validate-all --config-dir ./custom_configs

# Comprehensive workspace health check
cursus workspace health-check --fix-issues
```

### Automation and Scripting

```bash
# JSON output for programmatic processing
cursus catalog list --format json | jq '.[] | .id'
cursus workspace list --format json | jq '.[] | select(.status == "healthy")'

# Automated validation workflows
cursus alignment validate-all --continue-on-error --output-dir ./validation
cursus validation registry --verbose > validation_report.txt
```

## Performance Considerations

### Optimization Strategies

- **Lazy Loading** - CLI modules loaded only when needed
- **Caching** - Validation results cached for repeated operations
- **Parallel Processing** - Multiple validations processed concurrently
- **Incremental Operations** - Only changed components validated when possible

### Best Practices

1. **Use specific filters** to reduce processing time
2. **Cache results** for batch operations
3. **Use JSON output** for programmatic processing
4. **Validate incrementally** during development
5. **Generate reports** for comprehensive analysis

## Contributing

### Adding New CLI Commands

1. Create new CLI module in `src/cursus/cli/`
2. Add command routing in main dispatcher
3. Create documentation following API reference style guide
4. Add integration tests and examples
5. Update this README with new command information

### Documentation Standards

- Follow API reference documentation style guide
- Include comprehensive code examples
- Document all parameters and options
- Provide integration points and error handling
- Maintain consistent formatting and structure

## Related Documentation

- [API Reference Documentation Style Guide](../../1_design/api_reference_documentation_style_guide.md) - Documentation standards and formatting
- [Developer Guide](../../0_developer_guide/README.md) - Development guidelines and best practices
- [Validation Framework](../validation/README.md) - Validation system documentation
- [Registry System](../registry/README.md) - Registry management documentation
- [Workspace Management](../workspace/README.md) - Workspace system documentation
