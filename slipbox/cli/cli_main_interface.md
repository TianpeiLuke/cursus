---
tags:
  - code
  - cli
  - interface
  - cursus
  - main_entry_point
keywords:
  - command line interface
  - CLI
  - cursus
  - click framework
  - pipeline compilation
  - DAG validation
  - step builder testing
topics:
  - CLI architecture
  - command structure
  - pipeline operations
  - testing framework
language: python
date of note: 2025-08-07
---

# CLI Main Interface Documentation

## Overview

The main CLI interface (`src/cursus/cli/__init__.py`) provides the primary command-line entry point for the Cursus system. Built using the Click framework, it offers a comprehensive set of commands for pipeline compilation, validation, testing, and project management.

## Architecture

### Command Structure

The CLI follows a hierarchical command structure:

```
cursus (main group)
├── compile          # Compile DAG to SageMaker pipeline
├── validate         # Validate DAG compatibility
├── preview          # Preview compilation results
├── list-steps       # List available step types
├── init             # Generate project templates
└── test (subgroup)  # Universal Step Builder Testing
    ├── all          # Run all tests
    ├── level        # Run specific level tests
    ├── variant      # Run variant-specific tests
    └── list-builders # List available builders
```

### Core Components

#### 1. Main Group (`main`)
- **Purpose**: Root command group with global options
- **Features**: Version display, verbose mode
- **Global Context**: Maintains verbose flag across subcommands

#### 2. Pipeline Operations
- **`compile`**: Transforms DAG files into SageMaker pipelines
- **`validate`**: Checks DAG structure and compatibility
- **`preview`**: Shows compilation preview without full generation
- **`list-steps`**: Displays available step types

#### 3. Testing Framework Integration
- **`test` group**: Provides access to Universal Step Builder Tests
- **Multi-level testing**: Supports 4 levels of validation
- **Variant testing**: Specialized tests for different step types
- **Builder discovery**: Automatic detection of available builders

#### 4. Project Management
- **`init`**: Creates new projects from templates
- **Template support**: XGBoost, PyTorch, and basic templates
- **Project structure**: Generates DAGs, configs, and documentation

## Key Features

### 1. DAG Compilation
```bash
cursus compile my_dag.py --name fraud-detection --output pipeline.json
```
- Converts Python DAG definitions to SageMaker pipelines
- Supports JSON and YAML output formats
- Configuration file integration
- Error handling with verbose mode

### 2. Validation Engine
```bash
cursus validate my_dag.py --config config.yaml
```
- Pre-compilation validation
- Configuration compatibility checks
- Detailed error reporting
- Success/failure exit codes

### 3. Universal Testing System
```bash
cursus test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
cursus test level 1 <builder_class>
cursus test variant processing <builder_class>
```
- Integration with UniversalStepBuilderTestBase
- 4-level testing hierarchy
- Variant-specific testing
- Automatic builder discovery

### 4. Project Templates
```bash
cursus init --template xgboost --name fraud-detection
```
- Pre-configured project structures
- Multiple template types
- Automatic file generation
- Getting started documentation

## Implementation Details

### Click Framework Integration
- **Decorators**: Extensive use of Click decorators for command definition
- **Context passing**: Maintains state across command hierarchy
- **Type validation**: Built-in argument and option validation
- **Help generation**: Automatic help text and usage examples

### Error Handling
- **Exception catching**: Comprehensive error handling for all operations
- **Verbose mode**: Detailed stack traces when enabled
- **Exit codes**: Proper exit codes for scripting integration
- **User-friendly messages**: Clear error messages for common issues

### Dynamic Imports
- **Lazy loading**: Imports modules only when needed
- **Dependency management**: Graceful handling of missing dependencies
- **Module discovery**: Dynamic loading of DAG files and builder classes

## Configuration Management

### Global Options
- **`--verbose`**: Enables detailed output across all commands
- **Version display**: Shows Cursus version information
- **Context preservation**: Maintains settings across subcommands

### Command-Specific Options
- **File paths**: Consistent path handling with Click.Path
- **Output formats**: JSON/YAML format selection
- **Template selection**: Multiple project template options
- **Test levels**: Numeric level selection with validation

## Integration Points

### 1. Core API Integration
- **DAG Compiler**: Direct integration with compilation engine
- **Validation Engine**: Uses core validation components
- **Step Registry**: Accesses available step types

### 2. Testing Framework Integration
- **Builder Test CLI**: Delegates to specialized testing module
- **Universal Tests**: Full integration with 4-level testing system
- **Result Formatting**: Consistent output formatting

### 3. File System Operations
- **Project generation**: Creates directory structures and files
- **Template processing**: Generates customized project files
- **Path resolution**: Handles relative and absolute paths

## Usage Patterns

### Development Workflow
1. **Project Creation**: `cursus init --template xgboost --name my-project`
2. **DAG Development**: Edit generated DAG files
3. **Validation**: `cursus validate dags/main.py`
4. **Preview**: `cursus preview dags/main.py`
5. **Compilation**: `cursus compile dags/main.py --name my-project`

### Testing Workflow
1. **Builder Discovery**: `cursus test list-builders`
2. **Interface Testing**: `cursus test level 1 <builder>`
3. **Full Testing**: `cursus test all <builder>`
4. **Variant Testing**: `cursus test variant processing <builder>`

## Extension Points

### Adding New Commands
- **Command groups**: Easy addition of new command groups
- **Subcommands**: Simple subcommand registration
- **Option inheritance**: Automatic option inheritance from parent groups

### Template System
- **New templates**: Easy addition of project templates
- **Customization**: Template content customization
- **File generation**: Flexible file generation patterns

### Testing Integration
- **New test types**: Integration with additional test frameworks
- **Custom validators**: Addition of specialized validation logic
- **Result formatters**: Custom output formatting options

## Best Practices

### Command Design
- **Consistent naming**: Follow established naming conventions
- **Clear help text**: Provide comprehensive help and examples
- **Proper validation**: Validate inputs before processing
- **Error handling**: Provide meaningful error messages

### User Experience
- **Progress indication**: Show progress for long-running operations
- **Verbose mode**: Provide detailed output when requested
- **Exit codes**: Use appropriate exit codes for automation
- **Documentation**: Include usage examples in help text

## Future Enhancements

### Planned Features
- **Interactive mode**: Interactive DAG building
- **Pipeline monitoring**: Integration with SageMaker pipeline monitoring
- **Configuration validation**: Enhanced configuration validation
- **Plugin system**: Support for third-party extensions

### Performance Improvements
- **Lazy loading**: Further optimization of module loading
- **Caching**: Cache compilation results for faster iterations
- **Parallel processing**: Parallel validation and testing
- **Memory optimization**: Reduce memory footprint for large DAGs

This CLI interface serves as the primary user interaction point for the Cursus system, providing a comprehensive and user-friendly way to work with pipeline compilation, validation, and testing workflows.
