---
tags:
  - entry_point
  - code
  - cli
  - overview
  - documentation
keywords:
  - CLI overview
  - command line interface
  - cursus CLI
  - testing tools
  - validation tools
  - pipeline compilation
topics:
  - CLI architecture
  - command overview
  - usage guide
  - tool integration
language: python
date of note: 2025-08-07
---

# Cursus CLI Documentation Overview

## Introduction

The Cursus CLI provides a comprehensive command-line interface for the AutoPipe system, offering tools for pipeline compilation, validation, testing, and project management. Built with modern CLI best practices, it serves as the primary user interaction point for developers working with step builders and pipeline automation.

## Architecture Overview

### Module Structure

```
src/cursus/cli/
├── __init__.py          # Main CLI with Click framework integration
├── __main__.py          # Entry point for python -m execution
├── builder_test_cli.py  # Universal Step Builder Testing CLI
└── validation_cli.py    # Naming and interface validation CLI
```

### Command Hierarchy

```
cursus (main CLI)
├── compile              # DAG to SageMaker pipeline compilation
├── validate             # DAG structure and compatibility validation
├── preview              # Compilation preview without full generation
├── list-steps           # Available step types listing
├── init                 # Project template generation
└── test (subgroup)      # Universal Step Builder Testing
    ├── all              # Complete test suite execution
    ├── level <1-4>      # Level-specific testing (Interface/Spec/Path/Integration)
    ├── variant <type>   # Variant-specific testing (Processing/Training/etc.)
    └── list-builders    # Available builder discovery
```

## Core Components

### 1. Main CLI Interface (`__init__.py`)
**Purpose**: Primary command-line interface with pipeline operations
**Framework**: Click-based hierarchical command structure
**Key Features**:
- DAG compilation to SageMaker pipelines
- Pre-compilation validation and preview
- Project template generation
- Integration with testing framework

**Documentation**: [CLI Main Interface](cli_main_interface.md)

### 2. Entry Point Module (`__main__.py`)
**Purpose**: Standard Python module execution entry point
**Pattern**: Follows Python `-m` execution pattern
**Key Features**:
- Clean module invocation via `python -m cursus.cli`
- Proper error propagation and exit codes
- Cross-platform compatibility

**Documentation**: [CLI Entry Point](cli_entry_point.md)

### 3. Builder Test CLI (`builder_test_cli.py`)
**Purpose**: Universal Step Builder Testing interface
**Architecture**: 4-level testing hierarchy with variant support
**Key Features**:
- Automatic builder discovery with AST parsing fallback
- Level-specific testing (Interface/Specification/Path Mapping/Integration)
- Variant-specific testing (Processing/Training/Transform)
- Comprehensive result reporting with suggestions

**Documentation**: [Builder Test CLI](builder_test_cli.md)

### 4. Validation CLI (`validation_cli.py`)
**Purpose**: Naming standards and interface compliance validation
**Scope**: Registry, file naming, step naming, logical naming, interface compliance
**Key Features**:
- Automated naming convention enforcement
- Interface compliance validation
- Registry-wide validation capabilities
- Detailed violation reporting with suggestions

**Documentation**: [Validation CLI](validation_cli.md)

## Key Features

### 1. Pipeline Operations
```bash
# Compile DAG to SageMaker pipeline
cursus compile my_dag.py --name fraud-detection --output pipeline.json

# Validate DAG before compilation
cursus validate my_dag.py --config config.yaml

# Preview compilation results
cursus preview my_dag.py
```

### 2. Universal Step Builder Testing
```bash
# Discover available builders
cursus test list-builders

# Run complete test suite
cursus test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder

# Run specific test levels
cursus test level 1 <builder_class>  # Interface tests
cursus test level 2 <builder_class>  # Specification tests
cursus test level 3 <builder_class>  # Path mapping tests
cursus test level 4 <builder_class>  # Integration tests

# Run variant-specific tests
cursus test variant processing <builder_class>
```

### 3. Standards Validation
```bash
# Validate all registry entries
cursus validate registry

# Validate specific file names
cursus validate file builder_xgboost_training_step.py builder

# Validate step names
cursus validate step XGBoostTraining

# Validate interface compliance
cursus validate interface <builder_class>
```

### 4. Project Management
```bash
# Generate new project from template
cursus init --template xgboost --name fraud-detection

# List available step types
cursus list-steps
```

## Usage Patterns

### Development Workflow

#### 1. Project Setup
```bash
# Create new project
cursus init --template xgboost --name my-project
cd my-project

# Validate project structure
cursus validate registry
```

#### 2. Builder Development
```bash
# Discover available builders
cursus test list-builders

# Test new builder implementation
cursus test level 1 <new_builder>  # Interface compliance
cursus test level 2 <new_builder>  # Specification integration
cursus test all <new_builder>      # Complete validation
```

#### 3. Pipeline Development
```bash
# Validate DAG structure
cursus validate dags/main.py

# Preview compilation
cursus preview dags/main.py --config config/config.yaml

# Compile to SageMaker pipeline
cursus compile dags/main.py --name my-project --output pipeline.json
```

### Continuous Integration

#### Pre-commit Validation
```bash
#!/bin/bash
# Validate naming standards
cursus validate registry || exit 1

# Test all builders
for builder in $(cursus test list-builders); do
    cursus test all "$builder" || exit 1
done
```

#### Pipeline Validation
```bash
#!/bin/bash
# Validate all DAG files
for dag in dags/*.py; do
    cursus validate "$dag" || exit 1
done
```

### Quality Assurance

#### Code Review Process
```bash
# Validate new builder
cursus validate interface <new_builder> --verbose
cursus test all <new_builder> --verbose

# Validate naming conventions
cursus validate file <new_file> builder
cursus validate step <new_step_name>
```

## Integration Points

### 1. Universal Test Framework
- **Direct integration**: Seamless integration with UniversalStepBuilderTestBase
- **4-level architecture**: Complete support for hierarchical testing
- **Variant testing**: Specialized testing for different step types
- **Result aggregation**: Comprehensive result collection and reporting

### 2. Validation Framework
- **Naming standards**: Automated enforcement of naming conventions
- **Interface compliance**: Validation of step builder interfaces
- **Registry validation**: Cross-reference validation of registry entries
- **Standards enforcement**: Consistent application of coding standards

### 3. Pipeline Compilation
- **DAG processing**: Integration with DAG compilation engine
- **Configuration management**: Support for configuration-driven compilation
- **Template system**: Project template generation and management
- **Output formatting**: Multiple output format support

### 4. Development Tools
- **Builder discovery**: Automatic detection of available builders
- **Error reporting**: Comprehensive error reporting with suggestions
- **Verbose modes**: Detailed diagnostic information
- **Exit codes**: Proper exit codes for automation

## Best Practices

### Command Design
- **Consistent naming**: Follow established CLI naming conventions
- **Clear help text**: Provide comprehensive help and usage examples
- **Proper validation**: Validate inputs before processing
- **Meaningful errors**: Provide actionable error messages

### User Experience
- **Progress indication**: Show progress for long-running operations
- **Verbose modes**: Provide detailed output when requested
- **Exit codes**: Use appropriate exit codes for scripting
- **Documentation**: Include usage examples in help text

### Error Handling
- **Graceful degradation**: Handle missing dependencies gracefully
- **Clear messaging**: Provide clear, actionable error messages
- **Context preservation**: Maintain error context for debugging
- **Recovery guidance**: Offer suggestions for error resolution

## Extension Points

### Adding New Commands
```python
@main.command()
@click.argument('input_file')
@click.option('--output', '-o', help='Output file')
def new_command(input_file, output):
    """New command description."""
    # Implementation
```

### Adding New Test Types
```python
# In builder_test_cli.py
test_classes = {
    1: InterfaceTests,
    2: SpecificationTests,
    3: PathMappingTests,
    4: IntegrationTests,
    5: NewTestType,  # New test level
}
```

### Adding New Validation Types
```python
# In validation_cli.py
validation_parser = subparsers.add_parser(
    "new_validation",
    help="New validation type"
)
```

## Performance Considerations

### Optimization Strategies
- **Lazy loading**: Import modules only when needed
- **Caching**: Cache results for repeated operations
- **Parallel processing**: Process multiple operations concurrently
- **Memory management**: Efficient memory usage for large operations

### Scalability
- **Batch operations**: Support for bulk operations
- **Streaming**: Stream results for large datasets
- **Resource monitoring**: Monitor resource usage
- **Optimization hooks**: Hooks for performance optimization

## Future Enhancements

### Planned Features
- **Interactive mode**: Interactive CLI for guided operations
- **Configuration management**: Enhanced configuration handling
- **Plugin system**: Support for third-party extensions
- **Shell completion**: Auto-completion for commands and arguments

### Integration Improvements
- **IDE integration**: Better integration with development environments
- **CI/CD plugins**: Specialized integrations for CI/CD systems
- **Monitoring**: Integration with monitoring and telemetry
- **Notification systems**: Result notification capabilities

## Getting Started

### Installation
```bash
# Install from source
pip install -e .

# Or install from PyPI (when available)
pip install cursus
```

### Basic Usage
```bash
# Get help
cursus --help

# List available builders
cursus test list-builders

# Test a builder
cursus test all <builder_class>

# Validate naming standards
cursus validate registry
```

### Advanced Usage
```bash
# Compile a pipeline
cursus compile my_dag.py --name my-pipeline --config config.yaml

# Generate a new project
cursus init --template xgboost --name fraud-detection

# Run comprehensive validation
cursus validate registry --verbose
cursus test all <builder_class> --verbose
```

## Troubleshooting

### Common Issues
- **Module not found**: Ensure proper installation or PYTHONPATH
- **Import errors**: Check dependencies and package structure
- **Permission issues**: Verify file permissions and execution rights
- **Configuration errors**: Validate configuration file format and content

### Debug Mode
```bash
# Enable verbose output
cursus --verbose <command>

# Debug specific operations
cursus test all <builder_class> --verbose
cursus validate interface <builder_class> --verbose
```

This CLI system provides a comprehensive, user-friendly interface for all aspects of the AutoPipe development workflow, from initial project creation through testing, validation, and pipeline compilation.
