---
tags:
  - code
  - cli
  - entry_point
  - module_structure
  - python_packaging
keywords:
  - CLI entry point
  - module execution
  - python -m
  - package structure
  - command line interface
topics:
  - CLI architecture
  - module organization
  - package entry points
language: python
date of note: 2025-08-07
---

# CLI Entry Point Documentation

## Overview

The CLI Entry Point (`src/cursus/cli/__main__.py`) serves as the main execution entry point for the cursus CLI module. This simple but crucial file enables the CLI to be executed using Python's `-m` flag, providing a clean and standard way to invoke the command-line interface.

## Architecture

### Module Structure

```
src/cursus/cli/
├── __init__.py          # Main CLI implementation with Click commands
├── __main__.py          # Entry point for python -m execution
├── builder_test_cli.py  # Step builder testing CLI
└── validation_cli.py    # Validation CLI tools
```

### Entry Point Pattern

The `__main__.py` file follows Python's standard entry point pattern:

```python
#!/usr/bin/env python3
"""
Main entry point for the cursus CLI module.
"""

from . import main

if __name__ == '__main__':
    main()
```

## Key Features

### 1. Standard Python Module Execution
```bash
python -m cursus.cli
```
- **Standard pattern**: Follows Python's established `-m` execution pattern
- **Clean invocation**: Provides clean command-line invocation
- **Package integration**: Integrates seamlessly with Python packaging
- **Cross-platform**: Works consistently across different platforms

### 2. Delegation to Main CLI
- **Simple delegation**: Delegates execution to the main CLI function
- **Import isolation**: Isolates import logic from execution logic
- **Error propagation**: Properly propagates errors and exit codes
- **Clean separation**: Separates entry point from implementation

### 3. Shebang Support
```bash
#!/usr/bin/env python3
```
- **Direct execution**: Supports direct script execution on Unix systems
- **Environment detection**: Uses `env` for Python interpreter detection
- **Version specification**: Specifies Python 3 requirement
- **Portability**: Maintains portability across different systems

## Implementation Details

### Entry Point Mechanics

#### Module Discovery
- **Python -m flag**: Leverages Python's module execution mechanism
- **Package structure**: Relies on proper package structure with `__init__.py`
- **Import resolution**: Uses relative imports to access main CLI
- **Path independence**: Works regardless of current working directory

#### Execution Flow
1. **Python invocation**: `python -m cursus.cli` triggers module execution
2. **Module loading**: Python loads the `cursus.cli` package
3. **__main__.py execution**: Python executes `__main__.py` as the main module
4. **Function delegation**: `__main__.py` imports and calls `main()` from `__init__.py`
5. **CLI execution**: Main CLI function executes with command-line arguments

### Error Handling

#### Import Error Management
- **Graceful failure**: Handles import failures gracefully
- **Error propagation**: Propagates import errors to the user
- **Dependency checking**: Implicitly checks for required dependencies
- **Clear messaging**: Provides clear error messages for missing components

#### Exit Code Handling
- **Proper propagation**: Propagates exit codes from main CLI
- **Standard compliance**: Follows standard Unix exit code conventions
- **Error indication**: Properly indicates success (0) or failure (non-zero)
- **Automation support**: Supports automated scripting and CI/CD

## Usage Patterns

### Development Usage
```bash
# During development
cd /path/to/cursus
python -m src.cursus.cli --help

# After installation
python -m cursus.cli --help
```

### Production Usage
```bash
# Installed package
python -m cursus.cli compile my_dag.py
python -m cursus.cli test all <builder_class>
python -m cursus.cli validate registry
```

### Scripting Integration
```bash
#!/bin/bash
# CI/CD script example
python -m cursus.cli validate registry || exit 1
python -m cursus.cli test all <builder_class> || exit 1
```

## Integration Points

### 1. Package Structure Integration
- **Package hierarchy**: Integrates with overall package structure
- **Import system**: Uses Python's import system correctly
- **Module discovery**: Leverages Python's module discovery mechanism
- **Namespace management**: Maintains proper namespace isolation

### 2. CLI Framework Integration
- **Click framework**: Integrates with Click-based CLI in `__init__.py`
- **Command delegation**: Delegates all command processing to main CLI
- **Context preservation**: Maintains CLI context and state
- **Option handling**: Preserves command-line option handling

### 3. Packaging Integration
- **setuptools integration**: Works with setuptools entry points
- **pip installation**: Supports pip-based installation
- **Distribution packaging**: Compatible with Python distribution packaging
- **Virtual environment**: Works correctly in virtual environments

## Best Practices

### Entry Point Design
- **Minimal implementation**: Keep entry point minimal and focused
- **Clear delegation**: Delegate to main implementation clearly
- **Error handling**: Handle errors appropriately
- **Documentation**: Provide clear documentation for usage

### Module Organization
- **Separation of concerns**: Separate entry point from implementation
- **Import management**: Manage imports efficiently
- **Namespace clarity**: Maintain clear namespace organization
- **Dependency isolation**: Isolate dependencies appropriately

## Comparison with Alternative Approaches

### Console Scripts Entry Point
```python
# setup.py or pyproject.toml
entry_points = {
    'console_scripts': [
        'cursus=cursus.cli:main',
    ],
}
```
- **Pros**: Creates dedicated executable, simpler invocation
- **Cons**: Requires installation, less flexible for development

### Direct Script Execution
```python
# cursus_cli.py
if __name__ == '__main__':
    main()
```
- **Pros**: Simple, direct execution
- **Cons**: Not integrated with package structure, harder to distribute

### Module Entry Point (Current Approach)
```python
# __main__.py
from . import main
if __name__ == '__main__':
    main()
```
- **Pros**: Standard Python pattern, works in development and production
- **Cons**: Slightly more complex invocation

## Development Workflow

### Local Development
```bash
# Run from source
cd /path/to/cursus
python -m src.cursus.cli --help

# Test specific commands
python -m src.cursus.cli test list-builders
python -m src.cursus.cli validate registry
```

### Testing and Debugging
```bash
# Debug mode
python -m src.cursus.cli --verbose compile my_dag.py

# Error investigation
python -c "import cursus.cli; cursus.cli.main()" --help
```

### Installation Testing
```bash
# After pip install
python -m cursus.cli --version
python -m cursus.cli --help
```

## Future Enhancements

### Planned Improvements
- **Enhanced error handling**: More sophisticated error handling and reporting
- **Configuration support**: Support for configuration file-based execution
- **Plugin system**: Support for plugin-based CLI extensions
- **Interactive mode**: Support for interactive CLI modes

### Integration Enhancements
- **IDE integration**: Better integration with development environments
- **Shell completion**: Support for shell completion systems
- **Logging integration**: Enhanced logging and debugging support
- **Monitoring integration**: Integration with monitoring and telemetry systems

## Troubleshooting

### Common Issues

#### Module Not Found
```bash
# Error: No module named 'cursus.cli'
# Solution: Ensure package is installed or PYTHONPATH is set correctly
pip install -e .
# or
export PYTHONPATH=/path/to/cursus/src:$PYTHONPATH
```

#### Import Errors
```bash
# Error: ImportError in __main__.py
# Solution: Check package structure and __init__.py files
ls -la src/cursus/cli/
```

#### Permission Issues
```bash
# Error: Permission denied
# Solution: Check file permissions
chmod +x src/cursus/cli/__main__.py
```

This entry point module provides a clean, standard way to invoke the cursus CLI while maintaining proper separation between entry point logic and implementation details, following Python best practices for package organization and command-line tool distribution.
