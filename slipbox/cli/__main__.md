---
tags:
  - code
  - cli
  - entry-point
  - main
  - module-execution
keywords:
  - __main__
  - main entry point
  - module execution
  - CLI dispatcher
topics:
  - command-line interface
  - module execution
  - entry point
language: python
date of note: 2024-12-07
---

# CLI Main Entry Point

Main entry point for the cursus CLI module when executed as `python -m cursus.cli`.

## Overview

The `__main__.py` module serves as the entry point when the cursus CLI package is executed as a module using `python -m cursus.cli`. It provides a simple interface that delegates to the main CLI dispatcher function, enabling users to access all cursus command-line tools through the module execution pattern.

This module follows Python's standard convention for making packages executable as modules, providing a clean and consistent way to access the cursus CLI functionality without requiring direct script execution.

## Classes and Methods

### Functions
- [`main`](#main) - Entry point that delegates to the CLI dispatcher

## API Reference

### main

Entry point function that imports and calls the main CLI dispatcher when the module is executed.

```python
# Executed when running: python -m cursus.cli
if __name__ == '__main__':
    main()
```

## Usage Examples

### Module Execution

```bash
# Execute cursus CLI as a module
python -m cursus.cli --help

# Run specific commands through module execution
python -m cursus.cli alignment validate my_script --verbose
python -m cursus.cli builder-test all MyBuilder --scoring
python -m cursus.cli catalog find --tags training
python -m cursus.cli registry list-steps
python -m cursus.cli runtime-testing test_script my_script.py
python -m cursus.cli validation registry
python -m cursus.cli workspace setup --project my_project
```

### Alternative Execution Methods

```bash
# Direct module execution (equivalent to above)
python -m cursus.cli

# Using the installed package entry point
cursus --help  # If package provides console scripts

# Direct script execution (if available)
python src/cursus/cli/__main__.py
```

## Implementation Details

The `__main__.py` module contains minimal code that simply imports and calls the main dispatcher:

```python
#!/usr/bin/env python3
"""
Main entry point for the cursus CLI module.
"""

from . import main

if __name__ == '__main__':
    main()
```

This design pattern:
- **Delegates Responsibility**: All CLI logic remains in the main `__init__.py` module
- **Follows Conventions**: Uses Python's standard `__main__.py` pattern for executable modules
- **Maintains Simplicity**: Minimal code reduces maintenance overhead
- **Enables Flexibility**: Allows for easy modification of entry point behavior

## Integration Points

- **CLI Dispatcher**: Calls the main dispatcher function from `cursus.cli`
- **Command Routing**: Inherits all command routing and argument parsing from the main CLI
- **Error Handling**: Benefits from the comprehensive error handling in the main CLI
- **Help System**: Provides access to all help documentation and command examples

## Related Documentation

- [CLI Module](__init__.md) - Main CLI dispatcher and command routing
- [Alignment CLI](alignment_cli.md) - Comprehensive alignment validation tools
- [Builder Test CLI](builder_test_cli.md) - Step builder testing and validation
- [Catalog CLI](catalog_cli.md) - Pipeline catalog management tools
- [Registry CLI](registry_cli.md) - Registry management and workspace tools
- [Runtime Testing CLI](runtime_testing_cli.md) - Script runtime testing and benchmarking
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
- [Workspace CLI](workspace_cli.md) - Developer workspace management tools
