# Workspace Setup Guide for Isolated Project Development

**Version**: 1.0.0  
**Date**: September 5, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides comprehensive instructions for setting up isolated project workspaces for pipeline step development. Isolated project development allows developers to work in dedicated `development/projects/*/src/cursus_dev/` environments with hybrid registry access to both shared (`src/cursus`) and project-specific code.

## Prerequisites

Before setting up an isolated project workspace, ensure you have:

1. **Access to the main cursus repository** with the hybrid registry system
2. **CLI tools installed** with workspace management capabilities
3. **Understanding of the 6-layer architecture** (Step Specifications, Script Contracts, Processing Scripts, Step Builders, Configuration Classes, Hyperparameters)
4. **Basic familiarity** with the UnifiedRegistryManager system

## Workspace Setup Options

### Option 1: Create New Isolated Project

For completely new projects that need isolated development environments:

```bash
# Create new isolated project workspace
cursus init-workspace --project your_project_name --type isolated

# Navigate to the created project directory
cd development/projects/your_project_name

# Verify workspace creation
cursus list-workspaces
cursus current-workspace
```

### Option 2: Initialize Existing Project Directory

For existing project directories that need workspace capabilities:

```bash
# Navigate to existing project directory
cd development/projects/existing_project

# Initialize workspace in existing directory
cursus init-workspace --project existing_project --type isolated --existing

# Verify workspace initialization
cursus list-workspaces
cursus current-workspace
```

### Option 3: Clone and Setup from Template

For projects based on existing templates or patterns:

```bash
# Create project from template
cursus init-workspace --project your_project --type isolated --template standard

# Or create from existing project template
cursus init-workspace --project your_project --type isolated --template development/projects/project_alpha

# Navigate and verify
cd development/projects/your_project
cursus current-workspace
```

## Project Directory Structure

After successful workspace initialization, your project will have the following structure:

```
development/projects/your_project/
├── src/cursus_dev/              # Project-specific code (isolated from main src/cursus)
│   ├── __init__.py
│   ├── steps/                   # Project-specific step implementations
│   │   ├── __init__.py
│   │   ├── scripts/             # Project processing scripts
│   │   │   └── __init__.py
│   │   └── builders/            # Project step builders
│   │       └── __init__.py
│   ├── configs/                 # Project-specific configurations
│   │   └── __init__.py
│   └── utils/                   # Project utility functions
│       └── __init__.py
├── tests/                       # Project-specific tests
│   ├── __init__.py
│   ├── test_steps.py           # Step testing templates
│   ├── test_builders.py        # Builder testing templates
│   └── integration/            # Integration test directory
│       └── __init__.py
├── data/                        # Project data files
│   ├── input/                  # Input data samples
│   ├── output/                 # Output data samples
│   └── config/                 # Configuration files
├── docs/                        # Project-specific documentation
│   └── README.md
├── workspace_config.yaml        # Workspace configuration
├── requirements.txt             # Project-specific dependencies
└── README.md                    # Project overview
```

## Workspace Configuration

### workspace_config.yaml

The workspace configuration file defines project-specific settings:

```yaml
# workspace_config.yaml
workspace:
  name: "your_project"
  type: "isolated"
  version: "1.0.0"
  created: "2025-09-05"
  
project:
  description: "Project-specific pipeline step development"
  maintainer: "Your Name <your.email@company.com>"
  
registry:
  workspace_context: "your_project"
  fallback_to_main: true
  shared_component_access: true
  
paths:
  project_code: "src/cursus_dev"
  shared_code: "../../src/cursus"  # Relative path to main cursus code
  tests: "tests"
  data: "data"
  
dependencies:
  shared_components:
    - "cursus.core"
    - "cursus.registry.hybrid"
    - "cursus.validation"
  project_specific:
    - "cursus_dev.steps"
    - "cursus_dev.utils"
    
validation:
  enable_cross_workspace: true
  enable_shared_component_checks: true
  enable_integration_tests: true
  
cli:
  default_workspace: "your_project"
  auto_activate: true
```

### Environment Setup

Set up the Python environment for isolated development:

```bash
# Navigate to project directory
cd development/projects/your_project

# Create project-specific virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install cursus in development mode with project-specific extensions
pip install -e ../../  # Install main cursus package
pip install -e .       # Install project-specific extensions
```

## Workspace Activation and Context Management

### Activating Workspace

```bash
# Navigate to project directory
cd development/projects/your_project

# Activate workspace (sets context for CLI commands)
cursus activate-workspace your_project

# Verify activation
cursus current-workspace
cursus list-steps --workspace your_project
```

### Setting Workspace Context in Code

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("your_project")

# Verify context
current_workspace = registry.get_current_workspace()
print(f"Current workspace: {current_workspace}")

# List available components in current workspace
project_components = registry.list_components()
print(f"Project components: {project_components}")
```

### Context Management Patterns

```python
# Pattern 1: Temporary context switching
def access_shared_component():
    registry = UnifiedRegistryManager()
    
    # Save current workspace
    original_workspace = registry.get_current_workspace()
    
    try:
        # Switch to main workspace to access shared components
        registry.set_workspace_context("main")
        shared_component = registry.get_component("shared_utility")
        return shared_component
    finally:
        # Always restore original workspace
        registry.set_workspace_context(original_workspace)

# Pattern 2: Context manager for safe switching
from contextlib import contextmanager

@contextmanager
def workspace_context(workspace_name):
    registry = UnifiedRegistryManager()
    original_workspace = registry.get_current_workspace()
    try:
        registry.set_workspace_context(workspace_name)
        yield registry
    finally:
        registry.set_workspace_context(original_workspace)

# Usage
with workspace_context("main") as registry:
    shared_component = registry.get_component("shared_utility")
```

## Hybrid Registry Integration Setup

### Registry Configuration

Configure the hybrid registry for your project:

```python
# File: src/cursus_dev/__init__.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Initialize project registry
def setup_project_registry():
    """Initialize registry with project workspace context."""
    registry = UnifiedRegistryManager()
    
    # Set project workspace context
    registry.set_workspace_context("your_project")
    
    # Configure fallback behavior
    registry.configure_fallback(
        enable_main_fallback=True,
        enable_shared_access=True,
        cache_shared_components=True
    )
    
    return registry

# Auto-setup when importing project modules
PROJECT_REGISTRY = setup_project_registry()
```

### Component Registration Setup

Set up automatic component registration for your project:

```python
# File: src/cursus_dev/steps/__init__.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get project registry instance
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# Auto-register project components
def register_project_components():
    """Register all project-specific components."""
    
    # Import and register step builders
    from .builders import *  # This will trigger @register_builder decorators
    
    # Import and register other components as needed
    # Components will be registered with current workspace context
    
    print(f"Registered components in workspace: {registry.get_current_workspace()}")
    print(f"Available components: {registry.list_components()}")

# Auto-register on import
register_project_components()
```

## Testing Setup

### Project-Specific Test Configuration

```python
# File: tests/conftest.py

import pytest
from cursus.registry.hybrid.manager import UnifiedRegistryManager

@pytest.fixture(scope="session")
def project_registry():
    """Provide project registry for tests."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("your_project")
    return registry

@pytest.fixture(scope="session")
def shared_registry():
    """Provide shared registry for integration tests."""
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("main")
    return registry

@pytest.fixture
def workspace_context():
    """Provide workspace context manager for tests."""
    from contextlib import contextmanager
    
    @contextmanager
    def _workspace_context(workspace_name):
        registry = UnifiedRegistryManager()
        original = registry.get_current_workspace()
        try:
            registry.set_workspace_context(workspace_name)
            yield registry
        finally:
            registry.set_workspace_context(original)
    
    return _workspace_context
```

### Test Structure

```python
# File: tests/test_steps.py

import pytest
from cursus.registry.hybrid.manager import UnifiedRegistryManager

class TestProjectSteps:
    """Test project-specific step implementations."""
    
    def test_project_step_registration(self, project_registry):
        """Test that project steps are properly registered."""
        # Test project-specific step registration
        assert "your_custom_step" in project_registry.list_step_builders()
    
    def test_shared_component_access(self, workspace_context):
        """Test access to shared components."""
        with workspace_context("main") as registry:
            # Test access to shared components
            shared_step = registry.get_step_builder("preprocessing_step")
            assert shared_step is not None
    
    def test_hybrid_registry_resolution(self, project_registry):
        """Test hybrid registry resolution priority."""
        # Test that project-specific components take priority
        project_step = project_registry.get_step_builder("your_custom_step")
        assert project_step is not None
        
        # Test fallback to shared components
        shared_step = project_registry.get_step_builder("preprocessing_step")
        assert shared_step is not None
```

## CLI Integration

### Workspace-Aware CLI Commands

```bash
# Workspace management
cursus list-workspaces                    # List all available workspaces
cursus current-workspace                  # Show current active workspace
cursus activate-workspace your_project    # Activate specific workspace
cursus deactivate-workspace              # Deactivate current workspace

# Component management
cursus list-steps --workspace your_project           # List project-specific steps
cursus list-steps --workspace main                   # List shared steps
cursus list-steps --workspace your_project --all     # List all accessible steps

# Validation commands
cursus validate-workspace your_project               # Validate workspace setup
cursus validate-registry --workspace your_project   # Validate registry integration
cursus validate-alignment --workspace your_project  # Validate component alignment

# Testing commands
cursus test --workspace your_project                 # Run project-specific tests
cursus test --workspace your_project --integration   # Run integration tests
cursus test --workspace your_project --shared        # Include shared component tests
```

### Custom CLI Extensions

Create project-specific CLI extensions:

```python
# File: src/cursus_dev/cli/__init__.py

import click
from cursus.cli.main import cli

@cli.group()
def your_project():
    """Project-specific CLI commands."""
    pass

@your_project.command()
@click.option('--step-type', required=True, help='Step type to create')
def create_step(step_type):
    """Create a new project-specific step."""
    # Implementation for creating project-specific steps
    click.echo(f"Creating project step: {step_type}")

@your_project.command()
def validate_project():
    """Validate project-specific components."""
    # Implementation for project validation
    click.echo("Validating project components...")
```

## Troubleshooting Common Setup Issues

### Issue 1: Workspace Not Found

```bash
# Error: Workspace 'your_project' not found
# Solution: Verify workspace initialization
cursus list-workspaces
cursus init-workspace --project your_project --type isolated
```

### Issue 2: Registry Context Not Set

```python
# Error: No workspace context set
# Solution: Explicitly set workspace context
from cursus.registry.hybrid.manager import UnifiedRegistryManager

registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")
```

### Issue 3: Shared Component Access Issues

```python
# Error: Cannot access shared components
# Solution: Verify fallback configuration
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

# Check fallback configuration
config = registry.get_configuration()
print(f"Fallback enabled: {config.get('enable_main_fallback', False)}")

# Enable fallback if needed
registry.configure_fallback(enable_main_fallback=True)
```

### Issue 4: Import Path Issues

```python
# Error: Cannot import cursus_dev modules
# Solution: Verify Python path and installation
import sys
import os

# Add project path to Python path
project_path = os.path.join(os.getcwd(), 'src')
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Verify cursus_dev is importable
try:
    import cursus_dev
    print("cursus_dev imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
```

## Validation and Verification

### Workspace Setup Validation

```bash
# Comprehensive workspace validation
cursus validate-workspace your_project --comprehensive

# Check specific aspects
cursus validate-workspace your_project --check-structure
cursus validate-workspace your_project --check-registry
cursus validate-workspace your_project --check-dependencies
```

### Manual Verification Checklist

1. **Directory Structure**:
   - [ ] `src/cursus_dev/` directory exists
   - [ ] Required subdirectories created (`steps/`, `configs/`, `utils/`)
   - [ ] `__init__.py` files in all Python directories

2. **Configuration Files**:
   - [ ] `workspace_config.yaml` exists and is valid
   - [ ] `requirements.txt` includes necessary dependencies
   - [ ] Project `README.md` created

3. **Registry Integration**:
   - [ ] Workspace context can be set programmatically
   - [ ] Project components can be registered
   - [ ] Shared components can be accessed

4. **CLI Integration**:
   - [ ] Workspace can be activated via CLI
   - [ ] Workspace-specific commands work
   - [ ] Component listing works for project workspace

5. **Testing Setup**:
   - [ ] Test directory structure created
   - [ ] Test configuration files exist
   - [ ] Basic tests can be run

## Next Steps

After successful workspace setup:

1. **Read the Creation Process Guide**: [ws_creation_process.md](ws_creation_process.md)
2. **Understand Component Architecture**: [ws_component_guide.md](ws_component_guide.md)
3. **Learn Registry Integration**: [ws_hybrid_registry_integration.md](ws_hybrid_registry_integration.md)
4. **Set Up Testing**: [ws_testing_in_isolated_projects.md](ws_testing_in_isolated_projects.md)
5. **Start Development**: [ws_adding_new_pipeline_step.md](ws_adding_new_pipeline_step.md)

## Related Documentation

- [Adding a New Pipeline Step (Workspace-Aware)](ws_adding_new_pipeline_step.md) - Main development guide
- [Hybrid Registry Integration](ws_hybrid_registry_integration.md) - Registry usage patterns
- [Shared Code Access Patterns](ws_shared_code_access_patterns.md) - Accessing shared components
- [Workspace CLI Reference](ws_workspace_cli_reference.md) - CLI tools and commands
- [Troubleshooting Workspace Issues](ws_troubleshooting_workspace_issues.md) - Common problems and solutions
