# Step Builder Registry Guide: UnifiedRegistryManager

**Version**: 2.0  
**Date**: September 5, 2025  
**Author**: MODS Development Team

## Overview

This guide explains how to use the modern UnifiedRegistryManager system for step builder registration and discovery. The UnifiedRegistryManager provides workspace-aware caching, context management, and consolidated registry functionality that replaces the legacy registry system.

## Key Features

The UnifiedRegistryManager offers several key features:

1. **Workspace-Aware Registration**: Context-sensitive step builder registration and lookup
2. **Unified Registry**: Single consolidated registry replacing CoreStepRegistry, LocalStepRegistry, and HybridRegistryManager
3. **Workspace Context Management**: Automatic context switching and isolation
4. **Hybrid Resolution**: Priority-based resolution across workspace contexts
5. **Caching**: Intelligent caching with workspace-aware invalidation
6. **Legacy Compatibility**: Backward compatibility with @register_builder decorator

## UnifiedRegistryManager Basics

### Getting the Registry Instance

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance (singleton pattern)
registry = UnifiedRegistryManager()
```

### Workspace Context Management

The registry operates with workspace contexts that determine resolution priority:

```python
# Set workspace context
registry.set_workspace_context("main")  # Main workspace
registry.set_workspace_context("project_alpha")  # Project workspace

# Get current workspace context
current_context = registry.workspace_context
print(f"Current workspace: {current_context}")

# Clear workspace context (resets to default)
registry.set_workspace_context(None)
```

## Registration Patterns

### Main Workspace Registration

For development in the main codebase (`src/cursus/`):

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set main workspace context (default)
registry.set_workspace_context("main")

# Register step builder
registry.register_step_builder("XGBoostTraining", XGBoostTrainingStepBuilder)

# Register with additional metadata
registry.register_step_builder(
    "XGBoostTraining", 
    XGBoostTrainingStepBuilder,
    metadata={
        "description": "XGBoost model training step",
        "version": "2.0",
        "workspace": "main"
    }
)
```

### Isolated Project Registration

For development in isolated projects (`development/projects/*/src/cursus_dev/`):

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("project_alpha")

# Register project-specific step builder
registry.register_step_builder("CustomProcessing", CustomProcessingStepBuilder)

# The registry will prioritize project-specific builders over shared ones
```

### Legacy Decorator Support

The `@register_builder` decorator is still supported and uses UnifiedRegistryManager internally:

```python
from cursus.registry.decorators import register_builder

@register_builder("XGBoostTraining")
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """Builder for XGBoost training step."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Implementation here
```

The decorator automatically:
1. Uses the current workspace context
2. Registers with UnifiedRegistryManager
3. Maintains backward compatibility

## Registry Lookup and Resolution

### Basic Lookup

```python
# Get step builder by name
builder_class = registry.get_step_builder("XGBoostTraining")

# Check if step builder exists
if registry.has_step_builder("XGBoostTraining"):
    builder_class = registry.get_step_builder("XGBoostTraining")
    builder = builder_class(config)
```

### Workspace-Aware Resolution

The registry follows a priority-based resolution strategy:

1. **Project-specific builders** (`src/cursus_dev/`) - highest priority
2. **Shared builders** (`src/cursus/`) - fallback
3. **Default implementations** - last resort

```python
# Set project workspace
registry.set_workspace_context("project_alpha")

# This will first look in project_alpha workspace, then fall back to shared
builder_class = registry.get_step_builder("XGBoostTraining")
```

### Listing Available Builders

```python
# List all step builders in current workspace
builders = registry.list_step_builders()

# List builders in specific workspace
builders = registry.list_step_builders(workspace="project_alpha")

# List all builders across all workspaces
all_builders = registry.list_all_step_builders()
```

## Workspace Management

### Creating Workspace Context

```python
# Initialize workspace context for a new project
registry.initialize_workspace("project_beta")

# Set workspace context
registry.set_workspace_context("project_beta")

# Register project-specific builders
registry.register_step_builder("ProjectSpecificStep", ProjectSpecificStepBuilder)
```

### Workspace Isolation

Each workspace maintains its own registry state:

```python
# Register in main workspace
registry.set_workspace_context("main")
registry.register_step_builder("SharedStep", SharedStepBuilder)

# Register in project workspace
registry.set_workspace_context("project_alpha")
registry.register_step_builder("SharedStep", ProjectSpecificStepBuilder)

# Resolution depends on current context
registry.set_workspace_context("project_alpha")
builder = registry.get_step_builder("SharedStep")  # Gets ProjectSpecificStepBuilder

registry.set_workspace_context("main")
builder = registry.get_step_builder("SharedStep")  # Gets SharedStepBuilder
```

### Cross-Workspace Access

```python
# Access builders from specific workspace
builder = registry.get_step_builder("SharedStep", workspace="main")

# Check builder availability across workspaces
workspaces = registry.find_step_builder_workspaces("SharedStep")
print(f"SharedStep available in: {workspaces}")
```

## Caching and Performance

### Workspace-Aware Caching

The UnifiedRegistryManager implements intelligent caching:

```python
# Cache is automatically managed per workspace
registry.set_workspace_context("project_alpha")
builder1 = registry.get_step_builder("XGBoostTraining")  # Cache miss

# Subsequent calls use cache
builder2 = registry.get_step_builder("XGBoostTraining")  # Cache hit

# Cache is isolated per workspace
registry.set_workspace_context("main")
builder3 = registry.get_step_builder("XGBoostTraining")  # Different cache
```

### Cache Management

```python
# Clear cache for current workspace
registry.clear_cache()

# Clear cache for specific workspace
registry.clear_cache(workspace="project_alpha")

# Clear all caches
registry.clear_all_caches()

# Get cache statistics
stats = registry.get_cache_stats()
print(f"Cache hits: {stats['hits']}, misses: {stats['misses']}")
```

## Integration with CLI

The UnifiedRegistryManager integrates with CLI commands for workspace management:

### CLI Commands

```bash
# Set workspace context
cursus set-workspace project_alpha

# List available step builders
cursus list-steps --workspace project_alpha

# Validate registry integrity
cursus validate-registry --workspace project_alpha

# Clear registry cache
cursus clear-cache --workspace project_alpha
```

### CLI Integration in Code

```python
from cursus.cli.workspace_cli import WorkspaceCLI

# Initialize CLI integration
cli = WorkspaceCLI()

# Set workspace via CLI
cli.set_workspace("project_alpha")

# List steps via CLI
steps = cli.list_steps(workspace="project_alpha")
```

## Validation and Debugging

### Registry Validation

```python
# Validate registry integrity
validation_result = registry.validate()

if validation_result.is_valid:
    print("Registry is valid")
else:
    print("Registry validation errors:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### Debugging Registry State

```python
# Get registry state information
state = registry.get_state()
print(f"Current workspace: {state['current_workspace']}")
print(f"Registered builders: {state['builder_count']}")
print(f"Cache size: {state['cache_size']}")

# Get detailed workspace information
workspace_info = registry.get_workspace_info("project_alpha")
print(f"Builders in project_alpha: {workspace_info['builders']}")
```

### Registry Inspection

```python
# Inspect specific builder registration
builder_info = registry.inspect_step_builder("XGBoostTraining")
print(f"Builder class: {builder_info['class']}")
print(f"Workspace: {builder_info['workspace']}")
print(f"Registration time: {builder_info['registered_at']}")
print(f"Metadata: {builder_info['metadata']}")
```

## Complete Example: Isolated Project Development

Here's a complete example of using UnifiedRegistryManager in an isolated project:

```python
# File: development/projects/project_alpha/src/cursus_dev/steps/builders/custom_processing_builder.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.core.step_builder import StepBuilderBase
from ..custom_processing_step import CustomProcessingStep

class CustomProcessingStepBuilder(StepBuilderBase):
    """Builder for custom processing step in project_alpha."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
    
    def build_step(self, **kwargs):
        return CustomProcessingStep(self.config)
    
    def validate_config(self, config):
        # Project-specific validation logic
        required_fields = ['input_path', 'output_path', 'custom_param']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required field: {field}")
        return True

# Register in project workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_alpha")
registry.register_step_builder("CustomProcessing", CustomProcessingStepBuilder)

# Usage in project code
def create_custom_processing_step(config):
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("project_alpha")
    
    # This will get the project-specific builder
    builder_class = registry.get_step_builder("CustomProcessing")
    builder = builder_class(config)
    
    return builder.build_step()
```

## Best Practices

### 1. Always Set Workspace Context

```python
# Good: Explicit workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_alpha")
builder = registry.get_step_builder("MyStep")

# Avoid: Relying on default context
registry = UnifiedRegistryManager()
builder = registry.get_step_builder("MyStep")  # Uses default context
```

### 2. Use Workspace-Specific Registration

```python
# Good: Register in appropriate workspace
registry.set_workspace_context("project_alpha")
registry.register_step_builder("ProjectStep", ProjectStepBuilder)

# Avoid: Registering in wrong workspace
registry.set_workspace_context("main")
registry.register_step_builder("ProjectStep", ProjectStepBuilder)  # Wrong workspace
```

### 3. Handle Resolution Failures Gracefully

```python
# Good: Check availability before use
if registry.has_step_builder("MyStep"):
    builder_class = registry.get_step_builder("MyStep")
else:
    # Handle missing builder
    raise ValueError(f"Step builder 'MyStep' not found in workspace '{registry.workspace_context}'")

# Better: Use try-catch
try:
    builder_class = registry.get_step_builder("MyStep")
except KeyError as e:
    # Handle missing builder with context
    available_builders = registry.list_step_builders()
    raise ValueError(f"Step builder 'MyStep' not found. Available: {available_builders}") from e
```

### 4. Use Metadata for Documentation

```python
# Good: Include descriptive metadata
registry.register_step_builder(
    "CustomProcessing",
    CustomProcessingStepBuilder,
    metadata={
        "description": "Custom data processing for project_alpha",
        "version": "1.0",
        "author": "Project Alpha Team",
        "dependencies": ["pandas", "numpy"],
        "workspace": "project_alpha"
    }
)
```

### 5. Validate Registry State

```python
# Good: Regular validation
def setup_project_registry():
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("project_alpha")
    
    # Register builders
    registry.register_step_builder("CustomStep", CustomStepBuilder)
    
    # Validate after registration
    validation = registry.validate()
    if not validation.is_valid:
        raise RuntimeError(f"Registry validation failed: {validation.errors}")
    
    return registry
```

## Migration from Legacy Registry

### Old Pattern (Deprecated)

```python
# Old way - don't use
from src.pipeline_registry.builder_registry import register_builder, get_global_registry

@register_builder("XGBoostTraining")
class XGBoostTrainingStepBuilder(StepBuilderBase):
    pass

registry = get_global_registry()
builder = registry.get_builder_for_step_type("XGBoostTraining")
```

### New Pattern (Recommended)

```python
# New way - use this
from cursus.registry.hybrid.manager import UnifiedRegistryManager

class XGBoostTrainingStepBuilder(StepBuilderBase):
    pass

# Explicit registration
registry = UnifiedRegistryManager()
registry.set_workspace_context("main")
registry.register_step_builder("XGBoostTraining", XGBoostTrainingStepBuilder)

# Or use legacy decorator (still supported)
from cursus.registry.decorators import register_builder

@register_builder("XGBoostTraining")
class XGBoostTrainingStepBuilder(StepBuilderBase):
    pass
```

## Troubleshooting

### Common Issues

1. **Builder Not Found**
   ```python
   # Check workspace context
   print(f"Current workspace: {registry.workspace_context}")
   
   # List available builders
   builders = registry.list_step_builders()
   print(f"Available builders: {builders}")
   
   # Check other workspaces
   workspaces = registry.find_step_builder_workspaces("MyStep")
   print(f"MyStep found in workspaces: {workspaces}")
   ```

2. **Wrong Builder Returned**
   ```python
   # Check resolution order
   builder_info = registry.inspect_step_builder("MyStep")
   print(f"Resolved from workspace: {builder_info['workspace']}")
   
   # Force specific workspace
   builder = registry.get_step_builder("MyStep", workspace="main")
   ```

3. **Cache Issues**
   ```python
   # Clear cache if stale data
   registry.clear_cache()
   
   # Check cache stats
   stats = registry.get_cache_stats()
   print(f"Cache stats: {stats}")
   ```

## Related Resources

- [Adding a New Pipeline Step](./adding_new_pipeline_step.md)
- [Step Builder Registry Usage](./step_builder_registry_usage.md)
- [Workspace-Aware Development Guide](../01_workspace_aware_developer_guide/ws_README.md)
- [UnifiedRegistryManager Design](../1_design/hybrid_registry_standardization_enforcement_design.md)
- [Workspace-Aware Architecture](../1_design/workspace_aware_distributed_registry_design.md)

## Recent Changes

### Version 2.0 Updates

1. **Complete rewrite** for UnifiedRegistryManager
2. **Workspace-aware registration** and lookup
3. **Consolidated registry** replacing legacy system
4. **Enhanced caching** with workspace isolation
5. **CLI integration** for workspace management
6. **Improved validation** and debugging capabilities
7. **Backward compatibility** with legacy decorators

This guide reflects the modern UnifiedRegistryManager approach and should be used for all new development.
