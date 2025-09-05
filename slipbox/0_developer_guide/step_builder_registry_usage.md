# Step Builder Registry Usage Guide: UnifiedRegistryManager

**Version**: 2.0  
**Date**: September 5, 2025  
**Author**: MODS Development Team

## Overview

This guide provides practical examples of using the UnifiedRegistryManager system for step builder registration and usage in both main workspace and isolated project development scenarios. The examples demonstrate workspace-aware patterns and modern registry usage.

## Basic Usage Patterns

### Getting Registry Instance

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance (singleton)
registry = UnifiedRegistryManager()

# Set workspace context
registry.set_workspace_context("main")  # or "project_alpha", etc.
```

### Checking Builder Availability

```python
# Check if a step builder exists in current workspace
if registry.has_step_builder("XGBoostTraining"):
    print("XGBoostTraining builder is available")

# Check across all workspaces
workspaces = registry.find_step_builder_workspaces("XGBoostTraining")
print(f"XGBoostTraining available in: {workspaces}")

# List all available builders in current workspace
builders = registry.list_step_builders()
print(f"Available builders: {builders}")
```

### Getting and Using Builders

```python
# Get builder class
builder_class = registry.get_step_builder("XGBoostTraining")

# Create builder instance with proper XGBoost configuration
from cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig

# Create hyperparameters
hyperparams = XGBoostModelHyperparameters(
    full_field_list=["feature1", "feature2", "feature3", "label"],
    tab_field_list=["feature1", "feature2", "feature3"],
    cat_field_list=[],
    label_name="label",
    id_name="id",
    num_round=100,
    max_depth=6,
    eta=0.1
)

# Create config
config = XGBoostTrainingConfig(
    training_entry_point="train.py",
    hyperparameters=hyperparams,
    training_instance_type="ml.m5.xlarge",
    source_dir="scripts/"
)
builder = builder_class(config)

# Create step
step = builder.create_step()
```

## Main Workspace Development

### Adding a New Step Builder (Main Workspace)

For development directly in `src/cursus/`:

```python
# File: src/cursus/steps/builders/custom_processing_builder.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.core.step_builder import StepBuilderBase
from cursus.core.config import BasePipelineConfig

class CustomProcessingConfig(BasePipelineConfig):
    """Configuration for custom processing step."""
    
    def __init__(self, 
                 input_path: str,
                 output_path: str,
                 processing_mode: str = "standard",
                 **kwargs):
        super().__init__(**kwargs)
        self.input_path = input_path
        self.output_path = output_path
        self.processing_mode = processing_mode

class CustomProcessingStepBuilder(StepBuilderBase):
    """Builder for custom processing step in main workspace."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config: CustomProcessingConfig = config
    
    def build_step(self, **kwargs):
        # Implementation here
        from ..custom_processing_step import CustomProcessingStep
        return CustomProcessingStep(self.config)
    
    def validate_config(self, config):
        required_fields = ['input_path', 'output_path']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required field: {field}")
        return True

# Register in main workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("main")
registry.register_step_builder("CustomProcessing", CustomProcessingStepBuilder)
```

### Using Legacy Decorator (Main Workspace)

```python
# File: src/cursus/steps/builders/legacy_step_builder.py

from cursus.registry.decorators import register_builder
from cursus.core.step_builder import StepBuilderBase

@register_builder("LegacyStep")
class LegacyStepBuilder(StepBuilderBase):
    """Legacy step builder using decorator pattern."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        # The decorator automatically registers this with UnifiedRegistryManager
        # in the current workspace context
    
    def build_step(self, **kwargs):
        # Implementation here
        pass
```

### Verifying Main Workspace Registration

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry and set main workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("main")

# Verify registration
if registry.has_step_builder("CustomProcessing"):
    print("CustomProcessing successfully registered in main workspace")
    
    # Get builder info
    builder_info = registry.inspect_step_builder("CustomProcessing")
    print(f"Builder class: {builder_info['class']}")
    print(f"Workspace: {builder_info['workspace']}")
    print(f"Metadata: {builder_info.get('metadata', {})}")

# List all builders in main workspace
main_builders = registry.list_step_builders(workspace="main")
print(f"Main workspace builders: {main_builders}")
```

## Isolated Project Development

### Setting Up Project Workspace

```python
# Initialize project workspace
from cursus.registry.hybrid.manager import UnifiedRegistryManager

registry = UnifiedRegistryManager()

# Initialize workspace for new project
registry.initialize_workspace("project_alpha")
registry.set_workspace_context("project_alpha")

print(f"Current workspace: {registry.workspace_context}")
```

### Adding Project-Specific Step Builder

```python
# File: development/projects/project_alpha/src/cursus_dev/steps/builders/project_specific_builder.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.core.step_builder import StepBuilderBase
from cursus.core.config import BasePipelineConfig

class ProjectSpecificConfig(BasePipelineConfig):
    """Configuration for project-specific processing."""
    
    def __init__(self, 
                 project_data_path: str,
                 custom_algorithm: str,
                 project_params: dict,
                 **kwargs):
        super().__init__(**kwargs)
        self.project_data_path = project_data_path
        self.custom_algorithm = custom_algorithm
        self.project_params = project_params

class ProjectSpecificStepBuilder(StepBuilderBase):
    """Builder for project-specific step in isolated environment."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.config: ProjectSpecificConfig = config
    
    def build_step(self, **kwargs):
        # Project-specific implementation
        from ..project_specific_step import ProjectSpecificStep
        return ProjectSpecificStep(self.config)
    
    def validate_config(self, config):
        # Project-specific validation
        required_fields = ['project_data_path', 'custom_algorithm']
        for field in required_fields:
            if not hasattr(config, field):
                raise ValueError(f"Missing required field: {field}")
        
        # Validate custom algorithm
        valid_algorithms = ['custom_ml', 'project_transform', 'special_processing']
        if config.custom_algorithm not in valid_algorithms:
            raise ValueError(f"Invalid algorithm: {config.custom_algorithm}")
        
        return True

# Register in project workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_alpha")
registry.register_step_builder(
    "ProjectSpecificProcessing", 
    ProjectSpecificStepBuilder,
    metadata={
        "description": "Project-specific processing for project_alpha",
        "version": "1.0",
        "project": "project_alpha",
        "dependencies": ["custom_ml_lib", "project_utils"]
    }
)
```

### Using Hybrid Resolution

```python
# File: development/projects/project_alpha/src/cursus_dev/pipeline/project_pipeline.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager

def create_project_pipeline():
    """Create pipeline using both shared and project-specific steps."""
    
    registry = UnifiedRegistryManager()
    registry.set_workspace_context("project_alpha")
    
    # Use shared step from main workspace (fallback resolution)
    preprocessing_builder = registry.get_step_builder("TabularPreprocessing")
    preprocessing_step = preprocessing_builder(preprocessing_config).build_step()
    
    # Use project-specific step (priority resolution)
    project_builder = registry.get_step_builder("ProjectSpecificProcessing")
    project_step = project_builder(project_config).build_step()
    
    # Use shared training step
    training_builder = registry.get_step_builder("XGBoostTraining")
    training_step = training_builder(training_config).build_step()
    
    return [preprocessing_step, project_step, training_step]
```

### Verifying Project Workspace Registration

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Set project workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_alpha")

# Check project-specific builders
project_builders = registry.list_step_builders(workspace="project_alpha")
print(f"Project-specific builders: {project_builders}")

# Check hybrid resolution
all_available = registry.list_step_builders()  # Includes fallback to shared
print(f"All available builders in project_alpha context: {all_available}")

# Verify specific builder resolution
if registry.has_step_builder("ProjectSpecificProcessing"):
    builder_info = registry.inspect_step_builder("ProjectSpecificProcessing")
    print(f"ProjectSpecificProcessing resolved from: {builder_info['workspace']}")

# Check shared builder access
if registry.has_step_builder("XGBoostTraining"):
    builder_info = registry.inspect_step_builder("XGBoostTraining")
    print(f"XGBoostTraining resolved from: {builder_info['workspace']}")
```

## Advanced Usage Patterns

### Cross-Workspace Builder Access

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

registry = UnifiedRegistryManager()

# Access builder from specific workspace regardless of current context
main_builder = registry.get_step_builder("XGBoostTraining", workspace="main")
project_builder = registry.get_step_builder("CustomProcessing", workspace="project_alpha")

# Find all workspaces containing a specific builder
workspaces = registry.find_step_builder_workspaces("XGBoostTraining")
print(f"XGBoostTraining available in workspaces: {workspaces}")

# Get builder with workspace preference
try:
    # Try project workspace first, fallback to main
    builder = registry.get_step_builder("SomeStep", workspace="project_alpha")
except KeyError:
    builder = registry.get_step_builder("SomeStep", workspace="main")
```

### Dynamic Builder Registration

```python
def register_dynamic_builder(step_name, builder_class, workspace="main"):
    """Dynamically register a step builder."""
    
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace)
    
    # Register with metadata
    registry.register_step_builder(
        step_name,
        builder_class,
        metadata={
            "registered_at": datetime.now().isoformat(),
            "dynamic": True,
            "workspace": workspace
        }
    )
    
    # Validate registration
    if registry.has_step_builder(step_name):
        print(f"Successfully registered {step_name} in {workspace}")
        return True
    else:
        print(f"Failed to register {step_name}")
        return False

# Usage
register_dynamic_builder("DynamicStep", DynamicStepBuilder, "project_beta")
```

### Builder Factory Pattern

```python
class StepBuilderFactory:
    """Factory for creating step builders with workspace awareness."""
    
    def __init__(self, workspace="main"):
        self.registry = UnifiedRegistryManager()
        self.registry.set_workspace_context(workspace)
        self.workspace = workspace
    
    def create_builder(self, step_type, config):
        """Create a step builder instance."""
        if not self.registry.has_step_builder(step_type):
            available = self.registry.list_step_builders()
            raise ValueError(f"Step type '{step_type}' not found. Available: {available}")
        
        builder_class = self.registry.get_step_builder(step_type)
        return builder_class(config)
    
    def list_available_steps(self):
        """List all available step types in current workspace."""
        return self.registry.list_step_builders()
    
    def switch_workspace(self, workspace):
        """Switch to different workspace."""
        self.registry.set_workspace_context(workspace)
        self.workspace = workspace

# Usage
factory = StepBuilderFactory("project_alpha")
builder = factory.create_builder("CustomProcessing", config)
step = builder.build_step()
```

## CLI Integration Examples

### Using CLI Commands

```bash
# Set workspace context
cursus set-workspace project_alpha

# List available step builders
cursus list-steps --workspace project_alpha

# Validate registry integrity
cursus validate-registry --workspace project_alpha

# Get builder information
cursus inspect-step XGBoostTraining --workspace project_alpha

# Clear registry cache
cursus clear-cache --workspace project_alpha
```

### CLI Integration in Python

```python
from cursus.cli.workspace_cli import WorkspaceCLI

# Initialize CLI
cli = WorkspaceCLI()

# Set workspace
cli.set_workspace("project_alpha")

# List steps
steps = cli.list_steps(workspace="project_alpha")
print(f"Available steps: {steps}")

# Validate registry
validation_result = cli.validate_registry(workspace="project_alpha")
if validation_result.is_valid:
    print("Registry is valid")
else:
    print(f"Validation errors: {validation_result.errors}")
```

## Best Practices

### 1. Always Set Workspace Context

```python
# Good: Explicit workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_alpha")
builder = registry.get_step_builder("MyStep")

# Avoid: Implicit context
registry = UnifiedRegistryManager()
builder = registry.get_step_builder("MyStep")  # Uses default context
```

### 2. Use Descriptive Metadata

```python
# Good: Include comprehensive metadata
registry.register_step_builder(
    "CustomProcessing",
    CustomProcessingStepBuilder,
    metadata={
        "description": "Custom data processing for project_alpha",
        "version": "1.2.0",
        "author": "Data Science Team",
        "dependencies": ["pandas>=1.3.0", "scikit-learn>=1.0.0"],
        "workspace": "project_alpha",
        "tags": ["preprocessing", "custom", "ml"]
    }
)
```

### 3. Handle Missing Builders Gracefully

```python
# Good: Graceful error handling
def get_builder_safely(step_type, workspace="main"):
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace)
    
    try:
        return registry.get_step_builder(step_type)
    except KeyError:
        available = registry.list_step_builders()
        raise ValueError(
            f"Step builder '{step_type}' not found in workspace '{workspace}'. "
            f"Available builders: {available}"
        )
```

### 4. Validate Registry State

```python
# Good: Regular validation
def validate_workspace_registry(workspace):
    registry = UnifiedRegistryManager()
    registry.set_workspace_context(workspace)
    
    validation = registry.validate()
    if not validation.is_valid:
        print(f"Registry validation failed for {workspace}:")
        for error in validation.errors:
            print(f"  - {error}")
        return False
    
    print(f"Registry for {workspace} is valid")
    return True

# Run validation for all workspaces
for workspace in ["main", "project_alpha", "project_beta"]:
    validate_workspace_registry(workspace)
```

### 5. Use Workspace-Specific Naming

```python
# Good: Clear workspace-specific naming
registry.set_workspace_context("project_alpha")
registry.register_step_builder("AlphaCustomProcessing", AlphaCustomStepBuilder)

# Avoid: Generic naming that could conflict
registry.register_step_builder("CustomProcessing", SomeStepBuilder)  # Could conflict
```

## Troubleshooting

### Builder Not Found

```python
# Debug missing builder
def debug_missing_builder(step_type):
    registry = UnifiedRegistryManager()
    
    print(f"Current workspace: {registry.workspace_context}")
    print(f"Available builders: {registry.list_step_builders()}")
    
    # Check other workspaces
    workspaces = registry.find_step_builder_workspaces(step_type)
    if workspaces:
        print(f"{step_type} found in workspaces: {workspaces}")
    else:
        print(f"{step_type} not found in any workspace")
    
    # Check cache
    stats = registry.get_cache_stats()
    print(f"Cache stats: {stats}")

# Usage
debug_missing_builder("MissingStep")
```

### Wrong Builder Resolution

```python
# Debug resolution order
def debug_builder_resolution(step_type):
    registry = UnifiedRegistryManager()
    
    # Check resolution in different workspaces
    for workspace in ["main", "project_alpha", "project_beta"]:
        try:
            registry.set_workspace_context(workspace)
            builder_info = registry.inspect_step_builder(step_type)
            print(f"In {workspace}: {step_type} resolves to {builder_info['class']} from {builder_info['workspace']}")
        except KeyError:
            print(f"In {workspace}: {step_type} not found")

# Usage
debug_builder_resolution("XGBoostTraining")
```

### Cache Issues

```python
# Debug cache problems
def debug_cache_issues():
    registry = UnifiedRegistryManager()
    
    # Get cache statistics
    stats = registry.get_cache_stats()
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Cache size: {stats['size']}")
    
    # Clear cache if needed
    if stats['size'] > 1000:  # Arbitrary threshold
        print("Cache size large, clearing...")
        registry.clear_cache()
    
    # Check cache after clearing
    new_stats = registry.get_cache_stats()
    print(f"Cache after clearing: {new_stats}")

# Usage
debug_cache_issues()
```

## Migration from Legacy Registry

### Old Pattern (Deprecated)

```python
# Old way - don't use
from src.pipeline_registry.builder_registry import get_global_registry, register_builder

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

# Get builder
builder_class = registry.get_step_builder("XGBoostTraining")
builder = builder_class(config)
```

## Related Resources

- [Step Builder Registry Guide](./step_builder_registry_guide.md)
- [Adding a New Pipeline Step](./adding_new_pipeline_step.md)
- [Workspace-Aware Development Guide](../01_workspace_aware_developer_guide/ws_README.md)
- [UnifiedRegistryManager Design](../1_design/hybrid_registry_standardization_enforcement_design.md)

This guide demonstrates practical usage patterns for the modern UnifiedRegistryManager system with workspace awareness and should be used for all new development.
