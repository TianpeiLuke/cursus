# Step Catalog Integration Guide

**Version**: 1.0.0  
**Date**: September 27, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides comprehensive documentation for integrating with the **StepCatalog System**, which serves as the unified foundation for step discovery, builder resolution, and pipeline construction. The StepCatalog system has replaced the legacy StepBuilderRegistry with a more robust, scalable, and maintainable architecture.

## StepCatalog System Architecture

The StepCatalog system follows a clean two-system design with proper separation of concerns:

```mermaid
graph TB
    subgraph "Registry System (Single Source of Truth)"
        REG[Registry System]
        REG --> |"Canonical step definitions"| NAMES[get_step_names()]
        REG --> |"Config-to-step mapping"| CONFIG[get_config_step_registry()]
        REG --> |"Workspace context"| WS[Workspace Management]
        REG --> |"Validation authority"| VAL[validate_step_name()]
    end
    
    subgraph "Step Catalog System (Discovery + Bidirectional Mapping)"
        CAT[Step Catalog System]
        CAT --> |"Multi-component discovery"| DISC[Component Discovery]
        CAT --> |"Config ↔ Builder mapping"| MAP[Bidirectional Mapping]
        CAT --> |"Job type variants"| JOB[Job Type Handling]
        CAT --> |"Workspace-aware discovery"| WSDISC[Workspace Discovery]
        CAT --> |"Pipeline construction"| PIPE[Pipeline Support]
    end
    
    REG --> |"References"| CAT
    
    classDef registry fill:#e1f5fe
    classDef catalog fill:#f3e5f5
    
    class REG,NAMES,CONFIG,WS,VAL registry
    class CAT,DISC,MAP,JOB,WSDISC,PIPE catalog
```

### System Responsibilities

**Registry System: Single Source of Truth**
- **Canonical Step Definitions**: Maintain authoritative step name → definition mappings
- **Workspace Context Management**: Support multiple workspace contexts with proper isolation
- **Derived Registry Generation**: Provide config-to-step-name and other derived mappings
- **Validation Authority**: Serve as validation source for all step-related operations

**Step Catalog System: Comprehensive Discovery & Bidirectional Mapping**
- **Multi-Component Discovery**: Scripts, contracts, specs, builders, configs across workspaces
- **Bidirectional Mapping**: Step name/type ↔ Components, Config ↔ Builder, Builder ↔ Step name
- **Job Type Variant Handling**: Support variants like "CradleDataLoading_training"
- **Workspace-Aware Discovery**: Project-specific component discovery and resolution
- **Pipeline Construction Support**: All builder-related operations for pipeline construction

## Key Features

### 1. Automatic Builder Discovery

The StepCatalog automatically discovers step builders through file system scanning:

```python
from cursus.step_catalog import StepCatalog

# Initialize catalog - automatically discovers all builders
catalog = StepCatalog()

# List all discovered step types
step_types = catalog.list_supported_step_types()
print(f"Discovered {len(step_types)} step types")

# Check if specific step type is supported
is_supported = catalog.is_step_type_supported("BatchTransform")
print(f"BatchTransform supported: {is_supported}")
```

**Discovery Rules**:
- **File naming convention**: `builder_*.py` files in `src/cursus/steps/builders/`
- **Class naming convention**: Classes ending with `StepBuilder`
- **Registry integration**: Step definitions from `src/cursus/registry/step_names.py`

### 2. Config-to-Builder Resolution

Direct mapping from configuration instances to builder classes:

```python
from cursus.step_catalog import StepCatalog
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingStepConfig

# Initialize catalog
catalog = StepCatalog()

# Create configuration
config = XGBoostTrainingStepConfig(
    job_type="training",
    region="us-west-2",
    # ... other config parameters
)

# Get builder class for config
builder_class = catalog.get_builder_for_config(config)
if builder_class:
    # Create builder instance
    builder = builder_class(config=config, role=role, sagemaker_session=session)
    print(f"Created builder: {builder.__class__.__name__}")
else:
    print("No builder found for config")
```

### 3. Step Type to Builder Resolution

Map step types to builder classes:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Get builder class by step type
builder_class = catalog.get_builder_for_step_type("XGBoostTraining")
if builder_class:
    print(f"Builder class: {builder_class.__name__}")
    
    # Create instance with config
    config = XGBoostTrainingStepConfig(...)
    builder = builder_class(config=config, role=role, sagemaker_session=session)
```

### 4. Legacy Alias Support

Backward compatibility for legacy step names:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Legacy aliases are automatically resolved
legacy_builder = catalog.get_builder_for_step_type("PytorchTraining")  # Legacy name
canonical_builder = catalog.get_builder_for_step_type("PyTorchTraining")  # Canonical name

# Both return the same builder class
assert legacy_builder == canonical_builder
```

**Supported Legacy Aliases**:
- `MIMSPackaging` → `Package`
- `MIMSPayload` → `Payload`
- `ModelRegistration` → `Registration`
- `PytorchTraining` → `PyTorchTraining`
- `PytorchModel` → `PyTorchModel`

### 5. Pipeline Construction Interface

Complete interface for pipeline construction:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Get complete builder map for pipeline construction
builder_map = catalog.get_builder_map()
print(f"Available builders: {list(builder_map.keys())}")

# Validate builder availability for multiple step types
step_types = ["XGBoostTraining", "BatchTransform", "Package"]
availability = catalog.validate_builder_availability(step_types)

for step_type, available in availability.items():
    status = "✅" if available else "❌"
    print(f"{status} {step_type}: {'Available' if available else 'Not Available'}")

# Get config types for a step type
config_types = catalog.get_config_types_for_step_type("XGBoostTraining")
print(f"Config types for XGBoostTraining: {config_types}")
```

## Integration Patterns

### 1. DAG Compiler Integration

```python
from cursus.step_catalog import StepCatalog
from cursus.core.dag.dag_compiler import PipelineDAGCompiler

# Create DAG compiler with StepCatalog
catalog = StepCatalog()
compiler = PipelineDAGCompiler(step_catalog=catalog)

# Get supported step types
supported_types = compiler.get_supported_step_types()
print(f"Compiler supports {len(supported_types)} step types")

# Create pipeline template
template = compiler.create_template(
    dag=pipeline_dag,
    config_path="config.json",
    step_catalog=catalog  # Pass catalog to template
)
```

### 2. Pipeline Assembler Integration

```python
from cursus.step_catalog import StepCatalog
from cursus.core.assembly.pipeline_assembler import PipelineAssembler

# Create assembler with StepCatalog
catalog = StepCatalog()
assembler = PipelineAssembler(
    dag=pipeline_dag,
    config_map=config_map,
    step_catalog=catalog  # Direct config-to-builder resolution
)

# Assembler automatically uses catalog for builder resolution
pipeline = assembler.assemble()
```

### 3. Dynamic Template Integration

```python
from cursus.step_catalog import StepCatalog
from cursus.core.templates.dynamic_template import DynamicPipelineTemplate

# Create template with StepCatalog
catalog = StepCatalog()
template = DynamicPipelineTemplate(
    dag=pipeline_dag,
    config_path="config.json",
    step_catalog=catalog  # Builder map generation
)

# Get catalog statistics
stats = template.get_step_catalog_stats()
print(f"Catalog stats: {stats}")
```

## Step Builder Development

### 1. Modern Step Builder Pattern

No decorators or manual registration required:

```python
from typing import Dict, List, Any, Optional
from sagemaker.workflow.steps import ProcessingStep
from cursus.core.base.builder_base import StepBuilderBase

class YourNewStepBuilder(StepBuilderBase):
    """
    Builder for YourNewStep - automatically discovered by StepCatalog.
    
    No @register_builder decorator needed!
    """
    
    def __init__(
        self,
        config,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        """Initialize builder - automatic discovery handles registration."""
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config = config
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the SageMaker step."""
        # Implementation here
        pass
```

### 2. File Organization

Proper file organization for automatic discovery:

```
src/cursus/steps/builders/
├── __init__.py                          # Export builders
├── builder_your_new_step.py            # Your builder (follows naming convention)
├── builder_xgboost_training_step.py    # Example existing builder
└── ...
```

### 3. Registry Integration

Step definitions in the registry system:

```python
# In src/cursus/registry/step_names.py
STEP_NAMES = {
    "YourNewStep": {
        "config_class": "YourNewStepConfig",
        "builder_step_name": "YourNewStepBuilder",
        "spec_type": "YourNewStep",
        "sagemaker_step_type": "Processing",
        "description": "Description of your new step"
    }
}
```

## Advanced Features

### 1. Workspace-Aware Discovery

Support for multiple workspace contexts:

```python
from cursus.step_catalog import StepCatalog
from pathlib import Path

# Initialize with specific workspace directories
workspace_dirs = [
    Path("development/projects/project_alpha"),
    Path("development/projects/project_beta")
]

catalog = StepCatalog(workspace_dirs=workspace_dirs)

# Catalog discovers builders from both shared and project-specific locations
step_types = catalog.list_supported_step_types()
```

### 2. Job Type Variant Handling

Automatic handling of job type variants:

```python
from cursus.step_catalog import StepCatalog

catalog = StepCatalog()

# Handles variants like "CradleDataLoading_training", "CradleDataLoading_calibration"
training_builder = catalog.get_builder_for_step_type("CradleDataLoading_training")
calibration_builder = catalog.get_builder_for_step_type("CradleDataLoading_calibration")

# Both resolve to the same builder class but with different specifications
assert training_builder == calibration_builder  # Same class
```

### 3. Performance Optimization

Built-in caching and performance optimization:

```python
from cursus.step_catalog import StepCatalog
import time

# Measure catalog initialization performance
start_time = time.time()
catalog = StepCatalog()
init_time = time.time() - start_time

print(f"Catalog initialized in {init_time:.3f}s")

# Measure discovery performance
start_time = time.time()
step_types = catalog.list_supported_step_types()
discovery_time = time.time() - start_time

print(f"Discovered {len(step_types)} step types in {discovery_time:.3f}s")
```

## Migration from StepBuilderRegistry

### 1. Code Updates

**Old Pattern (StepBuilderRegistry)**:
```python
from cursus.registry.builder_registry import StepBuilderRegistry

# Old approach
registry = StepBuilderRegistry()
builder = registry.get_builder_for_config(config)
```

**New Pattern (StepCatalog)**:
```python
from cursus.step_catalog import StepCatalog

# New approach
catalog = StepCatalog()
builder = catalog.get_builder_for_config(config)
```

### 2. Builder Class Updates

**Old Pattern (with decorator)**:
```python
from cursus.registry.builder_registry import register_builder

@register_builder()
class YourStepBuilder(StepBuilderBase):
    pass
```

**New Pattern (no decorator)**:
```python
class YourStepBuilder(StepBuilderBase):
    """Automatically discovered by StepCatalog - no decorator needed!"""
    pass
```

### 3. Import Updates

Update all imports across your codebase:

```python
# OLD IMPORTS - Remove these
from cursus.registry.builder_registry import StepBuilderRegistry
from cursus.registry.builder_registry import register_builder

# NEW IMPORTS - Use these
from cursus.step_catalog import StepCatalog
```

## Testing and Validation

### 1. Catalog Functionality Testing

```python
import unittest
from cursus.step_catalog import StepCatalog

class TestStepCatalogIntegration(unittest.TestCase):
    def setUp(self):
        self.catalog = StepCatalog()
    
    def test_step_discovery(self):
        """Test that catalog discovers step types."""
        step_types = self.catalog.list_supported_step_types()
        self.assertGreater(len(step_types), 0)
        self.assertIn("BatchTransform", step_types)
    
    def test_builder_resolution(self):
        """Test builder resolution for known step types."""
        builder_class = self.catalog.get_builder_for_step_type("BatchTransform")
        self.assertIsNotNone(builder_class)
        self.assertTrue(hasattr(builder_class, 'create_step'))
    
    def test_config_to_builder_mapping(self):
        """Test config-to-builder resolution."""
        from cursus.steps.configs.config_batch_transform_step import BatchTransformStepConfig
        
        config = BatchTransformStepConfig(job_type="training")
        builder_class = self.catalog.get_builder_for_config(config)
        self.assertIsNotNone(builder_class)
    
    def test_legacy_alias_support(self):
        """Test legacy alias resolution."""
        legacy_builder = self.catalog.get_builder_for_step_type("PytorchTraining")
        canonical_builder = self.catalog.get_builder_for_step_type("PyTorchTraining")
        self.assertEqual(legacy_builder, canonical_builder)
    
    def test_builder_availability_validation(self):
        """Test builder availability validation."""
        step_types = ["BatchTransform", "XGBoostTraining"]
        availability = self.catalog.validate_builder_availability(step_types)
        
        self.assertEqual(len(availability), 2)
        for step_type, available in availability.items():
            self.assertIsInstance(available, bool)
```

### 2. Performance Testing

```python
import time
import unittest
from cursus.step_catalog import StepCatalog

class TestStepCatalogPerformance(unittest.TestCase):
    def test_initialization_performance(self):
        """Test catalog initialization performance."""
        start_time = time.time()
        catalog = StepCatalog()
        init_time = time.time() - start_time
        
        # Should initialize quickly
        self.assertLess(init_time, 1.0)  # Less than 1 second
    
    def test_discovery_performance(self):
        """Test step discovery performance."""
        catalog = StepCatalog()
        
        start_time = time.time()
        step_types = catalog.list_supported_step_types()
        discovery_time = time.time() - start_time
        
        # Should discover quickly
        self.assertLess(discovery_time, 0.1)  # Less than 100ms
        self.assertGreater(len(step_types), 0)
    
    def test_builder_resolution_performance(self):
        """Test builder resolution performance."""
        catalog = StepCatalog()
        
        # Test multiple resolutions
        step_types = ["BatchTransform", "XGBoostTraining", "PyTorchModel"]
        
        start_time = time.time()
        for step_type in step_types:
            builder_class = catalog.get_builder_for_step_type(step_type)
        resolution_time = time.time() - start_time
        
        # Should resolve quickly
        self.assertLess(resolution_time, 0.1)  # Less than 100ms for 3 resolutions
```

## Best Practices

### 1. Catalog Usage

- **Single Instance**: Use a single StepCatalog instance per application
- **Early Initialization**: Initialize catalog early in application lifecycle
- **Caching**: Leverage built-in caching for performance
- **Error Handling**: Handle cases where builders are not found

### 2. Builder Development

- **Follow Naming Conventions**: Use `builder_*.py` files and `*StepBuilder` classes
- **No Manual Registration**: Let automatic discovery handle registration
- **Registry Integration**: Ensure step definitions exist in `step_names.py`
- **Proper Inheritance**: Extend `StepBuilderBase` for all builders

### 3. Testing

- **Integration Testing**: Test catalog integration in your components
- **Performance Testing**: Validate catalog performance in your use cases
- **Builder Testing**: Test that your builders are discoverable
- **Availability Testing**: Validate builder availability for your step types

## Troubleshooting

### Common Issues

**1. Builder Not Found**
```python
# Issue: get_builder_for_step_type returns None
builder = catalog.get_builder_for_step_type("MyStep")  # Returns None

# Solutions:
# - Check file naming: builder_my_step.py
# - Check class naming: MyStepBuilder
# - Check registry entry in step_names.py
# - Verify file is in src/cursus/steps/builders/
```

**2. Config Resolution Fails**
```python
# Issue: get_builder_for_config returns None
builder = catalog.get_builder_for_config(config)  # Returns None

# Solutions:
# - Check config class name matches registry
# - Verify job_type attribute if using variants
# - Check registry mapping in step_names.py
```

**3. Import Errors**
```python
# Issue: ImportError when loading builders
# Solutions:
# - Remove old @register_builder imports
# - Check builder file syntax
# - Verify all dependencies are available
```

### Debugging

Enable debug logging to troubleshoot issues:

```python
import logging
from cursus.step_catalog import StepCatalog

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Initialize catalog with debug info
catalog = StepCatalog()

# Check catalog state
print(f"Discovered step types: {catalog.list_supported_step_types()}")
print(f"Builder map: {list(catalog.get_builder_map().keys())}")
```

## Related Documentation

- **[Step Builder Implementation](step_builder.md)** - Detailed guide to step builder implementation patterns
- **[Adding a New Pipeline Step](adding_new_pipeline_step.md)** - Complete guide to adding new steps
- **[Creation Process](creation_process.md)** - Step-by-step creation process
- **[Validation Framework Guide](validation_framework_guide.md)** - Comprehensive validation system
- **[Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md)** - Pipeline catalog integration

## Conclusion

The StepCatalog system provides a robust, scalable foundation for step discovery and builder resolution. By following the patterns and practices outlined in this guide, you can effectively integrate with the StepCatalog system and build maintainable pipeline components.

Key benefits of the StepCatalog system:
- **Automatic Discovery**: No manual registration required
- **Performance**: Built-in caching and optimization
- **Flexibility**: Support for job type variants and workspace contexts
- **Backward Compatibility**: Legacy alias support for smooth migration
- **Clean Architecture**: Clear separation of concerns with registry system

The StepCatalog system represents a significant improvement over the legacy StepBuilderRegistry, providing better maintainability, performance, and developer experience.
