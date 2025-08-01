# Pipeline Dependencies

This module provides dependency resolution and specification management for pipeline components. It enables automatic dependency resolution between pipeline steps based on their input/output specifications.

## Overview

The Pipeline Dependencies module is a key component of the specification-driven pipeline building system. It provides the infrastructure for automatic dependency resolution, which allows pipeline steps to be connected based on their specifications rather than manual wiring.

## Core Components

### Base Specifications
- **[Base Specifications](base_specifications.md)** - Foundation classes for dependency specifications using Pydantic V2 models
- **[Specification Registry](specification_registry.md)** - Registry for managing step specifications with context awareness
- **[Registry Manager](registry_manager.md)** - Multi-context registry management with complete isolation

### Dependency Resolution
- **[Dependency Resolver](dependency_resolver.md)** - Core dependency resolution algorithms with compatibility scoring
- **[Semantic Matcher](semantic_matcher.md)** - Multi-metric semantic matching for compatible specifications
- **[Factory](factory.md)** - Factory functions for creating and managing dependency resolution components

## Key Features

1. **Specification-Driven Dependency Resolution** - Automatically connects pipeline steps based on their declared specifications
2. **Semantic Matching** - Intelligent matching of dependencies and outputs using multiple similarity metrics
3. **Context Isolation** - Registry isolation for multiple pipelines with independent dependency resolution
4. **Thread Safety** - Thread-local component management for concurrent pipeline building
5. **Property Reference System** - Bridges between definition-time and runtime property references
6. **Compatibility Scoring** - Multi-factor compatibility scoring for optimal dependency matching
7. **Step Type Awareness** - Recognizes compatible step types for more intelligent matching
8. **Data Type Compatibility** - Ensures matched dependencies have compatible data types

## How Specification-Driven Dependency Resolution Works

The dependency resolution process follows these steps:

1. **Specification Registration**:
   - Each step builder registers its specification with the registry
   - Specifications declare dependencies (inputs) and outputs

2. **Dependency Analysis**:
   - The dependency resolver analyzes the specifications of all steps
   - It identifies required inputs and available outputs

3. **Compatibility Scoring**:
   - For each dependency, the resolver calculates compatibility scores with all available outputs
   - Factors include name similarity, type compatibility, and explicit compatibility declarations

4. **Optimal Matching**:
   - The resolver selects the best match for each dependency based on compatibility score
   - It ensures all required dependencies are satisfied

5. **Property Reference Creation**:
   - The resolver creates property references to bridge definition-time and runtime
   - These references are converted to SageMaker property references during pipeline assembly

## Dependency Resolution Example

```python
from src.pipeline_deps.registry_manager import RegistryManager
from src.pipeline_deps.dependency_resolver import create_dependency_resolver

# Create a registry manager and get a registry
registry_manager = RegistryManager()
registry = registry_manager.get_registry("my_pipeline")

# Create a dependency resolver
resolver = create_dependency_resolver(registry)

# Register step specifications
resolver.register_specification("preprocessing", preprocessing_spec)
resolver.register_specification("training", training_spec)

# Resolve dependencies between steps
dependencies = resolver.resolve_dependencies(source_step="preprocessing", target_step="training")

# Output: {
#    "training_data": PropertyReference(step_name="preprocessing", output_name="processed_data"),
#    "validation_data": PropertyReference(step_name="preprocessing", output_name="validation_data")
# }
```

## Using the Factory Module

The factory module provides convenient functions for creating and managing dependency resolution components:

```python
from src.pipeline_deps.factory import create_pipeline_components, dependency_resolution_context

# Create components with a named context
components = create_pipeline_components("my_pipeline")
registry_manager = components["registry_manager"]
resolver = components["resolver"]

# Or use a context manager for scoped components
with dependency_resolution_context("my_pipeline") as components:
    registry_manager = components["registry_manager"]
    resolver = components["resolver"]
    
    # Use components to build pipeline
    # ...

# Components are cleaned up when context exits
```

## Thread Safety

The module provides thread-safe component management for concurrent pipeline building:

```python
from src.pipeline_deps.factory import get_thread_components

# In Thread 1
components1 = get_thread_components()
resolver1 = components1["resolver"]

# In Thread 2
components2 = get_thread_components()
resolver2 = components2["resolver"]

# Each thread gets its own isolated components
assert resolver1 is not resolver2
```

## Integration with Pipeline Builder

This module integrates directly with the Pipeline Builder system:

```python
from src.pipeline_builder.pipeline_template_base import PipelineTemplateBase

class MyPipelineTemplate(PipelineTemplateBase):
    # Template definition...
    pass

# The template automatically uses dependency resolution components
template = MyPipelineTemplate.build_with_context(config_path="config.json")
```

## File Structure

```
slipbox/v2/pipeline_deps/
├── README.md                    # This overview document
├── base_specifications.md       # Core specification data structures
├── dependency_resolver.md       # Dependency resolution algorithms
├── factory.md                   # Component factory functions
├── registry_manager.md          # Multi-context registry management
├── semantic_matcher.md          # Semantic similarity matching
└── specification_registry.md    # Context-aware specification registry
```

## Related Documentation

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Core abstract class for pipeline templates
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Assembles pipeline steps using specifications
- [Pipeline Builder Overview](../pipeline_builder/README.md): Complete pipeline building system
- [Template Implementation](../pipeline_builder/template_implementation.md): Template implementation details

### Pipeline Structure
- [Pipeline DAG](../pipeline_dag/pipeline_dag.md): DAG structure for pipeline steps
- [Pipeline DAG Overview](../pipeline_dag/README.md): DAG-based pipeline structure concepts

### Pipeline Components
- [Pipeline Steps](../pipeline_steps/README.md): Available steps and their specifications
- [Script Contracts](../pipeline_script_contracts/README.md): Script contracts and validation
- [Base Script Contract](../pipeline_script_contracts/base_script_contract.md): Foundation for script contracts

### Dependency Components
- [Dependency Resolver](dependency_resolver.md): Resolves step dependencies
- [Base Specifications](base_specifications.md): Core specification structures
- [Semantic Matcher](semantic_matcher.md): Name matching algorithms
- [Property Reference](property_reference.md): Runtime property bridge
