---
tags:
  - code
  - deps
  - specification_registry
  - context_isolation
  - step_management
keywords:
  - SpecificationRegistry
  - step specifications
  - context isolation
  - specification management
  - dependency compatibility
  - step type tracking
  - registry operations
topics:
  - specification registry
  - context management
  - step specification storage
language: python
date of note: 2024-12-07
---

# Specification Registry

Specification registry for managing step specifications with context isolation, providing core registry functionality for storing, retrieving, and managing step specifications within isolated contexts.

## Overview

The `SpecificationRegistry` class provides a context-aware registry system for managing step specifications with complete isolation between different contexts. This registry serves as the central storage and retrieval mechanism for step specifications, enabling organized management of pipeline step definitions within specific contexts such as pipelines, environments, or experiments.

The registry supports comprehensive specification management including step specification storage with validation, step type tracking and categorization, compatibility analysis between dependencies and outputs, context-aware isolation for multi-pipeline environments, and efficient retrieval operations with various query patterns.

The system provides advanced features including automatic specification validation using Pydantic models, step type to name mapping for efficient lookups, compatibility scoring algorithms for dependency resolution, semantic keyword matching for intelligent output discovery, and comprehensive logging for debugging and monitoring.

## Classes and Methods

### Classes
- [`SpecificationRegistry`](#specificationregistry) - Context-aware registry for managing step specifications with isolation

## API Reference

### SpecificationRegistry

_class_ cursus.core.deps.specification_registry.SpecificationRegistry(_context_name="default"_)

Context-aware registry for managing step specifications with isolation. This class provides a centralized storage and retrieval system for step specifications, ensuring complete isolation between different contexts while supporting efficient querying and compatibility analysis.

**Parameters:**
- **context_name** (_str_) – Name of the context this registry belongs to (e.g., pipeline name, environment). Defaults to "default".

```python
from cursus.core.deps.specification_registry import SpecificationRegistry

# Create registry for specific context
registry = SpecificationRegistry(context_name="ml-pipeline-v1")

# Create default registry
default_registry = SpecificationRegistry()

print(f"Registry context: {registry.context_name}")
```

#### register

register(_step_name_, _specification_)

Register a step specification. This method stores a step specification in the registry with validation and automatic step type tracking.

**Parameters:**
- **step_name** (_str_) – Name of the step to register.
- **specification** (_StepSpecification_) – Step specification instance containing dependencies, outputs, and metadata.

```python
from cursus.core.base import StepSpecification, DependencySpec, OutputSpec, DependencyType

# Create step specification
preprocessing_spec = StepSpecification(
    step_type="TabularPreprocessing",
    node_type="processing",
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.TRAINING_DATA,
            data_type="S3Uri",
            required=True
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.Outputs['DATA']",
            output_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
    }
)

# Register specification
registry.register("preprocessing_step", preprocessing_spec)
```

#### get_specification

get_specification(_step_name_)

Get specification by step name. This method retrieves a previously registered step specification by its name.

**Parameters:**
- **step_name** (_str_) – Name of the step to retrieve.

**Returns:**
- **Optional[StepSpecification]** – Step specification if found, None otherwise.

```python
# Retrieve specification
spec = registry.get_specification("preprocessing_step")
if spec:
    print(f"Found specification for {spec.step_type}")
else:
    print("Specification not found")
```

#### get_specifications_by_type

get_specifications_by_type(_step_type_)

Get all specifications of a given step type. This method retrieves all step specifications that match the specified step type.

**Parameters:**
- **step_type** (_str_) – Step type to filter by (e.g., "TabularPreprocessing", "Training").

**Returns:**
- **List[StepSpecification]** – List of step specifications matching the step type.

```python
# Get all preprocessing specifications
preprocessing_specs = registry.get_specifications_by_type("TabularPreprocessing")
print(f"Found {len(preprocessing_specs)} preprocessing specifications")

for spec in preprocessing_specs:
    print(f"  - {spec.step_type}: {len(spec.dependencies)} deps, {len(spec.outputs)} outputs")
```

#### list_step_names

list_step_names()

Get list of all registered step names. This method returns all step names currently registered in the registry.

**Returns:**
- **List[str]** – List of all registered step names.

```python
# List all registered steps
step_names = registry.list_step_names()
print(f"Registered steps: {step_names}")
```

#### list_step_types

list_step_types()

Get list of all registered step types. This method returns all unique step types currently registered in the registry.

**Returns:**
- **List[str]** – List of all registered step types.

```python
# List all step types
step_types = registry.list_step_types()
print(f"Available step types: {step_types}")
```

#### find_compatible_outputs

find_compatible_outputs(_dependency_spec_)

Find outputs compatible with a dependency specification. This method searches through all registered specifications to find outputs that are compatible with the given dependency specification.

**Parameters:**
- **dependency_spec** (_DependencySpec_) – Dependency specification to find compatible outputs for.

**Returns:**
- **List[Tuple[str, str, OutputSpec, float]]** – List of tuples containing (step_name, output_name, output_spec, compatibility_score) sorted by compatibility score (highest first).

```python
# Create dependency specification
training_data_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.TRAINING_DATA,
    data_type="S3Uri",
    required=True,
    compatible_sources=["TabularPreprocessing"],
    semantic_keywords=["processed", "clean", "transformed"]
)

# Find compatible outputs
compatible = registry.find_compatible_outputs(training_data_dep)
for step_name, output_name, output_spec, score in compatible:
    print(f"Compatible: {step_name}.{output_name} (score: {score:.3f})")
```

## Usage Examples

### Basic Registry Operations
```python
from cursus.core.deps.specification_registry import SpecificationRegistry
from cursus.core.base import StepSpecification, DependencySpec, OutputSpec, DependencyType

# Create registry for specific pipeline
pipeline_registry = SpecificationRegistry("training-pipeline-v2")

# Create multiple step specifications
preprocessing_spec = StepSpecification(
    step_type="TabularPreprocessing",
    node_type="processing",
    dependencies={
        "raw_data": DependencySpec(
            logical_name="raw_data",
            dependency_type=DependencyType.TRAINING_DATA,
            data_type="S3Uri",
            required=True
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            property_path="properties.Outputs['DATA']",
            output_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
    }
)

training_spec = StepSpecification(
    step_type="Training",
    node_type="training",
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.TRAINING_DATA,
            data_type="S3Uri",
            required=True
        )
    },
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            data_type="S3Uri"
        )
    }
)

# Register specifications
pipeline_registry.register("preprocessing", preprocessing_spec)
pipeline_registry.register("training", training_spec)

# Query registry
print(f"Registered steps: {pipeline_registry.list_step_names()}")
print(f"Step types: {pipeline_registry.list_step_types()}")
```

### Compatibility Analysis
```python
# Find compatible outputs for training step
training_data_dependency = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.TRAINING_DATA,
    data_type="S3Uri",
    required=True,
    compatible_sources=["TabularPreprocessing"],
    semantic_keywords=["processed", "clean"]
)

# Search for compatible outputs
compatible_outputs = pipeline_registry.find_compatible_outputs(training_data_dependency)

print("Compatible outputs for training data:")
for step_name, output_name, output_spec, score in compatible_outputs:
    print(f"  {step_name}.{output_name}: {score:.3f}")
    print(f"    Type: {output_spec.output_type}")
    print(f"    Path: {output_spec.property_path}")
```

### Multi-Context Management
```python
# Create separate registries for different contexts
training_registry = SpecificationRegistry("training-pipeline")
inference_registry = SpecificationRegistry("inference-pipeline")

# Register different specifications in each context
training_registry.register("preprocessing", training_preprocessing_spec)
inference_registry.register("preprocessing", inference_preprocessing_spec)

# Contexts are completely isolated
training_steps = training_registry.list_step_names()
inference_steps = inference_registry.list_step_names()

print(f"Training context: {training_steps}")
print(f"Inference context: {inference_steps}")
```

### Step Type Analysis
```python
# Analyze step types in registry
step_types = pipeline_registry.list_step_types()

for step_type in step_types:
    specs = pipeline_registry.get_specifications_by_type(step_type)
    print(f"\nStep type: {step_type}")
    print(f"  Count: {len(specs)}")
    
    for spec in specs:
        print(f"  - Dependencies: {len(spec.dependencies)}")
        print(f"  - Outputs: {len(spec.outputs)}")
        print(f"  - Required deps: {len(spec.list_required_dependencies())}")
```

## Related Documentation

- [Registry Manager](registry_manager.md) - Manages multiple specification registries with context isolation
- [Step Specification](../base/step_specification.md) - Step specification system used by the registry
- [Dependency Resolver](dependency_resolver.md) - Uses specification registry for dependency resolution
- [Factory](factory.md) - Factory functions that create specification registries
- [Semantic Matcher](semantic_matcher.md) - Used for intelligent compatibility analysis
