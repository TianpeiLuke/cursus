---
tags:
  - code
  - core
  - base
  - enums
  - type_definitions
keywords:
  - enumeration
  - dependency types
  - node types
  - pipeline classification
  - type safety
topics:
  - type system
  - pipeline architecture
  - dependency management
language: python
date of note: 2025-08-07
---

# Core Base Enumerations

## Overview

The `enums.py` module defines shared enumeration types used across multiple base classes in the cursus framework. These enums provide type safety and avoid circular imports by serving as a single source of truth for common type definitions.

## Purpose

This module contains enums that are used across multiple base classes to:
- Classify different types of dependencies in the pipeline
- Categorize nodes based on their input/output characteristics
- Provide consistent type definitions without circular dependencies
- Enable type-safe operations and comparisons

## Enumerations

### DependencyType

Defines the types of dependencies that can exist in the pipeline.

```python
class DependencyType(Enum):
    MODEL_ARTIFACTS = "model_artifacts"
    PROCESSING_OUTPUT = "processing_output"
    TRAINING_DATA = "training_data"
    HYPERPARAMETERS = "hyperparameters"
    PAYLOAD_SAMPLES = "payload_samples"
    CUSTOM_PROPERTY = "custom_property"
```

#### Values

- **MODEL_ARTIFACTS**: Model artifacts produced by training steps
- **PROCESSING_OUTPUT**: Output from data processing steps
- **TRAINING_DATA**: Training datasets for model training
- **HYPERPARAMETERS**: Model hyperparameters and configuration
- **PAYLOAD_SAMPLES**: Sample payloads for testing and validation
- **CUSTOM_PROPERTY**: Custom properties defined by specific steps

#### Usage Examples

```python
from cursus.core.base.enums import DependencyType

# Creating dependency specifications
training_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.TRAINING_DATA,
    required=True
)

# Type checking
if dep.dependency_type == DependencyType.MODEL_ARTIFACTS:
    handle_model_artifacts(dep)
```

### NodeType

Classifies nodes in the pipeline based on their dependency and output characteristics.

```python
class NodeType(Enum):
    SOURCE = "source"      # No dependencies, has outputs
    INTERNAL = "internal"  # Has both dependencies and outputs
    SINK = "sink"         # Has dependencies, no outputs
    SINGULAR = "singular" # No dependencies, no outputs
```

#### Values

- **SOURCE**: Nodes that produce outputs but have no dependencies (e.g., data loading steps)
- **INTERNAL**: Nodes that consume inputs and produce outputs (e.g., processing, training steps)
- **SINK**: Nodes that consume inputs but produce no outputs (e.g., model registration)
- **SINGULAR**: Standalone nodes with no inputs or outputs (e.g., cleanup operations)

#### Usage Examples

```python
from cursus.core.base.enums import NodeType

# Defining step specifications
data_load_spec = StepSpecification(
    step_type="DataLoadingStep",
    node_type=NodeType.SOURCE,
    outputs={"data": output_spec}
)

training_spec = StepSpecification(
    step_type="TrainingStep", 
    node_type=NodeType.INTERNAL,
    dependencies={"data": dep_spec},
    outputs={"model": model_spec}
)
```

## Implementation Details

### Custom Equality and Hashing

Both enums implement custom `__eq__` and `__hash__` methods to ensure proper behavior:

```python
def __eq__(self, other):
    """Compare enum instances by value."""
    if isinstance(other, DependencyType):  # or NodeType
        return self.value == other.value
    return super().__eq__(other)
    
def __hash__(self):
    """Ensure hashability is maintained when used as dictionary keys."""
    return hash(self.value)
```

#### Benefits

1. **Value-based Comparison**: Enums compare by their string values, not object identity
2. **Dictionary Keys**: Can be safely used as dictionary keys
3. **Serialization**: Consistent behavior when serializing/deserializing
4. **Cross-instance Equality**: Enum instances from different contexts compare correctly

### Design Rationale

#### Why Separate Enums Module?

1. **Circular Import Prevention**: Enums have no dependencies and can be imported anywhere
2. **Single Source of Truth**: All type definitions in one place
3. **Reusability**: Can be used across multiple modules without coupling
4. **Type Safety**: Provides compile-time type checking

#### Value Choices

- **String Values**: Human-readable and serialization-friendly
- **Snake Case**: Consistent with Python naming conventions
- **Descriptive Names**: Clear meaning without additional documentation

## Usage Patterns

### In Specifications

```python
# Dependency specification
dep_spec = DependencySpec(
    logical_name="model_input",
    dependency_type=DependencyType.MODEL_ARTIFACTS,
    required=True
)

# Output specification  
output_spec = OutputSpec(
    logical_name="processed_data",
    output_type=DependencyType.PROCESSING_OUTPUT,
    property_path="properties.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri"
)
```

### In Step Classification

```python
# Validate node type constraints
if spec.node_type == NodeType.SOURCE:
    if spec.dependencies:
        raise ValueError("SOURCE nodes cannot have dependencies")
    if not spec.outputs:
        raise ValueError("SOURCE nodes must have outputs")
```

### In Dependency Resolution

```python
# Filter dependencies by type
model_deps = [
    dep for dep in dependencies 
    if dep.dependency_type == DependencyType.MODEL_ARTIFACTS
]

# Route based on node type
if step.node_type == NodeType.SINK:
    # Final step in pipeline
    finalize_pipeline(step)
```

## Validation and Constraints

### DependencyType Validation

The enum values are used to validate:
- Compatibility between step outputs and dependencies
- Proper data flow through the pipeline
- Semantic matching during dependency resolution

### NodeType Validation

Node types enforce structural constraints:
- **SOURCE**: Must have outputs, cannot have dependencies
- **INTERNAL**: Must have both dependencies and outputs  
- **SINK**: Must have dependencies, cannot have outputs
- **SINGULAR**: Cannot have dependencies or outputs

## Extension Guidelines

When adding new enum values:

1. **Naming**: Use descriptive, snake_case names
2. **Values**: Use string values matching the name in snake_case
3. **Documentation**: Add clear descriptions and usage examples
4. **Validation**: Update validation logic in dependent classes
5. **Backward Compatibility**: Consider impact on existing code

## Integration Points

These enums are used throughout the framework:

- **Specifications**: Define dependency and output types
- **Validation**: Enforce pipeline structure constraints
- **Resolution**: Match dependencies to outputs
- **Serialization**: Convert to/from JSON and YAML
- **Documentation**: Generate pipeline diagrams and documentation

The enums serve as the foundation for type-safe pipeline definition and validation across the entire cursus framework.
