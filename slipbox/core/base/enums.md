---
tags:
  - code
  - base
  - enums
  - dependency_types
  - node_types
keywords:
  - DependencyType
  - NodeType
  - pipeline enums
  - dependency classification
  - node classification
  - enum types
topics:
  - base enums
  - dependency types
  - node types
language: python
date of note: 2024-12-07
---

# Base Enums

Shared enums for the cursus core base classes that provide standardized type definitions for dependencies and nodes across the pipeline system.

## Overview

This module contains enums that are used across multiple base classes to avoid circular imports and provide a single source of truth for type definitions. These enums establish the fundamental classification systems for pipeline components, enabling consistent type checking and dependency resolution throughout the system.

The enums support advanced features including value-based equality comparison for consistent behavior across different instances, hash functionality for use as dictionary keys and set members, comprehensive type coverage for all pipeline dependency patterns, and integration with the dependency resolution system for intelligent matching.

## Classes and Methods

### Enums
- [`DependencyType`](#dependencytype) - Types of dependencies in the pipeline system
- [`NodeType`](#nodetype) - Types of nodes based on dependency/output characteristics

## API Reference

### DependencyType

_enum_ cursus.core.base.enums.DependencyType

Types of dependencies in the pipeline. This enum defines the standard dependency types used throughout the pipeline system for classification and compatibility checking.

**Values:**
- **MODEL_ARTIFACTS** – Model artifacts produced by training steps, typically containing trained model files and metadata.
- **PROCESSING_OUTPUT** – Output from processing steps, including transformed data, feature engineering results, and preprocessed datasets.
- **TRAINING_DATA** – Training datasets used by training steps, including processed and raw training data.
- **HYPERPARAMETERS** – Hyperparameter configurations and tuning results used for model training and optimization.
- **PAYLOAD_SAMPLES** – Sample payloads used for testing, validation, and inference endpoint configuration.
- **CUSTOM_PROPERTY** – Custom properties and metadata that don't fit standard categories.

```python
from cursus.core.base.enums import DependencyType

# Use dependency types in specifications
model_dep_type = DependencyType.MODEL_ARTIFACTS
data_dep_type = DependencyType.TRAINING_DATA
config_dep_type = DependencyType.HYPERPARAMETERS

# Compare dependency types
if dep_type == DependencyType.MODEL_ARTIFACTS:
    print("This is a model artifact dependency")

# Use as dictionary keys
dependency_handlers = {
    DependencyType.MODEL_ARTIFACTS: handle_model_artifacts,
    DependencyType.TRAINING_DATA: handle_training_data,
    DependencyType.HYPERPARAMETERS: handle_hyperparameters
}
```

### NodeType

_enum_ cursus.core.base.enums.NodeType

Types of nodes in the pipeline based on their dependency/output characteristics. This enum classifies pipeline nodes according to their input/output patterns, enabling proper DAG construction and validation.

**Values:**
- **SOURCE** – No dependencies, has outputs. Examples include data loading steps, parameter initialization, and external data ingestion.
- **INTERNAL** – Has both dependencies and outputs. Examples include processing steps, training steps, and transformation operations.
- **SINK** – Has dependencies, no outputs. Examples include model registration, result publishing, and final output steps.
- **SINGULAR** – No dependencies, no outputs. Examples include standalone operations, cleanup tasks, and independent utilities.

```python
from cursus.core.base.enums import NodeType

# Classify nodes by their characteristics
data_loader_type = NodeType.SOURCE      # Loads data, no dependencies
preprocessing_type = NodeType.INTERNAL  # Processes data, depends on loader
training_type = NodeType.INTERNAL       # Trains model, depends on preprocessing
registration_type = NodeType.SINK       # Registers model, depends on training

# Use in DAG validation
def validate_node_connections(node_type, has_dependencies, has_outputs):
    if node_type == NodeType.SOURCE:
        return not has_dependencies and has_outputs
    elif node_type == NodeType.INTERNAL:
        return has_dependencies and has_outputs
    elif node_type == NodeType.SINK:
        return has_dependencies and not has_outputs
    elif node_type == NodeType.SINGULAR:
        return not has_dependencies and not has_outputs
```

## Usage Examples

### Dependency Type Classification
```python
from cursus.core.base.enums import DependencyType

# Define dependency types for different pipeline components
def classify_dependency(dependency_name, data_source):
    """Classify a dependency based on its characteristics."""
    if "model" in dependency_name.lower():
        return DependencyType.MODEL_ARTIFACTS
    elif "data" in dependency_name.lower() or "dataset" in dependency_name.lower():
        return DependencyType.TRAINING_DATA
    elif "config" in dependency_name.lower() or "param" in dependency_name.lower():
        return DependencyType.HYPERPARAMETERS
    elif "sample" in dependency_name.lower() or "payload" in dependency_name.lower():
        return DependencyType.PAYLOAD_SAMPLES
    elif "output" in dependency_name.lower() or "result" in dependency_name.lower():
        return DependencyType.PROCESSING_OUTPUT
    else:
        return DependencyType.CUSTOM_PROPERTY

# Example usage
dep_type = classify_dependency("training_data_processed", "s3://bucket/data/")
print(f"Classified as: {dep_type.value}")
```

### Node Type Analysis
```python
from cursus.core.base.enums import NodeType

# Analyze pipeline structure using node types
def analyze_pipeline_structure(nodes_with_types):
    """Analyze pipeline structure based on node types."""
    type_counts = {node_type: 0 for node_type in NodeType}
    
    for node_name, node_type in nodes_with_types.items():
        type_counts[node_type] += 1
    
    print("Pipeline Structure Analysis:")
    print(f"  Source nodes: {type_counts[NodeType.SOURCE]}")
    print(f"  Internal nodes: {type_counts[NodeType.INTERNAL]}")
    print(f"  Sink nodes: {type_counts[NodeType.SINK]}")
    print(f"  Singular nodes: {type_counts[NodeType.SINGULAR]}")
    
    # Validate pipeline structure
    if type_counts[NodeType.SOURCE] == 0:
        print("Warning: No source nodes found - pipeline may lack data inputs")
    if type_counts[NodeType.SINK] == 0:
        print("Warning: No sink nodes found - pipeline may not produce final outputs")

# Example pipeline analysis
pipeline_nodes = {
    "data_loader": NodeType.SOURCE,
    "preprocessing": NodeType.INTERNAL,
    "training": NodeType.INTERNAL,
    "evaluation": NodeType.INTERNAL,
    "model_registration": NodeType.SINK
}

analyze_pipeline_structure(pipeline_nodes)
```

### Enum Comparison and Hashing
```python
from cursus.core.base.enums import DependencyType, NodeType

# Demonstrate enum equality and hashing
def test_enum_behavior():
    """Test enum equality and hashing behavior."""
    
    # Value-based equality
    dep1 = DependencyType.MODEL_ARTIFACTS
    dep2 = DependencyType.MODEL_ARTIFACTS
    assert dep1 == dep2  # True - same value
    
    # Use as dictionary keys
    dependency_config = {
        DependencyType.MODEL_ARTIFACTS: {"s3_prefix": "models/"},
        DependencyType.TRAINING_DATA: {"s3_prefix": "data/"},
        DependencyType.HYPERPARAMETERS: {"s3_prefix": "configs/"}
    }
    
    # Access using enum instances
    model_config = dependency_config[DependencyType.MODEL_ARTIFACTS]
    print(f"Model config: {model_config}")
    
    # Use in sets
    required_types = {
        DependencyType.TRAINING_DATA,
        DependencyType.HYPERPARAMETERS
    }
    
    if DependencyType.TRAINING_DATA in required_types:
        print("Training data is required")

test_enum_behavior()
```

## Related Documentation

- [Specification Base](specification_base.md) - Uses these enums for dependency and output specifications
- [Config Base](config_base.md) - Base configuration classes that reference these enum types
- [Builder Base](builder_base.md) - Step builders that work with these dependency types
- [Dependency Resolver](../deps/dependency_resolver.md) - Uses these enums for compatibility checking
- [Contract Base](contract_base.md) - Contract specifications that utilize these enum classifications
