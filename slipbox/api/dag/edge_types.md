---
tags:
  - code
  - api
  - dag
  - edge-types
  - dependencies
keywords:
  - EdgeType
  - DependencyEdge
  - ConditionalEdge
  - ParallelEdge
  - EdgeCollection
  - typed dependencies
  - confidence scoring
  - property reference
topics:
  - edge types
  - typed dependencies
  - dependency management
  - confidence scoring
language: python
date of note: 2024-12-07
---

# Edge Types

Edge types for enhanced pipeline DAG with typed dependencies, defining various types of edges that can exist between pipeline steps with confidence scoring and property reference management.

## Overview

The Edge Types module defines a comprehensive system for representing typed dependencies between pipeline steps. It extends simple edge relationships with rich type information, confidence scoring for auto-resolved dependencies, and specialized edge types for different execution patterns.

The module provides Pydantic-based models for type safety and validation, confidence-based edge resolution for intelligent dependency matching, and a collection management system for efficient edge operations and statistics. It supports property reference generation for SageMaker integration and comprehensive validation and error reporting.

## Classes and Methods

### Classes
- [`EdgeType`](#edgetype) - Enumeration of edge types in pipeline DAG
- [`DependencyEdge`](#dependencyedge) - Typed dependency edge between step ports
- [`ConditionalEdge`](#conditionaledge) - Conditional dependency edge with condition expression
- [`ParallelEdge`](#paralleledge) - Parallel execution hint edge
- [`EdgeCollection`](#edgecollection) - Collection of edges with utility methods

## API Reference

### EdgeType

_class_ cursus.api.dag.edge_types.EdgeType(_Enum_)

Enumeration defining types of edges in the pipeline DAG.

**Values:**
- **DEPENDENCY** – Standard dependency edge
- **CONDITIONAL** – Conditional dependency
- **PARALLEL** – Parallel execution hint
- **SEQUENTIAL** – Sequential execution requirement

```python
from cursus.api.dag.edge_types import EdgeType

# Use edge types
edge_type = EdgeType.DEPENDENCY
conditional_type = EdgeType.CONDITIONAL
```

### DependencyEdge

_class_ cursus.api.dag.edge_types.DependencyEdge(_source_step_, _target_step_, _source_output_, _target_input_, _confidence=1.0_, _edge_type=EdgeType.DEPENDENCY_, _metadata=None_)

Represents a typed dependency edge between step ports with confidence scoring and metadata.

**Parameters:**
- **source_step** (_str_) – Name of the source step (min_length=1)
- **target_step** (_str_) – Name of the target step (min_length=1)
- **source_output** (_str_) – Logical name of source output (min_length=1)
- **target_input** (_str_) – Logical name of target input (min_length=1)
- **confidence** (_float_) – Confidence score for auto-resolved edges (0.0-1.0, default=1.0)
- **edge_type** (_EdgeType_) – Type of edge (default=EdgeType.DEPENDENCY)
- **metadata** (_Dict[str, Any]_) – Additional metadata (default=empty dict)

```python
from cursus.api.dag.edge_types import DependencyEdge, EdgeType

# Create standard dependency edge
edge = DependencyEdge(
    source_step="preprocessing",
    target_step="training",
    source_output="processed_data",
    target_input="training_data"
)

# Create auto-resolved edge with confidence
auto_edge = DependencyEdge(
    source_step="data_loading",
    target_step="preprocessing",
    source_output="raw_data",
    target_input="input_data",
    confidence=0.85,
    metadata={"auto_resolved": True}
)
```

#### to_property_reference_dict

to_property_reference_dict()

Convert edge to a property reference dictionary for SageMaker pipeline construction.

**Returns:**
- **Dict[str, Any]** – Property reference dictionary with SageMaker "Get" syntax

```python
edge = DependencyEdge(
    source_step="preprocessing",
    target_step="training", 
    source_output="processed_data",
    target_input="training_data"
)

prop_ref = edge.to_property_reference_dict()
# Returns: {"Get": "Steps.preprocessing.processed_data"}
```

#### is_high_confidence

is_high_confidence(_threshold=0.8_)

Check if this edge has high confidence above the specified threshold.

**Parameters:**
- **threshold** (_float_) – Confidence threshold (default=0.8)

**Returns:**
- **bool** – True if confidence >= threshold, False otherwise

```python
edge = DependencyEdge(
    source_step="step1", target_step="step2",
    source_output="out1", target_input="in1",
    confidence=0.9
)

is_high = edge.is_high_confidence()  # True (0.9 >= 0.8)
is_very_high = edge.is_high_confidence(0.95)  # False (0.9 < 0.95)
```

#### is_auto_resolved

is_auto_resolved()

Check if this edge was automatically resolved (confidence < 1.0).

**Returns:**
- **bool** – True if edge was auto-resolved, False if manual

```python
manual_edge = DependencyEdge(
    source_step="step1", target_step="step2",
    source_output="out1", target_input="in1",
    confidence=1.0
)

auto_edge = DependencyEdge(
    source_step="step1", target_step="step2", 
    source_output="out1", target_input="in1",
    confidence=0.85
)

print(manual_edge.is_auto_resolved())  # False
print(auto_edge.is_auto_resolved())    # True
```

### ConditionalEdge

_class_ cursus.api.dag.edge_types.ConditionalEdge(_source_step_, _target_step_, _source_output_, _target_input_, _condition=""_, _**kwargs_)

Represents a conditional dependency edge with condition expression.

**Parameters:**
- **condition** (_str_) – Condition expression (default="")
- **edge_type** (_EdgeType_) – Automatically set to EdgeType.CONDITIONAL
- **...** – All parameters from DependencyEdge

```python
from cursus.api.dag.edge_types import ConditionalEdge

# Create conditional edge
conditional_edge = ConditionalEdge(
    source_step="validation",
    target_step="deployment",
    source_output="validation_result",
    target_input="deploy_trigger",
    condition="validation_result.accuracy > 0.95",
    confidence=1.0
)
```

### ParallelEdge

_class_ cursus.api.dag.edge_types.ParallelEdge(_source_step_, _target_step_, _source_output_, _target_input_, _max_parallel=None_, _**kwargs_)

Represents a parallel execution hint edge with optional parallelism limits.

**Parameters:**
- **max_parallel** (_Optional[int]_) – Maximum parallel executions (>=1, default=None)
- **edge_type** (_EdgeType_) – Automatically set to EdgeType.PARALLEL
- **...** – All parameters from DependencyEdge

```python
from cursus.api.dag.edge_types import ParallelEdge

# Create parallel edge with limit
parallel_edge = ParallelEdge(
    source_step="data_split",
    target_step="parallel_training",
    source_output="data_partitions", 
    target_input="training_partition",
    max_parallel=4
)
```

### EdgeCollection

_class_ cursus.api.dag.edge_types.EdgeCollection()

Collection of edges with utility methods for management, indexing, and statistics.

```python
from cursus.api.dag.edge_types import EdgeCollection, DependencyEdge

# Create edge collection
collection = EdgeCollection()

# Add edges
edge1 = DependencyEdge(
    source_step="step1", target_step="step2",
    source_output="out1", target_input="in1"
)
edge_id = collection.add_edge(edge1)
```

#### add_edge

add_edge(_edge_)

Add an edge to the collection with automatic indexing and duplicate handling.

**Parameters:**
- **edge** (_DependencyEdge_) – DependencyEdge to add

**Returns:**
- **str** – Edge ID for the added edge

```python
collection = EdgeCollection()

edge = DependencyEdge(
    source_step="preprocessing",
    target_step="training",
    source_output="data",
    target_input="training_data"
)

edge_id = collection.add_edge(edge)
print(f"Added edge with ID: {edge_id}")
```

#### remove_edge

remove_edge(_edge_id_)

Remove an edge from the collection by its ID.

**Parameters:**
- **edge_id** (_str_) – ID of the edge to remove

**Returns:**
- **bool** – True if edge was removed, False if not found

```python
success = collection.remove_edge(edge_id)
if success:
    print("Edge removed successfully")
```

#### get_edges_from_step

get_edges_from_step(_step_name_)

Get all edges originating from a specific step.

**Parameters:**
- **step_name** (_str_) – Name of the source step

**Returns:**
- **List[DependencyEdge]** – List of edges from the step

```python
outgoing_edges = collection.get_edges_from_step("preprocessing")
print(f"Preprocessing has {len(outgoing_edges)} outgoing edges")
```

#### get_edges_to_step

get_edges_to_step(_step_name_)

Get all edges targeting a specific step.

**Parameters:**
- **step_name** (_str_) – Name of the target step

**Returns:**
- **List[DependencyEdge]** – List of edges to the step

```python
incoming_edges = collection.get_edges_to_step("training")
print(f"Training has {len(incoming_edges)} incoming edges")
```

#### get_edge

get_edge(_source_step_, _source_output_, _target_step_, _target_input_)

Get a specific edge by its components.

**Parameters:**
- **source_step** (_str_) – Name of source step
- **source_output** (_str_) – Name of source output
- **target_step** (_str_) – Name of target step
- **target_input** (_str_) – Name of target input

**Returns:**
- **Optional[DependencyEdge]** – Edge if found, None otherwise

```python
edge = collection.get_edge(
    source_step="preprocessing",
    source_output="data",
    target_step="training", 
    target_input="training_data"
)
```

#### list_all_edges

list_all_edges()

Get list of all edges in the collection.

**Returns:**
- **List[DependencyEdge]** – List of all edges

```python
all_edges = collection.list_all_edges()
print(f"Collection contains {len(all_edges)} edges")
```

#### list_auto_resolved_edges

list_auto_resolved_edges()

Get list of automatically resolved edges (confidence < 1.0).

**Returns:**
- **List[DependencyEdge]** – List of auto-resolved edges

```python
auto_edges = collection.list_auto_resolved_edges()
print(f"Found {len(auto_edges)} auto-resolved edges")
```

#### list_high_confidence_edges

list_high_confidence_edges(_threshold=0.8_)

Get list of high confidence edges above threshold.

**Parameters:**
- **threshold** (_float_) – Confidence threshold (default=0.8)

**Returns:**
- **List[DependencyEdge]** – List of high confidence edges

```python
high_conf_edges = collection.list_high_confidence_edges(0.9)
print(f"Found {len(high_conf_edges)} high confidence edges")
```

#### list_low_confidence_edges

list_low_confidence_edges(_threshold=0.6_)

Get list of low confidence edges that may need review.

**Parameters:**
- **threshold** (_float_) – Confidence threshold (default=0.6)

**Returns:**
- **List[DependencyEdge]** – List of low confidence edges

```python
low_conf_edges = collection.list_low_confidence_edges()
for edge in low_conf_edges:
    print(f"Low confidence edge: {edge} (confidence: {edge.confidence})")
```

#### get_step_dependencies

get_step_dependencies(_step_name_)

Get all dependencies for a step as a dictionary mapping input names to edges.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **Dict[str, DependencyEdge]** – Dictionary mapping input names to dependency edges

```python
dependencies = collection.get_step_dependencies("training")
for input_name, edge in dependencies.items():
    print(f"Input '{input_name}' comes from {edge.source_step}.{edge.source_output}")
```

#### validate_edges

validate_edges()

Validate all edges and return list of validation errors.

**Returns:**
- **List[str]** – List of validation error messages

```python
errors = collection.validate_edges()
if errors:
    print("Validation errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("All edges are valid")
```

#### get_statistics

get_statistics()

Get comprehensive statistics about the edge collection.

**Returns:**
- **Dict[str, Any]** – Dictionary containing collection statistics

```python
stats = collection.get_statistics()
print(f"Total edges: {stats['total_edges']}")
print(f"Auto-resolved: {stats['auto_resolved_edges']}")
print(f"Average confidence: {stats['average_confidence']:.3f}")
print(f"Edge types: {stats['edge_types']}")
```

## Usage Examples

### Basic Edge Creation and Management

```python
from cursus.api.dag.edge_types import DependencyEdge, EdgeCollection, EdgeType

# Create edge collection
collection = EdgeCollection()

# Create and add edges
edge1 = DependencyEdge(
    source_step="data_loading",
    target_step="preprocessing",
    source_output="raw_data",
    target_input="input_data",
    confidence=1.0
)

edge2 = DependencyEdge(
    source_step="preprocessing", 
    target_step="training",
    source_output="processed_data",
    target_input="training_data",
    confidence=0.85,
    metadata={"auto_resolved": True}
)

# Add edges to collection
collection.add_edge(edge1)
collection.add_edge(edge2)

# Query edges
training_deps = collection.get_edges_to_step("training")
print(f"Training dependencies: {len(training_deps)}")
```

### Confidence-Based Edge Analysis

```python
# Analyze edge confidence
high_conf = collection.list_high_confidence_edges(0.9)
low_conf = collection.list_low_confidence_edges(0.7)

print(f"High confidence edges (>0.9): {len(high_conf)}")
print(f"Low confidence edges (<0.7): {len(low_conf)}")

# Review low confidence edges
for edge in low_conf:
    print(f"Review needed: {edge} (confidence: {edge.confidence:.3f})")
```

### Specialized Edge Types

```python
from cursus.api.dag.edge_types import ConditionalEdge, ParallelEdge

# Create conditional edge
conditional = ConditionalEdge(
    source_step="validation",
    target_step="deployment", 
    source_output="metrics",
    target_input="deploy_signal",
    condition="metrics.accuracy > 0.95"
)

# Create parallel edge
parallel = ParallelEdge(
    source_step="data_split",
    target_step="parallel_process",
    source_output="partitions",
    target_input="partition_data",
    max_parallel=8
)

collection.add_edge(conditional)
collection.add_edge(parallel)
```

### SageMaker Integration

```python
# Generate property references for SageMaker
edge = DependencyEdge(
    source_step="preprocessing",
    target_step="training",
    source_output="processed_data", 
    target_input="training_data"
)

# Get SageMaker property reference
prop_ref = edge.to_property_reference_dict()
print(f"SageMaker reference: {prop_ref}")
# Output: {"Get": "Steps.preprocessing.processed_data"}
```

## Validation and Error Handling

### Edge Validation

```python
# Validate edge collection
errors = collection.validate_edges()

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
        
    # Handle specific error types
    for error in errors:
        if "Self-dependency" in error:
            print("Found circular dependency - needs resolution")
        elif "Invalid confidence" in error:
            print("Confidence score out of bounds")
```

### Statistics and Monitoring

```python
# Get comprehensive statistics
stats = collection.get_statistics()

print("Edge Collection Statistics:")
print(f"  Total edges: {stats['total_edges']}")
print(f"  Auto-resolved: {stats['auto_resolved_edges']}")
print(f"  High confidence: {stats['high_confidence_edges']}")
print(f"  Average confidence: {stats['average_confidence']:.3f}")
print(f"  Edge types: {stats['edge_types']}")
print(f"  Unique source steps: {stats['unique_source_steps']}")
print(f"  Unique target steps: {stats['unique_target_steps']}")
```

## Integration Points

### Enhanced DAG Integration
EdgeCollection integrates with EnhancedPipelineDAG for typed dependency management and intelligent resolution.

### Property Reference System
DependencyEdge provides SageMaker property reference generation for pipeline construction.

### Dependency Resolver Integration
Edge types support the dependency resolution system with confidence scoring and metadata.

## Related Documentation

- [Enhanced Pipeline DAG](enhanced_dag.md) - Uses EdgeCollection for typed dependency management
- [Base Pipeline DAG](base_dag.md) - Foundation DAG implementation
- [Pipeline DAG Resolver](pipeline_dag_resolver.md) - Execution planning with edge information
- [Dependency System](../../core/deps/README.md) - Dependency resolution and specification system
- [Property Reference](../../core/deps/property_reference.md) - SageMaker property reference system
