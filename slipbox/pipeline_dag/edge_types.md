# Pipeline DAG: Edge Types

## Overview

The Edge Types module provides rich typing and management capabilities for pipeline DAG edges. It defines various types of edges that can exist between pipeline steps, including typed dependency edges with confidence scoring, and provides utilities for edge collection management.

## Key Components

### EdgeType Enum

```python
class EdgeType(Enum):
    """Types of edges in the pipeline DAG."""
    DEPENDENCY = "dependency"        # Standard dependency edge
    CONDITIONAL = "conditional"      # Conditional dependency
    PARALLEL = "parallel"           # Parallel execution hint
    SEQUENTIAL = "sequential"       # Sequential execution requirement
```

The EdgeType enum defines the different types of relationships that can exist between pipeline steps:

- **DEPENDENCY**: Standard input/output dependency between steps
- **CONDITIONAL**: Dependency that is only active under certain conditions
- **PARALLEL**: Hint that steps can be executed in parallel
- **SEQUENTIAL**: Requirement that steps must be executed sequentially

### DependencyEdge Model

```python
class DependencyEdge(BaseModel):
    """Represents a typed dependency edge between step ports."""
    source_step: str
    target_step: str
    source_output: str
    target_input: str
    confidence: float = 1.0
    edge_type: EdgeType = EdgeType.DEPENDENCY
    metadata: Dict[str, Any] = Field(default_factory=dict)
```

The DependencyEdge class represents a directed edge between the output port of one step and the input port of another step:

- **source_step**: Name of the step providing output
- **target_step**: Name of the step consuming input
- **source_output**: Logical name of the output port
- **target_input**: Logical name of the input port
- **confidence**: Confidence score for auto-resolved edges (0.0 to 1.0)
- **edge_type**: Type of edge (from EdgeType enum)
- **metadata**: Additional metadata for the edge

This model enables precise specification of dependencies between steps at the port level, which is critical for the automatic dependency resolution system.

### Specialized Edge Types

#### ConditionalEdge

```python
class ConditionalEdge(DependencyEdge):
    """Represents a conditional dependency edge."""
    condition: str = ""
    edge_type: EdgeType = EdgeType.CONDITIONAL
```

The ConditionalEdge class extends DependencyEdge to include a condition expression that determines when the dependency is active. This enables dynamic pipeline behavior based on runtime conditions.

#### ParallelEdge

```python
class ParallelEdge(DependencyEdge):
    """Represents a parallel execution hint edge."""
    max_parallel: Optional[int] = None
    edge_type: EdgeType = EdgeType.PARALLEL
```

The ParallelEdge class extends DependencyEdge to include information about parallel execution capabilities, such as the maximum number of parallel executions.

### EdgeCollection

```python
class EdgeCollection:
    """Collection of edges with utility methods."""
```

The EdgeCollection class provides a rich set of utilities for managing collections of dependency edges, including:

- **Adding and removing edges**: `add_edge()`, `remove_edge()`
- **Querying edges**: `get_edges_from_step()`, `get_edges_to_step()`, `get_edge()`
- **Filtering edges**: `list_all_edges()`, `list_auto_resolved_edges()`, `list_high_confidence_edges()`, `list_low_confidence_edges()`
- **Dependency management**: `get_step_dependencies()`
- **Validation**: `validate_edges()`
- **Statistics**: `get_statistics()`

## Key Features

### Confidence Scoring

Dependency edges include a confidence score (0.0 to 1.0) that indicates the level of certainty in automatically resolved dependencies. This enables:

- **High-confidence acceptance**: Automatically accepting dependencies with high confidence
- **Low-confidence review**: Flagging dependencies with low confidence for manual review
- **Progressive resolution**: Implementing a stepped resolution process starting with high-confidence edges

### Edge Type Differentiation

Different edge types enable sophisticated pipeline behavior:

- **Dependency edges** for standard data flow
- **Conditional edges** for dynamic behavior
- **Parallel edges** for execution optimization
- **Sequential edges** for enforcing order constraints

### Efficient Indexing

The EdgeCollection maintains efficient indices for fast access:

- **Source index**: Maps source steps to outgoing edges
- **Target index**: Maps target steps to incoming edges
- **Edge ID index**: Enables direct lookup of edges by ID

### Edge Validation

Edge validation ensures the integrity of the dependency graph:

- **Self-dependency detection**: Prevents steps from depending on themselves
- **Confidence bound validation**: Ensures confidence scores are within valid range
- **Empty component detection**: Ensures all edge components are properly specified

### Duplicate Edge Handling

Intelligent handling of duplicate edges:

- **Higher confidence priority**: Replacing edges with higher-confidence versions
- **Duplicate detection**: Preventing redundant edges between the same ports
- **Logging**: Detailed logging of edge additions and replacements

## Usage Examples

### Creating and Managing Dependency Edges

```python
# Create a standard dependency edge
edge = DependencyEdge(
    source_step="preprocessing",
    target_step="training",
    source_output="processed_data",
    target_input="training_data",
    confidence=0.95,
    metadata={"auto_resolved": True}
)

# Create an edge collection
edges = EdgeCollection()

# Add the edge to the collection
edge_id = edges.add_edge(edge)

# Get all edges from the preprocessing step
outgoing_edges = edges.get_edges_from_step("preprocessing")

# Get all edges to the training step
incoming_edges = edges.get_edges_to_step("training")

# Get a specific edge
specific_edge = edges.get_edge(
    source_step="preprocessing", 
    source_output="processed_data",
    target_step="training", 
    target_input="training_data"
)

# Validate all edges
validation_errors = edges.validate_edges()

# Get edge statistics
stats = edges.get_statistics()
print(f"Average confidence: {stats['average_confidence']:.2f}")
```

### Working with Conditional Edges

```python
# Create a conditional edge
conditional_edge = ConditionalEdge(
    source_step="evaluation",
    target_step="registration",
    source_output="evaluation_results",
    target_input="model_metrics",
    condition="context.model_accuracy > 0.9"  # Only activate if accuracy is high enough
)

# Add to the collection
edges.add_edge(conditional_edge)
```

### Edge Collection Statistics

```python
# Get comprehensive statistics about the edge collection
statistics = edges.get_statistics()

print(f"Total edges: {statistics['total_edges']}")
print(f"Auto-resolved edges: {statistics['auto_resolved_edges']}")
print(f"High confidence edges: {statistics['high_confidence_edges']}")
print(f"Low confidence edges: {statistics['low_confidence_edges']}")
print(f"Average confidence: {statistics['average_confidence']:.2f}")
print(f"Edge types: {statistics['edge_types']}")
```

## Integration with Other Components

### With Enhanced DAG

The Edge Types module is a core component of the [Enhanced DAG](enhanced_dag.md) system:

```python
# In EnhancedPipelineDAG
self.dependency_edges = EdgeCollection()

# During auto-resolution
edge = DependencyEdge(
    source_step=prop_ref.step_name,
    target_step=consumer_step,
    source_output=output_spec.logical_name,
    target_input=dep_name,
    confidence=confidence,
    metadata={'auto_resolved': True}
)

edge_id = self.dependency_edges.add_edge(edge)
```

### With Pipeline Assembler

The Edge Types module integrates with the [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md) for pipeline construction:

```python
# In PipelineAssembler
for edge in dag.dependency_edges.list_all_edges():
    source_prop_ref = PropertyReference(
        step_name=edge.source_step,
        output_spec=source_spec.get_output(edge.source_output)
    )
    
    # Use this property reference to connect steps
    inputs[edge.target_input] = source_prop_ref.to_runtime_property(step_instances)
```

### With Dependency Resolver

The Edge Types module works with the [Dependency Resolver](../pipeline_deps/dependency_resolver.md) to represent resolved dependencies:

```python
# Converting resolver results to edges
for consumer_step, dependencies in resolved_deps.items():
    for dep_name, prop_ref in dependencies.items():
        edge = DependencyEdge(
            source_step=prop_ref.step_name,
            target_step=consumer_step,
            source_output=prop_ref.output_spec.logical_name,
            target_input=dep_name,
            confidence=resolver_confidence,
        )
        edge_collection.add_edge(edge)
```

## Related Documentation

### Pipeline DAG Components
- [Pipeline DAG Overview](README.md): Introduction to the DAG-based pipeline structure
- [Base Pipeline DAG](base_dag.md): Core DAG implementation
- [Enhanced Pipeline DAG](enhanced_dag.md): Advanced DAG with port-level dependency resolution

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Uses the edge system for pipeline structure
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Consumes edge information for step wiring
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md): Examples showing edge usage in pipelines

### Dependency System
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Creates edges through dependency resolution
- [Base Specifications](../pipeline_deps/base_specifications.md): Defines port specifications used by edges
- [Property Reference](../pipeline_deps/property_reference.md): Referenced by edges to create runtime connections
