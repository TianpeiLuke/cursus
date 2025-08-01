# Pipeline DAG

## Overview

The Pipeline DAG (Directed Acyclic Graph) module provides the core graph structure representation for pipeline construction. It defines the relationships between pipeline steps and enables automatic dependency resolution and proper step ordering.

## Key Components

### [Base Pipeline DAG](base_dag.md)

The Base Pipeline DAG provides a lightweight, efficient representation of directed acyclic graphs for pipeline construction. It defines the core graph structure and algorithms for:

- Adding nodes and edges
- Querying dependencies
- Topological sorting
- Cycle detection

### [Edge Types](edge_types.md)

The Edge Types module provides rich typing and management capabilities for pipeline DAG edges including:

- Various edge types (dependency, conditional, parallel, sequential)
- Confidence scoring for auto-resolved edges
- Edge collection management utilities
- Edge validation and statistics

### [Enhanced Pipeline DAG](enhanced_dag.md)

The Enhanced Pipeline DAG extends the base Pipeline DAG with port-based dependency management and intelligent resolution:

- Typed input/output ports via step specifications
- Automatic dependency resolution with confidence scoring
- Property reference management
- Enhanced validation and reporting

## Core Functionality

### Directed Graph Representation

The PipelineDAG provides a lightweight, efficient representation of directed graphs with:
- Fast node and edge insertion
- Constant-time dependency lookups
- Linear-time topological sorting

### Topological Ordering

The PipelineDAG implements Kahn's algorithm for topological sorting, which:
1. Calculates in-degree for each node (number of incoming edges)
2. Identifies source nodes (those with in-degree zero)
3. Processes nodes in a queue, decrementing in-degree of neighboring nodes
4. Detects cycles by checking if all nodes are processed

### Dependency Management

The PipelineDAG supports:
- Forward dependency tracking (which nodes depend on the current node)
- Backward dependency tracking (which nodes the current node depends on)
- Easy traversal in both directions

## Usage with Other Components

### Integration with Pipeline Builder

The PipelineDAG is a core component of the [Pipeline Builder Template](../pipeline_builder/pipeline_builder_template.md). It provides the structure that the template uses to:

1. Define the relationships between steps
2. Determine the order of step instantiation
3. Guide the dependency resolution process

### Integration with Dependency Resolution

The PipelineDAG works in conjunction with the [Dependency Resolver](../pipeline_deps/dependency_resolver.md) to:

1. Define potential dependency relationships
2. Guide message propagation between steps
3. Ensure topologically valid execution order

## Example Usage

```python
# Create an empty DAG
dag = PipelineDAG()

# Data loading and preprocessing branch
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_edge("data_loading", "preprocessing")

# Model training branch
dag.add_node("training")
dag.add_edge("preprocessing", "training")

# Model evaluation branch
dag.add_node("evaluation")
dag.add_edge("training", "evaluation")
dag.add_edge("preprocessing", "evaluation")  # Evaluation also needs raw data

# Model registration branch
dag.add_node("registration")
dag.add_edge("evaluation", "registration")

# Determine build order
build_order = dag.topological_sort()
print(f"Build order: {build_order}")
# Output: ['data_loading', 'preprocessing', 'training', 'evaluation', 'registration']
```

## Best Practices

1. **Step Naming**: Use descriptive names for nodes that reflect their function
2. **Direct Dependencies Only**: Create edges only between steps with direct dependencies
3. **Avoid Redundant Dependencies**: If A→B→C, don't add A→C unless C directly depends on A
4. **Validation**: Always validate the DAG by calling topological_sort() to detect cycles
5. **Modular Construction**: Build complex DAGs by combining smaller sub-graphs
6. **DAG Visualization**: Consider visualizing the DAG for complex pipelines using tools like Graphviz

## Related Documentation

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Core abstract class for templates
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Assembles steps using DAGs
- [Pipeline Builder Overview](../pipeline_builder/README.md): Complete pipeline building system
- [Template Implementation](../pipeline_builder/template_implementation.md): Template implementation details
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md): Example pipelines using DAGs

### Dependency Resolution
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Uses DAG structure for resolution
- [Base Specifications](../pipeline_deps/base_specifications.md): Core specification structures
- [Property Reference](../pipeline_deps/property_reference.md): Runtime property bridge

### Pipeline Components
- [Pipeline Steps](../pipeline_steps/README.md): Steps connected via DAG
- [Pipeline Dependencies](../pipeline_deps/README.md): Dependency resolution system
