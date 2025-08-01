# Base Pipeline DAG

## Overview

The Base Pipeline DAG provides a lightweight, efficient representation of directed acyclic graphs for pipeline construction. It defines the core graph structure and algorithms used throughout the pipeline building system, serving as the foundation for more advanced DAG implementations.

## Class Definition

```python
class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.
    """
    def __init__(self, nodes: Optional[List[str]] = None, edges: Optional[List[tuple]] = None):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        """
```

## Data Structures

The PipelineDAG uses several internal data structures to represent the graph efficiently:

1. **nodes**: A list of all node names in the graph
   ```python
   self.nodes = nodes or []
   ```

2. **edges**: A list of all edges in the graph as (source, destination) tuples
   ```python
   self.edges = edges or []
   ```

3. **adj_list**: An adjacency list representing outgoing edges (forward graph)
   ```python
   self.adj_list = {n: [] for n in self.nodes}
   ```

4. **reverse_adj**: An adjacency list representing incoming edges (backward graph)
   ```python
   self.reverse_adj = {n: [] for n in self.nodes}
   ```

This dual representation with both forward and backward adjacency lists enables efficient traversal in both directions, which is essential for dependency analysis.

## Key Methods

### Adding Nodes and Edges

```python
def add_node(self, node: str) -> None:
    """Add a single node to the DAG."""
    
def add_edge(self, src: str, dst: str) -> None:
    """Add a directed edge from src to dst."""
```

These methods provide the core graph construction capabilities:

- **add_node**: Adds a single node to the graph if it doesn't already exist
- **add_edge**: Adds a directed edge from source to destination, automatically adding any missing nodes

### Querying Dependencies

```python
def get_dependencies(self, node: str) -> List[str]:
    """Return immediate dependencies (parents) of a node."""
```

This method returns the immediate dependencies (parent nodes) of a given node, which is essential for understanding what steps must be executed before a particular step.

### Topological Sorting

```python
def topological_sort(self) -> List[str]:
    """Return nodes in topological order."""
```

This method returns the nodes in topological order, which is critical for determining the execution order of pipeline steps. A topological ordering ensures that for every directed edge (u, v), node u comes before node v in the ordering.

The implementation uses Kahn's algorithm:

1. Calculate in-degree for each node (number of incoming edges)
2. Start with nodes that have in-degree zero (source nodes)
3. Process nodes in a queue, decrementing in-degree of neighboring nodes
4. Add nodes to the result list in the order they are processed

If the graph contains cycles, this method raises a `ValueError` with the message "DAG has cycles or disconnected nodes". This ensures that pipeline definitions are valid directed acyclic graphs.

## Topological Sorting Algorithm

The core of the PipelineDAG is the topological sorting algorithm, which is essential for pipeline execution:

```python
def topological_sort(self) -> List[str]:
    """Return nodes in topological order."""
    
    in_degree = {n: 0 for n in self.nodes}
    for src, dst in self.edges:
        in_degree[dst] += 1

    queue = deque([n for n in self.nodes if in_degree[n] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in self.adj_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(order) != len(self.nodes):
        raise ValueError("DAG has cycles or disconnected nodes")
    return order
```

This algorithm:
1. Calculates the in-degree (number of incoming edges) for each node
2. Starts with nodes that have in-degree zero (source nodes)
3. Processes nodes in a queue, decreasing the in-degree of neighboring nodes
4. When a node's in-degree becomes zero, adds it to the queue
5. If not all nodes are processed, detects cycles or disconnected components

## Usage Examples

### Creating a Simple Pipeline DAG

```python
# Create an empty DAG
dag = PipelineDAG()

# Add nodes and edges
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_edge("data_loading", "preprocessing")

dag.add_node("training")
dag.add_edge("preprocessing", "training")

dag.add_node("evaluation")
dag.add_edge("preprocessing", "evaluation")
dag.add_edge("training", "evaluation")

# Get the execution order
try:
    execution_order = dag.topological_sort()
    print(f"Execution order: {execution_order}")
except ValueError as e:
    print(f"Error: {e}")
```

This example:
1. Creates a pipeline DAG with 4 steps: data loading, preprocessing, training, and evaluation
2. Defines dependencies between these steps
3. Gets the topological sort to determine the execution order

### Checking Dependencies

```python
# Check dependencies for the evaluation step
evaluation_deps = dag.get_dependencies("evaluation")
print(f"Evaluation depends on: {evaluation_deps}")

# Check dependencies for the training step
training_deps = dag.get_dependencies("training")
print(f"Training depends on: {training_deps}")
```

This example retrieves the immediate dependencies (parent nodes) for specific steps in the pipeline.

### Initializing with Existing Nodes and Edges

```python
# Define nodes and edges upfront
nodes = ["data_loading", "preprocessing", "training", "evaluation"]
edges = [
    ("data_loading", "preprocessing"),
    ("preprocessing", "training"),
    ("preprocessing", "evaluation"),
    ("training", "evaluation")
]

# Create DAG with predefined structure
dag = PipelineDAG(nodes=nodes, edges=edges)

# Get execution order
execution_order = dag.topological_sort()
print(f"Execution order: {execution_order}")
```

This example creates a DAG with a predefined structure using the constructor parameters.

## Performance Characteristics

The PipelineDAG is designed for efficiency with the following complexity characteristics:

- **Space Complexity**: O(V + E) where V is the number of vertices (nodes) and E is the number of edges
- **Time Complexity**:
  - `add_node()`: O(1)
  - `add_edge()`: O(1)
  - `get_dependencies()`: O(1)
  - `topological_sort()`: O(V + E)

These characteristics make it suitable for pipelines with hundreds of steps and complex dependency relationships.

## Extension Points

The Base Pipeline DAG is designed to be extended with more sophisticated functionality:

- **Enhanced DAG**: The [Enhanced Pipeline DAG](enhanced_dag.md) extends the base DAG with port-based dependency management and intelligent dependency resolution.
- **Edge Types**: The [Edge Types](edge_types.md) system extends the simple (source, destination) edges with rich type information and confidence scoring.
- **Visualization**: The base DAG can be extended with visualization capabilities by implementing exporters to formats like DOT (Graphviz), JSON, or other visualization-friendly formats.

## Integration with Other Components

### With Pipeline Template Base

The Base Pipeline DAG integrates with the [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md) to define pipeline structure:

```python
# In PipelineTemplateBase._create_pipeline_dag
dag = PipelineDAG()

# Add nodes and edges based on template configuration
for step_name in self.steps:
    dag.add_node(step_name)

# Define dependencies based on template structure
for src_step, dst_step in self.dependencies:
    dag.add_edge(src_step, dst_step)

return dag
```

### With Pipeline Assembler

The Base Pipeline DAG integrates with the [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md) to determine step instantiation order:

```python
# In PipelineAssembler.generate_pipeline
try:
    build_order = self.dag.topological_sort()
    logger.info(f"Build order: {build_order}")
except ValueError as e:
    logger.error(f"Error in topological sort: {e}")
    raise ValueError(f"Failed to determine build order: {e}") from e

# Instantiate steps in topological order
for step_name in build_order:
    # ...instantiate step...
```

## Related Documentation

### Pipeline DAG Components
- [Pipeline DAG Overview](README.md): Introduction to the DAG-based pipeline structure
- [Enhanced Pipeline DAG](enhanced_dag.md): Advanced DAG with port-level dependency resolution
- [Edge Types](edge_types.md): Rich edge typing system for dependency representation

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Uses the DAG for template-based pipeline creation
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Consumes the DAG for step instantiation
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md): Examples showing the DAG in action

### Dependency System
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Works with the DAG for dependency resolution
- [Base Specifications](../pipeline_deps/base_specifications.md): Defines step specifications used with the DAG
- [Property Reference](../pipeline_deps/property_reference.md): Used to represent resolved dependencies in the DAG
