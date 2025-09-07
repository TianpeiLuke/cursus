---
tags:
  - code
  - api
  - dag
  - pipeline
  - graph
keywords:
  - PipelineDAG
  - directed acyclic graph
  - topological sort
  - pipeline topology
  - graph algorithms
  - dependency management
topics:
  - pipeline DAG
  - graph algorithms
  - pipeline topology
  - dependency management
language: python
date of note: 2024-12-07
---

# Base Pipeline DAG

Lightweight, efficient representation of directed acyclic graphs for pipeline construction, providing core graph structure and algorithms used throughout the pipeline building system.

## Overview

The Base Pipeline DAG provides a foundational implementation of directed acyclic graphs optimized for pipeline construction and execution planning. It serves as the core graph structure used throughout the pipeline building system, offering efficient node and edge management, dependency tracking, and topological sorting capabilities.

The module implements a dual adjacency list representation for efficient bidirectional graph traversal, essential for dependency analysis and pipeline execution planning. It provides robust cycle detection through topological sorting and maintains compatibility with more advanced DAG implementations in the system.

## Classes and Methods

### Classes
- [`PipelineDAG`](#pipelinedag) - Core directed acyclic graph implementation for pipeline topology

## API Reference

### PipelineDAG

_class_ cursus.api.dag.base_dag.PipelineDAG(_nodes=None_, _edges=None_)

Represents a pipeline topology as a directed acyclic graph (DAG) where each node is a step name and edges define dependencies.

**Parameters:**
- **nodes** (_Optional[List[str]]_) – List of step names to initialize the DAG with
- **edges** (_Optional[List[tuple]]_) – List of (from_step, to_step) tuples defining dependencies

```python
from cursus.api.dag.base_dag import PipelineDAG

# Create empty DAG
dag = PipelineDAG()

# Create DAG with predefined structure
nodes = ["data_loading", "preprocessing", "training"]
edges = [("data_loading", "preprocessing"), ("preprocessing", "training")]
dag = PipelineDAG(nodes=nodes, edges=edges)
```

#### add_node

add_node(_node_)

Add a single node to the DAG if it doesn't already exist.

**Parameters:**
- **node** (_str_) – Name of the step to add as a node

```python
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing")
```

#### add_edge

add_edge(_src_, _dst_)

Add a directed edge from source to destination, automatically adding any missing nodes.

**Parameters:**
- **src** (_str_) – Name of the source step
- **dst** (_str_) – Name of the destination step

```python
dag = PipelineDAG()
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")
```

#### get_dependencies

get_dependencies(_node_)

Return immediate dependencies (parents) of a node.

**Parameters:**
- **node** (_str_) – Name of the node to get dependencies for

**Returns:**
- **List[str]** – List of immediate dependency node names

```python
dag = PipelineDAG()
dag.add_edge("preprocessing", "training")
dag.add_edge("data_loading", "training")

dependencies = dag.get_dependencies("training")
# Returns: ["preprocessing", "data_loading"]
```

#### topological_sort

topological_sort()

Return nodes in topological order using Kahn's algorithm.

**Returns:**
- **List[str]** – List of node names in topological execution order

**Raises:**
- **ValueError** – If DAG has cycles or disconnected nodes

```python
dag = PipelineDAG()
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")
dag.add_edge("preprocessing", "evaluation")
dag.add_edge("training", "evaluation")

try:
    execution_order = dag.topological_sort()
    # Returns: ["data_loading", "preprocessing", "training", "evaluation"]
except ValueError as e:
    print(f"DAG validation error: {e}")
```

## Data Structures

The PipelineDAG uses efficient internal data structures for graph representation:

### Node and Edge Storage
- **nodes** – List of all node names in the graph
- **edges** – List of all edges as (source, destination) tuples

### Adjacency Lists
- **adj_list** – Forward adjacency list for outgoing edges (node → successors)
- **reverse_adj** – Reverse adjacency list for incoming edges (node → predecessors)

This dual representation enables O(1) dependency lookups and efficient bidirectional graph traversal.

## Algorithm Implementation

### Topological Sorting Algorithm

The core topological sorting implementation uses Kahn's algorithm:

1. **Initialize in-degrees** – Calculate incoming edge count for each node
2. **Queue source nodes** – Add nodes with zero in-degree to processing queue
3. **Process nodes** – Remove nodes from queue, add to result, update neighbor in-degrees
4. **Cycle detection** – If not all nodes processed, cycles exist

```python
def topological_sort(self) -> List[str]:
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

## Performance Characteristics

### Time Complexity
- **add_node()** – O(1) constant time node addition
- **add_edge()** – O(1) constant time edge addition
- **get_dependencies()** – O(1) constant time dependency lookup
- **topological_sort()** – O(V + E) linear in vertices and edges

### Space Complexity
- **Overall** – O(V + E) where V is vertices and E is edges
- **Adjacency Lists** – O(E) for storing edge relationships
- **Node Storage** – O(V) for storing node names

## Usage Examples

### Basic Pipeline Construction

```python
from cursus.api.dag.base_dag import PipelineDAG

# Create pipeline DAG
dag = PipelineDAG()

# Build ML pipeline structure
dag.add_node("data_loading")
dag.add_node("preprocessing") 
dag.add_node("feature_engineering")
dag.add_node("training")
dag.add_node("evaluation")

# Define dependencies
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "feature_engineering")
dag.add_edge("feature_engineering", "training")
dag.add_edge("feature_engineering", "evaluation")
dag.add_edge("training", "evaluation")

# Get execution order
execution_order = dag.topological_sort()
print(f"Pipeline execution order: {execution_order}")
```

### Dependency Analysis

```python
# Analyze step dependencies
for step in dag.nodes:
    dependencies = dag.get_dependencies(step)
    print(f"{step} depends on: {dependencies}")

# Check for specific dependency relationships
if "preprocessing" in dag.get_dependencies("training"):
    print("Training step depends on preprocessing")
```

### Error Handling

```python
# Handle cyclic dependencies
dag_with_cycle = PipelineDAG()
dag_with_cycle.add_edge("step_a", "step_b")
dag_with_cycle.add_edge("step_b", "step_c")
dag_with_cycle.add_edge("step_c", "step_a")  # Creates cycle

try:
    order = dag_with_cycle.topological_sort()
except ValueError as e:
    print(f"Cycle detected: {e}")
    # Handle cycle resolution or report error
```

## Integration Points

### Pipeline Template Integration
The PipelineDAG integrates with pipeline templates for structure definition and step ordering.

### Enhanced DAG Extension
Serves as the foundation for EnhancedPipelineDAG with port-based dependency management and intelligent resolution.

### Pipeline Assembler Integration
Provides topological ordering for step instantiation and pipeline construction.

## Related Documentation

- [Enhanced Pipeline DAG](enhanced_dag.md) - Advanced DAG with port-based dependency management
- [Edge Types](edge_types.md) - Rich edge typing system for dependency representation
- [Pipeline DAG Resolver](pipeline_dag_resolver.md) - Execution planning and configuration resolution
- [Workspace DAG](workspace_dag.md) - Workspace-aware DAG implementation
- [DAG README](README.md) - Overview of DAG-based pipeline architecture
