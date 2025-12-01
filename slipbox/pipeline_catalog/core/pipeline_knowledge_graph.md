---
tags:
  - code
  - implementation
  - pipeline_catalog
  - knowledge_graph
  - relationship_navigation
keywords:
  - Knowledge graph
  - Pipeline relationships
  - Evolution paths
  - Graph traversal
  - Relationship tracking
topics:
  - Pipeline catalog knowledge graph
  - Relationship navigation
  - Evolution tracking
  - Graph algorithms
language: python
date of note: 2025-12-01
---

# Pipeline Knowledge Graph

## Overview

The `PipelineKnowledgeGraph` class provides tools for navigating pipeline relationships, tracking evolution paths, and understanding the pipeline ecosystem through graph-based analysis.

Key capabilities include relationship tracking (extends, similar_to, used_by, alternative_to), evolution path discovery (simple → standard → comprehensive), graph traversal algorithms, cluster detection, and visual graph generation.

## Purpose and Major Tasks

### Primary Purpose
Navigate and analyze pipeline relationships through graph-based structures, enabling discovery of evolution paths, alternatives, and ecosystem understanding.

### Major Tasks

1. **Relationship Tracking**: Track connections between pipelines (extends, similar_to, etc.)
2. **Evolution Paths**: Discover simple → standard → comprehensive progressions
3. **Graph Traversal**: Navigate relationships with BFS/DFS algorithms
4. **Shortest Path**: Find shortest connection path between pipelines
5. **Cluster Detection**: Identify pipeline families and groups
6. **Subgraph Extraction**: Extract related pipeline subsets
7. **Visual Generation**: Create graph visualizations
8. **Relationship Analysis**: Analyze connection patterns

## Module Contract

### Entry Point
```python
from cursus.pipeline_catalog.core.pipeline_knowledge_graph import PipelineKnowledgeGraph
```

### Class Initialization

```python
graph = PipelineKnowledgeGraph(
    registry: CatalogRegistry,          # Catalog registry for relationships
    discovery: DAGAutoDiscovery         # DAG discovery for pipeline data
)
```

### Relationship Types

```python
class RelationshipType(Enum):
    EXTENDS = "extends"              # Pipeline extends another
    SIMILAR_TO = "similar_to"        # Similar characteristics
    ALTERNATIVE_TO = "alternative_to" # Alternative implementation
    USED_BY = "used_by"              # Used by another pipeline
    DEPENDS_ON = "depends_on"        # Dependency relationship
```

### Key Methods

```python
# Get relationships for a pipeline
relationships = graph.get_relationships(
    dag_id: str,
    relationship_type: Optional[RelationshipType] = None
) -> List[Dict[str, Any]]

# Find evolution path
path = graph.find_evolution_path(
    start_dag_id: str,
    target_complexity: str
) -> List[str]

# Find shortest path between pipelines
path = graph.find_shortest_path(
    start_dag_id: str,
    end_dag_id: str
) -> Optional[List[str]]

# Get connected components (clusters)
clusters = graph.find_clusters() -> List[List[str]]

# Extract subgraph
subgraph = graph.get_subgraph(
    dag_id: str,
    depth: int = 2
) -> Dict[str, Any]

# Generate visualization
graph.visualize(
    output_path: str,
    format: str = "png"
) -> None
```

## Key Functions and Algorithms

### Relationship Tracking

#### `get_relationships(dag_id, relationship_type) -> List[Dict]`
**Purpose**: Get all relationships for a pipeline

**Algorithm**:
```python
1. Load pipeline node from registry
2. Extract relationships section
3. Filter by relationship_type if provided
4. For each relationship:
   a. Load target pipeline metadata
   b. Build relationship info dict
5. Return list of relationships
```

**Returns**:
```python
[
    {
        "type": "extends",
        "target_id": "xgboost_simple",
        "target_framework": "xgboost",
        "target_complexity": "simple",
        "description": "Adds calibration step"
    },
    ...
]
```

### Evolution Path Discovery

#### `find_evolution_path(start_dag_id, target_complexity) -> List[str]`
**Purpose**: Find progression path from starting pipeline to target complexity

**Algorithm**:
```python
1. Start with current pipeline
2. Check if already at target complexity
3. Use BFS to explore relationships:
   a. Priority: "extends" relationships
   b. Filter by same framework
   c. Move toward target complexity
4. Build path list
5. Return ordered path from simple → target
```

**Complexity Ordering**: simple < standard < comprehensive

**Example**:
```python
# Find path from simple to comprehensive
path = graph.find_evolution_path(
    start_dag_id="xgboost_simple",
    target_complexity="comprehensive"
)
# Returns: ["xgboost_simple", "xgboost_training", "xgboost_complete_e2e"]
```

### Shortest Path Algorithm

#### `find_shortest_path(start_dag_id, end_dag_id) -> Optional[List[str]]`
**Purpose**: Find shortest connection path between two pipelines

**Algorithm** (Bidirectional BFS):
```python
1. Initialize forward queue from start_dag_id
2. Initialize backward queue from end_dag_id
3. While both queues have elements:
   a. Expand forward search one level
   b. Expand backward search one level
   c. Check for intersection
   d. If found: construct path and return
4. If no intersection: return None (no path exists)
```

**Complexity**: O(b^(d/2)) where b=branching factor, d=path depth

**Example**:
```python
path = graph.find_shortest_path(
    "xgboost_simple",
    "pytorch_complete_e2e"
)
# Returns: ["xgboost_simple", "xgboost_complete_e2e", "pytorch_complete_e2e"]
# Via "alternative_to" relationships
```

### Cluster Detection

#### `find_clusters() -> List[List[str]]`
**Purpose**: Identify pipeline families/groups using connected components

**Algorithm** (Union-Find):
```python
1. Initialize each pipeline as its own cluster
2. For each pipeline and its relationships:
   a. Union pipeline with related pipelines
3. Collect final clusters
4. Sort clusters by size (descending)
5. Return list of clusters
```

**Returns**:
```python
[
    ["xgboost_simple", "xgboost_training", "xgboost_complete_e2e"],  # XGBoost family
    ["pytorch_training", "pytorch_complete_e2e"],                     # PyTorch family
    ["lightgbm_simple", "lightgbm_training"],                        # LightGBM family
    ...
]
```

### Subgraph Extraction

#### `get_subgraph(dag_id, depth) -> Dict[str, Any]`
**Purpose**: Extract local neighborhood of a pipeline

**Algorithm** (BFS with depth limit):
```python
1. Start with root pipeline
2. Explore relationships up to depth:
   a. Add each connected pipeline
   b. Add relationship edges
3. Build subgraph structure:
   a. nodes: pipeline metadata
   b. edges: relationship information
4. Return subgraph dictionary
```

**Parameters**:
- `dag_id` (str): Root pipeline
- `depth` (int): Maximum relationship hops (default: 2)

**Returns**:
```python
{
    "nodes": {
        "xgboost_simple": {...},
        "xgboost_training": {...},
        ...
    },
    "edges": [
        {"source": "xgboost_simple", "target": "xgboost_training", "type": "extends"},
        ...
    ],
    "center": "xgboost_simple"
}
```

### Visualization

#### `visualize(output_path, format) -> None`
**Purpose**: Generate visual graph representation

**Algorithm**:
```python
1. Build graph structure using networkx or graphviz
2. Apply layout algorithm (e.g., force-directed)
3. Style nodes by:
   a. Framework (color)
   b. Complexity (size)
   c. Features (shape)
4. Style edges by relationship type
5. Render to output format
6. Save to output_path
```

**Supported Formats**: png, svg, pdf, dot

**Example**:
```python
# Generate PNG visualization
graph.visualize(
    output_path="pipeline_graph.png",
    format="png"
)

# Generate interactive SVG
graph.visualize(
    output_path="pipeline_graph.svg",
    format="svg"
)
```

## Integration Patterns

### With PipelineExplorer

```python
from cursus.pipeline_catalog.core import PipelineExplorer, PipelineKnowledgeGraph

explorer = PipelineExplorer()
graph = PipelineKnowledgeGraph(explorer.registry, explorer.discovery)

# Explore then navigate relationships
pipelines = explorer.filter(framework="xgboost")
for p in pipelines:
    relationships = graph.get_relationships(p['dag_id'])
    print(f"{p['dag_id']}: {len(relationships)} relationships")
```

### Evolution-Based Discovery

```python
# Find progression from simple to comprehensive
path = graph.find_evolution_path(
    start_dag_id="xgboost_simple",
    target_complexity="comprehensive"
)

print("Evolution path:")
for dag_id in path:
    info = explorer.get_pipeline_info(dag_id)
    print(f"  {dag_id}: {info['complexity']}")
```

### Cluster Analysis

```python
# Identify pipeline families
clusters = graph.find_clusters()

print(f"Found {len(clusters)} pipeline families:")
for i, cluster in enumerate(clusters, 1):
    print(f"\nFamily {i}: {len(cluster)} pipelines")
    for dag_id in cluster:
        info = explorer.get_pipeline_info(dag_id)
        print(f"  - {dag_id} ({info['framework']})")
```

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| get_relationships | O(k) | O(k) |
| find_evolution_path | O(n) | O(n) |
| find_shortest_path | O(b^(d/2)) | O(b^d) |
| find_clusters | O(n * α(n)) | O(n) |
| get_subgraph | O(b^d) | O(b^d) |

Where:
- n = number of pipelines
- k = number of relationships
- b = branching factor
- d = depth
- α = inverse Ackermann function (nearly constant)

## Best Practices

### 1. Use Evolution Paths for Learning

```python
# ✅ Good: Show progression
path = graph.find_evolution_path("xgboost_simple", "comprehensive")

print("Learning path:")
for i, dag_id in enumerate(path):
    print(f"{i+1}. {dag_id}")
    if i < len(path) - 1:
        print(f"   ↓ (adds complexity)")
```

### 2. Explore Alternatives

```python
# ✅ Good: Find alternatives
alternatives = graph.get_relationships(
    "xgboost_training",
    relationship_type=RelationshipType.ALTERNATIVE_TO
)

for alt in alternatives:
    print(f"Alternative: {alt['target_id']}")
    print(f"  Framework: {alt['target_framework']}")
```

### 3. Visualize Before Complex Changes

```python
# ✅ Good: Understand ecosystem first
graph.visualize("current_state.png")

# Make changes...

graph.visualize("new_state.png")
# Compare visualizations
```

## Examples

### Example 1: Evolution Path Discovery

```python
from cursus.pipeline_catalog.core import PipelineKnowledgeGraph

graph = PipelineKnowledgeGraph(registry, discovery)

# Find XGBoost evolution
path = graph.find_evolution_path(
    start_dag_id="xgboost_simple",
    target_complexity="comprehensive"
)

print("XGBoost Evolution:")
for step in path:
    info = discovery.load_dag_info(step)
    print(f"  {step}")
    print(f"    Complexity: {info.complexity}")
    print(f"    Features: {info.features}")
```

### Example 2: Relationship Analysis

```python
# Analyze pipeline ecosystem
all_relationships = {}

for dag_id in discovery.list_available_dags():
    relationships = graph.get_relationships(dag_id)
    all_relationships[dag_id] = relationships

# Find most connected pipelines
sorted_by_connections = sorted(
    all_relationships.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

print("Top 5 most connected pipelines:")
for dag_id, rels in sorted_by_connections[:5]:
    print(f"{dag_id}: {len(rels)} connections")
```

### Example 3: Subgraph Exploration

```python
# Explore neighborhood
center = "xgboost_complete_e2e"
subgraph = graph.get_subgraph(center, depth=2)

print(f"Subgraph around {center}:")
print(f"  Nodes: {len(subgraph['nodes'])}")
print(f"  Edges: {len(subgraph['edges'])}")

print("\nConnected pipelines:")
for node_id in subgraph['nodes']:
    if node_id != center:
        print(f"  - {node_id}")
```

## References

### Related Components

- **[DAG Discovery](dag_discovery.md)**: Provides pipeline catalog
- **[Pipeline Explorer](pipeline_explorer.md)**: Interactive exploration
- **[Catalog Registry](../../src/cursus/pipeline_catalog/core/catalog_registry.py)**: Stores relationships

### Design Documents

- **[Pipeline Catalog Redesign](../../1_design/pipeline_catalog_redesign.md)**: Overall system design

### External References

- **[NetworkX](https://networkx.org/)**: Graph algorithms library
- **[Graphviz](https://graphviz.org/)**: Graph visualization
