---
tags:
  - code
  - implementation
  - pipeline_catalog
  - pipeline_explorer
  - interactive_discovery
keywords:
  - Pipeline exploration
  - Interactive discovery
  - Jupyter integration
  - Pipeline filtering
  - Similarity search
topics:
  - Pipeline catalog exploration
  - Interactive pipeline discovery
  - User-friendly interfaces
language: python
date of note: 2025-12-01
---

# Pipeline Explorer

## Overview

The `PipelineExplorer` class provides interactive tools for discovering and exploring available pipelines. It offers multiple filtering dimensions, detailed information display, similarity search, and Jupyter notebook integration for user-friendly pipeline discovery.

Key capabilities include multi-dimensional filtering (framework, complexity, features), detailed pipeline information with formatted output, similarity-based recommendations, interactive Jupyter displays, and comprehensive pipeline comparisons.

## Purpose and Major Tasks

### Primary Purpose
Provide user-friendly, interactive tools for discovering and exploring pipelines through multiple filtering dimensions and rich information displays.

### Major Tasks

1. **Multi-Dimensional Filtering**: Filter by framework, complexity, features simultaneously
2. **Pipeline Listing**: Display available pipelines with rich formatting
3. **Detailed Information**: Show comprehensive pipeline metadata
4. **Similarity Search**: Find similar pipelines based on features/characteristics
5. **Comparison**: Compare multiple pipelines side-by-side
6. **Jupyter Integration**: Interactive displays for notebook environments
7. **Search Helpers**: Assist users in finding appropriate pipelines
8. **Statistics**: Provide catalog statistics and breakdowns

## Module Contract

### Entry Point
```python
from cursus.pipeline_catalog.core.pipeline_explorer import PipelineExplorer
```

### Class Initialization

```python
explorer = PipelineExplorer(
    factory: Optional[PipelineFactory] = None,    # Pipeline factory instance
    discovery: Optional[DAGAutoDiscovery] = None  # DAG discovery instance
)
```

### Key Methods

```python
# Filter pipelines by multiple criteria
pipelines = explorer.filter(
    framework: Optional[str] = None,
    complexity: Optional[str] = None,
    features: Optional[List[str]] = None,
    has_all_features: bool = True
) -> List[Dict[str, Any]]

# Display pipeline information
explorer.display_pipeline(dag_id: str) -> None

# Find similar pipelines
similar = explorer.find_similar(
    dag_id: str,
    limit: int = 5
) -> List[Dict[str, Any]]

# Compare pipelines
explorer.compare_pipelines(
    dag_ids: List[str]
) -> None

# Get catalog statistics
stats = explorer.get_statistics() -> Dict[str, Any]
```

## Key Functions and Algorithms

### Multi-Dimensional Filtering

#### `filter(framework, complexity, features, has_all_features) -> List[Dict]`
**Purpose**: Filter pipelines by multiple criteria simultaneously

**Algorithm**:
```python
1. Get all available pipelines from discovery
2. Apply framework filter if provided
3. Apply complexity filter if provided
4. Apply features filter:
   a. If has_all_features=True: pipeline must have ALL features
   b. If has_all_features=False: pipeline must have ANY feature
5. Return filtered results with metadata
```

**Parameters**:
- `framework` (str): Framework name filter
- `complexity` (str): Complexity level filter
- `features` (List[str]): Feature list filter
- `has_all_features` (bool): Require all features (True) or any feature (False)

**Returns**: List of pipeline information dictionaries

**Examples**:
```python
# Filter by framework
xgb_pipelines = explorer.filter(framework="xgboost")

# Filter by complexity
comprehensive = explorer.filter(complexity="comprehensive")

# Filter by features (all required)
training_eval = explorer.filter(
    features=["training", "evaluation"],
    has_all_features=True
)

# Filter by features (any match)
training_or_eval = explorer.filter(
    features=["training", "evaluation"],
    has_all_features=False
)

# Combined filters
specific = explorer.filter(
    framework="xgboost",
    complexity="comprehensive",
    features=["calibration"]
)
```

### Similarity Search

#### `find_similar(dag_id, limit) -> List[Dict]`
**Purpose**: Find pipelines similar to a given pipeline

**Algorithm**:
```python
1. Load reference pipeline metadata
2. Calculate similarity score for each other pipeline:
   a. Framework match: +10 points
   b. Complexity match: +5 points
   c. Shared features: +3 points each
   d. Node count proximity: +1 to +5 points (closer = higher)
3. Sort by similarity score (descending)
4. Return top N results (limited by limit parameter)
```

**Similarity Scoring**:
| Match Type | Points | Notes |
|------------|--------|-------|
| Same framework | 10 | Exact match |
| Same complexity | 5 | Exact match |
| Shared feature | 3 each | Per common feature |
| Node count proximity | 1-5 | Inverse of difference |

**Parameters**:
- `dag_id` (str): Reference pipeline ID
- `limit` (int): Maximum number of results (default: 5)

**Returns**: List of similar pipelines with similarity scores

**Example**:
```python
# Find pipelines similar to xgboost_complete_e2e
similar = explorer.find_similar("xgboost_complete_e2e", limit=5)

for pipeline in similar:
    print(f"{pipeline['dag_id']}: "
          f"similarity={pipeline['similarity_score']}")
```

### Pipeline Display

#### `display_pipeline(dag_id) -> None`
**Purpose**: Display detailed pipeline information in formatted output

**Output Includes**:
- DAG ID and generated class name
- Framework and complexity
- Features list
- Node and edge counts
- Description
- File location
- Workspace information

**Example**:
```python
explorer.display_pipeline("xgboost_complete_e2e")
# Output:
# Pipeline: xgboost_complete_e2e
# Class: XgboostCompleteE2EPipeline
# Framework: xgboost
# Complexity: comprehensive
# Features: training, evaluation, calibration
# Nodes: 10, Edges: 11
# Description: Complete XGBoost pipeline...
```

### Pipeline Comparison

#### `compare_pipelines(dag_ids) -> None`
**Purpose**: Display side-by-side comparison of multiple pipelines

**Comparison Includes**:
- Framework
- Complexity
- Features (with highlighting of unique vs shared)
- Node/edge counts
- Performance characteristics
- Use case recommendations

**Example**:
```python
explorer.compare_pipelines([
    "xgboost_complete_e2e",
    "pytorch_complete_e2e",
    "xgboost_simple"
])

# Output shows side-by-side comparison table
```

### Statistics

#### `get_statistics() -> Dict[str, Any]`
**Purpose**: Get comprehensive catalog statistics

**Returns**:
```python
{
    "total_pipelines": 34,
    "by_framework": {
        "xgboost": 11,
        "pytorch": 6,
        "lightgbm": 5,
        ...
    },
    "by_complexity": {
        "simple": 8,
        "standard": 15,
        "comprehensive": 11
    },
    "by_features": {
        "training": 25,
        "evaluation": 18,
        "calibration": 8,
        ...
    },
    "avg_node_count": 7.2,
    "avg_edge_count": 6.5
}
```

## Integration Patterns

### With PipelineFactory

```python
from cursus.pipeline_catalog.core import PipelineFactory, PipelineExplorer

# Create factory
factory = PipelineFactory()

# Create explorer with factory
explorer = PipelineExplorer(factory=factory)

# Explore and create
pipelines = explorer.filter(framework="xgboost")
pipeline = factory.create(pipelines[0]['dag_id'])
```

### Jupyter Notebook Integration

```python
# In Jupyter notebook
from cursus.pipeline_catalog.core import PipelineExplorer

explorer = PipelineExplorer()

# Interactive display with rich formatting
explorer.display_all_pipelines()

# Interactive filtering
explorer.interactive_filter()

# Click to create pipeline
explorer.create_from_selection()
```

## Best Practices

### 1. Start Broad, Then Narrow

```python
# ✅ Good: Progressive filtering
all_xgb = explorer.filter(framework="xgboost")
print(f"Found {len(all_xgb)} XGBoost pipelines")

comprehensive_xgb = explorer.filter(
    framework="xgboost",
    complexity="comprehensive"
)
print(f"Narrowed to {len(comprehensive_xgb)} comprehensive pipelines")
```

### 2. Use Similarity for Discovery

```python
# ✅ Good: Find alternatives
current = "xgboost_training"
alternatives = explorer.find_similar(current, limit=3)

print(f"Alternatives to {current}:")
for alt in alternatives:
    print(f"  - {alt['dag_id']} (score: {alt['similarity_score']})")
```

### 3. Compare Before Choosing

```python
# ✅ Good: Compare candidates
candidates = ["xgboost_complete_e2e", "xgboost_training_with_calibration"]
explorer.compare_pipelines(candidates)

# Make informed decision
chosen = candidates[0]
pipeline = factory.create(chosen)
```

## Examples

### Example 1: Basic Exploration

```python
from cursus.pipeline_catalog.core import PipelineExplorer

explorer = PipelineExplorer()

# Get statistics
stats = explorer.get_statistics()
print(f"Total pipelines: {stats['total_pipelines']}")

# List by framework
for framework, count in stats['by_framework'].items():
    print(f"{framework}: {count} pipelines")
```

### Example 2: Feature-Based Discovery

```python
# Find pipelines with specific features
training_pipelines = explorer.filter(
    features=["training", "evaluation"],
    has_all_features=True
)

print(f"Found {len(training_pipelines)} pipelines with training + evaluation")

# Display details
for p in training_pipelines:
    explorer.display_pipeline(p['dag_id'])
```

### Example 3: Similarity-Based Exploration

```python
# Start with known pipeline
reference = "xgboost_simple"

# Find similar pipelines
similar = explorer.find_similar(reference, limit=5)

print(f"Pipelines similar to {reference}:")
for p in similar:
    print(f"  {p['dag_id']}: {p['framework']} - {p['complexity']}")
    print(f"    Similarity: {p['similarity_score']}")
    print(f"    Common features: {p['common_features']}")
```

## References

### Related Components

- **[DAG Discovery](dag_discovery.md)**: Provides pipeline catalog
- **[Pipeline Factory](pipeline_factory.md)**: Creates pipelines from exploration results
- **[Pipeline Knowledge Graph](pipeline_knowledge_graph.md)**: Relationship navigation

### Design Documents

- **[Pipeline Catalog Redesign](../../1_design/pipeline_catalog_redesign.md)**: Overall system design
