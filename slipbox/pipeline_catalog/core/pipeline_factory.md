---
tags:
  - code
  - implementation
  - pipeline_catalog
  - pipeline_factory
  - dynamic_creation
  - knowledge_driven
keywords:
  - Pipeline factory
  - Dynamic class generation
  - Search-driven creation
  - Criteria-based filtering
  - Pipeline caching
topics:
  - Pipeline catalog factory
  - Dynamic pipeline creation
  - Knowledge-driven design
  - Factory pattern
language: python
date of note: 2025-12-01
---

# Pipeline Factory

## Overview

The `PipelineFactory` class implements a knowledge-driven factory pattern for dynamic pipeline creation. Instead of maintaining separate hardcoded pipeline classes, the factory generates pipeline classes on-the-fly using DAG definitions discovered by `DAGAutoDiscovery`.

This eliminates redundant pipeline implementations (previously 7+ classes with ~70 lines each), reduces maintenance burden, enables flexible pipeline creation through multiple interfaces, and provides intelligent search and recommendation capabilities.

Key capabilities include dynamic class generation from DAG definitions, multiple creation interfaces (direct ID, natural language search, structured criteria), class caching for performance, full BasePipeline integration, and registry-enriched metadata.

## Purpose and Major Tasks

### Primary Purpose
Dynamically generate and instantiate pipeline classes from DAG definitions, providing flexible creation interfaces while eliminating redundant pipeline class implementations.

### Major Tasks

1. **DAG Discovery Integration**: Initialize and use DAGAutoDiscovery for DAG catalog
2. **Dynamic Class Generation**: Generate BasePipeline subclasses at runtime
3. **Direct Creation**: Create pipelines by DAG ID
4. **Search-Based Creation**: Natural language search for pipeline creation
5. **Criteria-Based Creation**: Structured filtering for pipeline selection
6. **Class Caching**: Cache generated classes for performance
7. **Metadata Enrichment**: Combine DAG and registry metadata
8. **Pipeline Listing**: Enumerate available pipelines with metadata
9. **Information Retrieval**: Get detailed pipeline information
10. **Statistics**: Provide factory and discovery statistics

## Module Contract

### Entry Point
```python
from cursus.pipeline_catalog.core.pipeline_factory import PipelineFactory
```

### Class Initialization

```python
factory = PipelineFactory(
    package_root: Optional[Path] = None,           # Package root for discovery
    workspace_dirs: Optional[List[Path]] = None,   # Workspace directories
    registry_path: Optional[str] = None,           # Catalog registry path
    enable_caching: bool = True                    # Enable class caching
)
```

### Creation Methods

#### Direct Creation by ID
```python
pipeline = factory.create(
    dag_id: str,                                # Required: DAG identifier
    config_path: Optional[str] = None,          # Configuration file path
    sagemaker_session: Optional[PipelineSession] = None,
    execution_role: Optional[str] = None,
    enable_mods: bool = True,
    validate: bool = True,
    pipeline_parameters: Optional[list] = None,
    **kwargs
) -> BasePipeline
```

#### Search-Based Creation
```python
pipeline = factory.create_by_search(
    query: str,                                 # Natural language query
    config_path: Optional[str] = None,
    **kwargs
) -> BasePipeline
```

#### Criteria-Based Creation
```python
pipeline = factory.create_by_criteria(
    framework: Optional[str] = None,            # Framework filter
    complexity: Optional[str] = None,           # Complexity filter
    features: Optional[List[str]] = None,       # Required features
    config_path: Optional[str] = None,
    **kwargs
) -> BasePipeline
```

### Query Methods

```python
# List all available pipelines
pipelines = factory.list_available_pipelines() -> List[Dict[str, Any]]

# Get specific pipeline info
info = factory.get_pipeline_info(dag_id: str) -> Dict[str, Any]

# Get cache statistics
cache_stats = factory.get_cache_stats() -> Dict[str, Any]

# Get discovery statistics
discovery_stats = factory.get_discovery_stats() -> Dict[str, Any]
```

### Cache Management

```python
# Clear class cache
factory.clear_cache() -> None
```

## Key Functions and Algorithms

### Initialization

#### `__init__(package_root, workspace_dirs, registry_path, enable_caching)`
**Purpose**: Initialize factory with DAG discovery

**Algorithm**:
```python
1. Initialize DAGAutoDiscovery with provided parameters
2. Initialize empty class cache dictionary
3. Run initial DAG discovery
4. Log number of discovered DAGs
```

**Example**:
```python
factory = PipelineFactory(
    package_root=Path("/path/to/cursus"),
    enable_caching=True
)
# Output: "PipelineFactory initialized with 34 DAGs"
```

### Direct Pipeline Creation

#### `create(dag_id, config_path, **kwargs) -> BasePipeline`
**Purpose**: Create pipeline directly by DAG ID

**Algorithm**:
```python
1. Load DAG info from discovery
2. If DAG not found: raise ValueError with available DAGs
3. Get or generate pipeline class:
   a. Check cache if enabled
   b. Generate new class if not cached
   c. Cache class if caching enabled
4. Instantiate pipeline with provided parameters
5. Log creation success
6. Return pipeline instance
```

**Parameters**:
- `dag_id` (str): DAG identifier (e.g., "xgboost_complete_e2e")
- `config_path` (str): Path to configuration JSON
- `sagemaker_session` (PipelineSession): SageMaker session
- `execution_role` (str): IAM execution role
- `enable_mods` (bool): Enable MODS features
- `validate` (bool): Validate DAG before compilation
- `pipeline_parameters` (list): Custom pipeline parameters
- `**kwargs`: Additional BasePipeline arguments

**Returns**: `BasePipeline` - Instantiated pipeline

**Raises**: `ValueError` if DAG not found

**Example**:
```python
pipeline = factory.create(
    dag_id="xgboost_complete_e2e",
    config_path="config.json",
    execution_role="arn:aws:iam::123456789:role/SageMakerRole"
)

# Pipeline ready for use
pipeline.generate_pipeline()
```

### Search-Based Creation

#### `create_by_search(query, config_path, **kwargs) -> BasePipeline`
**Purpose**: Create pipeline using natural language search

**Algorithm**:
```python
1. Tokenize query (lowercase, split by spaces)
2. Score each DAG:
   a. Framework match: +10 points
   b. Complexity match: +5 points
   c. Feature match: +3 points each
   d. DAG ID token match: +2 points each
3. Collect DAGs with score > 0
4. If no matches: raise ValueError
5. Find best match (highest score)
6. Check for ambiguous matches (multiple DAGs with same score)
7. If ambiguous: raise ValueError with suggestions
8. Log matched DAG and score
9. Create pipeline using matched DAG ID
```

**Scoring System**:
| Match Type | Points | Example |
|------------|--------|---------|
| Framework | 10 | "xgboost" in query → xgboost DAG |
| Complexity | 5 | "comprehensive" in query |
| Feature | 3 each | "training", "evaluation" |
| DAG ID Token | 2 each | "complete", "e2e" |

**Parameters**:
- `query` (str): Natural language search query
- `config_path` (str): Configuration file path
- `**kwargs`: Additional create() arguments

**Returns**: `BasePipeline` - Instantiated pipeline

**Raises**: `ValueError` if no matches or ambiguous matches

**Examples**:
```python
# Example 1: Framework + features
pipeline = factory.create_by_search(
    query="xgboost training calibration",
    config_path="config.json"
)
# Matches: xgboost_training_with_calibration

# Example 2: Comprehensive pipeline
pipeline = factory.create_by_search(
    query="pytorch comprehensive end-to-end",
    config_path="config.json"
)
# Matches: pytorch_complete_e2e

# Example 3: Ambiguous query
pipeline = factory.create_by_search(
    query="training",  # ❌ Too generic
    config_path="config.json"
)
# Raises ValueError: Multiple matches - be more specific
```

### Criteria-Based Creation

#### `create_by_criteria(framework, complexity, features, config_path, **kwargs) -> BasePipeline`
**Purpose**: Create pipeline using structured criteria

**Algorithm**:
```python
1. Call discovery.search_dags() with criteria:
   a. Filter by framework if provided
   b. Filter by complexity if provided
   c. Filter by features if provided (all must match)
2. If no matches: raise ValueError with criteria
3. If multiple matches: raise ValueError with match list
4. Get single matched DAG ID
5. Log match success
6. Create pipeline using matched DAG ID
```

**Parameters**:
- `framework` (str): Framework name ("xgboost", "pytorch", etc.)
- `complexity` (str): Complexity level ("simple", "standard", "comprehensive")
- `features` (List[str]): Required features (all must be present)
- `config_path` (str): Configuration file path
- `**kwargs`: Additional create() arguments

**Returns**: `BasePipeline` - Instantiated pipeline

**Raises**: `ValueError` if no matches or multiple matches

**Examples**:
```python
# Example 1: Framework + complexity
pipeline = factory.create_by_criteria(
    framework="xgboost",
    complexity="comprehensive",
    config_path="config.json"
)
# Matches: xgboost_complete_e2e

# Example 2: Framework + features
pipeline = factory.create_by_criteria(
    framework="pytorch",
    features=["training", "evaluation"],
    config_path="config.json"
)
# Matches pipelines with both features

# Example 3: All criteria
pipeline = factory.create_by_criteria(
    framework="xgboost",
    complexity="standard",
    features=["training", "calibration"],
    config_path="config.json"
)
# Matches: xgboost_training_with_calibration
```

### Dynamic Class Generation

#### `_generate_pipeline_class(dag_info) -> Type[BasePipeline]`
**Purpose**: Dynamically generate pipeline class from DAG information

**Algorithm**:
```python
1. Load DAG functions if not already loaded:
   a. create_*_dag function
   b. get_dag_metadata function
2. Generate class name from DAG ID:
   a. Split by underscore
   b. Capitalize each part
   c. Append "Pipeline"
   d. Example: "xgboost_complete_e2e" → "XgboostCompleteE2EPipeline"
3. Define create_dag method:
   a. Calls dag_info.create_function()
   b. Returns PipelineDAG
4. Define get_enhanced_dag_metadata method:
   a. Get basic metadata from dag_info.metadata_function()
   b. Convert to EnhancedDAGMetadata
   c. Add enrichment (title, tags, etc.)
   d. Return enhanced metadata
5. Create class dynamically using type():
   a. Class name
   b. Base class tuple: (BasePipeline,)
   c. Method dictionary
   d. Module and docstring
6. Store dag_info reference in class
7. Log generation success
8. Return generated class
```

**Dynamic Class Structure**:
```python
# Generated class equivalent to:
class XgboostCompleteE2EPipeline(BasePipeline):
    """Dynamically generated pipeline class for xgboost_complete_e2e"""
    
    _dag_info = dag_info  # Reference to DAG info
    
    def create_dag(self) -> PipelineDAG:
        return dag_info.create_function()
    
    def get_enhanced_dag_metadata(self) -> EnhancedDAGMetadata:
        # Load and enhance metadata
        ...
```

**Parameters**:
- `dag_info` (DAGInfo): DAG information from discovery

**Returns**: `Type[BasePipeline]` - Generated pipeline class

**Complexity**: O(1) - Class generation is constant time

### Class Name Conversion

#### `_dag_id_to_class_name(dag_id) -> str`
**Purpose**: Convert DAG ID to Python class name

**Algorithm**:
```python
1. Split DAG ID by underscore
2. Capitalize each part
3. Join parts together
4. Append "Pipeline"
5. Return class name
```

**Examples**:
```python
"xgboost_complete_e2e" → "XgboostCompleteE2EPipeline"
"pytorch_training" → "PytorchTrainingPipeline"
"simple" → "SimplePipeline"
"lightgbm_multi_target" → "LightgbmMultiTargetPipeline"
```

### Caching System

#### `_get_or_create_pipeline_class(dag_info) -> Type[BasePipeline]`
**Purpose**: Get cached or generate new pipeline class

**Algorithm**:
```python
1. Check if caching enabled
2. If enabled and class in cache:
   a. Log cache hit
   b. Return cached class
3. Generate new pipeline class:
   a. Call _generate_pipeline_class()
4. If caching enabled:
   a. Store in cache
   b. Log cache store
5. Return generated class
```

**Cache Benefits**:
- First creation: ~10-20ms (generation time)
- Subsequent creations: <1ms (cache lookup)
- Memory overhead: ~10KB per cached class

**Example**:
```python
# First creation - generates class
pipeline1 = factory.create("xgboost_complete_e2e")  # ~15ms

# Second creation - uses cache
pipeline2 = factory.create("xgboost_complete_e2e")  # <1ms

# Same class used
assert type(pipeline1) is type(pipeline2)  # True
```

### Information Methods

#### `list_available_pipelines() -> List[Dict[str, Any]]`
**Purpose**: List all available pipelines with metadata

**Returns**: List of dictionaries with:
```python
{
    "dag_id": "xgboost_complete_e2e",
    "class_name": "XgboostCompleteE2EPipeline",
    "framework": "xgboost",
    "complexity": "comprehensive",
    "features": ["training", "evaluation", "calibration"],
    "node_count": 10,
    "edge_count": 11,
    "workspace": "package",
    "description": "Complete XGBoost pipeline..."
}
```

**Example**:
```python
pipelines = factory.list_available_pipelines()
for p in pipelines:
    print(f"{p['dag_id']}: {p['framework']} - {p['complexity']}")
```

#### `get_pipeline_info(dag_id) -> Dict[str, Any]`
**Purpose**: Get detailed information about specific pipeline

**Parameters**:
- `dag_id` (str): DAG identifier

**Returns**: Dictionary with detailed information

**Raises**: `ValueError` if DAG not found

**Example**:
```python
info = factory.get_pipeline_info("xgboost_complete_e2e")
print(f"Framework: {info['framework']}")
print(f"Features: {info['features']}")
print(f"File: {info['file_path']}")
```

## Performance Characteristics

### Creation Performance

| Operation | First Call | Cached Call | Memory |
|-----------|-----------|-------------|---------|
| create() | ~15ms | <1ms | ~10KB |
| create_by_search() | ~20ms | <1ms | ~10KB |
| create_by_criteria() | ~18ms | <1ms | ~10KB |

### Search Performance

| Query Complexity | DAG Count | Search Time |
|-----------------|-----------|-------------|
| Simple (1-2 tokens) | 34 | ~2ms |
| Medium (3-4 tokens) | 34 | ~3ms |
| Complex (5+ tokens) | 34 | ~4ms |

**Scaling**: Search time is O(n * t) where n=DAG count, t=token count

## Integration Patterns

### With DAG Discovery

```python
# Factory uses discovery internally
factory = PipelineFactory()

# Discovery happens during initialization
# All 34 DAGs available immediately
pipeline = factory.create("xgboost_complete_e2e")
```

### With Workspace Development

```python
# Factory respects workspace priority
factory = PipelineFactory(
    workspace_dirs=[Path("workspace")]
)

# Workspace DAGs override package DAGs
pipeline = factory.create("custom_xgboost")  # Uses workspace version
```

### With SageMaker

```python
from sagemaker.workflow.pipeline_context import PipelineSession

session = PipelineSession()

# Create pipeline with SageMaker integration
pipeline = factory.create(
    dag_id="xgboost_complete_e2e",
    config_path="config.json",
    sagemaker_session=session,
    execution_role="arn:aws:iam::123456789:role/SageMakerRole"
)

# Generate SageMaker pipeline
sm_pipeline = pipeline.generate_pipeline()
```

## Error Handling

### DAG Not Found

```python
try:
    pipeline = factory.create("nonexistent_dag")
except ValueError as e:
    # Error message includes available DAGs
    print(e)
    # "DAG 'nonexistent_dag' not found. Available DAGs: [...]"
```

### Ambiguous Search

```python
try:
    pipeline = factory.create_by_search("training")  # Too generic
except ValueError as e:
    # Error suggests being more specific
    print(e)
    # "Ambiguous query 'training' matches multiple DAGs..."
```

### No Criteria Matches

```python
try:
    pipeline = factory.create_by_criteria(
        framework="invalid_framework"
    )
except ValueError as e:
    print(e)
    # "No DAGs match criteria: framework=invalid_framework..."
```

## Best Practices

### 1. Use Appropriate Creation Method

```python
# ✅ Known DAG ID - use direct creation
pipeline = factory.create("xgboost_complete_e2e")

# ✅ Natural language - use search
pipeline = factory.create_by_search("comprehensive xgboost pipeline")

# ✅ Structured requirements - use criteria
pipeline = factory.create_by_criteria(
    framework="xgboost",
    features=["training", "calibration"]
)
```

### 2. Enable Caching for Reuse

```python
# ✅ Caching enabled (default)
factory = PipelineFactory(enable_caching=True)

# Create same pipeline multiple times - uses cache
for i in range(100):
    pipeline = factory.create("xgboost_complete_e2e")  # <1ms after first
```

### 3. Use Specific Search Queries

```python
# ✅ Specific query
pipeline = factory.create_by_search("xgboost comprehensive calibration")

# ❌ Generic query
pipeline = factory.create_by_search("training")  # May match multiple
```

### 4. Provide All Required Criteria

```python
# ✅ Multiple criteria for precise match
pipeline = factory.create_by_criteria(
    framework="pytorch",
    complexity="standard",
    features=["training", "evaluation"]
)

# ⚠️ Single criterion may match multiple
pipeline = factory.create_by_criteria(framework="xgboost")  # May error
```

## Troubleshooting

### Issue 1: Ambiguous Search Results

**Symptom**: ValueError with "Ambiguous query matches multiple DAGs"

**Solution**:
```python
# Add more specific terms
pipeline = factory.create_by_search(
    "xgboost training calibration"  # More specific
)

# Or use direct creation
pipeline = factory.create("xgboost_training_with_calibration")
```

### Issue 2: No Cache Performance Gain

**Symptom**: Subsequent creations still slow

**Cause**: Caching disabled or cache cleared

**Solution**:
```python
# Ensure caching enabled
factory = PipelineFactory(enable_caching=True)

# Check cache stats
stats = factory.get_cache_stats()
print(f"Caching enabled: {stats['caching_enabled']}")
print(f"Cached classes: {stats['cached_classes']}")
```

### Issue 3: Pipeline Creation Fails

**Symptom**: Error during pipeline instantiation

**Causes**:
1. Invalid config_path
2. Missing SageMaker session
3. Invalid execution role

**Solution**:
```python
# Verify configuration exists
config_path = Path("config.json")
assert config_path.exists()

# Provide valid SageMaker session
session = PipelineSession()

# Provide valid execution role
role = "arn:aws:iam::123456789:role/SageMakerRole"

pipeline = factory.create(
    dag_id="xgboost_complete_e2e",
    config_path=str(config_path),
    sagemaker_session=session,
    execution_role=role
)
```

## Examples

### Example 1: Basic Factory Usage

```python
from pathlib import Path
from cursus.pipeline_catalog.core.pipeline_factory import PipelineFactory

# Initialize factory
factory = PipelineFactory()

# Create pipeline
pipeline = factory.create(
    dag_id="xgboost_complete_e2e",
    config_path="config.json"
)

# Use pipeline
sm_pipeline = pipeline.generate_pipeline()
```

### Example 2: Search-Driven Development

```python
# Developer doesn't know exact DAG ID
# Use natural language search
pipeline = factory.create_by_search(
    query="xgboost with calibration and evaluation",
    config_path="config.json"
)

# Factory finds best match automatically
```

### Example 3: Criteria-Based Selection

```python
# Build pipeline based on requirements
requirements = {
    "framework": "pytorch",
    "complexity": "comprehensive",
    "features": ["training", "evaluation", "registration"]
}

pipeline = factory.create_by_criteria(
    **requirements,
    config_path="config.json"
)
```

### Example 4: Pipeline Exploration

```python
# List all available pipelines
all_pipelines = factory.list_available_pipelines()

# Filter by framework
xgb_pipelines = [
    p for p in all_pipelines 
    if p['framework'] == 'xgboost'
]

print(f"XGBoost pipelines: {len(xgb_pipelines)}")
for p in xgb_pipelines:
    print(f"  - {p['dag_id']}: {p['features']}")
```

### Example 5: Performance Monitoring

```python
import time

# Measure first creation
start = time.time()
pipeline1 = factory.create("xgboost_complete_e2e")
first_time = time.time() - start

# Measure cached creation
start = time.time()
pipeline2 = factory.create("xgboost_complete_e2e")
cached_time = time.time() - start

print(f"First creation: {first_time*1000:.2f}ms")
print(f"Cached creation: {cached_time*1000:.2f}ms")
print(f"Speed improvement: {first_time/cached_time:.1f}x")
```

## References

### Related Code

- **Implementation**: `src/cursus/pipeline_catalog/core/pipeline_factory.py`
- **Tests**: `tests/pipeline_catalog/test_phase1_integration.py`
- **Base Pipeline**: `src/cursus/pipeline_catalog/core/base_pipeline.py`

### Related Components

- **[DAG Discovery](dag_discovery.md)**: Provides DAG catalog for factory
- **[Base Pipeline](../../src/cursus/pipeline_catalog/core/base_pipeline.py)**: Base class for all pipelines
- **[Catalog Registry](../../src/cursus/pipeline_catalog/core/catalog_registry.py)**: Provides metadata enrichment

### Design Documents

- **[Pipeline Catalog Redesign](../../1_design/pipeline_catalog_redesign.md)**: Overall system design
- **[Factory Pattern](../../1_design/pipeline_factory_design.md)**: Factory implementation details

### External References

- **[Python type() Function](https://docs.python.org/3/library/functions.html#type)**: Dynamic class creation
- **[Factory Pattern](https://refactoring.guru/design-patterns/factory-method)**: Design pattern reference
