---
tags:
  - code
  - implementation
  - pipeline_catalog
  - dag_discovery
  - auto_discovery
  - ast_parsing
keywords:
  - DAG auto-discovery
  - AST parsing
  - workspace-aware
  - pipeline discovery
  - metadata extraction
  - naming conventions
topics:
  - Pipeline catalog discovery
  - AST-based discovery
  - Workspace integration
  - Dynamic DAG loading
language: python
date of note: 2025-12-01
---

# DAG Auto-Discovery System

## Overview

The `DAGAutoDiscovery` class implements an AST-based DAG discovery system that automatically finds and catalogs all DAG definitions in both package and workspace locations. It follows proven patterns from the step catalog system and provides workspace-aware prioritization for local development.

The discovery system eliminates manual DAG registration, supports flexible naming conventions, enables workspace-based development, and integrates seamlessly with the pipeline catalog registry. It uses AST parsing to avoid import overhead and circular dependencies, making discovery fast and reliable.

Key capabilities include automatic DAG detection via file scanning, AST-based metadata extraction without imports, workspace priority (local overrides package), registry integration for enrichment, function-level caching for performance, and convention enforcement for consistency.

## Purpose and Major Tasks

### Primary Purpose
Automatically discover and catalog all DAG definitions following naming conventions (`create_*_dag` + `get_dag_metadata`), providing workspace-aware prioritization and registry-enriched metadata without requiring manual registration.

### Major Tasks

1. **Package DAG Discovery**: Scan `pipeline_catalog/shared_dags` directory recursively
2. **Workspace DAG Discovery**: Scan workspace `dags/` directories with priority
3. **AST-Based Parsing**: Extract DAG information without importing modules
4. **Metadata Extraction**: Parse `get_dag_metadata()` function returns
5. **Registry Integration**: Enrich discovered DAGs with catalog registry data
6. **Naming Convention Validation**: Enforce `create_*_dag` function naming
7. **ID Generation**: Convert function names to DAG IDs
8. **Priority Management**: Workspace DAGs override package DAGs
9. **Caching**: Cache discovered DAGs for performance
10. **Statistics**: Provide discovery metrics and summaries

## Module Contract

### Entry Point
```python
from cursus.pipeline_catalog.core.dag_discovery import DAGAutoDiscovery
```

### Class Initialization

```python
discovery = DAGAutoDiscovery(
    package_root: Path,              # Root of cursus package
    workspace_dirs: Optional[List[Path]] = None,  # Optional workspace directories
    registry_path: str = "catalog_index.json"     # Registry for enrichment
)
```

### Key Data Structures

#### DAGInfo Dataclass
```python
@dataclass
class DAGInfo:
    """Rich metadata about a discovered DAG."""
    dag_id: str                    # Unique DAG identifier
    dag_name: str                  # Function name (create_*_dag)
    dag_path: Path                 # Path to DAG file
    workspace_id: str              # "package" or workspace path
    framework: str                 # "xgboost", "pytorch", etc.
    complexity: str                # "simple", "standard", "comprehensive"
    features: List[str]            # ["training", "evaluation", ...]
    node_count: int                # Number of nodes in DAG
    edge_count: int                # Number of edges in DAG
    create_function: Optional[Callable]  # Loaded on-demand
    metadata: Dict[str, Any]       # Full metadata dictionary
```

### Discovery Methods

#### Primary Discovery
```python
# Discover all DAGs from package and workspaces
all_dags: Dict[str, DAGInfo] = discovery.discover_all_dags()

# Load specific DAG information
dag_info: Optional[DAGInfo] = discovery.load_dag_info("xgboost_complete_e2e")

# Get statistics
stats: Dict[str, Any] = discovery.get_discovery_stats()
```

#### Query Methods
```python
# List all discovered DAG IDs
dag_ids: List[str] = discovery.list_available_dags()

# Filter by framework
xgb_dags: List[str] = discovery.get_dags_by_framework("xgboost")

# Search by pattern
training_dags: List[str] = discovery.search_dags("training")
```

### Required Naming Conventions

**DAG Files**:
- Must end with `_dag.py`
- Must contain `create_*_dag()` function
- Must contain `get_dag_metadata()` function

**Example**:
```python
# File: xgboost/complete_e2e_dag.py

def create_xgboost_complete_e2e_dag() -> PipelineDAG:
    """Create XGBoost complete end-to-end DAG."""
    dag = PipelineDAG()
    # ... DAG construction
    return dag

def get_dag_metadata() -> DAGMetadata:
    """Get metadata for this DAG."""
    return DAGMetadata(
        description="Complete XGBoost pipeline...",
        complexity="comprehensive",
        features=["training", "evaluation", "calibration"],
        framework="xgboost",
        node_count=10,
        edge_count=11
    )
```

### Directory Structure

**Package Structure**:
```
src/cursus/pipeline_catalog/
├── shared_dags/
│   ├── xgboost/
│   │   ├── complete_e2e_dag.py      # ✅ Discovered
│   │   ├── simple_dag.py            # ✅ Discovered
│   │   └── training_dag.py          # ✅ Discovered
│   ├── pytorch/
│   │   ├── complete_e2e_dag.py      # ✅ Discovered
│   │   └── training_dag.py          # ✅ Discovered
│   └── ...
```

**Workspace Structure**:
```
workspace/
├── dags/                             # Workspace DAGs (highest priority)
│   ├── custom_xgboost_dag.py        # ✅ Overrides package if same ID
│   ├── experimental_dag.py          # ✅ Workspace-only DAG
│   └── ...
```

## Key Functions and Algorithms

### Discovery Orchestration

#### `discover_all_dags() -> Dict[str, DAGInfo]`
**Purpose**: Main discovery method that scans all locations and builds DAG catalog

**Algorithm**:
```python
1. Check if discovery already complete (return cached results)
2. Scan workspace directories (highest priority):
   a. For each workspace_dir:
      - Look for dags/ subdirectory
      - Scan recursively for *_dag.py files
      - Parse each file with AST
      - Add to cache (workspace DAGs override package)
3. Scan package shared_dags directory (lower priority):
   a. Locate package_root/pipeline_catalog/shared_dags
   b. Scan recursively for *_dag.py files
   c. Parse each file with AST
   d. Add to cache only if not already present (workspace wins)
4. Mark discovery as complete
5. Return complete DAG cache
```

**Returns**: `Dict[str, DAGInfo]` - DAG ID → DAG information

**Caching**: Results cached after first call, cleared only on explicit cache_clear()

**Example**:
```python
discovery = DAGAutoDiscovery(package_root=Path("/path/to/cursus"))
all_dags = discovery.discover_all_dags()

print(f"Discovered {len(all_dags)} DAGs")
# Output: Discovered 34 DAGs

for dag_id, dag_info in all_dags.items():
    print(f"{dag_id}: {dag_info.framework} - {dag_info.complexity}")
```

### AST-Based Parsing

#### `_extract_dag_from_ast(file_path: Path, workspace_id: str) -> Optional[DAGInfo]`
**Purpose**: Extract DAG information from Python file using AST parsing

**Algorithm**:
```python
1. Read file contents as text
2. Parse source code into AST tree
3. Find create_*_dag functions:
   a. Walk AST tree
   b. Match FunctionDef nodes
   c. Check name starts with "create_" and ends with "_dag"
   d. If no match: return None (not a valid DAG file)
4. Find get_dag_metadata function:
   a. Walk AST tree
   b. Match FunctionDef named "get_dag_metadata"
   c. Parse return statement for metadata
5. Extract DAG ID from function name:
   a. Remove "create_" prefix
   b. Remove "_dag" suffix
   c. Result: xgboost_complete_e2e
6. Enrich with registry data if available
7. Build and return DAGInfo object
```

**Parameters**:
- `file_path` (Path): Path to DAG file
- `workspace_id` (str): "package" or workspace path

**Returns**: `Optional[DAGInfo]` - DAG information or None if invalid

**Error Handling**: Logs warning and returns None on parse errors

**Example**:
```python
# File: xgboost/complete_e2e_dag.py
# Function: create_xgboost_complete_e2e_dag()
# Result: DAG ID = "xgboost_complete_e2e"

dag_info = discovery._extract_dag_from_ast(
    Path("shared_dags/xgboost/complete_e2e_dag.py"),
    "package"
)
# DAGInfo(
#     dag_id="xgboost_complete_e2e",
#     dag_name="create_xgboost_complete_e2e_dag",
#     framework="xgboost",
#     ...
# )
```

### ID Generation

#### `_extract_dag_id(function_name: str) -> str`
**Purpose**: Convert function name to DAG ID

**Algorithm**:
```python
1. Start with function name (e.g., "create_xgboost_complete_e2e_dag")
2. Remove "create_" prefix → "xgboost_complete_e2e_dag"
3. Remove "_dag" suffix → "xgboost_complete_e2e"
4. Return result
```

**Parameters**:
- `function_name` (str): Function name following convention

**Returns**: `str` - DAG ID

**Examples**:
```python
"create_xgboost_complete_e2e_dag" → "xgboost_complete_e2e"
"create_pytorch_training_dag" → "pytorch_training"
"create_simple_dag" → "simple"
```

### Metadata Extraction

#### `_extract_metadata_from_ast(tree: ast.AST) -> Dict[str, Any]`
**Purpose**: Extract metadata from get_dag_metadata() function

**Algorithm**:
```python
1. Walk AST tree looking for FunctionDef nodes
2. Find function named "get_dag_metadata"
3. If found:
   a. Locate Return statement in function body
   b. Parse return value (typically DAGMetadata constructor)
   c. Extract field values from constructor call
   d. Build metadata dictionary
4. If not found: return empty dict
5. Return metadata dictionary
```

**Parameters**:
- `tree` (ast.AST): Parsed AST tree

**Returns**: `Dict[str, Any]` - Extracted metadata

**Extracted Fields**:
- description: str
- complexity: str ("simple", "standard", "comprehensive")
- features: List[str]
- framework: str
- node_count: int
- edge_count: int
- extra_metadata: Dict[str, Any]

### Workspace Priority Management

#### `_scan_workspace_directory(workspace_dir: Path) -> Dict[str, DAGInfo]`
**Purpose**: Scan workspace DAGs directory with highest priority

**Algorithm**:
```python
1. Check for workspace_dir/dags/ directory
2. If not exists: return empty dict
3. Scan dags/ directory recursively:
   a. Find all *_dag.py files
   b. Parse each with AST
   c. Build DAGInfo with workspace_id=str(workspace_dir)
4. Return discovered DAGs
```

**Priority Rules**:
- Workspace DAGs always override package DAGs with same ID
- Multiple workspaces: first workspace wins
- Package DAGs: only used if no workspace override

**Example**:
```python
# Workspace: /home/user/my_workspace/dags/custom_xgboost_dag.py
# Package:   /cursus/pipeline_catalog/shared_dags/xgboost/complete_e2e_dag.py

# If both define "xgboost_complete_e2e" → workspace wins
```

### Query and Filter Methods

#### `list_available_dags() -> List[str]`
**Purpose**: List all discovered DAG IDs

**Returns**: `List[str]` - Sorted list of DAG IDs

**Example**:
```python
dag_ids = discovery.list_available_dags()
# ["bedrock_pytorch_e2e", "dummy_e2e_basic", "xgboost_complete_e2e", ...]
```

#### `get_dags_by_framework(framework: str) -> List[str]`
**Purpose**: Filter DAGs by framework

**Parameters**:
- `framework` (str): Framework name ("xgboost", "pytorch", etc.)

**Returns**: `List[str]` - DAG IDs matching framework

**Example**:
```python
xgb_dags = discovery.get_dags_by_framework("xgboost")
# ["xgboost_complete_e2e", "xgboost_simple", "xgboost_training", ...]
```

#### `search_dags(query: str) -> List[str]`
**Purpose**: Search DAGs by query string

**Algorithm**:
```python
1. Convert query to lowercase
2. For each discovered DAG:
   a. Check if query in dag_id (lowercase)
   b. Check if query in description (lowercase)
   c. Check if query in any feature
   d. If match: include in results
3. Return matching DAG IDs
```

**Parameters**:
- `query` (str): Search query

**Returns**: `List[str]` - Matching DAG IDs

**Example**:
```python
training_dags = discovery.search_dags("training")
# ["xgboost_training", "pytorch_training", "lightgbm_training", ...]

e2e_dags = discovery.search_dags("e2e")
# ["xgboost_complete_e2e", "pytorch_complete_e2e", "dummy_e2e_basic", ...]
```

### Statistics and Reporting

#### `get_discovery_stats() -> Dict[str, Any]`
**Purpose**: Get discovery statistics and summaries

**Returns**: Dictionary with:
```python
{
    "total_dags": 34,
    "package_dags": 34,
    "workspace_dags": 0,
    "frameworks": {
        "xgboost": 11,
        "pytorch": 6,
        "lightgbm": 5,
        "bedrock": 7,
        "dummy": 2,
        "generic": 3
    },
    "complexities": {
        "simple": 8,
        "standard": 15,
        "comprehensive": 11
    },
    "discovery_complete": true
}
```

**Example**:
```python
stats = discovery.get_discovery_stats()
print(f"Total: {stats['total_dags']}")
print(f"Frameworks: {stats['frameworks']}")
```

## Performance Characteristics

### Discovery Performance

| DAG Count | Initial Discovery | Cached Access | Memory Usage |
|-----------|------------------|---------------|--------------|
| 10 DAGs   | ~50ms           | <1ms          | ~500KB       |
| 34 DAGs   | ~100ms          | <1ms          | ~1.5MB       |
| 100 DAGs  | ~300ms          | <1ms          | ~5MB         |
| 1000 DAGs | ~3s             | <1ms          | ~50MB        |

**Optimization Features**:
- AST parsing (no imports, fast)
- Result caching (subsequent calls instant)
- Lazy function loading (functions loaded on-demand)
- Efficient file scanning (glob patterns)

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| discover_all_dags | O(n * m) | O(n) |
| load_dag_info | O(1) | O(1) |
| list_available_dags | O(n log n) | O(n) |
| get_dags_by_framework | O(n) | O(k) |
| search_dags | O(n * m) | O(k) |

Where:
- n = number of DAG files
- m = average lines per file
- k = matching results

## Integration Patterns

### With Pipeline Factory

```python
# Discovery feeds Factory
discovery = DAGAutoDiscovery(package_root=Path.cwd())
all_dags = discovery.discover_all_dags()

factory = PipelineFactory()
# Factory uses discovery internally to find DAGs
pipeline = factory.create("xgboost_complete_e2e", "config.json")
```

### With Workspace Development

```python
# Developer creates custom DAG in workspace
# workspace/dags/my_custom_dag.py:
def create_my_custom_dag() -> PipelineDAG:
    ...

def get_dag_metadata() -> DAGMetadata:
    return DAGMetadata(
        description="My custom pipeline",
        framework="xgboost",
        ...
    )

# Discovery automatically finds it
discovery = DAGAutoDiscovery(
    package_root=package_root,
    workspace_dirs=[Path("workspace")]
)
all_dags = discovery.discover_all_dags()
assert "my_custom" in all_dags  # ✅ Found!
```

### With Registry Enrichment

```python
# Discovery enriches DAGs with registry data
discovery = DAGAutoDiscovery(
    package_root=package_root,
    registry_path="catalog_index.json"
)

# DAG metadata enriched with registry information
dag_info = discovery.load_dag_info("xgboost_complete_e2e")
# dag_info.metadata includes registry data
```

## Error Handling

### Parse Errors

**Invalid Python Syntax**:
```python
# DAG file has syntax error
# → Warning logged, file skipped, discovery continues
```
**Cause**: Malformed Python code

**Handling**: Logged as warning, file excluded from results

### Missing Functions

**No create_*_dag function**:
```python
# File has get_dag_metadata but no create_*_dag
# → Warning logged, file skipped
```
**Cause**: Incomplete DAG file

**Handling**: File not considered valid DAG

### Path Errors

**Invalid package_root**:
```python
discovery = DAGAutoDiscovery(package_root=Path("/invalid/path"))
# → Warning about missing shared_dags, empty results
```
**Cause**: Incorrect package root path

**Handling**: No package DAGs discovered, only workspaces (if any)

### Registry Errors

**Registry not found**:
```python
discovery = DAGAutoDiscovery(registry_path="missing.json")
# → Warning logged, discovery continues without enrichment
```
**Cause**: Registry file doesn't exist

**Handling**: Discovery works, but no registry enrichment

## Best Practices

### For Package DAG Development

1. **Follow Naming Conventions**
   ```python
   # ✅ Good
   def create_xgboost_complete_e2e_dag() -> PipelineDAG:
       ...
   
   # ❌ Bad
   def create_dag() -> PipelineDAG:  # Too generic
       ...
   ```

2. **Always Include get_dag_metadata()**
   ```python
   # ✅ Required
   def get_dag_metadata() -> DAGMetadata:
       return DAGMetadata(
           description="...",
           complexity="standard",
           features=["training"],
           framework="xgboost",
           node_count=5,
           edge_count=4
       )
   ```

3. **Use Descriptive DAG IDs**
   ```python
   # ✅ Good: create_xgboost_training_with_evaluation_dag
   # → xgboost_training_with_evaluation

   # ❌ Bad: create_training_dag
   # → training  # Too generic
   ```

### For Workspace Development

1. **Organize in dags/ Directory**
   ```
   workspace/
   └── dags/
       ├── custom_xgboost_dag.py
       ├── experimental_pytorch_dag.py
       └── ...
   ```

2. **Use Unique DAG IDs**
   ```python
   # If overriding package DAG, use same ID
   # If new DAG, use unique ID to avoid conflicts
   ```

3. **Test Discovery**
   ```python
   discovery = DAGAutoDiscovery(
       package_root=package_root,
       workspace_dirs=[workspace_path]
   )
   dag_info = discovery.load_dag_info("my_custom")
   assert dag_info.workspace_id != "package"
   ```

### For Performance

1. **Cache Discovery Results**
   ```python
   # Discover once
   discovery = DAGAutoDiscovery(package_root=package_root)
   all_dags = discovery.discover_all_dags()  # ~100ms
   
   # Subsequent accesses cached
   dag_info = discovery.load_dag_info("xgboost_complete_e2e")  # <1ms
   ```

2. **Use Specific Queries**
   ```python
   # ✅ Efficient
   xgb_dags = discovery.get_dags_by_framework("xgboost")
   
   # ❌ Less efficient
   all_dags = discovery.discover_all_dags()
   xgb_dags = [d for d in all_dags if all_dags[d].framework == "xgboost"]
   ```

## Troubleshooting

### Issue 1: DAG Not Discovered

**Symptom**: DAG file exists but not found by discovery

**Common Causes**:
1. Filename doesn't end with `_dag.py`
2. No `create_*_dag()` function
3. No `get_dag_metadata()` function
4. File not in scanned directory

**Solution**:
```python
# 1. Check filename
assert file.endswith("_dag.py")

# 2. Check function exists
def create_my_custom_dag() -> PipelineDAG:  # ✅ Required
    ...

# 3. Check metadata function
def get_dag_metadata() -> DAGMetadata:  # ✅ Required
    ...

# 4. Check location
# Must be in: shared_dags/ or workspace/dags/
```

### Issue 2: Workspace DAG Not Overriding

**Symptom**: Package DAG used instead of workspace DAG

**Cause**: DAG IDs don't match exactly

**Solution**:
```python
# Ensure same DAG ID

# Package: create_xgboost_complete_e2e_dag
# → ID: xgboost_complete_e2e

# Workspace must use same function name
def create_xgboost_complete_e2e_dag():  # ✅ Same ID
    ...
```

### Issue 3: Slow Discovery

**Symptom**: Discovery takes longer than expected

**Common Causes**:
1. Large number of files in directory
2. Deep directory nesting
3. Large Python files

**Solution**:
```python
# 1. Use specific workspace directories
discovery = DAGAutoDiscovery(
    package_root=package_root,
    workspace_dirs=[Path("workspace/dags")]  # Specific path
)

# 2. Cache results
all_dags = discovery.discover_all_dags()  # Once
# Subsequent accesses use cache
```

## Examples

### Example 1: Basic Discovery

```python
from pathlib import Path
from cursus.pipeline_catalog.core.dag_discovery import DAGAutoDiscovery

# Initialize discovery
discovery = DAGAutoDiscovery(package_root=Path("/path/to/cursus"))

# Discover all DAGs
all_dags = discovery.discover_all_dags()
print(f"Found {len(all_dags)} DAGs")

# List DAG IDs
for dag_id in discovery.list_available_dags():
    print(f"- {dag_id}")
```

### Example 2: Framework Filtering

```python
# Get all XGBoost DAGs
xgb_dags = discovery.get_dags_by_framework("xgboost")
print(f"XGBoost DAGs: {xgb_dags}")

# Get specific DAG info
dag_info = discovery.load_dag_info("xgboost_complete_e2e")
print(f"Framework: {dag_info.framework}")
print(f"Complexity: {dag_info.complexity}")
print(f"Features: {dag_info.features}")
```

### Example 3: Workspace Development

```python
# Developer creates custom DAG
# workspace/dags/my_custom_xgboost_dag.py

from pathlib import Path

# Discover with workspace
discovery = DAGAutoDiscovery(
    package_root=Path("/path/to/cursus"),
    workspace_dirs=[Path("/path/to/workspace")]
)

all_dags = discovery.discover_all_dags()

# Check if custom DAG found
custom_dag = discovery.load_dag_info("my_custom_xgboost")
assert custom_dag is not None
assert custom_dag.workspace_id != "package"
print(f"Custom DAG from: {custom_dag.workspace_id}")
```

### Example 4: Discovery Statistics

```python
# Get discovery statistics
stats = discovery.get_discovery_stats()

print(f"Total DAGs: {stats['total_dags']}")
print(f"Package DAGs: {stats['package_dags']}")
print(f"Workspace DAGs: {stats['workspace_dags']}")

print("\nFramework breakdown:")
for fw, count in stats['frameworks'].items():
    print(f"  {fw}: {count}")

print("\nComplexity breakdown:")
for comp, count in stats['complexities'].items():
    print(f"  {comp}: {count}")
```

## References

### Related Code

- **Implementation**: `src/cursus/pipeline_catalog/core/dag_discovery.py`
- **Tests**: `tests/pipeline_catalog/test_phase1_integration.py`
- **DAG Files**: `src/cursus/pipeline_catalog/shared_dags/`

### Related Components

- **[Pipeline Factory](pipeline_factory.md)**: Uses DAG discovery for pipeline creation
- **[Base Pipeline](../../src/cursus/pipeline_catalog/core/base_pipeline.py)**: Foundation for all pipelines
- **[Catalog Registry](../../src/cursus/pipeline_catalog/core/catalog_registry.py)**: Provides enrichment data

### Design Documents

- **[Pipeline Catalog Redesign](../../1_design/pipeline_catalog_redesign.md)**: Overall system design
- **[Discovery Patterns](../../0_developer_guide/step_catalog_integration_guide.md)**: Discovery pattern reference

### External References

- **[Python AST Module](https://docs.python.org/3/library/ast.html)**: AST parsing documentation
- **[Path Glob Patterns](https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob)**: File pattern matching
