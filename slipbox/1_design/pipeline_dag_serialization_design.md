---
tags:
  - design
  - serialization
  - dag
  - knowledge_transfer
  - persistence
keywords:
  - pipeline serialization
  - dag export
  - dag import
  - json format
  - knowledge transfer
  - pipeline persistence
topics:
  - pipeline architecture
  - data serialization
  - knowledge management
language: python
date of note: 2026-01-16
---

# Pipeline DAG Serialization Design

## What is the Purpose of Pipeline DAG Serialization?

Pipeline DAG Serialization provides **persistence and knowledge transfer capabilities** for pipeline topologies. It enables saving, sharing, and reconstructing pipeline structures across different environments, teams, and time periods, facilitating collaboration and documentation.

## Core Purpose

Pipeline DAG Serialization provides the **persistence layer** that:

1. **Knowledge Transfer** - Enable sharing pipeline architectures between teams and projects
2. **Version Control** - Track pipeline evolution through serializable representations
3. **Documentation** - Generate human-readable pipeline structure documentation
4. **Reproducibility** - Recreate exact pipeline topologies from saved configurations
5. **Analysis and Optimization** - Enable offline analysis of pipeline structures

## Key Components

### 1. PipelineDAGWriter

The writer component serializes PipelineDAG instances to JSON format with metadata:

```python
class PipelineDAGWriter:
    """
    Writer for serializing PipelineDAG to various formats.
    
    Features:
    - JSON format with metadata
    - Pretty printing for readability
    - Validation before writing
    - Metadata tracking (creation time, version, etc.)
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, dag: PipelineDAG, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize writer with a PipelineDAG instance.
        
        Args:
            dag: PipelineDAG instance to serialize
            metadata: Optional metadata to include in serialization
        """
        self.dag = dag
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PipelineDAG to dictionary representation"""
        return {
            "version": self.VERSION,
            "created_at": datetime.now().isoformat(),
            "metadata": self.metadata,
            "dag": {
                "nodes": self.dag.nodes,
                "edges": [[src, dst] for src, dst in self.dag.edges],
            },
            "statistics": self._compute_statistics(),
        }
    
    def write_to_file(
        self,
        filepath: Union[str, Path],
        pretty: bool = True,
        validate: bool = True,
    ) -> None:
        """Write PipelineDAG to a JSON file with validation"""
        if validate:
            self._validate_dag()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        json_str = self.to_json(pretty=pretty)
        with open(filepath, 'w') as f:
            f.write(json_str)
```

### 2. PipelineDAGReader

The reader component deserializes JSON back to PipelineDAG instances:

```python
class PipelineDAGReader:
    """
    Reader for deserializing PipelineDAG from various formats.
    
    Features:
    - JSON format support
    - Validation during reading
    - Metadata extraction
    - Version compatibility checking
    """
    
    SUPPORTED_VERSIONS = ["1.0.0"]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> PipelineDAG:
        """Create PipelineDAG from dictionary representation"""
        if validate:
            cls._validate_data(data)
        
        dag_data = data.get("dag", {})
        nodes = dag_data.get("nodes", [])
        edges = [tuple(edge) for edge in dag_data.get("edges", [])]
        
        return PipelineDAG(nodes=nodes, edges=edges)
    
    @classmethod
    def read_from_file(cls, filepath: Union[str, Path], validate: bool = True) -> PipelineDAG:
        """Read PipelineDAG from a JSON file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r') as f:
            json_str = f.read()
        
        return cls.from_json(json_str, validate=validate)
    
    @classmethod
    def extract_metadata(cls, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata without loading full DAG"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return {
            "version": data.get("version"),
            "created_at": data.get("created_at"),
            "metadata": data.get("metadata", {}),
            "statistics": data.get("statistics", {}),
        }
```

### 3. JSON Format Structure

The serialization format includes comprehensive metadata and statistics:

```json
{
  "version": "1.0.0",
  "created_at": "2026-01-16T20:00:00",
  "metadata": {
    "project": "ml_pipeline",
    "author": "data_scientist",
    "description": "XGBoost training pipeline",
    "team": "ml_platform"
  },
  "dag": {
    "nodes": [
      "data_ingestion",
      "data_validation",
      "feature_engineering",
      "xgboost_training",
      "model_evaluation",
      "model_registration"
    ],
    "edges": [
      ["data_ingestion", "data_validation"],
      ["data_validation", "feature_engineering"],
      ["feature_engineering", "xgboost_training"],
      ["xgboost_training", "model_evaluation"],
      ["model_evaluation", "model_registration"]
    ]
  },
  "statistics": {
    "node_count": 6,
    "edge_count": 5,
    "has_cycles": false,
    "entry_nodes": ["data_ingestion"],
    "exit_nodes": ["model_registration"],
    "max_depth": 5,
    "isolated_nodes": []
  }
}
```

### 4. Validation System

Comprehensive validation ensures data integrity:

```python
class DAGValidator:
    """Validation for DAG serialization"""
    
    @staticmethod
    def validate_before_write(dag: PipelineDAG) -> None:
        """Validate DAG structure before writing"""
        # Check for cycles using topological sort
        try:
            dag.topological_sort()
        except ValueError as e:
            raise ValueError(f"DAG validation failed: {e}")
        
        # Check for empty nodes
        if not dag.nodes:
            raise ValueError("Cannot write empty DAG (no nodes)")
        
        # Check for dangling edges
        node_set = set(dag.nodes)
        for src, dst in dag.edges:
            if src not in node_set:
                raise ValueError(f"Edge references non-existent source: {src}")
            if dst not in node_set:
                raise ValueError(f"Edge references non-existent destination: {dst}")
    
    @staticmethod
    def validate_after_read(data: Dict[str, Any]) -> None:
        """Validate data structure after reading"""
        # Check version compatibility
        version = data.get("version")
        if version not in PipelineDAGReader.SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}")
        
        # Check required fields
        if "dag" not in data:
            raise ValueError("Missing required field: 'dag'")
        
        dag_data = data["dag"]
        if "nodes" not in dag_data or "edges" not in dag_data:
            raise ValueError("Missing required DAG fields")
        
        # Validate edge format
        for edge in dag_data["edges"]:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError(f"Invalid edge format: {edge}")
```

### 5. Statistics Computation

Automatic computation of DAG statistics for analysis:

```python
class StatisticsComputer:
    """Compute statistics about the DAG"""
    
    @staticmethod
    def compute_statistics(dag: PipelineDAG) -> Dict[str, Any]:
        """Compute comprehensive DAG statistics"""
        try:
            topo_order = dag.topological_sort()
            has_cycles = False
        except ValueError:
            topo_order = []
            has_cycles = True
        
        # Count nodes by degree
        in_degrees = {
            node: len(dag.reverse_adj.get(node, [])) 
            for node in dag.nodes
        }
        out_degrees = {
            node: len(dag.adj_list.get(node, [])) 
            for node in dag.nodes
        }
        
        # Identify entry and exit nodes
        entry_nodes = [node for node, deg in in_degrees.items() if deg == 0]
        exit_nodes = [node for node, deg in out_degrees.items() if deg == 0]
        
        # Compute maximum depth
        max_depth = StatisticsComputer._compute_max_depth(dag, topo_order)
        
        return {
            "node_count": len(dag.nodes),
            "edge_count": len(dag.edges),
            "has_cycles": has_cycles,
            "entry_nodes": entry_nodes,
            "exit_nodes": exit_nodes,
            "max_depth": max_depth,
            "isolated_nodes": [
                node for node in dag.nodes 
                if in_degrees[node] == 0 and out_degrees[node] == 0
            ],
        }
    
    @staticmethod
    def _compute_max_depth(dag: PipelineDAG, topo_order: List[str]) -> int:
        """Compute maximum depth using dynamic programming"""
        if not topo_order:
            return 0
        
        depths = {node: 0 for node in dag.nodes}
        
        for node in topo_order:
            for predecessor in dag.reverse_adj.get(node, []):
                depths[node] = max(depths[node], depths[predecessor] + 1)
        
        return max(depths.values()) if depths else 0
```

## Convenience Functions

Simple interface for common operations:

```python
def export_dag_to_json(
    dag: PipelineDAG,
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
    pretty: bool = True,
) -> None:
    """
    Convenience function to export a PipelineDAG to JSON file.
    
    Example:
        >>> dag = PipelineDAG(nodes=["step1", "step2"], edges=[("step1", "step2")])
        >>> export_dag_to_json(
        ...     dag, 
        ...     "my_pipeline.json",
        ...     metadata={"project": "ml_platform"}
        ... )
    """
    writer = PipelineDAGWriter(dag, metadata=metadata)
    writer.write_to_file(filepath, pretty=pretty)


def import_dag_from_json(filepath: Union[str, Path]) -> PipelineDAG:
    """
    Convenience function to import a PipelineDAG from JSON file.
    
    Example:
        >>> dag = import_dag_from_json("my_pipeline.json")
        >>> print(dag.nodes)
        ['step1', 'step2']
    """
    return PipelineDAGReader.read_from_file(filepath)
```

## Integration with Other Components

### With Pipeline DAG

Direct integration with PipelineDAG class:

```python
# Create and serialize a DAG
dag = PipelineDAG(
    nodes=["data_prep", "train", "evaluate"],
    edges=[("data_prep", "train"), ("train", "evaluate")]
)

# Export
writer = PipelineDAGWriter(dag, metadata={"project": "xgboost_pipeline"})
writer.write_to_file("pipeline.json")

# Import
reader = PipelineDAGReader()
loaded_dag = reader.read_from_file("pipeline.json")

# Verify
assert dag.nodes == loaded_dag.nodes
assert dag.edges == loaded_dag.edges
```

### With Pipeline Compilation

Enable pipeline compilation from serialized DAGs:

```python
class SerializablePipelineCompiler:
    """Compiler that works with serialized DAGs"""
    
    def compile_from_file(self, dag_path: str, config_path: str):
        """Compile pipeline from serialized DAG"""
        # Load DAG structure
        dag = import_dag_from_json(dag_path)
        
        # Load configurations
        with open(config_path) as f:
            configs = json.load(f)
        
        # Create resolver and execution plan
        resolver = PipelineDAGResolver(
            dag=dag,
            available_configs=configs
        )
        
        execution_plan = resolver.create_execution_plan()
        
        return self._build_pipeline(execution_plan)
```

### With Version Control

Enable tracking of pipeline evolution:

```python
class PipelineVersionControl:
    """Track pipeline changes over time"""
    
    def save_version(self, dag: PipelineDAG, version: str, comment: str):
        """Save a versioned snapshot of the pipeline"""
        metadata = {
            "version": version,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
            "author": self.get_current_user()
        }
        
        filename = f"pipeline_v{version}.json"
        export_dag_to_json(dag, f"versions/{filename}", metadata=metadata)
    
    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two pipeline versions"""
        dag1 = import_dag_from_json(f"versions/pipeline_v{version1}.json")
        dag2 = import_dag_from_json(f"versions/pipeline_v{version2}.json")
        
        return {
            "nodes_added": set(dag2.nodes) - set(dag1.nodes),
            "nodes_removed": set(dag1.nodes) - set(dag2.nodes),
            "edges_added": set(dag2.edges) - set(dag1.edges),
            "edges_removed": set(dag1.edges) - set(dag2.edges),
        }
```

## Strategic Value

Pipeline DAG Serialization provides:

1. **Knowledge Transfer**: Share pipeline architectures across teams and projects
2. **Reproducibility**: Recreate exact pipeline topologies from saved files
3. **Documentation**: Generate human-readable pipeline documentation
4. **Version Control**: Track pipeline evolution over time
5. **Offline Analysis**: Enable analysis without running pipelines
6. **Collaboration**: Facilitate team collaboration on pipeline design

## Example Usage Patterns

### Pattern 1: Basic Export/Import

```python
from cursus.api.dag import PipelineDAG, export_dag_to_json, import_dag_from_json

# Create a pipeline DAG
dag = PipelineDAG(
    nodes=["data_ingestion", "preprocessing", "training", "evaluation"],
    edges=[
        ("data_ingestion", "preprocessing"),
        ("preprocessing", "training"),
        ("training", "evaluation"),
    ]
)

# Export to JSON
export_dag_to_json(
    dag,
    "my_pipeline.json",
    metadata={
        "project": "customer_churn",
        "team": "ml_platform",
        "description": "Customer churn prediction pipeline"
    }
)

# Import from JSON
loaded_dag = import_dag_from_json("my_pipeline.json")

# Verify topological order
execution_order = loaded_dag.topological_sort()
print(f"Execution order: {' → '.join(execution_order)}")
```

### Pattern 2: Metadata Extraction

```python
from cursus.api.dag import PipelineDAGReader

# Extract metadata without loading full DAG
metadata = PipelineDAGReader.extract_metadata("my_pipeline.json")

print(f"Pipeline Version: {metadata['version']}")
print(f"Created: {metadata['created_at']}")
print(f"Metadata: {metadata['metadata']}")
print(f"\nStatistics:")
for key, value in metadata['statistics'].items():
    print(f"  {key}: {value}")
```

### Pattern 3: Pipeline Comparison

```python
# Load two pipeline versions
dag_v1 = import_dag_from_json("pipeline_v1.json")
dag_v2 = import_dag_from_json("pipeline_v2.json")

# Compare structures
nodes_added = set(dag_v2.nodes) - set(dag_v1.nodes)
nodes_removed = set(dag_v1.nodes) - set(dag_v2.nodes)
edges_added = set(dag_v2.edges) - set(dag_v1.edges)

print(f"Added nodes: {nodes_added}")
print(f"Removed nodes: {nodes_removed}")
print(f"Added edges: {edges_added}")
```

### Pattern 4: Complex DAG with Branching

```python
# Create complex DAG with parallel branches
dag = PipelineDAG()

# Add nodes
for node in ["data", "validate", "feature_eng", "train_a", "train_b", "ensemble", "deploy"]:
    dag.add_node(node)

# Add edges (parallel training branches)
dag.add_edge("data", "validate")
dag.add_edge("validate", "feature_eng")
dag.add_edge("feature_eng", "train_a")
dag.add_edge("feature_eng", "train_b")
dag.add_edge("train_a", "ensemble")
dag.add_edge("train_b", "ensemble")
dag.add_edge("ensemble", "deploy")

# Export with detailed metadata
writer = PipelineDAGWriter(
    dag,
    metadata={
        "pipeline_name": "parallel_training",
        "description": "Pipeline with parallel model training",
        "models": ["xgboost", "lightgbm"],
    }
)
writer.write_to_file("parallel_pipeline.json")
```

### Pattern 5: Round-Trip Validation

```python
# Original DAG
original = PipelineDAG(
    nodes=["step1", "step2", "step3"],
    edges=[("step1", "step2"), ("step2", "step3")]
)

# Export to JSON
writer = PipelineDAGWriter(original)
json_str = writer.to_json()

# Import from JSON
imported = PipelineDAGReader.from_json(json_str)

# Verify equivalence
assert set(original.nodes) == set(imported.nodes)
assert set(original.edges) == set(imported.edges)
print("✓ Round-trip successful")
```

## Benefits and Use Cases

### Knowledge Transfer

- Share pipeline architectures between teams
- Onboard new team members with documented pipelines
- Create pipeline templates for common patterns

### Version Control

- Track pipeline evolution over time
- Compare different pipeline versions
- Roll back to previous pipeline versions

### Documentation

- Generate human-readable pipeline documentation
- Include metadata for context and provenance
- Analyze pipeline complexity and structure

### Reproducibility

- Recreate exact pipeline topologies
- Ensure consistent pipeline definitions
- Enable deterministic pipeline execution

### Analysis

- Perform offline pipeline analysis
- Identify optimization opportunities
- Validate pipeline structures before execution

## Related Documents

- [Pipeline DAG](pipeline_dag.md) - Core DAG implementation
- [Pipeline DAG Resolver](pipeline_dag_resolver_design.md) - DAG resolution and execution planning
- [Dynamic Template System](dynamic_template_system.md) - Template system using DAGs
- [Pipeline Catalog Design](pipeline_catalog_design.md) - Catalog of reusable pipelines
- [Step Config Resolver](step_config_resolver.md) - Configuration resolution for DAG nodes
