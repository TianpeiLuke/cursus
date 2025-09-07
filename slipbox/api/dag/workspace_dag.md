---
tags:
  - code
  - api
  - dag
  - workspace
  - cross-workspace
keywords:
  - WorkspaceAwareDAG
  - workspace steps
  - cross-workspace dependencies
  - developer workspaces
  - workspace validation
  - workspace configuration
topics:
  - workspace-aware DAG
  - cross-workspace dependencies
  - developer workspace management
  - workspace validation
language: python
date of note: 2024-12-07
---

# Workspace DAG

Workspace-aware DAG implementation extending PipelineDAG to support workspace step configurations and cross-workspace dependency validation while maintaining compatibility.

## Overview

The Workspace DAG extends the base PipelineDAG with comprehensive workspace support for multi-developer environments. It provides workspace step management with developer isolation, cross-workspace dependency validation and analysis, and workspace configuration generation for pipeline deployment.

The module supports workspace complexity analysis for performance optimization, workspace merging and cloning capabilities for collaboration, and comprehensive validation with cross-workspace dependency tracking. It maintains full compatibility with the base PipelineDAG while adding workspace-specific features.

## Classes and Methods

### Classes
- [`WorkspaceAwareDAG`](#workspaceawaredag) - DAG with workspace step support and cross-workspace dependencies

## API Reference

### WorkspaceAwareDAG

_class_ cursus.api.dag.workspace_dag.WorkspaceAwareDAG(_workspace_root_, _nodes=None_, _edges=None_)

DAG with workspace step support and cross-workspace dependency management.

**Parameters:**
- **workspace_root** (_str_) – Root path of the workspace
- **nodes** (_Optional[List[str]]_) – Optional list of step names
- **edges** (_Optional[List[tuple]]_) – Optional list of (from_step, to_step) tuples

```python
from cursus.api.dag.workspace_dag import WorkspaceAwareDAG

# Create workspace-aware DAG
dag = WorkspaceAwareDAG(workspace_root="/workspace/ml_project")

# Create with initial structure
nodes = ["preprocessing", "training", "evaluation"]
edges = [("preprocessing", "training"), ("training", "evaluation")]
dag = WorkspaceAwareDAG(
    workspace_root="/workspace/ml_project",
    nodes=nodes,
    edges=edges
)
```

#### add_workspace_step

add_workspace_step(_step_name_, _developer_id_, _step_type_, _config_data_, _dependencies=None_)

Add a workspace step to the DAG with developer and configuration information.

**Parameters:**
- **step_name** (_str_) – Name of the step
- **developer_id** (_str_) – Developer workspace identifier
- **step_type** (_str_) – Type of the step
- **config_data** (_Dict[str, Any]_) – Step configuration data
- **dependencies** (_Optional[List[str]]_) – Optional list of step dependencies

```python
# Add workspace step
dag.add_workspace_step(
    step_name="data_preprocessing",
    developer_id="alice",
    step_type="PreprocessingStep",
    config_data={
        "input_path": "/data/raw",
        "output_path": "/data/processed",
        "preprocessing_type": "standard"
    },
    dependencies=["data_loading"]
)

# Add step without dependencies (source step)
dag.add_workspace_step(
    step_name="data_loading",
    developer_id="alice",
    step_type="DataLoadingStep",
    config_data={"source": "s3://bucket/data"}
)
```

#### remove_workspace_step

remove_workspace_step(_step_name_)

Remove a workspace step from the DAG.

**Parameters:**
- **step_name** (_str_) – Name of the step to remove

**Returns:**
- **bool** – True if step was removed, False if not found

```python
# Remove workspace step
success = dag.remove_workspace_step("old_preprocessing")
if success:
    print("Step removed successfully")
else:
    print("Step not found")
```

#### get_workspace_step

get_workspace_step(_step_name_)

Get workspace step configuration by name.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **Optional[Dict[str, Any]]** – Workspace step configuration if found, None otherwise

```python
# Get workspace step information
step_info = dag.get_workspace_step("data_preprocessing")
if step_info:
    print(f"Developer: {step_info['developer_id']}")
    print(f"Step type: {step_info['step_type']}")
    print(f"Config: {step_info['config_data']}")
```

#### get_developers

get_developers()

Get list of unique developers in the DAG.

**Returns:**
- **List[str]** – List of unique developer IDs

```python
# Get all developers
developers = dag.get_developers()
print(f"Developers in workspace: {developers}")
```

#### get_steps_by_developer

get_steps_by_developer(_developer_id_)

Get all step names for a specific developer.

**Parameters:**
- **developer_id** (_str_) – Developer workspace identifier

**Returns:**
- **List[str]** – List of step names for the developer

```python
# Get steps for specific developer
alice_steps = dag.get_steps_by_developer("alice")
print(f"Alice's steps: {alice_steps}")

bob_steps = dag.get_steps_by_developer("bob")
print(f"Bob's steps: {bob_steps}")
```

#### get_steps_by_type

get_steps_by_type(_step_type_)

Get all step names of a specific type.

**Parameters:**
- **step_type** (_str_) – Type of steps to find

**Returns:**
- **List[str]** – List of step names of the specified type

```python
# Get all preprocessing steps
preprocessing_steps = dag.get_steps_by_type("PreprocessingStep")
print(f"Preprocessing steps: {preprocessing_steps}")

# Get all training steps
training_steps = dag.get_steps_by_type("TrainingStep")
print(f"Training steps: {training_steps}")
```

#### validate_workspace_dependencies

validate_workspace_dependencies()

Validate workspace dependencies and cross-workspace references.

**Returns:**
- **Dict[str, Any]** – Validation result dictionary with errors, warnings, and statistics

```python
# Validate workspace dependencies
validation_result = dag.validate_workspace_dependencies()

print(f"Validation passed: {validation_result['valid']}")

if validation_result['errors']:
    print("Errors:")
    for error in validation_result['errors']:
        print(f"  - {error}")

if validation_result['warnings']:
    print("Warnings:")
    for warning in validation_result['warnings']:
        print(f"  - {warning}")

# Check cross-workspace dependencies
cross_deps = validation_result['cross_workspace_dependencies']
print(f"Cross-workspace dependencies: {len(cross_deps)}")

for dep in cross_deps:
    print(f"  {dep['dependent_step']} ({dep['dependent_developer']}) -> "
          f"{dep['dependency_step']} ({dep['dependency_developer']})")

# View statistics
stats = validation_result['dependency_stats']
print(f"Total dependencies: {stats['total_dependencies']}")
print(f"Cross-workspace: {stats['cross_workspace_count']}")
print(f"Intra-workspace: {stats['intra_workspace_count']}")
```

#### to_workspace_pipeline_config

to_workspace_pipeline_config(_pipeline_name_)

Convert DAG to workspace pipeline configuration.

**Parameters:**
- **pipeline_name** (_str_) – Name for the pipeline

**Returns:**
- **Dict[str, Any]** – Dictionary representing workspace pipeline configuration

```python
# Convert to workspace pipeline configuration
config = dag.to_workspace_pipeline_config("ml_training_pipeline")

print(f"Pipeline name: {config['pipeline_name']}")
print(f"Workspace root: {config['workspace_root']}")
print(f"Number of steps: {len(config['steps'])}")

# Access step configurations
for step in config['steps']:
    print(f"Step: {step['step_name']} (Developer: {step['developer_id']})")
```

#### get_workspace_summary

get_workspace_summary()

Get summary of workspace DAG structure with developer and step type statistics.

**Returns:**
- **Dict[str, Any]** – Dictionary containing workspace summary information

```python
# Get workspace summary
summary = dag.get_workspace_summary()

print(f"Workspace root: {summary['workspace_root']}")
print(f"Total steps: {summary['total_steps']}")
print(f"Total edges: {summary['total_edges']}")
print(f"Developers: {summary['developers']}")

# Developer statistics
for dev_id, stats in summary['developer_stats'].items():
    print(f"Developer {dev_id}:")
    print(f"  Steps: {stats['step_count']}")
    print(f"  Types: {stats['step_types']}")

# Step type statistics
print("Step type distribution:")
for step_type, count in summary['step_type_stats'].items():
    print(f"  {step_type}: {count}")

# Dependency validation summary
dep_validation = summary['dependency_validation']
print(f"Dependencies valid: {dep_validation['valid']}")
```

#### get_execution_order

get_execution_order()

Get execution order with workspace context information.

**Returns:**
- **List[Dict[str, Any]]** – List of step execution information with workspace details

```python
# Get execution order with workspace context
execution_order = dag.get_execution_order()

print("Execution Order:")
for step_info in execution_order:
    print(f"{step_info['execution_index']}: {step_info['step_name']}")
    print(f"  Developer: {step_info['developer_id']}")
    print(f"  Type: {step_info['step_type']}")
    print(f"  Dependencies: {step_info['dependencies']}")
```

#### analyze_workspace_complexity

analyze_workspace_complexity()

Analyze complexity metrics for the workspace DAG.

**Returns:**
- **Dict[str, Any]** – Dictionary containing complexity analysis and recommendations

```python
# Analyze workspace complexity
analysis = dag.analyze_workspace_complexity()

# Basic metrics
basic = analysis['basic_metrics']
print(f"Nodes: {basic['node_count']}")
print(f"Edges: {basic['edge_count']}")
print(f"Developers: {basic['developer_count']}")
print(f"Step types: {basic['step_type_count']}")

# Complexity metrics
complexity = analysis['complexity_metrics']
print(f"Average dependencies per step: {complexity['avg_dependencies_per_step']:.2f}")
print(f"Max dependencies: {complexity['max_dependencies']}")
print(f"Max dependents: {complexity['max_dependents']}")

# Developer analysis
for dev_id, dev_analysis in analysis['developer_analysis'].items():
    print(f"Developer {dev_id}:")
    print(f"  Step count: {dev_analysis['step_count']}")
    print(f"  Step types: {dev_analysis['step_types']}")
    print(f"  Avg dependencies: {dev_analysis['avg_dependencies']:.2f}")
    print(f"  Cross-workspace deps: {dev_analysis['cross_workspace_deps']}")

# Recommendations
if analysis['recommendations']:
    print("Recommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
```

## Class Methods

#### from_workspace_config

from_workspace_config(_workspace_config_)

Create workspace-aware DAG from workspace configuration.

**Parameters:**
- **workspace_config** (_Dict[str, Any]_) – Dictionary representing workspace pipeline configuration

**Returns:**
- **WorkspaceAwareDAG** – WorkspaceAwareDAG instance

```python
# Create DAG from workspace configuration
config = {
    "workspace_root": "/workspace/ml_project",
    "steps": [
        {
            "step_name": "data_loading",
            "developer_id": "alice",
            "step_type": "DataLoadingStep",
            "config_data": {"source": "s3://bucket/data"},
            "dependencies": []
        },
        {
            "step_name": "preprocessing",
            "developer_id": "alice", 
            "step_type": "PreprocessingStep",
            "config_data": {"method": "standard"},
            "dependencies": ["data_loading"]
        }
    ]
}

dag = WorkspaceAwareDAG.from_workspace_config(config)
print(f"Created DAG with {len(dag.workspace_steps)} steps")
```

#### clone

clone()

Create a deep copy of the workspace DAG.

**Returns:**
- **WorkspaceAwareDAG** – Deep copy of the workspace DAG

```python
# Clone workspace DAG
original_dag = WorkspaceAwareDAG(workspace_root="/workspace/original")
# ... add steps to original_dag ...

cloned_dag = original_dag.clone()
print(f"Cloned DAG with {len(cloned_dag.workspace_steps)} steps")

# Modifications to cloned DAG don't affect original
cloned_dag.add_workspace_step("new_step", "bob", "NewStep", {})
print(f"Original: {len(original_dag.workspace_steps)} steps")
print(f"Cloned: {len(cloned_dag.workspace_steps)} steps")
```

#### merge_workspace_dag

merge_workspace_dag(_other_dag_)

Merge another workspace DAG into this one.

**Parameters:**
- **other_dag** (_WorkspaceAwareDAG_) – Another WorkspaceAwareDAG to merge

**Raises:**
- **ValueError** – If there are conflicting step names

```python
# Merge workspace DAGs
dag1 = WorkspaceAwareDAG(workspace_root="/workspace/project1")
dag1.add_workspace_step("step1", "alice", "Step1", {})

dag2 = WorkspaceAwareDAG(workspace_root="/workspace/project2")
dag2.add_workspace_step("step2", "bob", "Step2", {})

try:
    dag1.merge_workspace_dag(dag2)
    print(f"Merged DAG has {len(dag1.workspace_steps)} steps")
    print(f"Developers: {dag1.get_developers()}")
except ValueError as e:
    print(f"Merge failed: {e}")
```

## Usage Examples

### Multi-Developer Pipeline Construction

```python
from cursus.api.dag.workspace_dag import WorkspaceAwareDAG

# Create workspace DAG
dag = WorkspaceAwareDAG(workspace_root="/workspace/ml_pipeline")

# Alice adds data processing steps
dag.add_workspace_step(
    step_name="data_loading",
    developer_id="alice",
    step_type="DataLoadingStep",
    config_data={
        "source": "s3://ml-data/raw",
        "format": "parquet"
    }
)

dag.add_workspace_step(
    step_name="data_preprocessing",
    developer_id="alice", 
    step_type="PreprocessingStep",
    config_data={
        "scaling": "standard",
        "encoding": "one_hot"
    },
    dependencies=["data_loading"]
)

# Bob adds training steps
dag.add_workspace_step(
    step_name="model_training",
    developer_id="bob",
    step_type="TrainingStep",
    config_data={
        "algorithm": "xgboost",
        "max_depth": 6,
        "learning_rate": 0.1
    },
    dependencies=["data_preprocessing"]  # Cross-workspace dependency
)

# Charlie adds evaluation steps
dag.add_workspace_step(
    step_name="model_evaluation",
    developer_id="charlie",
    step_type="EvaluationStep", 
    config_data={
        "metrics": ["accuracy", "precision", "recall"],
        "test_size": 0.2
    },
    dependencies=["model_training"]  # Cross-workspace dependency
)

# Validate the multi-developer pipeline
validation = dag.validate_workspace_dependencies()
print(f"Pipeline valid: {validation['valid']}")
print(f"Cross-workspace dependencies: {len(validation['cross_workspace_dependencies'])}")
```

### Workspace Analysis and Optimization

```python
# Get comprehensive workspace summary
summary = dag.get_workspace_summary()

print("Workspace Summary:")
print(f"Total steps: {summary['total_steps']}")
print(f"Developers: {len(summary['developers'])}")

# Analyze each developer's contribution
for dev_id, stats in summary['developer_stats'].items():
    print(f"\nDeveloper {dev_id}:")
    print(f"  Steps: {stats['step_count']}")
    print(f"  Step types: {stats['step_types']}")

# Perform complexity analysis
complexity = dag.analyze_workspace_complexity()

if complexity['recommendations']:
    print("\nOptimization Recommendations:")
    for rec in complexity['recommendations']:
        print(f"  - {rec}")
```

### Cross-Workspace Dependency Management

```python
# Validate cross-workspace dependencies
validation = dag.validate_workspace_dependencies()

print("Cross-Workspace Dependencies:")
for dep in validation['cross_workspace_dependencies']:
    print(f"  {dep['dependent_step']} ({dep['dependent_developer']}) depends on")
    print(f"  {dep['dependency_step']} ({dep['dependency_developer']})")

# Check dependency statistics
stats = validation['dependency_stats']
cross_ratio = stats['cross_workspace_count'] / stats['total_dependencies']
print(f"\nCross-workspace dependency ratio: {cross_ratio:.2%}")

if cross_ratio > 0.5:
    print("Warning: High cross-workspace coupling detected")
```

### Pipeline Configuration Generation

```python
# Generate workspace pipeline configuration
config = dag.to_workspace_pipeline_config("multi_dev_ml_pipeline")

print("Generated Pipeline Configuration:")
print(f"Pipeline: {config['pipeline_name']}")
print(f"Workspace: {config['workspace_root']}")

# Export for deployment
import json
with open("pipeline_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Pipeline configuration exported to pipeline_config.json")
```

### Workspace Collaboration Workflows

```python
# Clone workspace for experimentation
experimental_dag = dag.clone()

# Add experimental steps
experimental_dag.add_workspace_step(
    step_name="experimental_feature",
    developer_id="alice",
    step_type="ExperimentalStep",
    config_data={"experiment_type": "new_algorithm"}
)

# Merge successful experiments back
try:
    dag.merge_workspace_dag(experimental_dag)
    print("Experimental changes merged successfully")
except ValueError as e:
    print(f"Merge conflict: {e}")
```

## Integration Points

### Base DAG Compatibility
Maintains full compatibility with base PipelineDAG methods and functionality.

### Workspace Management System
Integrates with workspace management tools for developer environment coordination.

### Configuration System
Supports workspace-specific configuration management and deployment.

### Validation Framework
Provides comprehensive validation for workspace isolation and cross-workspace dependencies.

## Performance Considerations

### Workspace Scaling
- Efficient indexing for multi-developer environments
- Optimized cross-workspace dependency tracking
- Memory-efficient storage of workspace metadata

### Complexity Management
- Automated complexity analysis and recommendations
- Performance monitoring for large workspace DAGs
- Optimization suggestions for cross-workspace coupling

## Related Documentation

- [Base Pipeline DAG](base_dag.md) - Foundation DAG implementation
- [Enhanced Pipeline DAG](enhanced_dag.md) - Advanced DAG with port-based dependencies
- [Edge Types](edge_types.md) - Edge type system for dependency representation
- [Workspace Management](../../workspace/README.md) - Workspace management system
- [Developer Guide](../../0_developer_guide/README.md) - Multi-developer workflow guidelines
