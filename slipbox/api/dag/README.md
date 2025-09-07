---
tags:
  - entry_point
  - code
  - api
  - dag
  - pipeline
  - graph
  - dependency
keywords:
  - PipelineDAG
  - EnhancedPipelineDAG
  - WorkspaceAwareDAG
  - EdgeType
  - DependencyEdge
  - PipelineDAGResolver
  - topological sort
  - dependency resolution
topics:
  - pipeline construction
  - graph algorithms
  - dependency management
  - workspace collaboration
language: python
date of note: 2024-12-07
---

# Pipeline DAG

Core directed acyclic graph structures for pipeline construction and dependency management.

## Overview

The Pipeline DAG module provides comprehensive graph structure representations for pipeline construction, enabling automatic dependency resolution, proper step ordering, and collaborative pipeline development. The module implements a hierarchical approach with three main DAG types: base DAG for core graph operations, enhanced DAG for intelligent dependency resolution, and workspace-aware DAG for multi-developer collaboration.

The module supports various edge types with confidence scoring, port-based dependency management, and cross-workspace dependency validation. It integrates seamlessly with the pipeline builder system to provide robust foundation for complex pipeline construction workflows.

## Classes and Methods

### Classes
- [`PipelineDAG`](#pipelinedag) - Base directed acyclic graph with core graph operations
- [`EdgeType`](#edgetype) - Enumeration of edge types with confidence scoring
- [`DependencyEdge`](#dependencyedge) - Base class for typed pipeline dependencies
- [`ConditionalEdge`](#conditionaledge) - Conditional dependency with boolean evaluation
- [`ParallelEdge`](#paralleledge) - Parallel execution dependency
- [`EdgeCollection`](#edgecollection) - Collection management for pipeline edges
- [`EnhancedPipelineDAG`](#enhancedpipelinedag) - Enhanced DAG with port-based dependency management
- [`PipelineDAGResolver`](#pipelinedagresolver) - DAG resolution with execution planning
- [`PipelineExecutionPlan`](#pipelineexecutionplan) - Execution plan with configuration resolution
- [`WorkspaceAwareDAG`](#workspaceawaredag) - Multi-workspace DAG with collaboration support

### Functions
- [`create_execution_plan`](#create_execution_plan) - Create execution plan from DAG structure
- [`validate_cross_workspace_dependencies`](#validate_cross_workspace_dependencies) - Validate dependencies across workspaces

## API Reference

### PipelineDAG

_class_ cursus.api.dag.base_dag.PipelineDAG()

Lightweight, efficient directed acyclic graph for pipeline construction with fast node and edge operations, topological sorting, and cycle detection.

```python
# Create pipeline DAG with dependencies
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing") 
dag.add_edge("data_loading", "preprocessing")
build_order = dag.topological_sort()
```

#### add_node

add_node(_node_id_)

Add a node to the DAG.

**Parameters:**
- **node_id** (_str_) – Unique identifier for the node.

```python
dag.add_node("training_step")
```

#### add_edge

add_edge(_from_node_, _to_node_)

Add a directed edge between two nodes.

**Parameters:**
- **from_node** (_str_) – Source node identifier.
- **to_node** (_str_) – Target node identifier.

```python
dag.add_edge("preprocessing", "training")
```

#### topological_sort

topological_sort()

Perform topological sort using Kahn's algorithm.

**Returns:**
- **List[str]** – Topologically sorted list of node identifiers.

```python
execution_order = dag.topological_sort()
```

### EdgeType

_class_ cursus.api.dag.edge_types.EdgeType(_value_)

Enumeration of edge types with associated confidence scores for dependency resolution.

**Parameters:**
- **value** (_str_) – Edge type identifier.

```python
# Access edge types with confidence scores
edge_type = EdgeType.DEPENDENCY
confidence = edge_type.confidence_score
```

### DependencyEdge

_class_ cursus.api.dag.edge_types.DependencyEdge(_from_node_, _to_node_, _edge_type=EdgeType.DEPENDENCY_, _confidence=None_)

Base class for typed pipeline dependencies with confidence scoring and validation.

**Parameters:**
- **from_node** (_str_) – Source node identifier.
- **to_node** (_str_) – Target node identifier.
- **edge_type** (_EdgeType_) – Type of dependency edge.
- **confidence** (_Optional[float]_) – Confidence score for auto-resolved edges.

```python
# Create dependency edge with confidence
edge = DependencyEdge("step1", "step2", EdgeType.DEPENDENCY, confidence=0.9)
```

### ConditionalEdge

_class_ cursus.api.dag.edge_types.ConditionalEdge(_from_node_, _to_node_, _condition_, _confidence=None_)

Conditional dependency that activates based on boolean evaluation.

**Parameters:**
- **from_node** (_str_) – Source node identifier.
- **to_node** (_str_) – Target node identifier.
- **condition** (_str_) – Boolean condition expression.
- **confidence** (_Optional[float]_) – Confidence score for the condition.

```python
# Create conditional dependency
edge = ConditionalEdge("validation", "deployment", "accuracy > 0.95")
```

### ParallelEdge

_class_ cursus.api.dag.edge_types.ParallelEdge(_from_node_, _to_node_, _parallel_group_, _confidence=None_)

Parallel execution dependency for concurrent step execution.

**Parameters:**
- **from_node** (_str_) – Source node identifier.
- **to_node** (_str_) – Target node identifier.
- **parallel_group** (_str_) – Parallel execution group identifier.
- **confidence** (_Optional[float]_) – Confidence score for parallel execution.

```python
# Create parallel execution edge
edge = ParallelEdge("feature_eng", "model_train", "training_group")
```

### EdgeCollection

_class_ cursus.api.dag.edge_types.EdgeCollection()

Collection management for pipeline edges with statistics and validation.

```python
# Manage edge collections
collection = EdgeCollection()
collection.add_edge(dependency_edge)
stats = collection.get_statistics()
```

#### add_edge

add_edge(_edge_)

Add an edge to the collection.

**Parameters:**
- **edge** (_DependencyEdge_) – Edge to add to the collection.

```python
collection.add_edge(DependencyEdge("step1", "step2"))
```

#### get_statistics

get_statistics()

Get statistical summary of edges in the collection.

**Returns:**
- **Dict[str, Any]** – Statistics including edge counts and confidence scores.

```python
stats = collection.get_statistics()
```

### EnhancedPipelineDAG

_class_ cursus.api.dag.enhanced_dag.EnhancedPipelineDAG()

Enhanced DAG with port-based dependency management and intelligent resolution capabilities.

```python
# Create enhanced DAG with automatic resolution
dag = EnhancedPipelineDAG()
dag.add_step_with_spec("training", training_spec)
dag.resolve_dependencies()
```

#### add_step_with_spec

add_step_with_spec(_step_name_, _step_spec_)

Add a step with its specification for intelligent dependency resolution.

**Parameters:**
- **step_name** (_str_) – Name of the pipeline step.
- **step_spec** (_StepSpecification_) – Step specification with input/output ports.

```python
dag.add_step_with_spec("preprocessing", preprocessing_spec)
```

#### resolve_dependencies

resolve_dependencies()

Automatically resolve dependencies based on step specifications and port compatibility.

**Returns:**
- **List[DependencyEdge]** – List of resolved dependency edges with confidence scores.

```python
resolved_edges = dag.resolve_dependencies()
```

### PipelineDAGResolver

_class_ cursus.api.dag.pipeline_dag_resolver.PipelineDAGResolver(_dag_)

DAG resolution system with execution planning and configuration management.

**Parameters:**
- **dag** (_PipelineDAG_) – Pipeline DAG to resolve.

```python
# Create resolver with execution planning
resolver = PipelineDAGResolver(pipeline_dag)
execution_plan = resolver.create_execution_plan()
```

#### create_execution_plan

create_execution_plan(_config=None_)

Create execution plan with configuration resolution and dependency validation.

**Parameters:**
- **config** (_Optional[Dict[str, Any]]_) – Pipeline configuration parameters.

**Returns:**
- **PipelineExecutionPlan** – Execution plan with resolved configurations.

```python
plan = resolver.create_execution_plan({"batch_size": 32})
```

### PipelineExecutionPlan

_class_ cursus.api.dag.pipeline_dag_resolver.PipelineExecutionPlan(_execution_order_, _configurations_)

Execution plan with step ordering and resolved configurations.

**Parameters:**
- **execution_order** (_List[str]_) – Topologically sorted execution order.
- **configurations** (_Dict[str, Any]_) – Resolved step configurations.

```python
# Access execution plan details
plan = PipelineExecutionPlan(execution_order, step_configs)
next_step = plan.get_next_step()
```

#### get_next_step

get_next_step()

Get the next step in the execution sequence.

**Returns:**
- **Optional[str]** – Next step identifier or None if complete.

```python
next_step = plan.get_next_step()
```

### WorkspaceAwareDAG

_class_ cursus.api.dag.workspace_dag.WorkspaceAwareDAG(_workspace_id_)

Multi-workspace DAG supporting cross-workspace dependencies and collaborative pipeline development.

**Parameters:**
- **workspace_id** (_str_) – Unique workspace identifier.

```python
# Create workspace-aware DAG for collaboration
dag = WorkspaceAwareDAG("team_alpha")
dag.add_cross_workspace_dependency("shared_data", "team_beta")
```

#### add_cross_workspace_dependency

add_cross_workspace_dependency(_local_step_, _remote_workspace_, _remote_step=None_)

Add dependency on step from another workspace.

**Parameters:**
- **local_step** (_str_) – Local step that depends on remote step.
- **remote_workspace** (_str_) – Remote workspace identifier.
- **remote_step** (_Optional[str]_) – Remote step identifier (defaults to local_step).

```python
dag.add_cross_workspace_dependency("training", "data_team", "processed_data")
```

#### validate_cross_workspace_dependencies

validate_cross_workspace_dependencies()

Validate all cross-workspace dependencies for consistency and availability.

**Returns:**
- **Dict[str, bool]** – Validation results for each cross-workspace dependency.

```python
validation_results = dag.validate_cross_workspace_dependencies()
```

### create_execution_plan

create_execution_plan(_dag_, _config=None_)

Create execution plan from DAG structure with configuration resolution.

**Parameters:**
- **dag** (_PipelineDAG_) – Pipeline DAG to create execution plan from.
- **config** (_Optional[Dict[str, Any]]_) – Pipeline configuration parameters.

**Returns:**
- **PipelineExecutionPlan** – Execution plan with resolved configurations.

```python
plan = create_execution_plan(pipeline_dag, {"model_type": "xgboost"})
```

### validate_cross_workspace_dependencies

validate_cross_workspace_dependencies(_workspace_dag_)

Validate dependencies across workspaces for consistency and availability.

**Parameters:**
- **workspace_dag** (_WorkspaceAwareDAG_) – Workspace-aware DAG to validate.

**Returns:**
- **Dict[str, bool]** – Validation results for each dependency.

```python
results = validate_cross_workspace_dependencies(workspace_dag)
```

## Related Documentation

- [Base DAG](base_dag.md) - Core PipelineDAG implementation with graph algorithms
- [Edge Types](edge_types.md) - Edge type system with confidence scoring
- [Enhanced DAG](enhanced_dag.md) - Port-based dependency management
- [Pipeline DAG Resolver](pipeline_dag_resolver.md) - Execution planning and configuration resolution
- [Workspace DAG](workspace_dag.md) - Multi-workspace collaboration support
- [Pipeline Builder](../../core/assembler/README.md) - Pipeline assembly using DAG structures
- [Dependency Resolution](../../core/deps/README.md) - Dependency resolution system integration
