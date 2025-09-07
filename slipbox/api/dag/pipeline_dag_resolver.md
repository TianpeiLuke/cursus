---
tags:
  - code
  - api
  - dag
  - resolver
  - execution-planning
keywords:
  - PipelineDAGResolver
  - PipelineExecutionPlan
  - execution planning
  - configuration resolution
  - topological ordering
  - data flow mapping
topics:
  - pipeline execution planning
  - DAG resolution
  - configuration management
  - execution ordering
language: python
date of note: 2024-12-07
---

# Pipeline DAG Resolver

Pipeline DAG resolver for execution planning, providing topological ordering, step configuration resolution, and data flow mapping for pipeline construction.

## Overview

The Pipeline DAG Resolver transforms pipeline DAGs into executable plans with comprehensive execution ordering, step configuration resolution, and data flow mapping. It integrates with the configuration resolution system for intelligent step config matching and provides contract-based data flow analysis for robust pipeline construction.

The module supports dynamic step contract discovery through registry integration, semantic output matching for dependency resolution, and comprehensive DAG integrity validation. It maintains compatibility with both simple DAGs and configuration-enhanced pipelines.

## Classes and Methods

### Classes
- [`PipelineExecutionPlan`](#pipelineexecutionplan) - Execution plan with topological ordering and configurations
- [`PipelineDAGResolver`](#pipelinedagresolver) - Resolves pipeline DAG into executable plan

## API Reference

### PipelineExecutionPlan

_class_ cursus.api.dag.pipeline_dag_resolver.PipelineExecutionPlan(_execution_order_, _step_configs_, _dependencies_, _data_flow_map_)

Execution plan for pipeline with topological ordering and step configurations.

**Parameters:**
- **execution_order** (_List[str]_) – List of step names in execution order
- **step_configs** (_Dict[str, dict]_) – Dictionary mapping step names to configuration dictionaries
- **dependencies** (_Dict[str, List[str]]_) – Dictionary mapping step names to their dependencies
- **data_flow_map** (_Dict[str, Dict[str, str]]_) – Dictionary mapping step names to their input data flow

```python
from cursus.api.dag.pipeline_dag_resolver import PipelineExecutionPlan

# Create execution plan
plan = PipelineExecutionPlan(
    execution_order=["data_loading", "preprocessing", "training"],
    step_configs={
        "data_loading": {"input_path": "/data/raw"},
        "preprocessing": {"output_path": "/data/processed"},
        "training": {"model_type": "xgboost"}
    },
    dependencies={
        "preprocessing": ["data_loading"],
        "training": ["preprocessing"]
    },
    data_flow_map={
        "preprocessing": {"input_data": "data_loading:output"},
        "training": {"training_data": "preprocessing:output"}
    }
)
```

### PipelineDAGResolver

_class_ cursus.api.dag.pipeline_dag_resolver.PipelineDAGResolver(_dag_, _config_path=None_, _available_configs=None_, _metadata=None_)

Resolves pipeline DAG into executable plan with optional step configuration resolution.

**Parameters:**
- **dag** (_PipelineDAG_) – PipelineDAG instance defining pipeline structure
- **config_path** (_Optional[str]_) – Path to configuration file for step config resolution
- **available_configs** (_Optional[Dict[str, BasePipelineConfig]]_) – Pre-loaded configuration instances
- **metadata** (_Optional[Dict[str, Any]]_) – Configuration metadata for enhanced resolution

```python
from cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG
dag = PipelineDAG()
dag.add_edge("preprocessing", "training")
dag.add_edge("training", "evaluation")

# Create resolver without configuration
resolver = PipelineDAGResolver(dag)

# Create resolver with configuration file
resolver_with_config = PipelineDAGResolver(
    dag=dag,
    config_path="pipeline_config.json"
)

# Create resolver with pre-loaded configurations
from cursus.core.base.config_base import BasePipelineConfig

configs = {
    "training_config": BasePipelineConfig(),
    "evaluation_config": BasePipelineConfig()
}

resolver_with_configs = PipelineDAGResolver(
    dag=dag,
    available_configs=configs,
    metadata={"pipeline_type": "ml_training"}
)
```

#### create_execution_plan

create_execution_plan()

Create topologically sorted execution plan with optional step configuration resolution.

**Returns:**
- **PipelineExecutionPlan** – Complete execution plan with ordering and configurations

**Raises:**
- **ValueError** – If pipeline contains cycles

```python
# Create execution plan
try:
    plan = resolver.create_execution_plan()
    
    print(f"Execution order: {plan.execution_order}")
    print(f"Step configurations: {len(plan.step_configs)}")
    print(f"Dependencies: {plan.dependencies}")
    
    # Access specific step configuration
    if "training" in plan.step_configs:
        training_config = plan.step_configs["training"]
        print(f"Training config: {training_config}")
        
except ValueError as e:
    print(f"Cannot create execution plan: {e}")
```

#### get_step_dependencies

get_step_dependencies(_step_name_)

Get immediate dependencies for a specific step.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **List[str]** – List of immediate dependency step names

```python
# Get dependencies for specific step
training_deps = resolver.get_step_dependencies("training")
print(f"Training depends on: {training_deps}")

# Check if step has no dependencies
data_loading_deps = resolver.get_step_dependencies("data_loading")
if not data_loading_deps:
    print("Data loading is a source step")
```

#### get_dependent_steps

get_dependent_steps(_step_name_)

Get steps that depend on the given step.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **List[str]** – List of dependent step names

```python
# Get steps that depend on preprocessing
dependent_steps = resolver.get_dependent_steps("preprocessing")
print(f"Steps depending on preprocessing: {dependent_steps}")

# Find sink steps (no dependents)
evaluation_dependents = resolver.get_dependent_steps("evaluation")
if not evaluation_dependents:
    print("Evaluation is a sink step")
```

#### validate_dag_integrity

validate_dag_integrity()

Validate DAG integrity and return issues if found.

**Returns:**
- **Dict[str, List[str]]** – Dictionary mapping issue types to lists of issue descriptions

```python
# Validate DAG integrity
issues = resolver.validate_dag_integrity()

if not issues:
    print("DAG integrity validation passed")
else:
    print("DAG integrity issues found:")
    
    if "cycles" in issues:
        print("Cycles detected:")
        for cycle in issues["cycles"]:
            print(f"  - {cycle}")
    
    if "dangling_dependencies" in issues:
        print("Dangling dependencies:")
        for issue in issues["dangling_dependencies"]:
            print(f"  - {issue}")
    
    if "isolated_nodes" in issues:
        print("Isolated nodes:")
        for issue in issues["isolated_nodes"]:
            print(f"  - {issue}")
```

#### get_config_resolution_preview

get_config_resolution_preview()

Get a preview of how DAG nodes would be resolved to configurations.

**Returns:**
- **Optional[Dict[str, Any]]** – Preview information if config resolver available, None otherwise

```python
# Get configuration resolution preview
preview = resolver.get_config_resolution_preview()

if preview:
    print("Configuration Resolution Preview:")
    print(f"Resolvable steps: {preview.get('resolvable_steps', [])}")
    print(f"Unresolvable steps: {preview.get('unresolvable_steps', [])}")
    print(f"Resolution strategy: {preview.get('strategy', 'unknown')}")
else:
    print("No configuration resolver available")
```

## Data Flow Analysis

### Contract-Based Data Flow Mapping

The resolver provides sophisticated data flow analysis using step contracts:

```python
# The resolver automatically discovers step contracts and maps data flow
plan = resolver.create_execution_plan()

# Access data flow mapping
for step_name, inputs in plan.data_flow_map.items():
    print(f"Step '{step_name}' inputs:")
    for input_name, source in inputs.items():
        print(f"  {input_name} <- {source}")
```

### Semantic Output Matching

The resolver uses intelligent matching strategies for connecting step outputs to inputs:

- **Direct Channel Matching** – Exact input/output channel name matching
- **Path-Based Compatibility** – SageMaker path convention matching
- **Semantic Matching** – Common pattern recognition (e.g., model_path, data_path)
- **Fallback Matching** – Default to first available output when needed

## Usage Examples

### Basic Pipeline Resolution

```python
from cursus.api.dag.base_dag import PipelineDAG
from cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver

# Create pipeline DAG
dag = PipelineDAG()
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "feature_engineering")
dag.add_edge("feature_engineering", "training")
dag.add_edge("feature_engineering", "validation")
dag.add_edge("training", "evaluation")
dag.add_edge("validation", "evaluation")

# Create resolver
resolver = PipelineDAGResolver(dag)

# Validate DAG integrity
issues = resolver.validate_dag_integrity()
if issues:
    print("DAG has issues - cannot proceed")
    for issue_type, issue_list in issues.items():
        print(f"{issue_type}: {issue_list}")
else:
    # Create execution plan
    plan = resolver.create_execution_plan()
    
    print("Pipeline Execution Plan:")
    print(f"Execution order: {plan.execution_order}")
    
    # Analyze dependencies
    for step in plan.execution_order:
        deps = plan.dependencies[step]
        if deps:
            print(f"{step} depends on: {deps}")
        else:
            print(f"{step} is a source step")
```

### Configuration-Enhanced Resolution

```python
# Create resolver with configuration support
resolver_with_config = PipelineDAGResolver(
    dag=dag,
    config_path="ml_pipeline_config.json",
    metadata={
        "pipeline_type": "ml_training",
        "framework": "xgboost"
    }
)

# Preview configuration resolution
preview = resolver_with_config.get_config_resolution_preview()
if preview:
    print("Configuration Preview:")
    for key, value in preview.items():
        print(f"  {key}: {value}")

# Create execution plan with configurations
plan = resolver_with_config.create_execution_plan()

# Access resolved configurations
for step_name, config in plan.step_configs.items():
    if config:  # Non-empty configuration
        print(f"Step '{step_name}' configuration:")
        for param, value in config.items():
            print(f"  {param}: {value}")
```

### Data Flow Analysis

```python
# Analyze data flow between steps
plan = resolver.create_execution_plan()

print("Data Flow Analysis:")
for step_name, inputs in plan.data_flow_map.items():
    if inputs:
        print(f"\nStep '{step_name}' receives:")
        for input_channel, source in inputs.items():
            source_step, source_output = source.split(":")
            print(f"  {input_channel} from {source_step}.{source_output}")
    else:
        print(f"\nStep '{step_name}' has no inputs (source step)")
```

### Dependency Analysis

```python
# Analyze step dependencies
for step in dag.nodes:
    immediate_deps = resolver.get_step_dependencies(step)
    dependent_steps = resolver.get_dependent_steps(step)
    
    print(f"\nStep '{step}':")
    print(f"  Depends on: {immediate_deps if immediate_deps else 'None (source step)'}")
    print(f"  Depended on by: {dependent_steps if dependent_steps else 'None (sink step)'}")
```

### Error Handling and Validation

```python
# Comprehensive validation and error handling
try:
    # Validate DAG structure
    issues = resolver.validate_dag_integrity()
    
    if issues:
        print("DAG Validation Issues:")
        for issue_type, issue_list in issues.items():
            print(f"  {issue_type}:")
            for issue in issue_list:
                print(f"    - {issue}")
        
        # Handle specific issue types
        if "cycles" in issues:
            raise ValueError("Cannot resolve DAG with cycles")
    
    # Create execution plan
    plan = resolver.create_execution_plan()
    
    # Validate execution plan
    if not plan.execution_order:
        raise ValueError("Empty execution plan generated")
    
    if len(plan.execution_order) != len(dag.nodes):
        raise ValueError("Execution plan missing steps")
    
    print("Pipeline resolution successful!")
    
except ValueError as e:
    print(f"Pipeline resolution failed: {e}")
except Exception as e:
    print(f"Unexpected error during resolution: {e}")
```

## Integration Points

### Configuration Resolution System
Integrates with StepConfigResolver for intelligent step configuration matching and resolution.

### Step Contract Discovery
Uses registry helper functions for dynamic step contract discovery and validation.

### NetworkX Integration
Leverages NetworkX for advanced graph algorithms and cycle detection.

### SageMaker Path Conventions
Supports SageMaker path compatibility checking for data flow validation.

## Performance Considerations

### Graph Algorithm Efficiency
- Uses NetworkX for optimized topological sorting
- Efficient cycle detection with early termination
- Cached graph construction for repeated operations

### Configuration Resolution Optimization
- Lazy loading of configuration files
- Cached configuration resolution results
- Efficient metadata-based matching

### Memory Management
- Minimal memory footprint for large DAGs
- Efficient data structure usage
- Proper cleanup of temporary objects

## Related Documentation

- [Base Pipeline DAG](base_dag.md) - Foundation DAG implementation used by resolver
- [Enhanced Pipeline DAG](enhanced_dag.md) - Advanced DAG with enhanced resolution capabilities
- [Edge Types](edge_types.md) - Edge type system for dependency representation
- [Configuration Resolution](../../core/compiler/config_resolver.md) - Step configuration resolution system
- [Step Contracts](../../core/base/contract_base.md) - Script contract system for data flow analysis
- [Registry System](../../registry/README.md) - Registry integration for step discovery
