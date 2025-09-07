---
tags:
  - code
  - api
  - dag
  - enhanced
  - port-based
keywords:
  - EnhancedPipelineDAG
  - port-based dependencies
  - intelligent resolution
  - typed edges
  - step specifications
  - property references
  - dependency resolver
topics:
  - enhanced pipeline DAG
  - port-based dependency management
  - intelligent dependency resolution
  - typed pipeline edges
language: python
date of note: 2024-12-07
---

# Enhanced Pipeline DAG

Enhanced version of PipelineDAG with port-based dependency management, intelligent dependency resolution, typed edges, and declarative step specifications.

## Overview

The Enhanced Pipeline DAG extends the base PipelineDAG with sophisticated dependency management capabilities. It provides port-based dependency resolution through step specifications, intelligent auto-resolution with confidence scoring, and comprehensive validation and error reporting.

The module integrates with the dependency resolution system for semantic matching, supports property reference management for SageMaker integration, and provides enhanced validation including port compatibility checking. It maintains backward compatibility with the base PipelineDAG while adding advanced features for complex pipeline construction.

## Classes and Methods

### Classes
- [`EnhancedPipelineDAG`](#enhancedpipelinedag) - Enhanced DAG with port-based dependency management

## API Reference

### EnhancedPipelineDAG

_class_ cursus.api.dag.enhanced_dag.EnhancedPipelineDAG(_nodes=None_, _edges=None_)

Enhanced version of PipelineDAG with port-based dependency management and intelligent dependency resolution.

**Parameters:**
- **nodes** (_Optional[List[str]]_) – Optional list of initial node names
- **edges** (_Optional[List[tuple]]_) – Optional list of initial edges for compatibility

```python
from cursus.api.dag.enhanced_dag import EnhancedPipelineDAG

# Create enhanced DAG
dag = EnhancedPipelineDAG()

# Create with initial structure
nodes = ["preprocessing", "training", "evaluation"]
edges = [("preprocessing", "training"), ("training", "evaluation")]
dag = EnhancedPipelineDAG(nodes=nodes, edges=edges)
```

#### register_step_specification

register_step_specification(_step_name_, _specification_)

Register a step specification defining its input/output ports and dependencies.

**Parameters:**
- **step_name** (_str_) – Name of the step
- **specification** (_StepSpecification_) – Step specification with dependencies and outputs

**Raises:**
- **ValueError** – If specification is not a StepSpecification instance

```python
from cursus.core.deps import StepSpecification, DependencySpecification, OutputSpecification
from cursus.core.deps import DependencyType, OutputType

# Create step specification
spec = StepSpecification(
    step_type="TrainingStep",
    dependencies={
        "training_data": DependencySpecification(
            logical_name="training_data",
            dependency_type=DependencyType.DATASET,
            required=True
        )
    },
    outputs={
        "model": OutputSpecification(
            logical_name="model",
            output_type=OutputType.MODEL,
            description="Trained model artifact"
        )
    }
)

# Register specification
dag.register_step_specification("training", spec)
```

#### auto_resolve_dependencies

auto_resolve_dependencies(_confidence_threshold=0.6_)

Automatically resolve dependencies based on port compatibility and semantic matching.

**Parameters:**
- **confidence_threshold** (_float_) – Minimum confidence threshold for auto-resolution (default=0.6)

**Returns:**
- **List[DependencyEdge]** – List of resolved dependency edges

**Raises:**
- **DependencyResolutionError** – If dependency resolution fails

```python
# Auto-resolve dependencies with default threshold
resolved_edges = dag.auto_resolve_dependencies()
print(f"Resolved {len(resolved_edges)} dependencies")

# Use higher confidence threshold
high_conf_edges = dag.auto_resolve_dependencies(confidence_threshold=0.8)
print(f"High confidence edges: {len(high_conf_edges)}")
```

#### add_manual_dependency

add_manual_dependency(_source_step_, _source_output_, _target_step_, _target_input_)

Manually add a dependency edge between steps with full confidence.

**Parameters:**
- **source_step** (_str_) – Name of the source step
- **source_output** (_str_) – Logical name of the source output
- **target_step** (_str_) – Name of the target step
- **target_input** (_str_) – Logical name of the target input

**Returns:**
- **DependencyEdge** – Created dependency edge

**Raises:**
- **ValueError** – If steps are not registered or ports don't exist

```python
# Add manual dependency
edge = dag.add_manual_dependency(
    source_step="preprocessing",
    source_output="processed_data",
    target_step="training",
    target_input="training_data"
)

print(f"Added manual dependency: {edge}")
```

#### get_step_dependencies

get_step_dependencies(_step_name_)

Get resolved dependencies for a step as property references.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **Dict[str, PropertyReference]** – Dictionary mapping dependency names to property references

```python
# Get dependencies for training step
dependencies = dag.get_step_dependencies("training")

for dep_name, prop_ref in dependencies.items():
    print(f"Dependency '{dep_name}': {prop_ref.step_name}.{prop_ref.output_spec.logical_name}")
```

#### get_step_inputs_for_sagemaker

get_step_inputs_for_sagemaker(_step_name_)

Get step inputs formatted for SageMaker pipeline construction.

**Parameters:**
- **step_name** (_str_) – Name of the step

**Returns:**
- **Dict[str, Any]** – Dictionary of inputs formatted for SageMaker

```python
# Get SageMaker-formatted inputs
sagemaker_inputs = dag.get_step_inputs_for_sagemaker("training")

for input_name, sagemaker_ref in sagemaker_inputs.items():
    print(f"Input '{input_name}': {sagemaker_ref}")
    # Example output: {'Get': 'Steps.preprocessing.processed_data'}
```

#### validate_enhanced_dag

validate_enhanced_dag()

Enhanced validation including port compatibility and dependency resolution.

**Returns:**
- **List[str]** – List of validation errors

```python
# Validate enhanced DAG
errors = dag.validate_enhanced_dag()

if errors:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("DAG validation passed")
```

#### get_execution_order

get_execution_order()

Get execution order using inherited topological sort from base DAG.

**Returns:**
- **List[str]** – List of step names in execution order

```python
# Get execution order
execution_order = dag.get_execution_order()
print(f"Execution order: {execution_order}")
```

#### get_dag_statistics

get_dag_statistics()

Get comprehensive statistics about the enhanced DAG.

**Returns:**
- **Dict[str, Any]** – Dictionary containing DAG statistics

```python
# Get DAG statistics
stats = dag.get_dag_statistics()

print(f"Nodes: {stats['nodes']}")
print(f"Step specifications: {stats['step_specifications']}")
print(f"Dependency edges: {stats['dependency_edges']}")
print(f"Resolution rate: {stats['resolution_rate']:.2%}")
```

#### get_resolution_report

get_resolution_report()

Get detailed resolution report for debugging dependency resolution.

**Returns:**
- **Dict[str, Any]** – Detailed resolution report

```python
# Get resolution report for debugging
report = dag.get_resolution_report()

print("Resolution Report:")
print(f"Available steps: {report.get('available_steps', [])}")
print(f"Resolution strategy: {report.get('strategy', 'unknown')}")
```

#### clear_resolution_cache

clear_resolution_cache()

Clear dependency resolution cache and reset resolution state.

```python
# Clear resolution cache
dag.clear_resolution_cache()
print("Resolution cache cleared")
```

#### export_for_visualization

export_for_visualization()

Export DAG data for visualization tools with comprehensive node and edge information.

**Returns:**
- **Dict[str, Any]** – Dictionary containing visualization data

```python
# Export for visualization
viz_data = dag.export_for_visualization()

print(f"Nodes for visualization: {len(viz_data['nodes'])}")
print(f"Edges for visualization: {len(viz_data['edges'])}")

# Access node information
for node in viz_data['nodes']:
    print(f"Node {node['id']}: type={node['type']}, deps={node['dependencies']}")

# Access edge information
for edge in viz_data['edges']:
    print(f"Edge: {edge['source']}.{edge['source_output']} -> {edge['target']}.{edge['target_input']}")
    print(f"  Confidence: {edge['confidence']}, Auto-resolved: {edge['auto_resolved']}")
```

## Enhanced Features

### Port-Based Dependency Management

The Enhanced DAG provides sophisticated port-based dependency management:

- **Input/Output Ports** – Steps define typed input and output ports through specifications
- **Semantic Matching** – Intelligent matching based on port types and names
- **Confidence Scoring** – Auto-resolved dependencies include confidence scores
- **Manual Override** – Support for manual dependency specification

### Intelligent Dependency Resolution

```python
# Register multiple step specifications
preprocessing_spec = StepSpecification(
    step_type="PreprocessingStep",
    outputs={
        "processed_data": OutputSpecification(
            logical_name="processed_data",
            output_type=OutputType.DATASET
        )
    }
)

training_spec = StepSpecification(
    step_type="TrainingStep", 
    dependencies={
        "training_data": DependencySpecification(
            logical_name="training_data",
            dependency_type=DependencyType.DATASET,
            required=True
        )
    },
    outputs={
        "model": OutputSpecification(
            logical_name="model",
            output_type=OutputType.MODEL
        )
    }
)

# Register specifications
dag.register_step_specification("preprocessing", preprocessing_spec)
dag.register_step_specification("training", training_spec)

# Auto-resolve dependencies
resolved = dag.auto_resolve_dependencies()
# Automatically connects preprocessing.processed_data -> training.training_data
```

### Enhanced Validation

```python
# Comprehensive validation
errors = dag.validate_enhanced_dag()

# Validation checks include:
# - Base DAG structure (cycles, connectivity)
# - Step specification validity
# - Dependency edge validation
# - Unresolved required dependencies
# - Port type compatibility
```

## Usage Examples

### Complete Pipeline Construction

```python
from cursus.api.dag.enhanced_dag import EnhancedPipelineDAG
from cursus.core.deps import StepSpecification, DependencySpecification, OutputSpecification
from cursus.core.deps import DependencyType, OutputType

# Create enhanced DAG
dag = EnhancedPipelineDAG()

# Define data loading step
data_spec = StepSpecification(
    step_type="DataLoadingStep",
    outputs={
        "raw_data": OutputSpecification(
            logical_name="raw_data",
            output_type=OutputType.DATASET,
            description="Raw input data"
        )
    }
)

# Define preprocessing step
prep_spec = StepSpecification(
    step_type="PreprocessingStep",
    dependencies={
        "input_data": DependencySpecification(
            logical_name="input_data",
            dependency_type=DependencyType.DATASET,
            required=True
        )
    },
    outputs={
        "processed_data": OutputSpecification(
            logical_name="processed_data", 
            output_type=OutputType.DATASET,
            description="Preprocessed training data"
        )
    }
)

# Define training step
train_spec = StepSpecification(
    step_type="TrainingStep",
    dependencies={
        "training_data": DependencySpecification(
            logical_name="training_data",
            dependency_type=DependencyType.DATASET,
            required=True
        )
    },
    outputs={
        "model": OutputSpecification(
            logical_name="model",
            output_type=OutputType.MODEL,
            description="Trained ML model"
        )
    }
)

# Register all specifications
dag.register_step_specification("data_loading", data_spec)
dag.register_step_specification("preprocessing", prep_spec)
dag.register_step_specification("training", train_spec)

# Auto-resolve dependencies
resolved_edges = dag.auto_resolve_dependencies()
print(f"Auto-resolved {len(resolved_edges)} dependencies")

# Validate the complete DAG
errors = dag.validate_enhanced_dag()
if not errors:
    print("Pipeline construction successful!")
    
    # Get execution order
    order = dag.get_execution_order()
    print(f"Execution order: {order}")
```

### Manual Dependency Management

```python
# Add manual dependencies for specific requirements
manual_edge = dag.add_manual_dependency(
    source_step="preprocessing",
    source_output="processed_data",
    target_step="training", 
    target_input="training_data"
)

print(f"Manual dependency: {manual_edge}")
print(f"Confidence: {manual_edge.confidence}")  # 1.0 for manual edges
```

### SageMaker Integration

```python
# Get SageMaker-formatted inputs for pipeline construction
for step_name in dag.step_specifications:
    inputs = dag.get_step_inputs_for_sagemaker(step_name)
    if inputs:
        print(f"Step '{step_name}' inputs:")
        for input_name, sagemaker_ref in inputs.items():
            print(f"  {input_name}: {sagemaker_ref}")
```

### Debugging and Analysis

```python
# Get comprehensive statistics
stats = dag.get_dag_statistics()
print("DAG Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Get resolution report for debugging
report = dag.get_resolution_report()
print("\nResolution Report:")
print(f"Strategy used: {report.get('strategy', 'unknown')}")
print(f"Matches found: {report.get('total_matches', 0)}")

# Export for visualization
viz_data = dag.export_for_visualization()
# Use viz_data with visualization tools like Graphviz, D3.js, etc.
```

## Integration Points

### Dependency Resolution System
Integrates with UnifiedDependencyResolver for intelligent dependency matching and resolution.

### Property Reference System
Generates PropertyReference objects for SageMaker pipeline construction and step input management.

### Step Specification Registry
Uses SpecificationRegistry for step specification management and validation.

### Base DAG Compatibility
Maintains full compatibility with base PipelineDAG for existing pipeline code.

## Performance Considerations

### Resolution Caching
- Dependency resolution results are cached for performance
- Cache can be cleared when specifications change
- Resolution state tracking prevents unnecessary re-resolution

### Memory Management
- Efficient storage of step specifications and dependency edges
- Indexed edge collection for fast lookups
- Lazy resolution only when needed

## Related Documentation

- [Base Pipeline DAG](base_dag.md) - Foundation DAG implementation
- [Edge Types](edge_types.md) - Typed edge system used by Enhanced DAG
- [Pipeline DAG Resolver](pipeline_dag_resolver.md) - Execution planning integration
- [Dependency System](../../core/deps/README.md) - Dependency resolution framework
- [Step Specifications](../../core/deps/base_specifications.md) - Step specification system
- [Property Reference](../../core/deps/property_reference.md) - SageMaker property reference system
