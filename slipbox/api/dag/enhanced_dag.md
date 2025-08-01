# Enhanced Pipeline DAG

## Overview

The Enhanced Pipeline DAG extends the base Pipeline DAG with port-based dependency management, intelligent dependency resolution, and declarative step specifications. It provides a comprehensive system for defining, validating, and resolving dependencies between pipeline steps at the level of individual input and output ports.

## Class Definition

```python
class EnhancedPipelineDAG(PipelineDAG):
    """
    Enhanced version of PipelineDAG with port-based dependency management.
    
    Extends the existing PipelineDAG from pipeline_builder to add:
    - Typed input/output ports via step specifications
    - Intelligent dependency resolution
    - Property reference management
    - Semantic matching capabilities
    - Enhanced validation and error reporting
    """
```

## Key Features

### Step Specifications

The Enhanced Pipeline DAG uses step specifications to define the input and output ports of each step:

```python
def register_step_specification(self, step_name: str, specification: StepSpecification):
    """
    Register a step specification defining its input/output ports.
    
    Args:
        step_name: Name of the step
        specification: Step specification with dependencies and outputs
    """
```

Each specification defines:
- Input dependencies (ports) with their types, requirements, and compatibility information
- Output ports with their types, property paths, and aliases
- Node type and step type information

### Automatic Dependency Resolution

One of the most powerful features of the Enhanced Pipeline DAG is automatic dependency resolution based on port compatibility:

```python
def auto_resolve_dependencies(self, confidence_threshold: float = 0.6) -> List[DependencyEdge]:
    """
    Automatically resolve dependencies based on port compatibility.
    
    Args:
        confidence_threshold: Minimum confidence threshold for auto-resolution
            
    Returns:
        List of resolved dependency edges
    """
```

This method:
1. Uses the registered step specifications to identify compatible inputs and outputs
2. Calculates compatibility scores based on type matching, name similarity, and semantic matching
3. Creates dependency edges for connections with confidence above the threshold
4. Updates both the enhanced edges and the base DAG structure

### Manual Dependency Definition

In addition to automatic resolution, the Enhanced Pipeline DAG allows for manual definition of dependencies:

```python
def add_manual_dependency(self, source_step: str, source_output: str,
                         target_step: str, target_input: str) -> DependencyEdge:
    """
    Manually add a dependency edge between steps.
    """
```

This method:
1. Validates that the specified steps and ports exist
2. Creates a dependency edge with full confidence (1.0)
3. Updates both the enhanced edges and the base DAG structure
4. Creates the appropriate property reference

### Property Reference Management

The Enhanced Pipeline DAG manages property references that bridge between definition-time specifications and runtime properties:

```python
def get_step_dependencies(self, step_name: str) -> Dict[str, PropertyReference]:
    """
    Get resolved dependencies for a step.
    """
    
def get_step_inputs_for_sagemaker(self, step_name: str) -> Dict[str, Any]:
    """
    Get step inputs formatted for SageMaker pipeline construction.
    """
```

These methods provide access to resolved dependencies in formats suitable for different use cases:
- Raw property references for intermediate processing
- SageMaker-formatted inputs for pipeline construction

### Enhanced Validation

The Enhanced Pipeline DAG includes comprehensive validation that goes beyond the base DAG validation:

```python
def validate_enhanced_dag(self) -> List[str]:
    """
    Enhanced validation including port compatibility and dependency resolution.
    
    Returns:
        List of validation errors
    """
```

This validation includes:
1. Structure validation (cycle detection via topological sorting)
2. Step specification validation
3. Edge validation
4. Required dependency resolution check
5. Port compatibility validation

### Comprehensive Reporting

The Enhanced Pipeline DAG provides detailed statistics and reports:

```python
def get_dag_statistics(self) -> Dict[str, Any]:
    """Get comprehensive statistics about the DAG."""
    
def get_resolution_report(self) -> Dict[str, Any]:
    """Get detailed resolution report for debugging."""
    
def export_for_visualization(self) -> Dict[str, Any]:
    """Export DAG data for visualization tools."""
```

These methods generate information useful for:
- Performance monitoring
- Debugging dependency resolution issues
- Visualizing the DAG structure
- Generating reports for users

## Usage Examples

### Creating an Enhanced Pipeline DAG

```python
# Create the enhanced DAG
dag = EnhancedPipelineDAG()

# Register step specifications
dag.register_step_specification("data_loading", data_loading_spec)
dag.register_step_specification("preprocessing", preprocessing_spec)
dag.register_step_specification("training", training_spec)
dag.register_step_specification("evaluation", evaluation_spec)
```

### Auto-Resolving Dependencies

```python
# Automatically resolve dependencies
resolved_edges = dag.auto_resolve_dependencies(confidence_threshold=0.7)

# Log the resolved edges
for edge in resolved_edges:
    print(f"Resolved: {edge.source_step}.{edge.source_output} -> "
          f"{edge.target_step}.{edge.target_input} "
          f"(confidence: {edge.confidence:.2f})")
```

### Adding Manual Dependencies

```python
# Manually define a dependency
edge = dag.add_manual_dependency(
    source_step="preprocessing", 
    source_output="validation_data", 
    target_step="evaluation", 
    target_input="evaluation_data"
)

print(f"Manually added: {edge}")
```

### Validating the Enhanced DAG

```python
# Validate the DAG
validation_errors = dag.validate_enhanced_dag()

if validation_errors:
    print("Validation errors:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("DAG validation successful")
```

### Getting Dependencies for SageMaker Pipeline

```python
# Get dependencies for a specific step
training_inputs = dag.get_step_inputs_for_sagemaker("training")

# Use these inputs when creating the SageMaker step
training_step = TrainingStep(
    name="TrainingStep",
    inputs=training_inputs,
    # other parameters...
)
```

### Generating Statistics and Reports

```python
# Get DAG statistics
stats = dag.get_dag_statistics()
print(f"Nodes: {stats['nodes']}")
print(f"Dependency edges: {stats['dependency_edges']}")
print(f"Resolution rate: {stats['resolution_rate']:.2f}")

# Get detailed resolution report
report = dag.get_resolution_report()
print(f"Total dependencies: {report['resolution_summary']['total_dependencies']}")
print(f"Resolved dependencies: {report['resolution_summary']['resolved_dependencies']}")
print(f"Steps with errors: {report['resolution_summary']['steps_with_errors']}")
```

## Internal Components

### Dependency Resolution System

The Enhanced Pipeline DAG integrates with the dependency resolution system:

```python
# In __init__
self.resolver = UnifiedDependencyResolver()
self.registry = self.resolver.registry

# In register_step_specification
self.resolver.register_specification(step_name, specification)

# In auto_resolve_dependencies
resolved_deps = self.resolver.resolve_all_dependencies(available_steps)
```

This integration provides:
- Registration of step specifications
- Resolution of dependencies based on compatibility scoring
- Generation of property references for resolved dependencies

### Edge Collection

The Enhanced Pipeline DAG uses an EdgeCollection to manage dependency edges:

```python
# In __init__
self.dependency_edges = EdgeCollection()

# In auto_resolve_dependencies
edge_id = self.dependency_edges.add_edge(edge)

# In get_step_dependencies
edges = self.dependency_edges.get_edges_to_step(step_name)
```

The EdgeCollection provides:
- Efficient storage and retrieval of edges
- Validation of edge properties
- Indexing for quick access by source or target step
- Statistics about edge properties

### Property Reference Management

The Enhanced Pipeline DAG manages property references for resolved dependencies:

```python
# In auto_resolve_dependencies
self.property_references = resolved_deps

# In add_manual_dependency
self.property_references[target_step][target_input] = PropertyReference(
    step_name=source_step,
    output_spec=source_output_spec
)
```

These property references:
- Bridge between definition-time specifications and runtime properties
- Convert to SageMaker property references during pipeline construction
- Support both automatic and manual dependency definition

## Integration with Other Components

### With Pipeline Template Base

The Enhanced Pipeline DAG integrates with the [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md) to provide automatic dependency resolution:

```python
# In PipelineTemplateBase._create_pipeline_dag
dag = EnhancedPipelineDAG()

# Add nodes and edges based on template configuration
for step_name, config in self.configs.items():
    dag.add_node(step_name)

# Define dependencies based on template structure
for src_step, dst_step in self.dependencies:
    dag.add_edge(src_step, dst_step)

# In PipelineTemplateBase._register_step_specifications
for step_name, builder in self.step_builders.items():
    dag.register_step_specification(step_name, builder.spec)

# In PipelineTemplateBase._resolve_dependencies
resolved_edges = dag.auto_resolve_dependencies(confidence_threshold=0.7)
```

### With Pipeline Assembler

The Enhanced Pipeline DAG integrates with the [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md) to provide resolved dependencies:

```python
# In PipelineAssembler._instantiate_step
dependencies = dag.get_step_dependencies(step_name)

for input_name, prop_ref in dependencies.items():
    inputs[input_name] = prop_ref.to_runtime_property(step_instances)
```

### With Edge Types

The Enhanced Pipeline DAG uses the [Edge Types](edge_types.md) system for typed dependencies:

```python
# During auto-resolution
edge = DependencyEdge(
    source_step=prop_ref.step_name,
    target_step=consumer_step,
    source_output=output_spec.logical_name,
    target_input=dep_name,
    confidence=confidence,
    metadata={'auto_resolved': True}
)

edge_id = self.dependency_edges.add_edge(edge)
```

## Related Documentation

### Pipeline DAG Components
- [Pipeline DAG Overview](README.md): Introduction to the DAG-based pipeline structure
- [Base Pipeline DAG](base_dag.md): Foundation for the enhanced DAG
- [Edge Types](edge_types.md): Types of edges used in the enhanced DAG

### Pipeline Building
- [Pipeline Template Base](../pipeline_builder/pipeline_template_base.md): Uses the enhanced DAG for template-based pipeline creation
- [Pipeline Assembler](../pipeline_builder/pipeline_assembler.md): Consumes the enhanced DAG for step instantiation
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md): Examples showing the enhanced DAG in action

### Dependency System
- [Dependency Resolver](../pipeline_deps/dependency_resolver.md): Core component used by the enhanced DAG for dependency resolution
- [Base Specifications](../pipeline_deps/base_specifications.md): Defines step specifications used by the enhanced DAG
- [Property Reference](../pipeline_deps/property_reference.md): Used by the enhanced DAG to represent resolved dependencies
- [Semantic Matcher](../pipeline_deps/semantic_matcher.md): Used by the dependency resolver for name similarity calculation
