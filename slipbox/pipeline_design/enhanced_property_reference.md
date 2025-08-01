---
tags:
  - design
  - implementation
  - pipeline_deps
  - property_reference
keywords:
  - property references
  - SageMaker properties
  - property path parsing
  - runtime resolution
  - step output references
  - lazy evaluation
topics:
  - property path handling
  - pipeline step connections
  - runtime resolution
language: python
date of note: 2025-07-31
---

# Enhanced PropertyReference System

## Problem Statement

The pipeline system needs a robust way to handle references between different step outputs and inputs. At definition time, we specify logical connections, but at runtime, these must be translated into actual SageMaker property references. The challenge is bridging the gap between:

1. **Definition-time**: When we specify which outputs connect to which inputs
2. **Runtime**: When the actual SageMaker Properties objects must be created and passed

Specifically, the system needs to handle complex property paths like `properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri` and translate them into proper SageMaker property references.

## PropertyReference Class

The `PropertyReference` class serves as a **bridge between definition-time specifications and runtime property references**. It handles translation between logical output names and SageMaker's property path system.

```python
class PropertyReference(BaseModel):
    """Lazy evaluation reference bridging definition-time and runtime."""
    
    step_name: str = Field(
        ...,
        description="Name of the step that produces this output",
        min_length=1
    )
    output_spec: OutputSpec = Field(
        ...,
        description="Output specification for the referenced output"
    )
```

### Key Features

1. **Lazy Evaluation**: References are created at definition time but resolved only at runtime
2. **Property Path Parsing**: Handles complex property paths with attribute access, dictionary access, and array indexing
3. **SageMaker Integration**: Provides methods to convert to SageMaker property formats
4. **Error Handling**: Robust validation and clear error messages

## How It Works

### 1. Definition Time: Creating References

When defining a pipeline, we create property references to connect steps:

```python
# Create a property reference
prop_ref = PropertyReference(
    step_name="preprocessing_step",
    output_spec=OutputSpec(
        logical_name="processed_data",
        description="Preprocessed training data",
        property_path="properties.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri"
    )
)
```

### 2. Definition Time: Converting to SageMaker Format

At pipeline definition time, we convert to SageMaker's property dictionary format:

```python
# Get SageMaker property dictionary
sagemaker_prop = prop_ref.to_sagemaker_property()
# Returns: {"Get": "Steps.preprocessing_step.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri"}
```

### 3. Runtime: Resolving to Actual Properties

At runtime, when step instances exist, we resolve to actual SageMaker property objects:

```python
# Resolve to actual SageMaker property
actual_prop = prop_ref.to_runtime_property(step_instances)
# Returns a SageMaker Properties object that can be used as an input
```

## Property Path Parsing

The most complex part of the PropertyReference system is parsing property paths to handle various formats:

```python
def _parse_property_path(self, path: str) -> List[Union[str, Tuple[str, str]]]:
    """
    Parse a property path into a sequence of access operations.
    
    Handles various property path formats like:
    - Regular attribute access: "properties.ModelArtifacts.S3ModelArtifacts"
    - Dictionary access: "properties.Outputs['DATA']"
    - Array indexing: "properties.TrainingJobSummaries[0]"
    - Mixed patterns: "properties.Config.Outputs['data'].Sub[0].Value"
    """
```

The parser converts a property path string into a structured representation that can be used to navigate through SageMaker property objects.

### Supported Path Formats

The property path parser supports:

1. **Attribute Access**: `properties.ModelArtifacts.S3ModelArtifacts`
2. **Dictionary Access**: `Outputs['data']` or `Outputs["data"]`
3. **Array Indexing**: `TrainingJobSummaries[0]`
4. **Mixed Patterns**: `Config.Outputs['data'].Sub[0].Value`

### Path Navigation

Once a path is parsed, the system uses the structured representation to navigate through the actual SageMaker property objects:

```python
def _get_property_value(self, obj: Any, path_parts: List[Union[str, Tuple[str, str]]]) -> Any:
    """Navigate through the property path to get the final value."""
    
    current_obj = obj
    for part in path_parts:
        if isinstance(part, str):
            # Regular attribute access
            current_obj = getattr(current_obj, part)
        elif isinstance(part, tuple) and len(part) == 2:
            # Dictionary access or array indexing
            attr_name, key = part
            if attr_name:  
                current_obj = getattr(current_obj, attr_name)
            if isinstance(key, int) or key.isdigit():  
                # Array index
                idx = key if isinstance(key, int) else int(key)
                current_obj = current_obj[idx]
            else:  
                # Dictionary key
                current_obj = current_obj[key]
    
    return current_obj
```

## Integration with Step Builders

The `PropertyReference` class integrates with [Step Builders](step_builder.md) to facilitate automatic dependency resolution:

```python
# In a step builder's extract_inputs_from_dependencies method
resolved = resolver.resolve_step_dependencies(step_name, available_steps)
    
# Convert results to SageMaker properties
return {name: prop_ref.to_sagemaker_property() for name, prop_ref in resolved.items()}
```

## Integration with Dependency Resolver

The [Dependency Resolver](dependency_resolver.md) uses PropertyReference objects to represent connections between steps:

```python
# In UnifiedDependencyResolver
def resolve_step_dependencies(self, step_name: str, available_steps: List[str]) -> Dict[str, PropertyReference]:
    """Resolve all dependencies for a step and return property references."""
    # ... resolution logic ...
    
    # Create PropertyReference for the match
    property_ref = PropertyReference(
        step_name=src_step_name,
        output_spec=output_spec
    )
    
    result[dep_name] = property_ref
```

## Benefits

1. **Clear Separation**: Separates definition-time specification from runtime resolution
2. **Robust Property Handling**: Handles all SageMaker property path formats
3. **Error Detection**: Early validation and clear error messages
4. **Maintainability**: Centralizes property reference logic in one place
5. **Integration**: Works seamlessly with [step specifications](step_specification.md) and [dependency resolution](dependency_resolver.md)

## Advanced Usage

### Format-specific Property Paths

Some property paths may require special handling based on step type:

```python
# ModelStep property path format
model_property_path = "properties.ModelName"

# ProcessingStep property path format
processing_property_path = "properties.ProcessingOutputConfig.Outputs['data'].S3Output.S3Uri"
```

The `PropertyReference` class handles all these formats transparently.

### Error Handling

The class includes robust error handling:

1. **Step Name Validation**: Ensures step names are not empty
2. **Path Parsing Validation**: Validates property paths are well-formed
3. **Runtime Resolution**: Provides clear errors if paths can't be resolved

## Conclusion

The `PropertyReference` system is a critical component for connecting steps in SageMaker pipelines. By bridging definition-time specifications with runtime property objects, it enables declarative pipeline definitions that correctly resolve to executable SageMaker steps.

This approach enhances maintainability, readability, and robustness of pipeline definitions by providing a clear separation between logical connections and their runtime implementations.
