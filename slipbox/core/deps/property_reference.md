---
tags:
  - code
  - deps
  - property_reference
  - sagemaker_properties
  - runtime_references
keywords:
  - PropertyReference
  - SageMaker properties
  - property paths
  - runtime references
  - lazy evaluation
  - step instances
  - property navigation
topics:
  - property references
  - SageMaker integration
  - runtime property handling
language: python
date of note: 2024-12-07
---

# Property Reference

Property Reference module for SageMaker property path handling that bridges between definition-time specifications and runtime property references in the SageMaker pipeline context.

## Overview

The `PropertyReference` class provides a sophisticated system for handling complex property paths across various SageMaker step types. This class implements lazy evaluation references that bridge the gap between definition-time specifications and runtime property references, enabling dynamic property resolution during pipeline execution.

The system handles various SageMaker property path formats including regular attribute access, dictionary access with string keys, array indexing with numeric indices, and complex mixed patterns combining multiple access types. This comprehensive support ensures compatibility with all SageMaker step output formats and property structures.

The class supports advanced features including property path parsing with regex-based pattern matching, runtime property navigation through step instances, SageMaker Properties object creation for pipeline execution, and comprehensive validation of step names and property paths.

## Classes and Methods

### Classes
- [`PropertyReference`](#propertyreference) - Lazy evaluation reference for SageMaker property paths

## API Reference

### PropertyReference

_class_ cursus.core.deps.property_reference.PropertyReference(_step_name_, _output_spec_)

Lazy evaluation reference bridging definition-time and runtime. This class provides a bridge between specification-time output definitions and runtime SageMaker property references, enabling dynamic property resolution during pipeline execution.

**Parameters:**
- **step_name** (_str_) – Name of the step that produces this output. Must be non-empty and non-whitespace.
- **output_spec** (_OutputSpec_) – Output specification for the referenced output containing property path and metadata.

```python
from cursus.core.deps.property_reference import PropertyReference
from cursus.core.base import OutputSpec, DependencyType

# Create output specification
output_spec = OutputSpec(
    logical_name="model_artifacts",
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    output_type=DependencyType.MODEL_ARTIFACTS,
    data_type="S3Uri"
)

# Create property reference
prop_ref = PropertyReference(
    step_name="training_step",
    output_spec=output_spec
)
```

#### to_sagemaker_property

to_sagemaker_property()

Convert to SageMaker Properties dictionary format at pipeline definition time. This method creates a dictionary representation suitable for SageMaker pipeline definition, removing the "properties." prefix if present.

**Returns:**
- **Dict[str, str]** – SageMaker Properties dictionary with "Get" key containing the property path.

```python
# Convert to SageMaker property format
sagemaker_prop = prop_ref.to_sagemaker_property()
print(sagemaker_prop)
# Output: {"Get": "Steps.training_step.ModelArtifacts.S3ModelArtifacts"}
```

#### to_runtime_property

to_runtime_property(_step_instances_)

Create an actual SageMaker property reference using step instances. This method navigates the property path to create a proper SageMaker Properties object that can be used at runtime for dynamic property resolution.

**Parameters:**
- **step_instances** (_Dict[str, Any]_) – Dictionary mapping step names to step instances for property navigation.

**Returns:**
- **Any** – SageMaker Properties object for the referenced property that can be used in pipeline execution.

```python
# Create runtime property reference
step_instances = {
    "training_step": training_step_instance,
    "preprocessing_step": preprocessing_step_instance
}

runtime_prop = prop_ref.to_runtime_property(step_instances)
# Returns actual SageMaker Properties object for pipeline execution
```

## Property Path Formats

The PropertyReference class supports various SageMaker property path formats:

### Regular Attribute Access
```python
# Simple attribute navigation
output_spec = OutputSpec(
    logical_name="model_artifacts",
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    output_type=DependencyType.MODEL_ARTIFACTS,
    data_type="S3Uri"
)
```

### Dictionary Access
```python
# Dictionary access with string keys
output_spec = OutputSpec(
    logical_name="processing_output",
    property_path="properties.Outputs['DATA']",
    output_type=DependencyType.PROCESSING_OUTPUT,
    data_type="S3Uri"
)
```

### Array Indexing
```python
# Array indexing with numeric indices
output_spec = OutputSpec(
    logical_name="training_summary",
    property_path="properties.TrainingJobSummaries[0]",
    output_type=DependencyType.CUSTOM_PROPERTY,
    data_type="String"
)
```

### Complex Mixed Patterns
```python
# Complex patterns combining multiple access types
output_spec = OutputSpec(
    logical_name="config_value",
    property_path="properties.Config.Outputs['data'].Sub[0].Value",
    output_type=DependencyType.CUSTOM_PROPERTY,
    data_type="String"
)
```

## Usage Examples

### Basic Property Reference Creation
```python
from cursus.core.deps.property_reference import PropertyReference
from cursus.core.base import OutputSpec, DependencyType

# Create output specification for model artifacts
model_output_spec = OutputSpec(
    logical_name="trained_model",
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    output_type=DependencyType.MODEL_ARTIFACTS,
    data_type="S3Uri",
    aliases=["model", "artifacts"]
)

# Create property reference
model_ref = PropertyReference(
    step_name="training",
    output_spec=model_output_spec
)

print(f"Reference: {model_ref}")
# Output: training.trained_model
```

### Runtime Property Resolution
```python
# Assume we have step instances from pipeline execution
step_instances = {
    "training": training_step,
    "preprocessing": preprocessing_step,
    "evaluation": evaluation_step
}

# Create property reference for training output
training_output_ref = PropertyReference(
    step_name="training",
    output_spec=model_output_spec
)

# Resolve to runtime property
try:
    runtime_property = training_output_ref.to_runtime_property(step_instances)
    # Use runtime_property in subsequent pipeline steps
    print(f"Runtime property resolved successfully")
except ValueError as e:
    print(f"Resolution failed: {e}")
```

### Complex Property Path Navigation
```python
# Create reference for complex nested property
complex_output_spec = OutputSpec(
    logical_name="evaluation_metrics",
    property_path="properties.EvaluationReport.Outputs['metrics'].Results[0].Value",
    output_type=DependencyType.CUSTOM_PROPERTY,
    data_type="String"
)

complex_ref = PropertyReference(
    step_name="evaluation",
    output_spec=complex_output_spec
)

# Convert to SageMaker format
sagemaker_format = complex_ref.to_sagemaker_property()
print(sagemaker_format)
# Output: {"Get": "Steps.evaluation.EvaluationReport.Outputs['metrics'].Results[0].Value"}
```

## Related Documentation

- [Output Specification](../base/output_spec.md) - Output specification system used by property references
- [Dependency Resolver](dependency_resolver.md) - Uses property references for dependency resolution
- [Step Specification](../base/step_specification.md) - Step specification system that defines outputs
- [Pipeline Assembler](../assembler/pipeline_assembler.md) - Uses property references for step connection
- [SageMaker Properties Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-properties.html) - AWS documentation for SageMaker property system
