---
tags:
  - code
  - core
  - base
  - specifications
  - dependencies
keywords:
  - step specifications
  - dependency management
  - output specifications
  - declarative design
  - pydantic validation
topics:
  - specification-driven architecture
  - dependency resolution
  - pipeline validation
language: python
date of note: 2025-08-07
---

# Base Specification Classes

## Overview

The `specification_base.py` module provides the core classes for defining step dependencies and outputs in a declarative, type-safe manner using Pydantic V2 BaseModel. This module forms the foundation of the specification-driven architecture for pipeline definition and validation.

## Purpose

This module provides:
- **Declarative Dependency Management**: Type-safe specification of step dependencies
- **Output Specifications**: Formal definition of step outputs with property paths
- **Step Specifications**: Complete specification combining dependencies and outputs
- **Contract Alignment**: Validation between specifications and script contracts
- **Node Type Validation**: Enforcement of pipeline structure constraints

## Core Classes

### DependencySpec

Declarative specification for a step's dependency requirement.

```python
class DependencySpec(BaseModel):
    logical_name: str
    dependency_type: DependencyType
    required: bool = True
    compatible_sources: List[str] = Field(default_factory=list)
    semantic_keywords: List[str] = Field(default_factory=list)
    data_type: str = "S3Uri"
    description: str = ""
```

#### Key Features

- **Logical Naming**: Human-readable names for dependencies
- **Type Classification**: Uses `DependencyType` enum for categorization
- **Source Compatibility**: Lists compatible step types that can provide the dependency
- **Semantic Matching**: Keywords for intelligent dependency resolution
- **Validation**: Comprehensive field validation with custom validators

#### Usage Examples

```python
# Training data dependency
training_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.TRAINING_DATA,
    required=True,
    compatible_sources=["DataLoadingStep", "PreprocessingStep"],
    semantic_keywords=["data", "dataset", "training"],
    data_type="S3Uri",
    description="Training dataset for model training"
)

# Optional model dependency
model_dep = DependencySpec(
    logical_name="pretrained_model",
    dependency_type=DependencyType.MODEL_ARTIFACTS,
    required=False,
    compatible_sources=["TrainingStep", "ModelStep"],
    semantic_keywords=["model", "artifacts", "pretrained"],
    description="Optional pretrained model for transfer learning"
)
```

### OutputSpec

Declarative specification for a step's output.

```python
class OutputSpec(BaseModel):
    logical_name: str
    aliases: List[str] = Field(default_factory=list)
    output_type: DependencyType
    property_path: str
    data_type: str = "S3Uri"
    description: str = ""
```

#### Key Features

- **Logical Naming**: Primary name for the output
- **Alias Support**: Alternative names for flexible referencing
- **Property Paths**: Runtime SageMaker property paths for accessing outputs
- **Type Classification**: Uses `DependencyType` enum for output categorization
- **Validation**: Ensures property paths follow SageMaker conventions

#### Usage Examples

```python
# Processing output with aliases
processing_output = OutputSpec(
    logical_name="processed_data",
    aliases=["ProcessedData", "DATA"],
    output_type=DependencyType.PROCESSING_OUTPUT,
    property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
    data_type="S3Uri",
    description="Processed training data output"
)

# Model artifacts output
model_output = OutputSpec(
    logical_name="model_artifacts",
    aliases=["ModelArtifacts", "MODEL"],
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    data_type="S3Uri",
    description="Trained model artifacts"
)
```

### StepSpecification

Complete specification for a step's dependencies and outputs.

```python
class StepSpecification(BaseModel):
    step_type: str
    node_type: NodeType
    dependencies: Dict[str, DependencySpec]
    outputs: Dict[str, OutputSpec]
    script_contract: Optional[ScriptContract] = None
```

#### Key Features

- **Step Classification**: Identifies the step type and node classification
- **Dependency Management**: Dictionary of all step dependencies
- **Output Management**: Dictionary of all step outputs
- **Contract Integration**: Optional script contract for validation
- **Node Type Validation**: Enforces structural constraints based on node type

#### Node Type Constraints

The specification enforces these constraints based on `NodeType`:

- **SOURCE**: Must have outputs, cannot have dependencies
- **INTERNAL**: Must have both dependencies and outputs
- **SINK**: Must have dependencies, cannot have outputs
- **SINGULAR**: Cannot have dependencies or outputs

#### Usage Examples

```python
# Data loading step (SOURCE node)
data_load_spec = StepSpecification(
    step_type="DataLoadingStep",
    node_type=NodeType.SOURCE,
    dependencies={},  # SOURCE nodes have no dependencies
    outputs={
        "raw_data": OutputSpec(
            logical_name="raw_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['raw_data'].S3Output.S3Uri"
        )
    }
)

# Training step (INTERNAL node)
training_spec = StepSpecification(
    step_type="TrainingStep",
    node_type=NodeType.INTERNAL,
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True
        )
    },
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    }
)
```

## Validation Framework

### Field Validation

Each class includes comprehensive field validation:

#### DependencySpec Validation

```python
@field_validator('logical_name')
@classmethod
def validate_logical_name(cls, v: str) -> str:
    """Validate logical name follows naming conventions."""
    if not v or not v.strip():
        raise ValueError("logical_name cannot be empty or whitespace")
    
    if not v.replace('_', '').replace('-', '').isalnum():
        raise ValueError("logical_name should contain only alphanumeric characters, underscores, and hyphens")
    
    return v.strip()
```

#### OutputSpec Validation

```python
@field_validator('property_path')
@classmethod
def validate_property_path(cls, v: str) -> str:
    """Validate property path follows SageMaker conventions."""
    if not v or not v.strip():
        raise ValueError("property_path cannot be empty or whitespace")
    
    v = v.strip()
    if not v.startswith('properties.'):
        raise ValueError("property_path should start with 'properties.'")
    
    return v
```

### Model Validation

#### StepSpecification Validation

```python
@model_validator(mode='after')
def validate_node_type_constraints(self) -> 'StepSpecification':
    """Validate that dependencies and outputs match the node type."""
    has_deps = len(self.dependencies) > 0
    has_outputs = len(self.outputs) > 0
    
    if self.node_type == NodeType.SOURCE:
        if has_deps:
            raise ValueError(f"SOURCE node '{self.step_type}' cannot have dependencies")
        if not has_outputs:
            raise ValueError(f"SOURCE node '{self.step_type}' must have outputs")
    # ... additional node type validations
    
    return self
```

## Contract Alignment

### Alignment Validation

The `validate_contract_alignment()` method ensures specifications align with script contracts:

```python
def validate_contract_alignment(self) -> ValidationResult:
    """
    Validate that script contract aligns with step specification.
    
    Validation logic:
    - Specs can provide more inputs than contracts require (extra dependencies allowed)
    - Contracts can have fewer outputs than specs provide (aliases allowed)
    - For every contract input, there must be a matching spec dependency
    - For every contract output, there must be a matching spec output
    """
    if not self.script_contract:
        return ValidationResult.success("No contract to validate")
    
    errors = []
    
    # Validate input alignment
    contract_inputs = set(self.script_contract.expected_input_paths.keys())
    spec_dependency_names = set(dep.logical_name for dep in self.dependencies.values())
    
    missing_spec_dependencies = contract_inputs - spec_dependency_names
    if missing_spec_dependencies:
        errors.append(f"Contract inputs missing from specification dependencies: {missing_spec_dependencies}")
    
    # Validate output alignment
    contract_outputs = set(self.script_contract.expected_output_paths.keys())
    spec_output_names = set(output.logical_name for output in self.outputs.values())
    
    missing_spec_outputs = contract_outputs - spec_output_names
    if missing_spec_outputs:
        errors.append(f"Contract outputs missing from specification outputs: {missing_spec_outputs}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

## Query and Access Methods

### Dependency Queries

```python
def get_dependency(self, logical_name: str) -> Optional[DependencySpec]:
    """Get dependency specification by logical name."""
    return self.dependencies.get(logical_name)

def list_required_dependencies(self) -> List[DependencySpec]:
    """Get list of required dependencies."""
    return [dep for dep in self.dependencies.values() if dep.required]

def list_dependencies_by_type(self, dependency_type: DependencyType) -> List[DependencySpec]:
    """Get list of dependencies of a specific type."""
    return [dep for dep in self.dependencies.values() if dep.dependency_type == dependency_type]
```

### Output Queries

```python
def get_output(self, logical_name: str) -> Optional[OutputSpec]:
    """Get output specification by logical name."""
    return self.outputs.get(logical_name)

def get_output_by_name_or_alias(self, name: str) -> Optional[OutputSpec]:
    """Get output specification by logical name or alias."""
    # First try exact logical name match
    if name in self.outputs:
        return self.outputs[name]
    
    # Then search through aliases
    name_lower = name.lower()
    for output_spec in self.outputs.values():
        for alias in output_spec.aliases:
            if alias.lower() == name_lower:
                return output_spec
    
    return None

def list_all_output_names(self) -> List[str]:
    """Get list of all possible output names (logical names + aliases)."""
    all_names = []
    for output_spec in self.outputs.values():
        all_names.append(output_spec.logical_name)
        all_names.extend(output_spec.aliases)
    return all_names
```

## Design Patterns

### Specification-Driven Development

```python
# 1. Define the specification
spec = StepSpecification(
    step_type="ProcessingStep",
    node_type=NodeType.INTERNAL,
    dependencies={
        "input_data": DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
        )
    }
)

# 2. Use specification in step builder
builder = ProcessingStepBuilder(config, spec=spec)

# 3. Validate contract alignment
if spec.script_contract:
    result = spec.validate_contract_alignment()
    if not result.is_valid:
        raise ValueError(f"Contract alignment errors: {result.errors}")
```

### Flexible Output Referencing

```python
# Define output with aliases
output_spec = OutputSpec(
    logical_name="model_artifacts",
    aliases=["ModelArtifacts", "MODEL", "artifacts"],
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts"
)

# Reference by any name or alias
spec.get_output_by_name_or_alias("model_artifacts")  # Primary name
spec.get_output_by_name_or_alias("ModelArtifacts")   # Alias
spec.get_output_by_name_or_alias("MODEL")            # Alias
```

## Integration Points

### With Step Builders

Step builders use specifications for:
- Dependency validation
- Output property path resolution
- Contract alignment verification
- Type-safe step construction

### With Dependency Resolution

Dependency resolvers use specifications for:
- Semantic matching via keywords
- Type compatibility checking
- Source compatibility validation
- Automatic dependency resolution

### With Pipeline Validation

Pipeline validators use specifications for:
- Node type constraint enforcement
- Dependency graph validation
- Output-input compatibility checking
- Contract compliance verification

## Best Practices

### Specification Design

1. **Logical Names**: Use descriptive, consistent naming
2. **Type Classification**: Choose appropriate `DependencyType` values
3. **Property Paths**: Use correct SageMaker property path syntax
4. **Documentation**: Provide clear descriptions for all specifications

### Dependency Management

1. **Required vs Optional**: Carefully consider which dependencies are required
2. **Semantic Keywords**: Use relevant keywords for automatic resolution
3. **Compatible Sources**: List all step types that can provide the dependency
4. **Type Safety**: Use appropriate data types

### Output Management

1. **Aliases**: Provide useful aliases for flexible referencing
2. **Property Paths**: Ensure paths match actual SageMaker step properties
3. **Type Consistency**: Use consistent output types across related steps

## Error Handling

The module provides comprehensive error handling:

1. **Validation Errors**: Clear messages for invalid specifications
2. **Node Type Violations**: Specific errors for node type constraint violations
3. **Contract Misalignment**: Detailed errors for contract-specification mismatches
4. **Property Path Errors**: Validation of SageMaker property path syntax

This specification system provides a robust foundation for declarative, type-safe pipeline definition with comprehensive validation and flexible dependency management.
