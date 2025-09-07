---
tags:
  - code
  - base
  - specification_base
  - dependency_specs
  - output_specs
keywords:
  - StepSpecification
  - DependencySpec
  - OutputSpec
  - declarative specifications
  - type-safe dependencies
  - Pydantic models
  - specification validation
topics:
  - step specifications
  - dependency management
  - declarative configuration
language: python
date of note: 2024-12-07
---

# Specification Base

Base specifications for declarative dependency management that provide core classes for defining step dependencies and outputs in a declarative, type-safe manner using Pydantic V2 BaseModel.

## Overview

This module provides the foundational specification system for the cursus pipeline framework, enabling declarative definition of step dependencies and outputs with comprehensive validation and type safety. The specification system uses Pydantic V2 models to ensure data integrity and provide rich validation capabilities throughout the pipeline lifecycle.

The system supports advanced features including type-safe dependency and output specifications with enum validation, comprehensive validation with custom validators and model validators, alias support for flexible output referencing, semantic keyword matching for intelligent dependency resolution, and script contract integration for implementation validation.

The specification classes provide the foundation for the entire dependency resolution system, enabling intelligent pipeline assembly through declarative configuration rather than imperative wiring.

## Classes and Methods

### Classes
- [`DependencySpec`](#dependencyspec) - Declarative specification for step dependency requirements
- [`OutputSpec`](#outputspec) - Declarative specification for step outputs
- [`StepSpecification`](#stepspecification) - Complete specification for step dependencies and outputs

## API Reference

### DependencySpec

_class_ cursus.core.base.specification_base.DependencySpec(_logical_name_, _dependency_type_, _required=True_, _compatible_sources=[]_, _semantic_keywords=[]_, _data_type="S3Uri"_, _description=""_)

Declarative specification for a step's dependency requirement. This class defines what a step needs as input, including type information, compatibility constraints, and semantic hints for intelligent resolution.

**Parameters:**
- **logical_name** (_str_) – How this dependency is referenced within the step. Must be non-empty and follow naming conventions.
- **dependency_type** (_DependencyType_) – Type of dependency using the standard dependency type enumeration.
- **required** (_bool_) – Whether this dependency is required for step execution. Defaults to True.
- **compatible_sources** (_List[str]_) – Compatible step types that can provide this dependency. Defaults to empty list.
- **semantic_keywords** (_List[str]_) – Keywords for semantic matching during dependency resolution. Defaults to empty list.
- **data_type** (_str_) – Expected data type of the dependency. Defaults to "S3Uri".
- **description** (_str_) – Human-readable description of the dependency. Defaults to empty string.

```python
from cursus.core.base.specification_base import DependencySpec
from cursus.core.base.enums import DependencyType

# Create training data dependency
training_data_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.TRAINING_DATA,
    required=True,
    compatible_sources=["TabularPreprocessing", "DataLoading"],
    semantic_keywords=["processed", "clean", "transformed"],
    data_type="S3Uri",
    description="Processed training dataset for model training"
)

# Create optional hyperparameter dependency
hyperparams_dep = DependencySpec(
    logical_name="hyperparameters",
    dependency_type=DependencyType.HYPERPARAMETERS,
    required=False,
    compatible_sources=["HyperparameterTuning"],
    semantic_keywords=["config", "params", "tuning"],
    data_type="String",
    description="Optional hyperparameter configuration"
)
```

#### matches_name_or_alias

matches_name_or_alias(_name_)

Check if the given name matches the logical name or any alias. This method provides consistent interface for name matching across specification types.

**Parameters:**
- **name** (_str_) – The name to check against the dependency specification.

**Returns:**
- **bool** – True if the name matches the logical name or any alias.

```python
# Check name matching
if training_data_dep.matches_name_or_alias("training_data"):
    print("Name matches dependency specification")
```

### OutputSpec

_class_ cursus.core.base.specification_base.OutputSpec(_logical_name_, _output_type_, _property_path_, _aliases=[]_, _data_type="S3Uri"_, _description=""_)

Declarative specification for a step's output. This class defines what a step produces as output, including property paths for runtime access and aliases for flexible referencing.

**Parameters:**
- **logical_name** (_str_) – Primary name for this output. Must be non-empty and follow naming conventions.
- **output_type** (_DependencyType_) – Type of output using the standard dependency type enumeration.
- **property_path** (_str_) – Runtime SageMaker property path to access this output. Must start with "properties.".
- **aliases** (_List[str]_) – Alternative names that can be used to reference this output. Defaults to empty list.
- **data_type** (_str_) – Output data type. Defaults to "S3Uri".
- **description** (_str_) – Human-readable description of the output. Defaults to empty string.

```python
from cursus.core.base.specification_base import OutputSpec
from cursus.core.base.enums import DependencyType

# Create model artifacts output
model_output = OutputSpec(
    logical_name="model_artifacts",
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    aliases=["trained_model", "model", "artifacts"],
    data_type="S3Uri",
    description="Trained model artifacts and metadata"
)

# Create processing output with complex property path
processed_data_output = OutputSpec(
    logical_name="processed_data",
    output_type=DependencyType.PROCESSING_OUTPUT,
    property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
    aliases=["ProcessedData", "DATA"],
    data_type="S3Uri",
    description="Processed training data ready for model training"
)
```

#### matches_name_or_alias

matches_name_or_alias(_name_)

Check if the given name matches the logical name or any alias. This method enables flexible output referencing using either the primary name or any defined alias.

**Parameters:**
- **name** (_str_) – The name to check against the output specification.

**Returns:**
- **bool** – True if the name matches the logical name or any alias.

```python
# Check various name matches
if model_output.matches_name_or_alias("model_artifacts"):
    print("Matches logical name")

if model_output.matches_name_or_alias("trained_model"):
    print("Matches alias")

if model_output.matches_name_or_alias("artifacts"):
    print("Matches another alias")
```

### StepSpecification

_class_ cursus.core.base.specification_base.StepSpecification(_step_type_, _node_type_, _dependencies={}_, _outputs={}_, _script_contract=None_)

Complete specification for a step's dependencies and outputs. This class combines dependency and output specifications with step metadata to provide a complete declarative definition of a pipeline step.

**Parameters:**
- **step_type** (_str_) – Type identifier for this step. Must be non-empty.
- **node_type** (_NodeType_) – Node type classification for validation using the NodeType enumeration.
- **dependencies** (_Dict[str, DependencySpec]_) – Dictionary of dependency specifications keyed by logical name. Defaults to empty dict.
- **outputs** (_Dict[str, OutputSpec]_) – Dictionary of output specifications keyed by logical name. Defaults to empty dict.
- **script_contract** (_Optional[ScriptContract]_) – Optional script contract for implementation validation. Defaults to None.

```python
from cursus.core.base.specification_base import StepSpecification
from cursus.core.base.enums import NodeType, DependencyType

# Create complete step specification
preprocessing_spec = StepSpecification(
    step_type="TabularPreprocessing",
    node_type=NodeType.INTERNAL,
    dependencies={
        "raw_data": DependencySpec(
            logical_name="raw_data",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["DataLoading"],
            semantic_keywords=["raw", "input", "source"]
        )
    },
    outputs={
        "processed_data": OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.Outputs['ProcessedData']",
            aliases=["ProcessedData", "DATA"]
        )
    }
)
```

#### get_dependency

get_dependency(_logical_name_)

Get dependency specification by logical name. This method retrieves a specific dependency specification from the step.

**Parameters:**
- **logical_name** (_str_) – Logical name of the dependency to retrieve.

**Returns:**
- **Optional[DependencySpec]** – Dependency specification if found, None otherwise.

```python
# Retrieve specific dependency
raw_data_dep = preprocessing_spec.get_dependency("raw_data")
if raw_data_dep:
    print(f"Found dependency: {raw_data_dep.logical_name}")
```

#### get_output

get_output(_logical_name_)

Get output specification by logical name. This method retrieves a specific output specification from the step.

**Parameters:**
- **logical_name** (_str_) – Logical name of the output to retrieve.

**Returns:**
- **Optional[OutputSpec]** – Output specification if found, None otherwise.

```python
# Retrieve specific output
processed_output = preprocessing_spec.get_output("processed_data")
if processed_output:
    print(f"Found output: {processed_output.logical_name}")
```

#### get_output_by_name_or_alias

get_output_by_name_or_alias(_name_)

Get output specification by logical name or alias. This method provides flexible output retrieval using either the primary name or any defined alias.

**Parameters:**
- **name** (_str_) – The logical name or alias to search for.

**Returns:**
- **Optional[OutputSpec]** – Output specification if found, None otherwise.

```python
# Retrieve output by various names
output1 = preprocessing_spec.get_output_by_name_or_alias("processed_data")  # logical name
output2 = preprocessing_spec.get_output_by_name_or_alias("ProcessedData")   # alias
output3 = preprocessing_spec.get_output_by_name_or_alias("DATA")            # another alias

assert output1 == output2 == output3  # All return the same output spec
```

#### list_required_dependencies

list_required_dependencies()

Get list of required dependencies. This method filters dependencies to return only those marked as required.

**Returns:**
- **List[DependencySpec]** – List of required dependency specifications.

```python
# Get required dependencies
required_deps = preprocessing_spec.list_required_dependencies()
print(f"Required dependencies: {[dep.logical_name for dep in required_deps]}")
```

#### list_optional_dependencies

list_optional_dependencies()

Get list of optional dependencies. This method filters dependencies to return only those marked as optional.

**Returns:**
- **List[DependencySpec]** – List of optional dependency specifications.

```python
# Get optional dependencies
optional_deps = preprocessing_spec.list_optional_dependencies()
print(f"Optional dependencies: {[dep.logical_name for dep in optional_deps]}")
```

#### list_dependencies_by_type

list_dependencies_by_type(_dependency_type_)

Get list of dependencies of a specific type. This method filters dependencies by their dependency type.

**Parameters:**
- **dependency_type** (_DependencyType_) – Dependency type to filter by.

**Returns:**
- **List[DependencySpec]** – List of dependency specifications matching the type.

```python
# Get training data dependencies
training_deps = preprocessing_spec.list_dependencies_by_type(DependencyType.TRAINING_DATA)
print(f"Training data dependencies: {[dep.logical_name for dep in training_deps]}")
```

#### list_outputs_by_type

list_outputs_by_type(_output_type_)

Get list of outputs of a specific type. This method filters outputs by their output type.

**Parameters:**
- **output_type** (_DependencyType_) – Output type to filter by.

**Returns:**
- **List[OutputSpec]** – List of output specifications matching the type.

```python
# Get processing outputs
processing_outputs = preprocessing_spec.list_outputs_by_type(DependencyType.PROCESSING_OUTPUT)
print(f"Processing outputs: {[out.logical_name for out in processing_outputs]}")
```

#### validate_contract_alignment

validate_contract_alignment()

Validate that script contract aligns with step specification. This method checks that the script contract's inputs and outputs are compatible with the step specification.

**Returns:**
- **ValidationResult** – Result indicating whether the contract aligns with the specification.

```python
# Validate contract alignment
validation_result = preprocessing_spec.validate_contract_alignment()
if validation_result.is_valid:
    print("Contract aligns with specification")
else:
    print(f"Contract validation errors: {validation_result.errors}")
```

#### validate_script_compliance

validate_script_compliance(_script_path_)

Validate script implementation against contract. This method checks that the actual script implementation complies with the defined contract.

**Parameters:**
- **script_path** (_str_) – Path to the script file to validate.

**Returns:**
- **ValidationResult** – Result indicating whether the script complies with the contract.

```python
# Validate script implementation
script_validation = preprocessing_spec.validate_script_compliance("preprocessing_script.py")
if script_validation.is_valid:
    print("Script complies with contract")
else:
    print(f"Script validation errors: {script_validation.errors}")
```

## Usage Examples

### Complete Step Specification Creation
```python
from cursus.core.base.specification_base import StepSpecification, DependencySpec, OutputSpec
from cursus.core.base.enums import NodeType, DependencyType

# Create training step specification
training_spec = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing"],
            semantic_keywords=["processed", "clean", "features"],
            data_type="S3Uri",
            description="Processed training dataset with features"
        ),
        "hyperparameters": DependencySpec(
            logical_name="hyperparameters",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,
            compatible_sources=["HyperparameterTuning"],
            semantic_keywords=["config", "params", "tuning"],
            data_type="String",
            description="XGBoost hyperparameter configuration"
        )
    },
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            aliases=["trained_model", "model", "xgboost_model"],
            data_type="S3Uri",
            description="Trained XGBoost model artifacts"
        ),
        "training_metrics": OutputSpec(
            logical_name="training_metrics",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.FinalMetricDataList[0].Value",
            aliases=["metrics", "performance"],
            data_type="String",
            description="Training performance metrics"
        )
    }
)

print(f"Created specification for {training_spec.step_type}")
print(f"Dependencies: {len(training_spec.dependencies)}")
print(f"Outputs: {len(training_spec.outputs)}")
```

### Specification Analysis and Validation
```python
# Analyze specification structure
def analyze_specification(spec):
    """Analyze a step specification structure."""
    print(f"Step Type: {spec.step_type}")
    print(f"Node Type: {spec.node_type.value}")
    
    # Analyze dependencies
    required_deps = spec.list_required_dependencies()
    optional_deps = spec.list_optional_dependencies()
    
    print(f"\nDependencies ({len(spec.dependencies)} total):")
    print(f"  Required: {[dep.logical_name for dep in required_deps]}")
    print(f"  Optional: {[dep.logical_name for dep in optional_deps]}")
    
    # Analyze outputs
    print(f"\nOutputs ({len(spec.outputs)} total):")
    for output in spec.outputs.values():
        aliases_str = f" (aliases: {output.aliases})" if output.aliases else ""
        print(f"  {output.logical_name}: {output.output_type.value}{aliases_str}")
    
    # Validate node type constraints
    has_deps = len(spec.dependencies) > 0
    has_outputs = len(spec.outputs) > 0
    
    print(f"\nNode Type Validation:")
    print(f"  Has dependencies: {has_deps}")
    print(f"  Has outputs: {has_outputs}")
    print(f"  Node type constraints satisfied: {validate_node_constraints(spec.node_type, has_deps, has_outputs)}")

def validate_node_constraints(node_type, has_deps, has_outputs):
    """Validate node type constraints."""
    if node_type == NodeType.SOURCE:
        return not has_deps and has_outputs
    elif node_type == NodeType.INTERNAL:
        return has_deps and has_outputs
    elif node_type == NodeType.SINK:
        return has_deps and not has_outputs
    elif node_type == NodeType.SINGULAR:
        return not has_deps and not has_outputs
    return False

# Analyze the training specification
analyze_specification(training_spec)
```

### Flexible Output Access
```python
# Demonstrate flexible output access
def test_output_access(spec):
    """Test various ways to access outputs."""
    
    # Access by logical name
    model_output = spec.get_output("model_artifacts")
    print(f"By logical name: {model_output.logical_name if model_output else 'Not found'}")
    
    # Access by alias
    model_by_alias = spec.get_output_by_name_or_alias("trained_model")
    print(f"By alias: {model_by_alias.logical_name if model_by_alias else 'Not found'}")
    
    # List all possible names
    all_names = []
    for output in spec.outputs.values():
        all_names.append(output.logical_name)
        all_names.extend(output.aliases)
    
    print(f"All possible output names: {all_names}")
    
    # Test name matching
    test_names = ["model_artifacts", "trained_model", "xgboost_model", "nonexistent"]
    for name in test_names:
        found = spec.get_output_by_name_or_alias(name) is not None
        print(f"  '{name}': {'Found' if found else 'Not found'}")

test_output_access(training_spec)
```

## Related Documentation

- [Base Enums](enums.md) - Dependency and node type enumerations used by specifications
- [Config Base](config_base.md) - Configuration classes that work with specifications
- [Builder Base](builder_base.md) - Step builders that implement specifications
- [Contract Base](contract_base.md) - Script contracts for implementation validation
- [Dependency Resolver](../deps/dependency_resolver.md) - Uses specifications for intelligent dependency resolution
