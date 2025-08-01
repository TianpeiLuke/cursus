# Base Specifications

## Overview
The Base Specifications module provides foundation classes for defining pipeline step specifications using Pydantic V2 BaseModel for declarative dependency management. It enables type-safe specification of step dependencies and outputs with automatic validation.

## Core Classes

### DependencySpec
Declarative specification for a step's dependency requirement.

```python
class DependencySpec(BaseModel):
    logical_name: str = Field(..., description="How this dependency is referenced")
    dependency_type: DependencyType = Field(..., description="Type of dependency")
    required: bool = Field(default=True, description="Whether this dependency is required")
    compatible_sources: List[str] = Field(default_factory=list, description="Compatible step types")
    semantic_keywords: List[str] = Field(default_factory=list, description="Keywords for semantic matching")
    data_type: str = Field(default="S3Uri", description="Expected data type")
    description: str = Field(default="", description="Human-readable description")
```

### OutputSpec
Declarative specification for a step's output.

```python
class OutputSpec(BaseModel):
    logical_name: str = Field(..., description="How this output is referenced")
    output_type: DependencyType = Field(..., description="Type of output")
    property_path: str = Field(..., description="Runtime SageMaker property path")
    data_type: str = Field(default="S3Uri", description="Output data type")
    description: str = Field(default="", description="Human-readable description")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for this output")
```

### StepSpecification
Complete specification for a step's dependencies and outputs.

```python
class StepSpecification(BaseModel):
    step_type: str = Field(..., description="Type identifier for this step")
    node_type: NodeType = Field(..., description="Node type classification")
    dependencies: Dict[str, DependencySpec] = Field(default_factory=dict)
    outputs: Dict[str, OutputSpec] = Field(default_factory=dict)
    script_contract: Optional[ScriptContract] = Field(default=None, description="Associated script contract for validation")
```

### PropertyReference
Lazy evaluation reference bridging definition-time and runtime.

```python
class PropertyReference(BaseModel):
    step_name: str = Field(..., description="Name of the step that produces this output")
    output_spec: OutputSpec = Field(..., description="Output specification")
    
    def to_sagemaker_property(self) -> Dict[str, str]:
        """Convert to SageMaker Properties object at runtime."""
        return {"Get": f"Steps.{self.step_name}.{self.output_spec.property_path}"}
```

## Enumerations

### DependencyType
Types of dependencies in the pipeline.

```python
class DependencyType(Enum):
    MODEL_ARTIFACTS = "model_artifacts"
    PROCESSING_OUTPUT = "processing_output"
    TRAINING_DATA = "training_data"
    HYPERPARAMETERS = "hyperparameters"
    PAYLOAD_SAMPLES = "payload_samples"
    CUSTOM_PROPERTY = "custom_property"
```

### NodeType
Types of nodes based on their dependency/output characteristics.

```python
class NodeType(Enum):
    SOURCE = "source"      # No dependencies, has outputs (e.g., data loading)
    INTERNAL = "internal"  # Has both dependencies and outputs (e.g., processing, training)
    SINK = "sink"         # Has dependencies, no outputs (e.g., model registration)
    SINGULAR = "singular" # No dependencies, no outputs (e.g., standalone operations)
```

## Key Features

### 1. Pydantic V2 Integration
- **Type Safety** - Strong typing with automatic validation
- **Field Validation** - Custom validators for each field
- **Model Validation** - Cross-field validation rules
- **JSON Schema** - Automatic schema generation

### 2. Declarative Dependencies
- **Logical Names** - Human-readable dependency references
- **Semantic Keywords** - Keywords for intelligent matching
- **Compatible Sources** - Explicit compatibility declarations
- **Required/Optional** - Flexible dependency requirements

### 3. Runtime Property Resolution
- **Property Paths** - SageMaker property path specifications
- **Lazy Evaluation** - PropertyReference for runtime resolution
- **Type Conversion** - Automatic SageMaker Properties generation

### 4. Output Aliases System
- **Backward Compatibility** - Multiple names for the same output
- **Alias Resolution** - Automatic lookup by name or alias
- **Clean Architecture** - Consolidate duplicate outputs using aliases
- **Migration Support** - Gradual transition from legacy naming

### 5. Script Contract Validation
- **Contract Alignment** - Validate specification matches script contract
- **Input/Output Verification** - Ensure all contract paths are covered
- **Automatic Validation** - Built-in validation methods
- **Error Reporting** - Detailed alignment error messages

## Usage Examples

### Creating Dependency Specifications
```python
from src.pipeline_deps.base_specifications import DependencySpec, DependencyType

# Define a training data dependency
training_data_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    required=True,
    compatible_sources=["DataLoadingStep", "PreprocessingStep"],
    semantic_keywords=["data", "dataset", "training"],
    data_type="S3Uri",
    description="Training dataset for model training"
)
```

### Creating Output Specifications
```python
from src.pipeline_deps.base_specifications import OutputSpec, DependencyType

# Define a model artifacts output with aliases
model_output = OutputSpec(
    logical_name="model_output",
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    data_type="S3Uri",
    description="Trained model artifacts",
    aliases=["ModelArtifacts", "model_data", "output_path", "model_input"]
)
```

### Using Output Aliases
```python
# Access output by primary name
primary_output = step_spec.get_output("model_output")

# Access output by alias
alias_output = step_spec.get_output_by_name_or_alias("ModelArtifacts")
legacy_output = step_spec.get_output_by_name_or_alias("model_data")

# All references point to the same output
assert primary_output == alias_output == legacy_output

# List all available names for an output
all_names = [model_output.logical_name] + model_output.aliases
print(f"Available names: {all_names}")
# Output: ['model_output', 'ModelArtifacts', 'model_data', 'output_path', 'model_input']
```

### Script Contract Integration
```python
from src.pipeline_script_contracts.base_script_contract import ScriptContract

# Define script contract
contract = ScriptContract(
    script_name="train.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data"
    }
)

# Create specification with contract reference
step_spec = StepSpecification(
    step_type="TrainingStep",
    node_type=NodeType.INTERNAL,
    script_contract=contract,
    dependencies=[input_dep, config_dep],
    outputs=[model_output, data_output]
)

# Validate contract alignment
result = step_spec.validate_contract_alignment()
if result.is_valid:
    print("✅ Contract and specification are aligned!")
else:
    print("❌ Alignment errors:")
    for error in result.errors:
        print(f"  - {error}")
```

### Creating Step Specifications
```python
from src.pipeline_deps.base_specifications import StepSpecification, NodeType

# Create a complete step specification
step_spec = StepSpecification(
    step_type="XGBoostTrainingStep",
    node_type=NodeType.INTERNAL,
    dependencies=[training_data_dep],
    outputs=[model_output]
)

# Access dependencies and outputs
print(f"Dependencies: {list(step_spec.dependencies.keys())}")
print(f"Outputs: {list(step_spec.outputs.keys())}")
```

### Using Property References
```python
from src.pipeline_deps.base_specifications import PropertyReference

# Create property reference for runtime resolution
prop_ref = PropertyReference(
    step_name="training_step",
    output_spec=model_output
)

# Convert to SageMaker property
sagemaker_prop = prop_ref.to_sagemaker_property()
print(sagemaker_prop)
# Output: {"Get": "Steps.training_step.properties.ModelArtifacts.S3ModelArtifacts"}
```

## Validation Features

### Field Validation
```python
# Automatic validation on creation
try:
    invalid_dep = DependencySpec(
        logical_name="",  # Invalid: empty name
        dependency_type="invalid_type"  # Invalid: not a valid enum
    )
except ValidationError as e:
    print(f"Validation errors: {e}")
```

### Node Type Validation
```python
# Node type constraints are automatically validated
try:
    invalid_spec = StepSpecification(
        step_type="SourceStep",
        node_type=NodeType.SOURCE,
        dependencies=[training_data_dep],  # Invalid: SOURCE nodes cannot have dependencies
        outputs=[]
    )
except ValidationError as e:
    print(f"Node type validation failed: {e}")
```

### Custom Validation Methods
```python
# Get required vs optional dependencies
required_deps = step_spec.list_required_dependencies()
optional_deps = step_spec.list_optional_dependencies()

# Get dependencies by type
training_deps = step_spec.list_dependencies_by_type(DependencyType.TRAINING_DATA)
model_deps = step_spec.list_outputs_by_type(DependencyType.MODEL_ARTIFACTS)
```

## Node Type Constraints

### SOURCE Nodes
- **No Dependencies** - Cannot have any dependencies
- **Must Have Outputs** - Must produce at least one output
- **Examples**: Data loading steps, configuration generators

### INTERNAL Nodes
- **Must Have Dependencies** - Requires at least one dependency
- **Must Have Outputs** - Must produce at least one output
- **Examples**: Processing steps, training steps, transformation steps

### SINK Nodes
- **Must Have Dependencies** - Requires at least one dependency
- **No Outputs** - Cannot produce any outputs
- **Examples**: Model registration, deployment steps, notification steps

### SINGULAR Nodes
- **No Dependencies** - Cannot have any dependencies
- **No Outputs** - Cannot produce any outputs
- **Examples**: Standalone operations, cleanup tasks

## Integration Points

### With Dependency Resolver
```python
from src.pipeline_deps import DependencyResolver

# Specifications provide input for dependency resolution
resolver = DependencyResolver()
dependencies = resolver.resolve_dependencies([step_spec])
```

### With Specification Registry
```python
from src.pipeline_deps import SpecificationRegistry

# Register specifications for reuse
registry = SpecificationRegistry()
registry.register_specification("xgboost_training", step_spec)
```

### With Pipeline Builder
```python
# Specifications drive automatic pipeline construction
pipeline_builder.add_step_specification(step_spec)
```

## Related Design Documentation

For architectural context and design decisions, see:
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy and motivation
- **[Step Specification Design](../pipeline_design/step_specification.md)** - Detailed step specification patterns
- **[Step Contract Design](../pipeline_design/step_contract.md)** - Contract-based step definitions
- **[Registry Manager Design](../pipeline_design/registry_manager.md)** - Registry management architecture
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles and guidelines
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions

## Related Script Contract Documentation

For script contract implementation and validation details, see:
- **[Base Script Contract](../pipeline_script_contracts/base_script_contract.md)** - Foundation classes for script contracts
- **[Contract Validator](../pipeline_script_contracts/contract_validator.md)** - Contract validation mechanisms
- **[PyTorch Train Contract](../pipeline_script_contracts/pytorch_train_contract.md)** - PyTorch training contract example
- **[Script Contracts Overview](../pipeline_script_contracts/README.md)** - Complete script contracts documentation

## Best Practices

### 1. Naming Conventions
- Use descriptive logical names: `training_data`, `model_artifacts`
- Follow snake_case convention for consistency
- Avoid abbreviations unless widely understood

### 2. Semantic Keywords
- Include relevant domain keywords for matching
- Use lowercase for consistency
- Include both specific and general terms

### 3. Property Paths
- Use exact SageMaker property paths
- Test property paths with actual SageMaker steps
- Document property path structure for complex outputs

### 4. Dependency Types
- Choose appropriate dependency types for semantic matching
- Use CUSTOM_PROPERTY for domain-specific dependencies
- Be consistent across related steps

### 5. Validation
- Always validate specifications after creation
- Handle ValidationError exceptions appropriately
- Use model validation for complex business rules

## Error Handling

### Common Validation Errors
```python
from pydantic import ValidationError

try:
    spec = StepSpecification(
        step_type="",  # Empty step type
        node_type="invalid",  # Invalid node type
        dependencies={"dep1": "not_a_spec"}  # Invalid dependency spec
    )
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}, Error: {error['msg']}")
```

### Custom Error Handling
```python
def create_safe_specification(step_type: str, **kwargs) -> Optional[StepSpecification]:
    """Safely create specification with error handling."""
    try:
        return StepSpecification(step_type=step_type, **kwargs)
    except ValidationError as e:
        logger.error(f"Failed to create specification for {step_type}: {e}")
        return None
```

## Migration from Legacy Systems

### Backward Compatibility
The StepSpecification constructor supports both list and dict formats:

```python
# List format (legacy)
spec = StepSpecification(
    step_type="ProcessingStep",
    node_type=NodeType.INTERNAL,
    dependencies=[dep1, dep2],
    outputs=[out1, out2]
)

# Dict format (internal)
spec = StepSpecification(
    step_type="ProcessingStep", 
    node_type=NodeType.INTERNAL,
    dependencies={"dep1": dep1, "dep2": dep2},
    outputs={"out1": out1, "out2": out2}
)
```

### Legacy Validation Method
```python
# Legacy validate() method is maintained for compatibility
errors = step_spec.validate()
if errors:
    print(f"Validation errors: {errors}")
