# Standardization Rules

This document outlines the standardization rules that govern the development of pipeline components. These rules serve as enhanced architectural constraints that enforce universal patterns and consistency across all pipeline components.

## Purpose of Standardization Rules

Standardization Rules provide the enhanced constraint enforcement layer that:

1. **Universal Pattern Enforcement** - Ensure consistent patterns across all pipeline components
2. **Quality Gate Implementation** - Establish mandatory quality standards and validation rules
3. **Architectural Constraint Definition** - Define and enforce architectural boundaries and limitations
4. **Consistency Validation** - Provide automated checking of standardization compliance
5. **Evolution Governance** - Control how the system can evolve while maintaining standards

## Key Standardization Rules

### 1. Naming Conventions

All components must follow consistent naming conventions. These conventions are now centrally defined and enforced through the `step_names.py` registry, which serves as a single source of truth for step naming across the system.

| Component | Pattern | Examples | Counter-Examples |
|-----------|---------|----------|-----------------|
| Step Types | PascalCase | `CradleDataLoading`, `XGBoostTraining`, `PyTorchTraining` | `cradle_data_loading`, `xgboost_training`, `PytorchTraining` |
| Logical Names | snake_case | `input_data`, `model_artifacts` | `InputData`, `model-artifacts` |
| Config Classes | PascalCase + `Config` suffix | `CradleDataLoadConfig`, `XGBoostTrainingConfig` | `CradleDataLoadingConfiguration`, `XGBoostConfig` |
| Builder Classes | PascalCase + `StepBuilder` suffix | `CradleDataLoadingStepBuilder`, `XGBoostTrainingStepBuilder` | `DataLoadingBuilder`, `XGBoostStepBuilder` |

Additionally, all files must follow consistent naming patterns:

| File Type | Pattern | Examples | Counter-Examples |
|-----------|---------|----------|-----------------|
| Step Builder Files | `builder_xxx_step.py` | `builder_data_loading_step.py`, `builder_xgboost_training_step.py` | `DataLoadingStepBuilder.py`, `xgboost_step_builder.py` |
| Config Files | `config_xxx_step.py` | `config_data_loading_step.py`, `config_xgboost_training_step.py` | `DataLoadingConfig.py`, `xgboost_config.py` |
| Step Specification Files | `xxx_spec.py` | `data_loading_spec.py`, `xgboost_training_spec.py` | `DataLoadingSpecification.py`, `spec_xgboost.py` |
| Script Contract Files | `xxx_contract.py` | `data_loading_contract.py`, `xgboost_training_contract.py` | `DataLoadingContract.py`, `contract_xgboost.py` |

This consistency helps with:
- Auto-discovery of components
- Code navigation
- Understanding component relationships
- Automated validation

### 2. Interface Standardization

All components must implement standardized interfaces:

#### Step Builders

All step builders must:
- Inherit from `StepBuilderBase`
- Use the `@register_builder` decorator to register with the registry (or have their naming follow the standard pattern to be auto-discovered)
- Follow the strict naming convention `XXXStepBuilder` where XXX is the step type
- Implement the required methods:
  - `validate_configuration()`
  - `_get_inputs()`
  - `_get_outputs()`
  - `create_step()`

Example:

```python
from ..pipeline_registry.builder_registry import register_builder

@register_builder() # Step type will be auto-derived from class name (YourStepBuilder -> YourStep)
class YourStepBuilder(StepBuilderBase):
    """Builder for your processing step."""
    
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None):
        super().__init__(
            config=config,
            spec=YOUR_STEP_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config = config
    
    def validate_configuration(self):
        """Validate the configuration."""
        # Validation logic
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor."""
        # Input generation logic
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor."""
        # Output generation logic
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step."""
        # Step creation logic
```

#### Config Classes (Three-Tier Design)

All configuration classes must follow the three-tier field classification design:

1. **Tier 1 (Essential Fields)**:
   - Required inputs explicitly provided by users
   - No default values
   - Subject to validation
   - Public access
   - Example: `region: str = Field(..., description="Region code")`

2. **Tier 2 (System Fields)**:
   - Default values that can be overridden
   - Subject to validation
   - Public access
   - Example: `instance_type: str = Field(default="ml.m5.xlarge", description="Instance type")`

3. **Tier 3 (Derived Fields)**:
   - Private fields with leading underscores
   - Values calculated from other fields
   - Accessed through read-only properties
   - Example:
     ```python
     _pipeline_name: Optional[str] = Field(default=None, exclude=True)
     
     @property
     def pipeline_name(self) -> str:
         """Get derived pipeline name."""
         if self._pipeline_name is None:
             self._pipeline_name = f"{self.service_name}_{self.region}"
         return self._pipeline_name
     ```

All config classes must:
- Inherit from a base config class (e.g., `BasePipelineConfig`, `ProcessingStepConfigBase`)
- Use Pydantic for field declarations and validation
- Override `model_dump()` to include derived properties
- Implement required methods:
  - `get_script_contract()`
  - `get_script_path()` (for processing steps)
  - Additional getters as needed

Example:

```python
class YourStepConfig(BasePipelineConfig):
    """Configuration for your step."""
    
    # Tier 1: Essential fields
    region: str = Field(..., description="AWS region code")
    input_path: str = Field(..., description="Input data path")
    
    # Tier 2: System fields
    instance_type: str = Field(default="ml.m5.xlarge", description="Instance type")
    instance_count: int = Field(default=1, description="Number of instances")
    
    # Tier 3: Derived fields
    _output_path: Optional[str] = Field(default=None, exclude=True)
    
    @property
    def output_path(self) -> str:
        """Get output path based on input path."""
        if self._output_path is None:
            self._output_path = f"{self.input_path}/output"
        return self._output_path
    
    # Include derived fields in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        data["output_path"] = self.output_path
        return data
    
    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..pipeline_script_contracts.your_script_contract import YOUR_SCRIPT_CONTRACT
        return YOUR_SCRIPT_CONTRACT
    
    def get_script_path(self):
        """Return the path to the script."""
        return "your_script.py"
```

### 3. Documentation Standards

All components must have comprehensive, standardized documentation:

#### Class Documentation

All classes must include:
- Purpose description
- Key features
- Integration points
- Usage examples
- Related components

Example:

```python
class DataLoadingStepBuilder(StepBuilderBase):
    """
    Purpose: Build SageMaker processing steps for data loading operations.

    This builder creates ProcessingStep instances configured for data loading
    from various sources (S3, databases, APIs) with standardized output formats.

    Key Features:
    - Supports multiple data source types
    - Automatic schema validation
    - Standardized output formatting

    Integration:
    - Works with: PreprocessingStepBuilder, ValidationStepBuilder
    - Depends on: DataLoadingStepConfig, ProcessingStepFactory

    Example:
        ```python
        config = DataLoadingStepConfig(
            data_source="s3://bucket/data/",
            output_format="parquet"
        )
        builder = DataLoadingStepBuilder(config)
        step = builder.create_step({})
        ```

    See Also:
        PreprocessingStepBuilder, DataLoadingStepConfig
    """
```

#### Method Documentation

All methods must include:
- Brief description
- Parameters documentation
- Return value documentation
- Exception documentation
- Usage examples (for public methods)

Example:

```python
def build_step(self, inputs: Dict[str, Any]) -> ProcessingStep:
    """
    Build a SageMaker ProcessingStep for data loading.

    Parameters:
        inputs (Dict[str, Any]): Input parameters (typically empty for SOURCE steps)

    Returns:
        ProcessingStep: Configured SageMaker processing step

    Raises:
        ValidationError: If inputs don't meet specification requirements
        ConfigurationError: If configuration is invalid

    Example:
        ```python
        step = builder.build_step({})
        ```
    """
```

### 4. Error Handling Standards

All components must implement standardized error handling:

- Use the standard exception hierarchy
- Provide meaningful error messages
- Include error codes
- Add suggestions for resolution
- Log errors appropriately

Example:

```python
try:
    # Validation logic
    if not source_step:
        raise ValidationError(
            message="Source step cannot be None",
            error_code="CONN_001",
            suggestions=["Provide a valid source step instance"]
        )

    # Connection logic
    return create_connection(source_step, target_step)

except ValidationError:
    raise  # Re-raise validation errors as-is
except Exception as e:
    # Wrap unexpected errors in standard format
    raise ConnectionError(
        message=f"Unexpected error during step connection: {str(e)}",
        error_code="CONN_999",
        suggestions=["Check step compatibility", "Verify step specifications"]
    ) from e
```

### 5. Testing Standards

All components must have comprehensive, standardized tests:

- Unit tests for each component
- Integration tests for connected components
- Validation tests for specifications
- Error handling tests for edge cases
- Minimum test coverage threshold (85%)

Test classes should follow this structure:

```python
class TestYourStepBuilder(unittest.TestCase):
    """Tests for YourStepBuilder."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Setup code
    
    def test_build_step_success(self):
        """Test successful step building."""
        # Success test
    
    def test_build_step_validation_missing_input(self):
        """Test validation error for missing required input."""
        # Validation test
    
    def test_build_step_error_invalid_config(self):
        """Test error handling for invalid configuration."""
        # Error handling test
    
    def test_specification_compliance(self):
        """Test builder complies with specification."""
        # Compliance test
```

## Validation Tools

We provide tools to validate compliance with these standardization rules:

### Naming Convention Validation

```python
# Example validator usage
from src.tools.validation import NamingStandardValidator

validator = NamingStandardValidator()
errors = validator.validate_step_specification(YOUR_STEP_SPEC)
if errors:
    print("Naming convention violations:")
    for error in errors:
        print(f"  - {error}")
```

### Interface Validation

```python
# Example validator usage
from src.tools.validation import InterfaceStandardValidator

validator = InterfaceStandardValidator()
errors = validator.validate_step_builder_interface(YourStepBuilder)
if errors:
    print("Interface violations:")
    for error in errors:
        print(f"  - {error}")
```

### Documentation Validation

```python
# Example validator usage
from src.tools.validation import DocumentationStandardValidator

validator = DocumentationStandardValidator()
errors = validator.validate_class_documentation(YourStepBuilder)
if errors:
    print("Documentation violations:")
    for error in errors:
        print(f"  - {error}")
```

### Builder Registry Validation

```python
# Example registry validator usage
from src.pipeline_registry.builder_registry import get_global_registry

registry = get_global_registry()
validation = registry.validate_registry()

# Check validation results
print(f"Valid entries: {len(validation['valid'])}")
if validation['invalid']:
    print("Invalid entries:")
    for entry in validation['invalid']:
        print(f"  - {entry}")

if validation['missing']:
    print("Missing entries:")
    for entry in validation['missing']:
        print(f"  - {entry}")
```

### Job Type Handling

When working with step types that need to handle different job types (e.g., training, calibration), follow these patterns:

1. **Node Naming**: Use underscore suffix for job type variants:
   ```
   CradleDataLoading_training
   CradleDataLoading_calibration
   TabularPreprocessing_training
   ```

2. **Configuration Classes**: Job type should be a field in the config:
   ```python
   class CradleDataLoadConfig(BasePipelineConfig):
       job_type: str = Field(default="training", description="Job type (training, calibration)")
   ```

3. **Builder Resolution**: The builder registry will automatically resolve job types:
   ```python
   # This will resolve to CradleDataLoadingStepBuilder even though node name has _training suffix
   builder = registry.get_builder_for_config(config, node_name="CradleDataLoading_training") 
   ```

## Integration with Development Process

These standardization rules should be integrated into your development process:

1. **Initial Development**: Use as a reference when creating new components
2. **Pre-Commit Validation**: Run validation tools before committing code
3. **Code Review**: Include rule compliance in code review checklist
4. **Continuous Integration**: Add rule validation to CI/CD pipelines
5. **Documentation**: Include rule compliance in your documentation

By following these standardization rules, you'll contribute to a cohesive, maintainable pipeline architecture that is easier to understand, extend, and troubleshoot.

## See Also

- [Design Principles](design_principles.md)
- [Best Practices](best_practices.md)
- [Alignment Rules](alignment_rules.md)
- [Validation Checklist](validation_checklist.md)
