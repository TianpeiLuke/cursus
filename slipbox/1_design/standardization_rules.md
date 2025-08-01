# Standardization Rules

## What is the Purpose of Standardization Rules?

Standardization Rules serve as **enhanced architectural constraints** that enforce universal patterns and consistency across all pipeline components. They represent the evolution from basic design principles to comprehensive, enforceable standards that ensure system-wide coherence and quality.

## Core Purpose

Standardization Rules provide the **enhanced constraint enforcement layer** that:

1. **Universal Pattern Enforcement** - Ensure consistent patterns across all pipeline components
2. **Quality Gate Implementation** - Establish mandatory quality standards and validation rules
3. **Architectural Constraint Definition** - Define and enforce architectural boundaries and limitations
4. **Consistency Validation** - Provide automated checking of standardization compliance
5. **Evolution Governance** - Control how the system can evolve while maintaining standards

## Key Standardization Rules

### 1. Naming Conventions

**Rule**: All components must follow consistent naming conventions.

**Enforcement**: Automated validation during component registration.

```python
class NamingStandardValidator:
    """Enforce naming conventions across all components"""
    
    STEP_TYPE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')  # PascalCase
    LOGICAL_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]*$')  # snake_case
    CONFIG_CLASS_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*Config$')  # PascalCase + Config suffix
    BUILDER_CLASS_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*StepBuilder$')  # PascalCase + StepBuilder suffix
    
    def validate_step_specification(self, spec: StepSpecification):
        """Validate step specification naming"""
        errors = []
        
        # Validate step type naming
        if not self.STEP_TYPE_PATTERN.match(spec.step_type):
            errors.append(f"Step type '{spec.step_type}' must be PascalCase")
        
        # Validate dependency logical names
        for dep_name in spec.dependencies.keys():
            if not self.LOGICAL_NAME_PATTERN.match(dep_name):
                errors.append(f"Dependency name '{dep_name}' must be snake_case")
        
        # Validate output logical names
        for output_name in spec.outputs.keys():
            if not self.LOGICAL_NAME_PATTERN.match(output_name):
                errors.append(f"Output name '{output_name}' must be snake_case")
        
        return errors
    
    def validate_config_class(self, config_class: Type):
        """Validate configuration class naming"""
        class_name = config_class.__name__
        if not self.CONFIG_CLASS_PATTERN.match(class_name):
            raise StandardizationError(
                f"Config class '{class_name}' must follow pattern: PascalCaseConfig"
            )
    
    def validate_builder_class(self, builder_class: Type):
        """Validate builder class naming"""
        class_name = builder_class.__name__
        if not self.BUILDER_CLASS_PATTERN.match(class_name):
            raise StandardizationError(
                f"Builder class '{class_name}' must follow pattern: PascalCaseStepBuilder"
            )

# Centralized Registry
All naming standards are now enforced through the centralized `step_names.py` registry:

```python
# src/pipeline_registry/step_names.py
STEP_NAMES = {
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",           # For config registry
        "builder_step_name": "PyTorchTrainingStepBuilder", # For builder registry
        "spec_type": "PyTorchTraining",                    # For StepSpecification.step_type
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    # ... other steps
}
```

# Standard naming examples
```python
GOOD_NAMES = {
    "step_types": ["CradleDataLoading", "XGBoostTraining", "PyTorchTraining"],
    "logical_names": ["input_data", "model_artifacts", "processed_features"],
    "config_classes": ["CradleDataLoadConfig", "XGBoostTrainingConfig"],
    "builder_classes": ["CradleDataLoadingStepBuilder", "XGBoostTrainingStepBuilder"]
}

BAD_NAMES = {
    "step_types": ["cradle_data_loading", "xgboost_training", "PytorchTraining"],
    "logical_names": ["InputData", "model-artifacts", "processedFeatures"],
    "config_classes": ["CradleDataLoadingConfig", "XGBoostTrainingConfiguration"],
    "builder_classes": ["DataLoadingBuilder", "XGBoostTrainer"]
}
```

# Job Type Variants
Job type variants follow a consistent naming pattern:
```
{base_step_name}_{job_type}
```

Examples:
- `CradleDataLoading_training`
- `CradleDataLoading_calibration`
- `TabularPreprocessing_training`

### 2. Interface Standardization

**Rule**: All components must implement standardized interfaces.

**Enforcement**: Abstract base classes and interface validation.

```python
class InterfaceStandardValidator:
    """Enforce interface standardization"""
    
    REQUIRED_STEP_BUILDER_METHODS = [
        "get_specification",
        "build_step", 
        "validate_inputs",
        "get_output_reference"
    ]
    
    REQUIRED_CONFIG_METHODS = [
        "validate_configuration",
        "merge_with",
        "to_dict",
        "from_dict"
    ]
    
    def validate_step_builder_interface(self, builder_class: Type[BuilderStepBase]):
        """Validate step builder implements required interface"""
        errors = []
        
        for method_name in self.REQUIRED_STEP_BUILDER_METHODS:
            if not hasattr(builder_class, method_name):
                errors.append(f"Builder {builder_class.__name__} missing required method: {method_name}")
            else:
                method = getattr(builder_class, method_name)
                if not callable(method):
                    errors.append(f"Builder {builder_class.__name__}.{method_name} is not callable")
        
        # Validate get_specification returns StepSpecification
        if hasattr(builder_class, 'get_specification'):
            try:
                spec = builder_class.get_specification()
                if not isinstance(spec, StepSpecification):
                    errors.append(f"Builder {builder_class.__name__}.get_specification() must return StepSpecification")
            except Exception as e:
                errors.append(f"Builder {builder_class.__name__}.get_specification() failed: {e}")
        
        return errors
    
    def validate_config_interface(self, config_class: Type[ConfigBase]):
        """Validate config implements required interface"""
        errors = []
        
        for method_name in self.REQUIRED_CONFIG_METHODS:
            if not hasattr(config_class, method_name):
                errors.append(f"Config {config_class.__name__} missing required method: {method_name}")
        
        # Validate inheritance from ConfigBase
        if not issubclass(config_class, ConfigBase):
            errors.append(f"Config {config_class.__name__} must inherit from ConfigBase")
        
        return errors

# Standard interface enforcement
@dataclass
class StandardizedStepBuilder(BuilderStepBase):
    """Standardized base class enforcing interface compliance"""
    
    @classmethod
    @abstractmethod
    def get_specification(cls) -> StepSpecification:
        """REQUIRED: Return step specification"""
        pass
    
    @abstractmethod
    def build_step(self, inputs: Dict[str, Any]) -> Any:
        """REQUIRED: Build SageMaker step"""
        pass
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """REQUIRED: Validate inputs against specification"""
        spec = self.get_specification()
        errors = []
        
        # Standard validation logic
        for dep_name, dep_spec in spec.dependencies.items():
            if dep_spec.required and dep_name not in inputs:
                errors.append(f"Required input '{dep_name}' missing")
        
        return errors
    
    def get_output_reference(self, logical_name: str) -> Any:
        """REQUIRED: Get output reference by logical name"""
        spec = self.get_specification()
        if logical_name not in spec.outputs:
            raise ValueError(f"Output '{logical_name}' not found in specification")
        
        output_spec = spec.outputs[logical_name]
        return self._resolve_property_path(output_spec.property_path)
```

### 3. Documentation Standards

**Rule**: All components must have comprehensive, standardized documentation.

**Enforcement**: Documentation validation and auto-generation.

```python
class DocumentationStandardValidator:
    """Enforce documentation standards"""
    
    REQUIRED_DOCSTRING_SECTIONS = [
        "Purpose",
        "Parameters", 
        "Returns",
        "Raises",
        "Example"
    ]
    
    def validate_class_documentation(self, cls: Type):
        """Validate class has required documentation"""
        errors = []
        
        if not cls.__doc__:
            errors.append(f"Class {cls.__name__} missing docstring")
            return errors
        
        docstring = cls.__doc__.strip()
        
        # Check for purpose section
        if "Purpose:" not in docstring and "What is the purpose" not in docstring:
            errors.append(f"Class {cls.__name__} docstring missing Purpose section")
        
        # Check for example section
        if "Example:" not in docstring and "Usage:" not in docstring:
            errors.append(f"Class {cls.__name__} docstring missing Example section")
        
        return errors
    
    def validate_method_documentation(self, method: callable):
        """Validate method has required documentation"""
        errors = []
        
        if not method.__doc__:
            errors.append(f"Method {method.__name__} missing docstring")
            return errors
        
        docstring = method.__doc__.strip()
        
        # Check for parameter documentation
        if "Parameters:" not in docstring and "Args:" not in docstring:
            errors.append(f"Method {method.__name__} missing parameter documentation")
        
        # Check for return documentation
        if "Returns:" not in docstring and "Return:" not in docstring:
            errors.append(f"Method {method.__name__} missing return documentation")
        
        return errors

# Standard documentation template
STANDARD_CLASS_DOCSTRING_TEMPLATE = '''
"""
Purpose: {purpose}

This class {detailed_description}

Key Features:
- {feature_1}
- {feature_2}
- {feature_3}

Integration:
- Works with: {integration_points}
- Depends on: {dependencies}

Example:
    ```python
    {example_code}
    ```

See Also:
    {related_components}
"""
'''

STANDARD_METHOD_DOCSTRING_TEMPLATE = '''
"""
{brief_description}

Parameters:
    {parameter_name} ({parameter_type}): {parameter_description}
    {parameter_name} ({parameter_type}, optional): {parameter_description}

Returns:
    {return_type}: {return_description}

Raises:
    {exception_type}: {exception_description}

Example:
    ```python
    {example_code}
    ```
"""
'''
```

### 4. Error Handling Standards

**Rule**: All components must implement standardized error handling.

**Enforcement**: Error handling validation and standard exception hierarchy.

```python
# Standard exception hierarchy
class PipelineError(Exception):
    """Base exception for all pipeline errors"""
    pass

class ValidationError(PipelineError):
    """Raised when validation fails"""
    pass

class ConnectionError(PipelineError):
    """Raised when step connection fails"""
    pass

class ConfigurationError(PipelineError):
    """Raised when configuration is invalid"""
    pass

class SpecificationError(PipelineError):
    """Raised when specification is invalid"""
    pass

class StandardizationError(PipelineError):
    """Raised when standardization rules are violated"""
    pass

class ErrorHandlingStandardValidator:
    """Enforce error handling standards"""
    
    REQUIRED_ERROR_ATTRIBUTES = ["message", "error_code", "suggestions"]
    
    def validate_error_handling(self, method: callable):
        """Validate method implements standard error handling"""
        errors = []
        
        # Check if method has proper exception documentation
        if method.__doc__:
            docstring = method.__doc__
            if "Raises:" not in docstring:
                errors.append(f"Method {method.__name__} missing exception documentation")
        
        return errors
    
    def validate_exception_class(self, exception_class: Type[Exception]):
        """Validate exception class follows standards"""
        errors = []
        
        # Check inheritance from PipelineError
        if not issubclass(exception_class, PipelineError):
            errors.append(f"Exception {exception_class.__name__} must inherit from PipelineError")
        
        # Check for required attributes
        for attr in self.REQUIRED_ERROR_ATTRIBUTES:
            if not hasattr(exception_class, attr):
                errors.append(f"Exception {exception_class.__name__} missing attribute: {attr}")
        
        return errors

# Standard error handling pattern
class StandardizedError(PipelineError):
    """Standardized error with required attributes"""
    
    def __init__(self, message: str, error_code: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.suggestions = suggestions or []
    
    def _generate_error_code(self) -> str:
        """Generate standard error code"""
        class_name = self.__class__.__name__
        return f"{class_name.upper()}_{hash(self.message) % 10000:04d}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/serialization"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "suggestions": self.suggestions
        }

# Usage example with standard error handling
def connect_steps_with_standard_errors(source_step, target_step):
    """Connect steps with standardized error handling"""
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

**Rule**: All components must have comprehensive, standardized tests.

**Enforcement**: Test coverage validation and standard test patterns.

```python
class TestingStandardValidator:
    """Enforce testing standards"""
    
    REQUIRED_TEST_TYPES = [
        "unit_tests",
        "integration_tests", 
        "validation_tests",
        "error_handling_tests"
    ]
    
    MINIMUM_COVERAGE_THRESHOLD = 0.85  # 85% coverage required
    
    def validate_test_coverage(self, component_class: Type):
        """Validate component has required test coverage"""
        errors = []
        
        # Check for test file existence
        test_file_path = self._get_test_file_path(component_class)
        if not os.path.exists(test_file_path):
            errors.append(f"Missing test file for {component_class.__name__}: {test_file_path}")
        
        # Check coverage
        coverage = self._calculate_coverage(component_class)
        if coverage < self.MINIMUM_COVERAGE_THRESHOLD:
            errors.append(
                f"Test coverage for {component_class.__name__} is {coverage:.1%}, "
                f"minimum required: {self.MINIMUM_COVERAGE_THRESHOLD:.1%}"
            )
        
        return errors
    
    def validate_test_structure(self, test_class: Type):
        """Validate test class follows standard structure"""
        errors = []
        
        # Check for required test methods
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        required_patterns = [
            r'test_.*_success',      # Success case tests
            r'test_.*_validation',   # Validation tests
            r'test_.*_error',        # Error handling tests
        ]
        
        for pattern in required_patterns:
            if not any(re.match(pattern, method) for method in test_methods):
                errors.append(f"Test class {test_class.__name__} missing tests matching pattern: {pattern}")
        
        return errors

# Standard test base class
class StandardizedTestCase(unittest.TestCase):
    """Base class for standardized tests"""
    
    def setUp(self):
        """Standard test setup"""
        self.test_config = self._create_test_config()
        self.mock_dependencies = self._create_mock_dependencies()
    
    def tearDown(self):
        """Standard test cleanup"""
        self._cleanup_test_resources()
    
    def assert_validation_error(self, callable_obj, *args, **kwargs):
        """Standard assertion for validation errors"""
        with self.assertRaises(ValidationError) as context:
            callable_obj(*args, **kwargs)
        
        # Validate error has required attributes
        error = context.exception
        self.assertIsNotNone(error.message)
        self.assertIsNotNone(error.error_code)
        self.assertIsInstance(error.suggestions, list)
    
    def assert_specification_compliance(self, component, specification):
        """Standard assertion for specification compliance"""
        # Validate component implements specification
        errors = specification.validate_implementation(component)
        self.assertEqual([], errors, f"Specification compliance errors: {errors}")

# Standard test patterns
class TestXGBoostTrainingStepBuilder(StandardizedTestCase):
    """Example of standardized test structure"""
    
    def test_build_step_success(self):
        """Test successful step building"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        inputs = {"input_data": "s3://bucket/data"}
        
        step = builder.build_step(inputs)
        
        self.assertIsNotNone(step)
        self.assertEqual(step.name, self.test_config.step_name)
    
    def test_build_step_validation_missing_input(self):
        """Test validation error for missing required input"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        inputs = {}  # Missing required input
        
        self.assert_validation_error(builder.build_step, inputs)
    
    def test_build_step_error_invalid_config(self):
        """Test error handling for invalid configuration"""
        invalid_config = XGBoostTrainingStepConfig(instance_type="invalid")
        builder = XGBoostTrainingStepBuilder(invalid_config)
        
        with self.assertRaises(ConfigurationError):
            builder.build_step({"input_data": "s3://bucket/data"})
    
    def test_specification_compliance(self):
        """Test builder complies with specification"""
        builder = XGBoostTrainingStepBuilder(self.test_config)
        specification = builder.get_specification()
        
        self.assert_specification_compliance(builder, specification)
```

## Standardization Enforcement

### 1. Automated Validation Pipeline

```python
class StandardizationEnforcer:
    """Automated enforcement of standardization rules"""
    
    def __init__(self):
        self.validators = [
            NamingStandardValidator(),
            InterfaceStandardValidator(),
            DocumentationStandardValidator(),
            ErrorHandlingStandardValidator(),
            TestingStandardValidator()
        ]
    
    def validate_component(self, component_class: Type) -> ValidationResult:
        """Validate component against all standardization rules"""
        all_errors = []
        all_warnings = []
        
        for validator in self.validators:
            try:
                errors = validator.validate(component_class)
                all_errors.extend(errors)
            except Exception as e:
                all_warnings.append(f"Validator {validator.__class__.__name__} failed: {e}")
        
        return ValidationResult(
            component=component_class.__name__,
            errors=all_errors,
            warnings=all_warnings,
            is_compliant=len(all_errors) == 0
        )
    
    def enforce_standards_on_registration(self, registry: ComponentRegistry):
        """Enforce standards when components are registered"""
        original_register = registry.register_builder
        
        def validated_register(step_type: str, builder_class: Type[BuilderStepBase]):
            # Validate before registration
            result = self.validate_component(builder_class)
            if not result.is_compliant:
                raise StandardizationError(
                    f"Component {builder_class.__name__} fails standardization: {result.errors}"
                )
            
            # Proceed with registration
            return original_register(step_type, builder_class)
        
        registry.register_builder = validated_register

# Integration with CI/CD pipeline
class ContinuousStandardizationValidator:
    """Validate standardization in CI/CD pipeline"""
    
    def validate_pull_request(self, changed_files: List[str]) -> bool:
        """Validate changed components meet standards"""
        enforcer = StandardizationEnforcer()
        
        for file_path in changed_files:
            if self._is_component_file(file_path):
                component_classes = self._extract_component_classes(file_path)
                
                for component_class in component_classes:
                    result = enforcer.validate_component(component_class)
                    if not result.is_compliant:
                        print(f"❌ {component_class.__name__} fails standardization:")
                        for error in result.errors:
                            print(f"  - {error}")
                        return False
        
        print("✅ All components meet standardization requirements")
        return True
```

## Strategic Value

Standardization Rules provide:

1. **System-Wide Consistency**: Ensure uniform patterns across all components
2. **Quality Assurance**: Enforce mandatory quality standards automatically
3. **Maintainability**: Reduce cognitive load through consistent interfaces
4. **Onboarding Efficiency**: New developers can quickly understand standardized patterns
5. **Evolution Control**: Govern how the system can evolve while maintaining quality
6. **Automated Compliance**: Reduce manual review burden through automated validation

## Example Usage

```python
# Component development with standardization enforcement
@standardized_component
class DataLoadingStepBuilder(StandardizedStepBuilder):
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
        step = builder.build_step({})
        ```
    
    See Also:
        PreprocessingStepBuilder, DataLoadingStepConfig
    """
    
    @classmethod
    def get_specification(cls) -> StepSpecification:
        """Return the step specification for data loading."""
        return DATA_LOADING_SPEC
    
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
        # Standard validation
        validation_errors = self.validate_inputs(inputs)
        if validation_errors:
            raise ValidationError(
                message=f"Input validation failed: {validation_errors}",
                error_code="DATA_LOAD_001",
                suggestions=["Check input requirements in specification"]
            )
        
        # Build step with standard error handling
        try:
            return self._create_processing_step(inputs)
        except Exception as e:
            raise ConfigurationError(
                message=f"Failed to create data loading step: {str(e)}",
                error_code="DATA_LOAD_002",
                suggestions=["Verify configuration parameters", "Check AWS permissions"]
            ) from e

# Automatic validation during registration
registry = ComponentRegistry()
enforcer = StandardizationEnforcer()
enforcer.enforce_standards_on_registration(registry)

# This will automatically validate standardization compliance
registry.register_builder("DataLoading", DataLoadingStepBuilder)
```

Standardization Rules represent the **maturation of the architectural system** from flexible guidelines to enforceable standards that ensure system-wide quality, consistency, and maintainability while enabling controlled evolution and growth.
