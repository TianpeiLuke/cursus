---
tags:
  - test
  - builders
  - base
  - validation
  - framework
keywords:
  - base test class
  - universal step builder test
  - test framework foundation
  - mock factory integration
  - step info detection
  - validation infrastructure
  - test environment setup
topics:
  - test framework architecture
  - validation base classes
  - mock environment setup
  - step builder testing
language: python
date of note: 2025-08-18
---

# Base Test Framework for Universal Step Builder Tests

## Overview

The Base Test Framework provides the foundational infrastructure for universal step builder testing. The `UniversalStepBuilderTestBase` class serves as the abstract base class for all step builder test suites, providing common setup, utility methods, and mock environment configuration.

## Architecture

### Core Base Class: UniversalStepBuilderTestBase

The `UniversalStepBuilderTestBase` class provides the foundation for all step builder tests:

```python
class UniversalStepBuilderTestBase(ABC):
    """
    Base class for universal step builder tests.
    
    This class provides common setup and utility methods for testing step builders.
    Specific test suites inherit from this class to add their test methods.
    """
    
    def __init__(
        self, 
        builder_class: Type[StepBuilderBase],
        config: Optional[ConfigBase] = None,
        spec: Optional[StepSpecification] = None,
        contract: Optional[ScriptContract] = None,
        step_name: Optional[Union[str, StepName]] = None,
        verbose: bool = False,
        test_reporter: Optional[Callable] = None,
        **kwargs
    ):
```

### Key Components Integration

#### Step Info Detection
```python
# Detect step information using new detector
self.step_info_detector = StepInfoDetector(builder_class)
self.step_info = self.step_info_detector.detect_step_info()
```

#### Mock Factory Integration
```python
# Create mock factory based on step info
self.mock_factory = StepTypeMockFactory(self.step_info)
```

## Key Features

### Abstract Method Requirements

All test suites inheriting from the base class must implement:

```python
@abstractmethod
def get_step_type_specific_tests(self) -> List[str]:
    """Return step type-specific test methods."""
    pass

@abstractmethod
def _configure_step_type_mocks(self) -> None:
    """Configure step type-specific mock objects."""
    pass

@abstractmethod
def _validate_step_type_requirements(self) -> Dict[str, Any]:
    """Validate step type-specific requirements."""
    pass
```

### Comprehensive Mock Environment Setup

The base class provides extensive mock environment configuration:

#### SageMaker Session Mocking
```python
def _setup_test_environment(self) -> None:
    """Set up mock objects and test fixtures."""
    # Mock SageMaker session
    self.mock_session = MagicMock()
    self.mock_session.boto_session.client.return_value = MagicMock()
    self.mock_session.boto_region_name = 'us-east-1'
    
    # Configure S3-related methods to return strings, not MagicMock
    self.mock_session.default_bucket.return_value = 'test-bucket'
    self.mock_session.default_bucket_prefix = 'test-prefix'
```

#### S3 Operations Mocking
```python
# Mock S3 upload methods to prevent actual S3 operations
self.mock_session.upload_data.return_value = 's3://test-bucket/uploaded-code/model.tar.gz'

# Mock boto3 S3 client methods
mock_s3_client = MagicMock()
mock_s3_client.head_object.return_value = {'ContentLength': 1024}
mock_s3_client.download_file.return_value = None
self.mock_session.boto_session.client.return_value = mock_s3_client
```

#### File System Operations Mocking
```python
# Create a temporary directory that actually exists
self.temp_dir = tempfile.mkdtemp()

# Create a mock model tar.gz file
mock_model_file = os.path.join(self.temp_dir, 'model.tar.gz')
with open(mock_model_file, 'wb') as f:
    f.write(b'mock model data')
```

### Processor Run Mock Configuration

Critical fix for Processing step Pattern B validation:

```python
def _setup_processor_run_mock(self) -> None:
    """
    Configure processor.run() mock to return proper step arguments.
    
    This fixes the issue where processor.run() returns None, causing
    ProcessingStep creation to fail with "either step_args or processor 
    need to be given, but not both" error.
    """
    # Create a proper _StepArguments object that ProcessingStep expects
    mock_step_args_dict = {
        'ProcessingJobName': 'test-processing-job',
        'ProcessingResources': {
            'ClusterConfig': {
                'InstanceType': 'ml.m5.large',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            }
        },
        'RoleArn': 'arn:aws:iam::123456789012:role/MockRole',
        # ... additional configuration
    }
```

### Builder Instance Creation

Flexible builder instance creation with mock or provided components:

```python
def _create_builder_instance(self) -> StepBuilderBase:
    """Create a builder instance with mock configuration."""
    # Use provided config or create mock configuration
    config = self._provided_config if self._provided_config else self._create_mock_config()
    
    # Create builder instance
    builder = self.builder_class(
        config=config,
        sagemaker_session=self.mock_session,
        role=self.mock_role,
        registry_manager=self.mock_registry_manager,
        dependency_resolver=self.mock_dependency_resolver
    )
```

### Mock Dependencies Creation

Comprehensive mock dependency creation with proper property mapping:

```python
def _create_mock_dependencies(self) -> List[Step]:
    """Create mock dependencies for the builder with enhanced property mapping."""
    dependencies = []
    expected_deps = self._get_expected_dependencies()
    
    for i, dep_name in enumerate(expected_deps):
        step = MagicMock()
        step.name = f"Mock{dep_name.capitalize()}Step"
        step.properties = MagicMock()
        
        # Configure properties based on dependency type
        if dep_name.lower() in ['data', 'input_data']:
            # Processing step output for data dependencies
            step.properties.ProcessingOutputConfig = MagicMock()
            # ... detailed configuration
        elif dep_name.lower() in ['model_artifacts']:
            # Model artifacts from training steps
            step.properties.ModelArtifacts = MagicMock()
            step.properties.ModelArtifacts.S3ModelArtifacts = f"s3://test-bucket/model-artifacts/{dep_name}"
```

## Validation Infrastructure

### Validation Levels and Violations

```python
class ValidationLevel(Enum):
    """Validation violation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

class ValidationViolation(BaseModel):
    """Represents a validation violation."""
    level: ValidationLevel
    category: str
    message: str
    details: str = ""
```

### Test Execution Framework

```python
def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
    """
    Run all tests in this test suite.
    
    Returns:
        Dictionary mapping test names to their results
    """
    # Get all methods that start with "test_"
    test_methods = [
        getattr(self, name) for name in dir(self) 
        if name.startswith('test_') and callable(getattr(self, name))
    ]
    
    results = {}
    for test_method in test_methods:
        results[test_method.__name__] = self._run_test(test_method)
```

### Assertion and Logging Framework

```python
def _assert(self, condition: bool, message: str) -> None:
    """Assert that a condition is true."""
    # Add assertion to list
    self.assertions.append((condition, message))
    
    # Log message if verbose
    if self.verbose and not condition:
        print(f"❌ FAILED: {message}")
    elif self.verbose and condition:
        print(f"✅ PASSED: {message}")

def _log(self, message: str) -> None:
    """Log a message if verbose."""
    if self.verbose:
        print(f"ℹ️ INFO: {message}")
```

### Test Result Handling

```python
def _run_test(self, test_method: Callable) -> Dict[str, Any]:
    """Run a single test method and capture results."""
    # Reset assertions
    self.assertions = []
    
    # Run test
    try:
        # Log test start
        self._log(f"Running {test_method.__name__}...")
        
        # Run test method
        test_method()
        
        # Check if any assertions failed
        failed = [msg for cond, msg in self.assertions if not cond]
        
        # Return result
        if failed:
            return {
                "passed": False,
                "name": test_method.__name__,
                "error": "\n".join(failed)
            }
        else:
            return {
                "passed": True,
                "name": test_method.__name__,
                "assertions": len(self.assertions)
            }
    except Exception as e:
        # Return error result
        return {
            "passed": False,
            "name": test_method.__name__,
            "error": str(e),
            "exception": e
        }
```

## Mock Configuration Patterns

### Step Type-Specific Mock Creation

The base class uses the mock factory to create appropriate configurations:

```python
def _create_mock_config(self) -> SimpleNamespace:
    """Create a mock configuration for the builder using the factory."""
    # Use the mock factory to create step type-specific config
    return self.mock_factory.create_mock_config()

def _create_invalid_config(self) -> SimpleNamespace:
    """Create an invalid configuration for testing error handling."""
    # Create a minimal config without required attributes
    mock_config = SimpleNamespace()
    mock_config.region = 'NA'  # Include only the region
    return mock_config
```

### Dependency Resolution Patterns

```python
def _get_required_dependencies_from_spec(self, builder: StepBuilderBase) -> List[str]:
    """Extract required dependency logical names from builder specification."""
    required_deps = []
    
    if hasattr(builder, 'spec') and builder.spec and hasattr(builder.spec, 'dependencies'):
        for _, dependency_spec in builder.spec.dependencies.items():
            if dependency_spec.required:
                required_deps.append(dependency_spec.logical_name)
    
    return required_deps

def _create_mock_inputs_for_builder(self, builder: StepBuilderBase) -> Dict[str, str]:
    """Create mock inputs dictionary for a builder based on its required dependencies."""
    mock_inputs = {}
    
    # Get required dependencies from specification
    required_deps = self._get_required_dependencies_from_spec(builder)
    
    # Generate mock S3 URIs for each required dependency
    for logical_name in required_deps:
        mock_inputs[logical_name] = self._generate_mock_s3_uri(logical_name)
    
    return mock_inputs
```

### S3 URI Generation

```python
def _generate_mock_s3_uri(self, logical_name: str) -> str:
    """Generate a mock S3 URI for a given logical dependency name."""
    # Create appropriate S3 URI based on dependency type
    if logical_name.lower() in ['data', 'input_data']:
        uri = f"s3://test-bucket/processing-data/{logical_name}"
    elif logical_name.lower() in ['input_path']:
        uri = f"s3://test-bucket/training-data/{logical_name}"
    elif logical_name.lower() in ['model_input', 'model_artifacts', 'model_data']:
        uri = f"s3://test-bucket/model-artifacts/{logical_name}"
    elif logical_name.lower() in ['hyperparameters_s3_uri']:
        uri = f"s3://test-bucket/hyperparameters/{logical_name}/hyperparameters.json"
    else:
        uri = f"s3://test-bucket/generic/{logical_name}"
    
    # Ensure we return an actual string, not a MagicMock
    return str(uri)
```

## Advanced Mock Features

### Tarfile Operations Mocking

Critical for model artifact handling:

```python
def mock_tarfile_open(name, mode='r', **kwargs):
    # If it's our mock tar file, return a mock that can extract properly
    if name and (name.endswith('.tar.gz') or 'tar_file' in str(name)):
        mock_tar = MagicMock()
        
        # Mock extractall to create expected files
        def mock_extractall(path=None, members=None, numeric_owner=False, filter=None):
            if path:
                os.makedirs(path, exist_ok=True)
                # Create expected model files
                model_py = os.path.join(path, 'model.py')
                with open(model_py, 'w') as f:
                    f.write('# Mock model file\nprint("Mock model loaded")\n')
                
                requirements_txt = os.path.join(path, 'requirements.txt')
                with open(requirements_txt, 'w') as f:
                    f.write('torch>=1.0.0\n')
        
        mock_tar.extractall = mock_extractall
        return mock_tar
```

### Property Path Resolution

```python
def _get_property_path_for_dependency(self, dep_name: str) -> str:
    """Get the appropriate property path for a dependency based on its type."""
    if dep_name.lower() in ['data', 'input_data', 'model_input']:
        return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
    elif dep_name.lower() in ['input_path']:
        return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
    elif dep_name.lower() in ['model_artifacts']:
        return f"ModelArtifacts.S3ModelArtifacts"
    else:
        return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"

def _infer_step_type_from_dependency(self, dep_name: str) -> str:
    """Infer the step type that would produce this dependency."""
    if dep_name.lower() in ['data', 'input_data']:
        return 'Processing'
    elif dep_name.lower() in ['input_path']:
        return 'Processing'  # Training data usually comes from processing
    elif dep_name.lower() in ['model_input', 'model_artifacts']:
        return 'Training'  # Model artifacts come from training
    else:
        return 'Processing'  # Default to processing
```

## Integration Points

### With Step Info Detector
- Uses `StepInfoDetector` to automatically detect step information
- Provides step type classification and metadata extraction
- Enables step type-specific mock configuration

### With Mock Factory
- Integrates with `StepTypeMockFactory` for step type-specific mocks
- Provides appropriate mock configurations based on detected step type
- Handles dependency resolution and mock creation

### With Test Suites
- Serves as base class for all test suite implementations
- Provides common infrastructure and utilities
- Enables consistent test execution and reporting

## Usage Patterns

### Basic Test Suite Implementation

```python
class MyTestSuite(UniversalStepBuilderTestBase):
    """Custom test suite extending the base class."""
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return step type-specific test methods."""
        return ['test_custom_functionality']
    
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects."""
        # Configure custom mocks
        pass
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate step type-specific requirements."""
        return {"custom_validation": True}
    
    def test_custom_functionality(self):
        """Test custom functionality."""
        builder = self._create_builder_instance()
        self._assert(builder is not None, "Builder should be created")
```

### Error Handling Testing

```python
def test_error_handling(self):
    """Test error handling with invalid configuration."""
    invalid_config = self._create_invalid_config()
    
    with self._assert_raises(ValueError):
        builder = self.builder_class(
            config=invalid_config,
            sagemaker_session=self.mock_session,
            role=self.mock_role
        )
        builder.create_step()
```

## Best Practices

### Mock Configuration
- Use step type-specific mock factories for appropriate configurations
- Ensure S3 operations return strings, not MagicMock objects
- Configure comprehensive property mappings for dependencies

### Test Implementation
- Implement all abstract methods in test suite subclasses
- Use assertion framework for consistent test result tracking
- Leverage verbose logging for debugging and development

### Error Handling
- Test both success and failure scenarios
- Use context managers for exception testing
- Provide clear error messages in assertions

## Conclusion

The Base Test Framework provides a robust foundation for universal step builder testing. Through comprehensive mock environment setup, flexible configuration options, and integration with step detection and mock factory components, it enables consistent and reliable testing across all step builder implementations.

The framework's abstract base class pattern ensures that all test suites follow consistent patterns while allowing for step type-specific customization and validation requirements.
