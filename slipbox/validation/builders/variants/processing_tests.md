---
tags:
  - test
  - validation
  - builders
  - processing
  - sagemaker
keywords:
  - processing step validation
  - sagemaker processing
  - sklearn processor
  - xgboost processor
  - pattern validation
  - job type support
topics:
  - processing validation
  - step builder testing
  - sagemaker step types
  - processor validation
language: python
date of note: 2025-08-18
---

# Processing Step Builder Validation Tests

The Processing Step Builder Validation Tests provide comprehensive validation for Processing step builders using a specialized 4-level testing approach. These tests are specifically designed to validate Processing-specific patterns, processor types, and SageMaker Processing step requirements.

## Overview

The `ProcessingStepBuilderTest` class extends the `UniversalStepBuilderTest` to provide Processing-specific testing capabilities. It validates Processing step builders against SageMaker Processing step requirements and framework-specific patterns.

## Processing-Specific Features

### Processor Types

Processing step builders support two main processor types:

#### SKLearnProcessor (Pattern A)
- **Direct ProcessingStep Creation**: Creates ProcessingStep directly
- **Framework**: Scikit-learn based processing
- **Usage**: General data processing, feature engineering, model evaluation

#### XGBoostProcessor (Pattern B)
- **processor.run() + step_args**: Uses processor.run() with step arguments
- **Framework**: XGBoost-specific processing
- **Usage**: XGBoost model training, evaluation, and inference

### Job Type Support

Processing builders support multiple job types:

- **training**: Training data processing
- **validation**: Validation data processing
- **testing**: Test data processing
- **calibration**: Model calibration processing

### Special Features

#### Local Path Override
- **Purpose**: Package step pattern for inference scripts
- **Usage**: Override S3 paths with local paths for testing
- **Pattern**: Used in package and payload steps

#### File Upload
- **Purpose**: DummyTraining step pattern for model/config upload
- **Usage**: Upload local files to processing containers
- **Pattern**: Used in training and model steps

#### S3 Path Validation
- **Purpose**: S3 URI normalization and validation
- **Usage**: Ensure S3 paths are properly formatted
- **Pattern**: Used across all processing steps

## Test Architecture

### Level 1: Processing Interface Tests

**Purpose**: Validates Processing-specific interface compliance

**Key Validations**:
- Processor creation methods (`_create_processor`, `_get_processor`)
- Framework-specific attributes and methods
- Step creation pattern compliance
- Inheritance from Processing base classes

**Test Class**: `ProcessingInterfaceTests`

### Level 2: Processing Specification Tests

**Purpose**: Validates Processing specification and contract compliance

**Key Validations**:
- Job type-based specification loading
- Environment variable handling for Processing
- Processor configuration validation
- Contract alignment for Processing patterns

**Test Class**: `ProcessingSpecificationTests`

### Level 3: Processing Step Creation Tests

**Purpose**: Validates Processing path mapping and property path validation

**Key Validations**:
- ProcessingInput/ProcessingOutput creation
- Container path mapping from contracts
- S3 path validation and normalization
- Special input patterns (local paths, file uploads)

**Test Class**: `ProcessingStepCreationTests`

### Level 4: Processing Integration Tests

**Purpose**: Validates Processing integration and end-to-end workflow

**Key Validations**:
- Complete step creation workflow
- Pattern A/B validation
- Dependency resolution for Processing steps
- Integration with SageMaker Processing service

**Test Class**: `ProcessingIntegrationTests`

## Class Interface

### Constructor

```python
def __init__(self, builder_class, step_info: Optional[Dict[str, Any]] = None, 
             enable_scoring: bool = False, enable_structured_reporting: bool = False):
    """
    Initialize Processing step builder test suite.
    
    Args:
        builder_class: The Processing step builder class to test
        step_info: Optional step information dictionary
        enable_scoring: Whether to enable scoring functionality
        enable_structured_reporting: Whether to enable structured reporting
    """
```

### Key Methods

#### run_processing_validation()

Runs Processing-specific validation tests:

```python
def run_processing_validation(self, levels: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Run Processing-specific validation tests.
    
    Args:
        levels: Optional list of test levels to run (1-4). If None, runs all levels.
        
    Returns:
        Dictionary containing test results and Processing-specific information
    """
```

**Usage Example**:
```python
from cursus.validation.builders.variants.processing_test import ProcessingStepBuilderTest
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Create Processing-specific tester
tester = ProcessingStepBuilderTest(
    builder_class=TabularPreprocessingStepBuilder,
    enable_scoring=True
)

# Run Processing validation
results = tester.run_processing_validation()

# Check Processing-specific information
processing_info = results['processing_info']
print(f"Processor types: {processing_info['supported_processors']}")
print(f"Creation patterns: {processing_info['creation_patterns']}")
```

#### validate_processor_type()

Validates the processor type used by the Processing builder:

```python
def validate_processor_type(self, expected_processor: str = None) -> Dict[str, Any]:
    """
    Validate the processor type used by the Processing builder.
    
    Args:
        expected_processor: Expected processor type ('SKLearnProcessor' or 'XGBoostProcessor')
        
    Returns:
        Validation results for processor type
    """
```

**Usage Example**:
```python
# Validate processor type
processor_results = tester.validate_processor_type(expected_processor='SKLearnProcessor')

if processor_results['processor_validation']:
    print(f"✅ Processor type: {processor_results['processor_type']}")
    print(f"✅ Creation pattern: {processor_results['creation_pattern']}")
else:
    print("❌ Processor validation failed:")
    for detail in processor_results['validation_details']:
        print(f"  - {detail}")
```

#### validate_job_type_support()

Validates job type support for multi-job-type Processing builders:

```python
def validate_job_type_support(self, job_types: List[str] = None) -> Dict[str, Any]:
    """
    Validate job type support for multi-job-type Processing builders.
    
    Args:
        job_types: List of job types to test. Defaults to common Processing job types.
        
    Returns:
        Job type support validation results
    """
```

**Usage Example**:
```python
# Validate job type support
job_type_results = tester.validate_job_type_support(['training', 'validation', 'testing'])

print(f"Supported job types: {job_type_results['supported_job_types']}")
print(f"Unsupported job types: {job_type_results['unsupported_job_types']}")
```

## Processing-Specific Information

### get_processing_specific_info()

Returns comprehensive Processing-specific information:

```python
def get_processing_specific_info(self) -> Dict[str, Any]:
    """Get Processing-specific information for reporting."""
```

**Information Structure**:
```python
{
    'step_type': 'Processing',
    'supported_processors': ['SKLearnProcessor', 'XGBoostProcessor'],
    'creation_patterns': {
        'pattern_a': 'Direct ProcessingStep creation (SKLearnProcessor)',
        'pattern_b': 'processor.run() + step_args (XGBoostProcessor)'
    },
    'job_type_support': 'Multi-job-type (training/validation/testing/calibration)',
    'special_features': {
        'local_path_override': 'Package step pattern for inference scripts',
        'file_upload': 'DummyTraining step pattern for model/config upload',
        's3_path_validation': 'S3 URI normalization and validation',
        'environment_variables': 'JSON serialization for complex configurations'
    },
    'container_paths': {
        'input_base': '/opt/ml/processing/input',
        'output_base': '/opt/ml/processing/output',
        'code_base': '/opt/ml/processing/input/code'
    }
}
```

## Validation Patterns

### Pattern A: Direct ProcessingStep Creation

Used by SKLearnProcessor-based builders:

```python
# Pattern A implementation
def build_step(self, config):
    processor = self._create_processor()  # Returns SKLearnProcessor
    
    # Create ProcessingStep directly
    step = ProcessingStep(
        name=self.get_step_name(),
        processor=processor,
        inputs=[...],
        outputs=[...],
        job_arguments=[...]
    )
    
    return step
```

**Validation Points**:
- `_create_processor()` returns `SKLearnProcessor`
- Direct `ProcessingStep` instantiation
- Proper inputs/outputs configuration
- Job arguments properly formatted

### Pattern B: processor.run() + step_args

Used by XGBoostProcessor-based builders:

```python
# Pattern B implementation
def build_step(self, config):
    processor = self._create_processor()  # Returns XGBoostProcessor
    
    # Use processor.run() with step_args
    step_args = processor.run(
        inputs=[...],
        outputs=[...],
        arguments=[...]
    )
    
    return step_args
```

**Validation Points**:
- `_create_processor()` returns `XGBoostProcessor`
- Uses `processor.run()` method
- Returns step arguments instead of ProcessingStep
- Proper argument formatting for XGBoost

## Container Path Mapping

### Standard Container Paths

Processing steps use standardized container paths:

```python
CONTAINER_PATHS = {
    'input_base': '/opt/ml/processing/input',
    'output_base': '/opt/ml/processing/output',
    'code_base': '/opt/ml/processing/input/code'
}

# Path mapping examples
{
    'training_data': '/opt/ml/processing/input/training',
    'validation_data': '/opt/ml/processing/input/validation',
    'processed_output': '/opt/ml/processing/output/processed',
    'model_artifacts': '/opt/ml/processing/output/model'
}
```

### Path Validation Rules

1. **Input Paths**: Must start with `/opt/ml/processing/input`
2. **Output Paths**: Must start with `/opt/ml/processing/output`
3. **Code Paths**: Must start with `/opt/ml/processing/input/code`
4. **S3 Normalization**: S3 URIs properly formatted and validated

## Environment Variable Handling

### Processing Environment Variables

Processing steps use specific environment variables:

```python
# Common Processing environment variables
PROCESSING_ENV_VARS = {
    'required': [
        'SM_MODEL_DIR',
        'SM_CHANNEL_TRAINING',
        'SM_OUTPUT_DATA_DIR'
    ],
    'optional': {
        'SM_CHANNEL_VALIDATION': '/opt/ml/processing/input/validation',
        'SM_CHANNEL_TESTING': '/opt/ml/processing/input/testing'
    }
}
```

### JSON Serialization

Complex configurations are serialized as JSON:

```python
# Environment variable JSON serialization
env_var_value = json.dumps({
    'hyperparameters': {...},
    'model_config': {...},
    'processing_config': {...}
})

os.environ['PROCESSING_CONFIG'] = env_var_value
```

## Convenience Functions

### validate_processing_builder()

Quick validation for Processing builders:

```python
def validate_processing_builder(builder_class, enable_scoring: bool = False, 
                              enable_structured_reporting: bool = False) -> Dict[str, Any]:
    """
    Convenience function to validate a Processing step builder.
    
    Returns:
        Comprehensive validation results
    """
```

**Usage Example**:
```python
from cursus.validation.builders.variants.processing_test import validate_processing_builder
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Quick validation
results = validate_processing_builder(
    TabularPreprocessingStepBuilder,
    enable_scoring=True,
    enable_structured_reporting=True
)

print(f"Validation passed: {results.get('validation_passed', False)}")
```

### validate_processor_type()

Quick processor type validation:

```python
def validate_processor_type(builder_class, expected_processor: str = None) -> Dict[str, Any]:
    """
    Convenience function to validate processor type for a Processing builder.
    
    Args:
        builder_class: The Processing step builder class to validate
        expected_processor: Expected processor type ('SKLearnProcessor' or 'XGBoostProcessor')
        
    Returns:
        Processor type validation results
    """
```

### validate_job_type_support()

Quick job type support validation:

```python
def validate_job_type_support(builder_class, job_types: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to validate job type support for a Processing builder.
    
    Args:
        builder_class: The Processing step builder class to validate
        job_types: List of job types to test
        
    Returns:
        Job type support validation results
    """
```

## Comprehensive Reporting

### generate_processing_report()

Generates comprehensive Processing validation reports:

```python
def generate_processing_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
    """
    Generate a comprehensive Processing step builder validation report.
    
    Args:
        include_recommendations: Whether to include improvement recommendations
        
    Returns:
        Comprehensive validation report
    """
```

**Report Structure**:
```python
{
    'builder_class': 'TabularPreprocessingStepBuilder',
    'test_suite': 'ProcessingStepBuilderTest',
    'validation_timestamp': '2025-08-18T00:42:00',
    'validation_results': {...},           # Full validation results
    'processor_validation': {...},         # Processor type validation
    'job_type_validation': {...},          # Job type support validation
    'processing_info': {...},              # Processing-specific information
    'recommendations': [...]               # Improvement recommendations
}
```

## Processing-Specific Recommendations

The test suite generates Processing-specific recommendations:

### Processor-Specific Recommendations
- Fix processor type validation issues
- Implement Pattern A validation for SKLearnProcessor
- Implement Pattern B validation for XGBoostProcessor

### Job Type Recommendations
- Add support for missing job types
- Improve job type-specific configuration handling

### Feature Recommendations
- Ensure proper S3 path validation and normalization
- Implement comprehensive environment variable handling
- Validate container path mapping from contracts
- Test both Pattern A and Pattern B step creation if applicable

## Integration with Universal Test

The Processing tests integrate seamlessly with the Universal Step Builder Test:

```python
# Processing tests are automatically used for Processing step builders
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

# For Processing builders, this automatically uses Processing-specific tests
tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder)
results = tester.run_all_tests()

# Processing-specific information is included
if 'processing_info' in results:
    print("Processing-specific validation performed")
```

## Best Practices

### For Processing Builder Developers

1. **Implement Correct Pattern**: Use Pattern A for SKLearnProcessor, Pattern B for XGBoostProcessor
2. **Support Multiple Job Types**: Implement support for training, validation, testing, calibration
3. **Validate Container Paths**: Ensure proper container path mapping
4. **Handle Environment Variables**: Implement proper environment variable handling
5. **Test Both Patterns**: Test both local and S3 path scenarios

### For Test Authors

1. **Use Processing-Specific Tests**: Use `ProcessingStepBuilderTest` for Processing builders
2. **Validate Processor Types**: Always validate the correct processor type
3. **Test Job Type Support**: Validate support for all relevant job types
4. **Check Special Features**: Test local path override, file upload, S3 validation
5. **Generate Comprehensive Reports**: Use full reporting for detailed analysis

## Common Processing Patterns

### Tabular Preprocessing Pattern

```python
# Typical tabular preprocessing builder
class TabularPreprocessingStepBuilder(ProcessingStepBuilderBase):
    def _create_processor(self):
        return SKLearnProcessor(  # Pattern A
            framework_version=self.config.processing_framework_version,
            instance_type=self._get_instance_type(),
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            role=self.role
        )
    
    def build_step(self, config):
        processor = self._create_processor()
        return ProcessingStep(  # Direct ProcessingStep creation
            name=self.get_step_name(),
            processor=processor,
            inputs=self._get_inputs(),
            outputs=self._get_outputs(),
            job_arguments=self._get_job_arguments()
        )
```

### XGBoost Processing Pattern

```python
# Typical XGBoost processing builder
class XGBoostProcessingStepBuilder(ProcessingStepBuilderBase):
    def _create_processor(self):
        return XGBoostProcessor(  # Pattern B
            framework_version=self.config.xgboost_framework_version,
            instance_type=self._get_instance_type(),
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            role=self.role
        )
    
    def build_step(self, config):
        processor = self._create_processor()
        return processor.run(  # processor.run() + step_args
            inputs=self._get_inputs(),
            outputs=self._get_outputs(),
            arguments=self._get_arguments()
        )
```

## Error Handling

### Common Processing Validation Errors

#### Processor Type Mismatch
```
ValidationError: Processor type mismatch. Expected: SKLearnProcessor, Got: XGBoostProcessor
```

#### Missing Job Type Support
```
ValidationError: Job type 'calibration' not supported
```

#### Invalid Container Path
```
ValidationError: Container path '/invalid/path' does not start with '/opt/ml/processing/'
```

#### Missing Processor Method
```
ValidationError: No _create_processor method found
```

### Error Resolution

1. **Check Processor Type**: Ensure correct processor type for the pattern
2. **Implement Job Type Support**: Add support for all required job types
3. **Validate Container Paths**: Use proper container path prefixes
4. **Implement Required Methods**: Ensure all required methods are implemented

## Performance Considerations

### Processing-Specific Optimizations

- **Processor Caching**: Cache processor instances for reuse
- **Path Validation Caching**: Cache S3 path validation results
- **Job Type Configuration**: Optimize job type-specific configuration loading
- **Container Path Mapping**: Efficient container path resolution

### Test Performance

- **Selective Testing**: Run only relevant test levels when debugging
- **Mock Usage**: Use mocks for expensive operations during testing
- **Batch Validation**: Validate multiple Processing builders together
- **Result Caching**: Cache validation results for repeated tests

## Future Enhancements

Planned improvements to Processing validation:

1. **Enhanced Pattern Detection**: Better detection of Pattern A vs Pattern B
2. **Advanced Job Type Support**: More sophisticated job type validation
3. **Container Optimization**: Validation of container resource optimization
4. **Framework Integration**: Better integration with ML framework validation
5. **Performance Metrics**: Detailed performance analysis for Processing steps

The Processing Step Builder Validation Tests provide comprehensive, specialized validation for Processing step builders, ensuring they meet SageMaker Processing requirements and follow established patterns for reliable, efficient data processing workflows.
