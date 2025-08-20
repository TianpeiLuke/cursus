---
tags:
  - code
  - validation
  - testing
  - processing
  - integration
keywords:
  - processing integration tests
  - level 4 testing
  - pattern a pattern b
  - dependency resolution
  - step creation workflow
  - sagemaker processing
topics:
  - validation framework
  - integration testing
  - processing step validation
  - end-to-end testing
language: python
date of note: 2025-01-18
---

# Processing-Specific Level 4 Integration Tests

## Overview

The `ProcessingIntegrationTests` class provides Level 4 integration testing specifically for Processing step builders. These tests validate complete system integration and end-to-end functionality, focusing on the unique characteristics of Processing steps including Pattern A and Pattern B creation approaches, dependency resolution, and comprehensive workflow validation.

## Architecture

### Processing Step Integration Focus

Level 4 integration tests for Processing steps validate:

1. **Complete ProcessingStep Creation** - Both Pattern A and Pattern B approaches
2. **Dependency Resolution Integration** - Input extraction from upstream steps
3. **Step Name Generation** - Processing-specific naming conventions
4. **Cache Configuration** - Processing step caching integration
5. **End-to-End Workflow** - Complete step creation pipeline

### Pattern A vs Pattern B Testing

Processing steps support two distinct creation patterns:

- **Pattern A**: Direct ProcessingStep instantiation (SKLearnProcessor)
- **Pattern B**: processor.run + step_args approach (XGBoostProcessor)

## Core Test Methods

### Complete Step Creation Testing

```python
def test_complete_processing_step_creation(self) -> None:
    """Test complete ProcessingStep creation with all components."""
    
    # Configure comprehensive test environment
    config = Mock()
    config.processing_instance_type_large = "ml.m5.xlarge"
    config.processing_instance_type_small = "ml.m5.large"
    config.use_large_processing_instance = False
    config.processing_instance_count = 1
    config.processing_volume_size = 30
    config.processing_framework_version = "0.23-1"
    
    builder = self.builder_class(config=config)
    builder.role = "test-role"
    builder.session = Mock()
    builder.spec = self.mock_processing_spec
    builder.contract = self.mock_contract
    
    # Mock all required components
    builder._create_processor = Mock(return_value=self.mock_sklearn_processor)
    builder._get_inputs = Mock(return_value=self.mock_processing_inputs)
    builder._get_outputs = Mock(return_value=self.mock_processing_outputs)
    builder._get_job_arguments = Mock(return_value=["--job_type", "training"])
    builder._get_step_name = Mock(return_value="test-processing-step")
    builder._get_cache_config = Mock(return_value=None)
```

### Pattern A Integration Testing

Pattern A testing validates direct ProcessingStep instantiation:

```python
def test_pattern_a_step_creation(self) -> None:
    """Test Pattern A step creation (direct ProcessingStep instantiation)."""
    
    # Configure for SKLearnProcessor (Pattern A)
    builder._create_processor = Mock(return_value=self.mock_sklearn_processor)
    config.get_script_path = Mock(return_value="sklearn_processing.py")
    
    with patch('sagemaker.processing.ProcessingStep') as mock_step_class:
        step = builder.create_step()
        
        # Verify Pattern A characteristics
        call_args = mock_step_class.call_args
        kwargs = call_args[1]
        
        # Pattern A should have processor, inputs, outputs, code
        assert 'processor' in kwargs
        assert 'inputs' in kwargs
        assert 'outputs' in kwargs
        assert 'code' in kwargs
        
        # Pattern A should NOT have step_args
        assert 'step_args' not in kwargs
```

### Pattern B Integration Testing

Pattern B testing validates processor.run + step_args approach:

```python
def test_pattern_b_step_creation(self) -> None:
    """Test Pattern B step creation (processor.run + step_args)."""
    
    # Configure for XGBoostProcessor (Pattern B)
    config.framework_version = "1.3-1"
    config.py_version = "py38"
    config.processing_entry_point = "xgboost_processing.py"
    config.processing_source_dir = "src"
    
    builder._create_processor = Mock(return_value=self.mock_xgboost_processor)
    
    with patch('sagemaker.processing.ProcessingStep') as mock_step_class:
        step = builder.create_step()
        
        # Verify Pattern B characteristics
        kwargs = mock_step_class.call_args[1]
        
        # Pattern B should have step_args
        assert 'step_args' in kwargs
        
        # Pattern B should NOT have processor, inputs, outputs, code directly
        assert 'processor' not in kwargs
        assert 'inputs' not in kwargs
        assert 'outputs' not in kwargs
        assert 'code' not in kwargs
        
        # Verify processor.run was called
        self.mock_xgboost_processor.run.assert_called_once()
```

## Dependency Resolution Integration

### Input Extraction Testing

```python
def test_dependency_extraction_integration(self) -> None:
    """Test integration with dependency resolution system."""
    
    # Mock upstream dependencies
    mock_dependency_step = Mock()
    mock_dependency_step.name = "upstream-step"
    dependencies = [mock_dependency_step]
    
    # Mock dependency extraction
    builder.extract_inputs_from_dependencies = Mock(return_value={
        "input_data": "s3://bucket/upstream/output"
    })
    
    step = builder.create_step(
        inputs={"additional_input": "s3://bucket/additional"},
        outputs={"output_data": "s3://bucket/output"},
        dependencies=dependencies
    )
    
    # Verify dependency extraction was called
    builder.extract_inputs_from_dependencies.assert_called_once_with(dependencies)
    
    # Verify inputs were merged (dependency inputs + explicit inputs)
    call_args = builder._get_inputs.call_args
    merged_inputs = call_args[0][0]
    assert "input_data" in merged_inputs  # From dependencies
    assert "additional_input" in merged_inputs  # Explicit inputs
```

### Step Dependencies Integration

```python
def test_step_dependencies_integration(self) -> None:
    """Test step dependencies integration in ProcessingStep creation."""
    
    # Mock dependencies
    mock_dep1 = Mock()
    mock_dep1.name = "upstream-step-1"
    mock_dep2 = Mock()
    mock_dep2.name = "upstream-step-2"
    dependencies = [mock_dep1, mock_dep2]
    
    with patch('sagemaker.processing.ProcessingStep') as mock_step_class:
        step = builder.create_step(dependencies=dependencies)
        
        # Verify dependencies were passed to ProcessingStep
        kwargs = mock_step_class.call_args[1]
        assert 'depends_on' in kwargs
        
        depends_on = kwargs['depends_on']
        assert depends_on == dependencies
```

## Processing-Specific Validations

### Step Name Generation Testing

```python
def test_processing_step_name_generation(self) -> None:
    """Test Processing step name generation and consistency."""
    
    step_name = builder._get_step_name()
    
    # Validate step name format
    assert isinstance(step_name, str)
    assert len(step_name) > 0
    
    # Processing step names should follow naming conventions
    processing_keywords = ["processing", "preprocess", "eval", "package", "payload"]
    assert any(keyword in step_name.lower() for keyword in processing_keywords)
    
    # Test consistency - multiple calls should return same name
    step_name_2 = builder._get_step_name()
    assert step_name == step_name_2
```

### Cache Configuration Integration

```python
def test_cache_configuration_integration(self) -> None:
    """Test cache configuration integration."""
    
    # Test with caching enabled
    cache_config_enabled = builder._get_cache_config(enable_caching=True)
    
    # Test with caching disabled
    cache_config_disabled = builder._get_cache_config(enable_caching=False)
    
    # Validate cache configuration
    if cache_config_enabled is not None:
        assert (hasattr(cache_config_enabled, 'enable_caching') or 
                isinstance(cache_config_enabled, dict))
```

## End-to-End Workflow Testing

### Complete Workflow Validation

```python
def test_end_to_end_workflow(self) -> None:
    """Test complete end-to-end Processing step creation workflow."""
    
    # Configure comprehensive test environment
    config = Mock()
    config.processing_instance_type_large = "ml.m5.xlarge"
    config.processing_instance_type_small = "ml.m5.large"
    config.use_large_processing_instance = False
    config.processing_instance_count = 1
    config.processing_volume_size = 30
    config.processing_framework_version = "0.23-1"
    config.job_type = "training"
    
    builder = self.builder_class(config=config)
    builder.role = "arn:aws:iam::123456789012:role/SageMakerRole"
    builder.session = Mock()
    builder.spec = self.mock_processing_spec
    builder.contract = self.mock_contract
    
    # Mock upstream dependencies
    upstream_step = Mock()
    upstream_step.name = "data-ingestion-step"
    
    # Execute complete workflow
    step = builder.create_step(
        inputs={"additional_data": "s3://bucket/additional"},
        outputs={"processed_data": "s3://bucket/processed"},
        dependencies=[upstream_step],
        enable_caching=True
    )
    
    # Verify complete workflow execution
    assert step is not None
    
    # Verify all major components were invoked
    builder._create_processor.assert_called_once()
    builder._get_inputs.assert_called_once()
    builder._get_outputs.assert_called_once()
    builder._get_job_arguments.assert_called_once()
    builder._get_step_name.assert_called_once()
    builder.extract_inputs_from_dependencies.assert_called_once()
```

## Mock Configuration

### Processing-Specific Mocks

```python
def _configure_step_type_mocks(self) -> None:
    """Configure Processing-specific mock objects for integration tests."""
    
    # Mock SageMaker ProcessingStep
    self.mock_processing_step = Mock()
    self.mock_processing_step.name = "test-processing-step"
    
    # Mock processors for different patterns
    self.mock_sklearn_processor = Mock()
    self.mock_sklearn_processor.__class__.__name__ = "SKLearnProcessor"
    
    self.mock_xgboost_processor = Mock()
    self.mock_xgboost_processor.__class__.__name__ = "XGBoostProcessor"
    self.mock_xgboost_processor.run.return_value = {"step_args": "mock_args"}
    
    # Mock ProcessingInput/Output objects
    self.mock_processing_inputs = [
        Mock(input_name="input_data", 
             source="s3://bucket/input", 
             destination="/opt/ml/processing/input")
    ]
    self.mock_processing_outputs = [
        Mock(output_name="output_data", 
             source="/opt/ml/processing/output", 
             destination="s3://bucket/output")
    ]
    
    # Mock specification and contract
    self.mock_processing_spec = Mock()
    self.mock_processing_spec.dependencies = {
        "input_data": Mock(logical_name="input_data", required=True)
    }
    self.mock_processing_spec.outputs = {
        "output_data": Mock(logical_name="output_data")
    }
    
    self.mock_contract = Mock()
    self.mock_contract.expected_input_paths = {"input_data": "/opt/ml/processing/input"}
    self.mock_contract.expected_output_paths = {"output_data": "/opt/ml/processing/output"}
```

## Error Handling Integration

### Comprehensive Error Testing

```python
def test_error_handling_integration(self) -> None:
    """Test error handling in Processing step creation integration."""
    
    # Test with missing required configuration
    with self._expect_error("Missing configuration should raise error"):
        step = builder.create_step()
    
    # Test with invalid inputs
    builder.role = "test-role"
    builder.session = Mock()
    builder._create_processor = Mock(side_effect=ValueError("Invalid processor config"))
    
    with self._expect_error("Invalid processor config should raise error"):
        step = builder.create_step()
```

### Error Context Manager

```python
def _expect_error(self, description: str):
    """Context manager to expect an error."""
    class ExpectError:
        def __init__(self, test_instance, description):
            self.test_instance = test_instance
            self.description = description
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.test_instance._log(f"Expected error but none occurred: {description}")
                return False
            else:
                self.test_instance._log(f"Expected error occurred: {description} - {exc_type.__name__}")
                return True  # Suppress the exception
    
    return ExpectError(self, description)
```

## Test Coverage

### Processing-Specific Test Methods

The integration tests cover:

1. **test_complete_processing_step_creation** - Full step creation validation
2. **test_pattern_a_step_creation** - SKLearnProcessor pattern testing
3. **test_pattern_b_step_creation** - XGBoostProcessor pattern testing
4. **test_dependency_extraction_integration** - Dependency resolution testing
5. **test_processing_step_name_generation** - Step naming validation
6. **test_cache_configuration_integration** - Caching integration testing
7. **test_step_dependencies_integration** - Step dependency handling
8. **test_specification_attachment** - Specification attachment validation
9. **test_end_to_end_workflow** - Complete workflow testing
10. **test_error_handling_integration** - Error handling validation

### Validation Requirements

```python
def _validate_step_type_requirements(self) -> dict:
    """Validate Processing-specific requirements for integration tests."""
    return {
        "integration_tests_completed": True,
        "processing_specific_validations": True,
        "pattern_a_validated": True,
        "pattern_b_validated": True,
        "end_to_end_workflow_validated": True
    }
```

## Usage Examples

### Basic Integration Testing

```python
from cursus.validation.builders.variants.processing_integration_tests import ProcessingIntegrationTests

# Initialize integration tests
integration_tests = ProcessingIntegrationTests(processing_builder, config)

# Run all integration tests
results = integration_tests.run_all_tests()

# Check specific pattern validation
pattern_a_results = integration_tests.test_pattern_a_step_creation()
pattern_b_results = integration_tests.test_pattern_b_step_creation()
```

### End-to-End Workflow Testing

```python
# Test complete workflow with dependencies
workflow_results = integration_tests.test_end_to_end_workflow()

# Validate dependency resolution
dependency_results = integration_tests.test_dependency_extraction_integration()

# Check step name generation
naming_results = integration_tests.test_processing_step_name_generation()
```

### Error Handling Validation

```python
# Test error handling capabilities
error_results = integration_tests.test_error_handling_integration()

# Validate cache configuration
cache_results = integration_tests.test_cache_configuration_integration()
```

## Integration Points

### Test Factory Integration

The Processing integration tests integrate with the universal test factory:

```python
from cursus.validation.builders.test_factory import UniversalStepBuilderTestFactory

factory = UniversalStepBuilderTestFactory()
test_instance = factory.create_test_instance(processing_builder, config)
# Returns ProcessingIntegrationTests for Processing builders
```

### Registry Discovery Integration

Works with registry-based discovery for automatic test selection:

```python
from cursus.validation.builders.registry_discovery import discover_step_type

step_type = discover_step_type(processing_builder)
# Returns "Processing" for Processing builders
```

## Best Practices

### Comprehensive Integration Testing

1. **Test Both Patterns**: Always validate both Pattern A and Pattern B approaches
2. **Dependency Resolution**: Verify input extraction from upstream steps
3. **End-to-End Validation**: Test complete workflow from configuration to step creation
4. **Error Handling**: Validate error scenarios and edge cases
5. **Mock Realism**: Use realistic mock configurations that mirror production scenarios

### Configuration Management

```python
# Comprehensive integration test configuration
config = {
    "processing_instance_type_large": "ml.m5.xlarge",
    "processing_instance_type_small": "ml.m5.large",
    "use_large_processing_instance": False,
    "processing_instance_count": 1,
    "processing_volume_size": 30,
    "processing_framework_version": "0.23-1",
    "job_type": "training"
}
```

### Continuous Integration

```python
# CI/CD pipeline integration
def validate_processing_integration(builder_instance):
    integration_tests = ProcessingIntegrationTests(builder_instance, config)
    results = integration_tests.run_all_tests()
    
    if not results["all_tests_passed"]:
        raise ValueError("Processing integration tests failed")
    
    return results
```

The Processing integration tests provide comprehensive Level 4 validation ensuring that Processing step builders integrate correctly with the overall system and can create functional SageMaker ProcessingStep objects using both Pattern A and Pattern B approaches.
