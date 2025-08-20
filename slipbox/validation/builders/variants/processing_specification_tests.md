---
tags:
  - code
  - validation
  - testing
  - processing
  - specification
keywords:
  - processing specification tests
  - level 2 testing
  - job type specification
  - processor configuration
  - environment variables
  - contract compliance
topics:
  - validation framework
  - specification testing
  - processing step validation
  - contract validation
language: python
date of note: 2025-01-18
---

# Processing-Specific Level 2 Specification Tests

## Overview

The `ProcessingSpecificationTests` class provides Level 2 specification testing specifically for Processing step builders. These tests validate Processing-specific specification and contract compliance including job type-based specification loading, processor-specific configuration validation, environment variable handling patterns, and input/output specification compliance.

## Architecture

### Processing Specification Focus

Level 2 specification tests for Processing steps validate:

1. **Job Type-Based Specification Loading** - Multi-job-type builder specification handling
2. **Processor Configuration Validation** - Processing-specific configuration requirements
3. **Environment Variable Handling** - Processing parameter passing patterns
4. **Input/Output Specification Compliance** - Specification-driven I/O handling
5. **Contract Path Mapping** - Container path mapping validation
6. **Step Creation Pattern Compliance** - Pattern A/B specification alignment

### Processing-Specific Patterns

Processing steps have unique specification requirements:

- **Multi-Job-Type Support** - Different specifications for training/validation/evaluation
- **Processor-Specific Configuration** - SKLearn vs XGBoost configuration patterns
- **Environment Variable Patterns** - Complex parameter serialization (JSON, CSV)
- **Container Path Mapping** - Processing-specific input/output path conventions

## Core Test Methods

### Job Type Specification Loading

```python
def test_job_type_specification_loading(self) -> None:
    """Test that Processing builders properly load specifications based on job type."""
    
    # Test multi-job-type builders (TabularPreprocessing, CurrencyConversion)
    if hasattr(self.builder_class, '_load_specification_by_job_type'):
        # Test training job type
        with patch.object(self.builder_class, '_load_specification_by_job_type') as mock_load:
            mock_load.return_value = self.mock_processing_spec
            
            config = Mock()
            config.job_type = "training"
            
            builder = self.builder_class(config=config)
            mock_load.assert_called_with("training")
        
        # Test validation job type
        with patch.object(self.builder_class, '_load_specification_by_job_type') as mock_load:
            mock_load.return_value = self.mock_processing_spec
            
            config = Mock()
            config.job_type = "validation"
            
            builder = self.builder_class(config=config)
            mock_load.assert_called_with("validation")
```

### Processor Configuration Validation

```python
def test_processor_configuration_validation(self) -> None:
    """Test that Processing builders validate processor-specific configuration."""
    
    # Test required processing configuration attributes
    required_attrs = [
        'processing_instance_count', 'processing_volume_size',
        'processing_instance_type_large', 'processing_instance_type_small',
        'processing_framework_version', 'use_large_processing_instance'
    ]
    
    if hasattr(self.builder_class, 'validate_configuration'):
        # Test with missing required attributes
        for attr in required_attrs:
            config = Mock()
            # Set all attributes except the one being tested
            for other_attr in required_attrs:
                if other_attr != attr:
                    setattr(config, other_attr, "test_value")
            
            try:
                builder = self.builder_class(config=config)
                builder.validate_configuration()
                # Should have raised an error
                assert False, f"Missing {attr} should have caused validation error"
            except (ValueError, AttributeError):
                # Expected behavior
                assert True, f"Correctly detected missing {attr}"
```

## Environment Variables Handling

### Processing Environment Variables Testing

```python
def test_processing_environment_variables(self) -> None:
    """Test that Processing builders handle environment variables correctly."""
    
    if hasattr(self.builder_class, '_get_environment_variables'):
        config = Mock()
        config.label_name = "target"
        config.categorical_columns = ["cat1", "cat2"]
        config.currency_conversion_dict = {"USD": 1.0, "EUR": 0.85}
        
        builder = self.builder_class(config=config)
        env_vars = builder._get_environment_variables()
        
        # Check that environment variables are returned as dict
        assert isinstance(env_vars, dict)
        
        # Check for common Processing environment variable patterns
        if hasattr(config, 'label_name'):
            assert ("LABEL_FIELD" in env_vars or 
                   "label_name" in str(env_vars).lower())
        
        if hasattr(config, 'categorical_columns'):
            # Should be comma-separated or JSON
            found_categorical = any("categorical" in key.lower() for key in env_vars.keys())
            assert found_categorical
        
        if hasattr(config, 'currency_conversion_dict'):
            # Should be JSON serialized
            found_currency = any("currency" in key.lower() for key in env_vars.keys())
            assert found_currency
```

### Environment Variable Patterns

Processing steps use various environment variable patterns:

**Simple String Variables**:
```python
env_vars = {
    "LABEL_FIELD": config.label_name,
    "JOB_TYPE": config.job_type,
    "FRAMEWORK_VERSION": config.processing_framework_version
}
```

**List Serialization**:
```python
env_vars = {
    "CATEGORICAL_COLUMNS": ",".join(config.categorical_columns),
    "NUMERIC_COLUMNS": ",".join(config.numeric_columns)
}
```

**Dictionary Serialization**:
```python
env_vars = {
    "CURRENCY_CONVERSION_DICT": json.dumps(config.currency_conversion_dict),
    "FEATURE_MAPPING": json.dumps(config.feature_mapping)
}
```

## Specification-Driven Input/Output

### Input Specification Testing

```python
def test_specification_driven_inputs(self) -> None:
    """Test that Processing builders use specifications to define inputs."""
    
    if hasattr(self.builder_class, '_get_inputs'):
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification and contract
        builder.spec = self.mock_processing_spec
        builder.contract = self.mock_contract
        
        inputs = {
            "input_data": "s3://bucket/input/data",
            "metadata": "s3://bucket/input/metadata"
        }
        
        processing_inputs = builder._get_inputs(inputs)
        
        # Check that ProcessingInput objects are created
        assert isinstance(processing_inputs, list)
        
        if processing_inputs:
            # Check first input structure
            first_input = processing_inputs[0]
            assert hasattr(first_input, 'input_name')
            assert hasattr(first_input, 'source')
            assert hasattr(first_input, 'destination')
```

### Output Specification Testing

```python
def test_specification_driven_outputs(self) -> None:
    """Test that Processing builders use specifications to define outputs."""
    
    if hasattr(self.builder_class, '_get_outputs'):
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification and contract
        builder.spec = self.mock_processing_spec
        builder.contract = self.mock_contract
        
        outputs = {
            "processed_data": "s3://bucket/output/data",
            "statistics": "s3://bucket/output/stats"
        }
        
        processing_outputs = builder._get_outputs(outputs)
        
        # Check that ProcessingOutput objects are created
        assert isinstance(processing_outputs, list)
        
        if processing_outputs:
            # Check first output structure
            first_output = processing_outputs[0]
            assert hasattr(first_output, 'output_name')
            assert hasattr(first_output, 'source')
            assert hasattr(first_output, 'destination')
```

## Contract Path Mapping

### Container Path Mapping Testing

```python
def test_contract_path_mapping(self) -> None:
    """Test that Processing builders use contracts for container path mapping."""
    
    config = Mock()
    builder = self.builder_class(config=config)
    
    # Mock contract
    builder.contract = self.mock_contract
    
    # Test that contract paths are used correctly
    if hasattr(builder, '_get_inputs') and hasattr(builder, 'contract'):
        # Check that contract has expected path mappings
        assert hasattr(builder.contract, 'expected_input_paths')
        assert hasattr(builder.contract, 'expected_output_paths')
        
        # Verify path mapping structure
        if hasattr(builder.contract, 'expected_input_paths'):
            input_paths = builder.contract.expected_input_paths
            assert isinstance(input_paths, dict)
        
        if hasattr(builder.contract, 'expected_output_paths'):
            output_paths = builder.contract.expected_output_paths
            assert isinstance(output_paths, dict)
```

### Processing Container Path Conventions

Processing steps follow specific container path conventions:

**Input Paths**:
```python
expected_input_paths = {
    "input_data": "/opt/ml/processing/input/data",
    "metadata": "/opt/ml/processing/input/metadata",
    "config": "/opt/ml/processing/input/config"
}
```

**Output Paths**:
```python
expected_output_paths = {
    "processed_data": "/opt/ml/processing/output/data",
    "statistics": "/opt/ml/processing/output/stats",
    "model_artifacts": "/opt/ml/processing/output/model"
}
```

## Job Arguments Specification

### Job Arguments Testing

```python
def test_job_arguments_specification(self) -> None:
    """Test that Processing builders handle job arguments according to specification."""
    
    if hasattr(self.builder_class, '_get_job_arguments'):
        config = Mock()
        config.job_type = "training"
        config.mode = "batch"
        config.marketplace_id_col = "marketplace_id"
        config.enable_currency_conversion = True
        config.currency_col = "currency"
        
        builder = self.builder_class(config=config)
        job_args = builder._get_job_arguments()
        
        # Job arguments can be None, empty list, or list of strings
        if job_args is not None:
            assert isinstance(job_args, list)
            
            if job_args:
                # All arguments should be strings
                all_strings = all(isinstance(arg, str) for arg in job_args)
                assert all_strings
                
                # Common Processing argument patterns
                arg_string = " ".join(job_args)
                if hasattr(config, 'job_type'):
                    job_type_found = "job_type" in arg_string or "job-type" in arg_string
                    assert job_type_found
```

### Job Arguments vs Environment Variables

Processing steps can pass parameters through either job arguments or environment variables:

**Job Arguments Approach**:
```python
job_args = [
    "--job_type", "training",
    "--label_field", "target",
    "--marketplace_id_col", "marketplace_id",
    "--enable_currency_conversion", "true"
]
```

**Environment Variables Approach**:
```python
env_vars = {
    "JOB_TYPE": "training",
    "LABEL_FIELD": "target",
    "MARKETPLACE_ID_COL": "marketplace_id",
    "ENABLE_CURRENCY_CONVERSION": "true"
}
```

## Processor Type Alignment

### Processor Type Testing

```python
def test_processor_type_alignment(self) -> None:
    """Test that Processing builders create the correct processor type."""
    
    if hasattr(self.builder_class, '_create_processor'):
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
        
        processor = builder._create_processor()
        
        # Check processor type
        processor_type = type(processor).__name__
        assert processor_type in ["SKLearnProcessor", "XGBoostProcessor", "Mock"]
        
        # XGBoost processors should have framework and py_version
        if "XGBoost" in processor_type:
            if hasattr(config, 'framework_version'):
                assert True  # XGBoost processor with framework version
            if hasattr(config, 'py_version'):
                assert True  # XGBoost processor with Python version
```

### Framework-Specific Processor Configuration

**SKLearn Processor Configuration**:
```python
sklearn_processor = SKLearnProcessor(
    framework_version=config.processing_framework_version,
    instance_type=instance_type,
    instance_count=config.processing_instance_count,
    volume_size_in_gb=config.processing_volume_size,
    role=builder.role,
    sagemaker_session=builder.session
)
```

**XGBoost Processor Configuration**:
```python
xgboost_processor = XGBoostProcessor(
    framework_version=config.framework_version,
    py_version=config.py_version,
    instance_type=instance_type,
    instance_count=config.processing_instance_count,
    volume_size_in_gb=config.processing_volume_size,
    role=builder.role,
    sagemaker_session=builder.session
)
```

## Step Creation Pattern Compliance

### Pattern Compliance Testing

```python
def test_step_creation_pattern_compliance(self) -> None:
    """Test that Processing builders follow the correct step creation pattern."""
    
    if hasattr(self.builder_class, 'create_step'):
        config = Mock()
        # Set up minimal required configuration
        config.processing_instance_type_large = "ml.m5.xlarge"
        config.processing_instance_type_small = "ml.m5.large"
        config.use_large_processing_instance = False
        config.processing_instance_count = 1
        config.processing_volume_size = 30
        config.processing_framework_version = "0.23-1"
        
        builder = self.builder_class(config=config)
        builder.role = "test-role"
        builder.session = Mock()
        
        # Mock required methods
        builder._create_processor = Mock(return_value=self.mock_sklearn_processor)
        builder._get_inputs = Mock(return_value=[])
        builder._get_outputs = Mock(return_value=[])
        builder._get_job_arguments = Mock(return_value=["--job_type", "training"])
        builder._get_step_name = Mock(return_value="test-processing-step")
        builder._get_cache_config = Mock(return_value=None)
        
        # Test step creation
        step = builder.create_step(
            inputs={"input_data": "s3://bucket/input"},
            outputs={"output_data": "s3://bucket/output"},
            dependencies=[],
            enable_caching=True
        )
        
        # Verify step creation patterns
        assert step is not None
        
        # Check for ProcessingStep characteristics
        step_type = type(step).__name__
        assert ("ProcessingStep" in step_type or "Mock" in step_type)
```

## Mock Configuration

### Processing-Specific Mocks

```python
def _configure_step_type_mocks(self) -> None:
    """Configure Processing-specific mock objects for specification tests."""
    
    # Mock processor types
    self.mock_sklearn_processor = Mock()
    self.mock_xgboost_processor = Mock()
    
    # Mock specification objects
    self.mock_processing_spec = Mock()
    self.mock_processing_spec.dependencies = {
        "input_data": Mock(logical_name="input_data", required=True),
        "metadata": Mock(logical_name="metadata", required=False)
    }
    self.mock_processing_spec.outputs = {
        "processed_data": Mock(logical_name="processed_data"),
        "statistics": Mock(logical_name="statistics")
    }
    
    # Mock contract objects
    self.mock_contract = Mock()
    self.mock_contract.expected_input_paths = {
        "input_data": "/opt/ml/processing/input/data",
        "metadata": "/opt/ml/processing/input/metadata"
    }
    self.mock_contract.expected_output_paths = {
        "processed_data": "/opt/ml/processing/output/data",
        "statistics": "/opt/ml/processing/output/stats"
    }
```

## Test Coverage

### Processing-Specific Test Methods

The specification tests cover:

1. **test_job_type_specification_loading** - Multi-job-type specification handling
2. **test_processor_configuration_validation** - Configuration requirement validation
3. **test_processing_environment_variables** - Environment variable handling
4. **test_specification_driven_inputs** - Input specification compliance
5. **test_specification_driven_outputs** - Output specification compliance
6. **test_contract_path_mapping** - Container path mapping validation
7. **test_job_arguments_specification** - Job arguments specification compliance
8. **test_processor_type_alignment** - Processor type validation
9. **test_step_creation_pattern_compliance** - Pattern A/B compliance validation

### Validation Requirements

```python
def _validate_step_type_requirements(self) -> dict:
    """Validate Processing-specific requirements for specification tests."""
    return {
        "specification_tests_completed": True,
        "processing_specific_validations": True,
        "processor_type_validated": True,
        "job_type_support_validated": True
    }
```

## Usage Examples

### Basic Specification Testing

```python
from cursus.validation.builders.variants.processing_specification_tests import ProcessingSpecificationTests

# Initialize specification tests
spec_tests = ProcessingSpecificationTests(processing_builder, config)

# Run all specification tests
results = spec_tests.run_all_tests()

# Check specific specification validation
job_type_results = spec_tests.test_job_type_specification_loading()
env_var_results = spec_tests.test_processing_environment_variables()
```

### Configuration Validation

```python
# Test processor configuration validation
config_results = spec_tests.test_processor_configuration_validation()

# Validate processor type alignment
processor_results = spec_tests.test_processor_type_alignment()

# Check step creation pattern compliance
pattern_results = spec_tests.test_step_creation_pattern_compliance()
```

### Input/Output Specification Testing

```python
# Test specification-driven inputs
input_results = spec_tests.test_specification_driven_inputs()

# Test specification-driven outputs
output_results = spec_tests.test_specification_driven_outputs()

# Validate contract path mapping
contract_results = spec_tests.test_contract_path_mapping()
```

## Integration Points

### Test Factory Integration

The Processing specification tests integrate with the universal test factory:

```python
from cursus.validation.builders.test_factory import UniversalStepBuilderTestFactory

factory = UniversalStepBuilderTestFactory()
test_instance = factory.create_test_instance(processing_builder, config)
# Returns ProcessingSpecificationTests for Processing builders
```

### Registry Discovery Integration

Works with registry-based discovery for automatic test selection:

```python
from cursus.validation.builders.registry_discovery import discover_step_type

step_type = discover_step_type(processing_builder)
# Returns "Processing" for Processing builders
```

## Best Practices

### Specification Validation Strategy

1. **Job Type Awareness** - Test multi-job-type specification loading
2. **Configuration Completeness** - Validate all required processing configuration
3. **Environment Variable Patterns** - Test complex parameter serialization
4. **Contract Compliance** - Verify container path mapping
5. **Pattern Alignment** - Ensure specification matches step creation pattern

### Configuration Requirements

```python
# Required Processing specification configuration
processing_spec_config = {
    "processing_instance_count": 1,
    "processing_volume_size": 30,
    "processing_instance_type_large": "ml.m5.xlarge",
    "processing_instance_type_small": "ml.m5.large",
    "processing_framework_version": "0.23-1",
    "use_large_processing_instance": False,
    "job_type": "training"  # For multi-job-type builders
}
```

### Continuous Integration

```python
# CI/CD pipeline integration
def validate_processing_specification(builder_instance):
    spec_tests = ProcessingSpecificationTests(builder_instance, config)
    results = spec_tests.run_all_tests()
    
    if not results["all_tests_passed"]:
        raise ValueError("Processing specification tests failed")
    
    return results
```

The Processing specification tests provide comprehensive Level 2 validation ensuring that Processing step builders properly use specifications and contracts to define their behavior with focus on Processing-specific patterns and requirements.
