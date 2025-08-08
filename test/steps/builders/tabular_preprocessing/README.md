---
tags:
  - test
  - builders
  - tabular_preprocessing
  - validation
keywords:
  - step builder tests
  - tabular preprocessing
  - validation infrastructure
  - universal test framework
  - processing step validation
topics:
  - test suite documentation
  - validation framework
  - step builder testing
language: python
date of note: 2025-08-08
---

# Tabular Preprocessing Step Builder Tests

## Overview

This directory contains comprehensive tests for the `TabularPreprocessingStepBuilder` class using the existing validation infrastructure from `src/cursus/validation/builders`.

## Test Architecture

The test suite leverages the existing validation framework:

- **UniversalStepBuilderTest**: Comprehensive validation across all architectural levels
- **ProcessingStepBuilderTest**: Processing-specific validation patterns

## Test Configuration

The test case uses the exact configuration specified in the requirements:

- **source_dir**: `'src/cursus/steps/scripts'`
- **processing_entry_point**: `'tabular_preprocess.py'`
- **job_type**: `'training'` (uses `PREPROCESSING_TRAINING_SPEC`)
- **SageMaker step type**: `Processing` (uses Processing variant tests)

## Test Files

### Core Test Module
- **File**: `test_tabular_preprocessing.py`
- **Description**: Comprehensive test suite using existing validation infrastructure
- **Classes**:
  - `TestTabularPreprocessingWithExistingValidators`: Main validation tests
  - `TestTabularPreprocessingMultipleJobTypes`: Multi job-type validation

### Test Runner
- **File**: `run_test.py`
- **Description**: Simple script to execute the test suite with detailed output

### Package Configuration
- **File**: `__init__.py`
- **Description**: Package setup with easy import access to test functions

## Usage

### Method 1: Run with the runner script (Recommended)
```bash
cd test/steps/builders/tabular_preprocessing
python run_test.py
```

### Method 2: Run with Python unittest
```bash
cd test/steps/builders/tabular_preprocessing
python -m unittest test_tabular_preprocessing -v
```

### Method 3: Run specific test classes
```bash
cd test/steps/builders/tabular_preprocessing
python -m unittest test_tabular_preprocessing.TestTabularPreprocessingWithExistingValidators -v
python -m unittest test_tabular_preprocessing.TestTabularPreprocessingMultipleJobTypes -v
```

### Method 4: Import and run programmatically
```python
from test_tabular_preprocessing import run_comprehensive_test
result = run_comprehensive_test()
```

## Test Validation Levels

The existing validation infrastructure provides comprehensive testing across multiple levels:

### Level 1: Interface Tests
- ✅ Inheritance from `StepBuilderBase`
- ✅ Required methods implementation (`validate_configuration`, `_get_inputs`, `_get_outputs`, `create_step`)
- ✅ Method signatures and return types
- ✅ Error handling and exception management

### Level 2: Specification Tests
- ✅ `PREPROCESSING_TRAINING_SPEC` usage validation
- ✅ Contract alignment with `TABULAR_PREPROCESS_CONTRACT`
- ✅ Dependency specification (`DATA` input)
- ✅ Output specification (`processed_data` output)

### Level 3: Path Mapping Tests
- ✅ Input path mapping: `DATA` → `/opt/ml/processing/input/data`
- ✅ Output path mapping: `processed_data` → `/opt/ml/processing/output`
- ✅ Property path validation for SageMaker integration

### Level 4: Integration Tests
- ✅ ProcessingStep creation with realistic inputs/outputs
- ✅ Mock dependency resolution
- ✅ SageMaker step validation
- ✅ Step specification attachment

### Processing Variant Tests
- ✅ SKLearnProcessor creation and configuration
- ✅ Framework version validation (`0.23-1`)
- ✅ Instance type selection (large vs small)
- ✅ Processing job arguments construction
- ✅ Environment variable setup

## Configuration Details

The test creates a comprehensive `TabularPreprocessingConfig` with:

### Essential User Inputs (Tier 1)
- `label_name`: "target"

### System Fields with Defaults (Tier 2)
- `processing_entry_point`: "tabular_preprocess.py"
- `job_type`: "training"
- `train_ratio`: 0.7
- `test_val_ratio`: 0.5

### Processing Configuration
- `processing_instance_count`: 1
- `processing_volume_size`: 30
- `processing_instance_type_large`: "ml.m5.xlarge"
- `processing_instance_type_small`: "ml.m5.large"
- `processing_framework_version`: "0.23-1"
- `use_large_processing_instance`: False
- `py_version`: "py3"

### Optional Column Configurations
- `categorical_columns`: ["category_col1", "category_col2"]
- `numerical_columns`: ["num_col1", "num_col2"]
- `text_columns`: ["text_col1"]
- `date_columns`: ["date_col1"]

## Test Dependencies

The tests require the following components to be implemented:

### Required Classes
- `TabularPreprocessingStepBuilder` (in `src/cursus/steps/builders/`)
- `TabularPreprocessingConfig` (in `src/cursus/steps/configs/`)
- `PREPROCESSING_TRAINING_SPEC` (in `src/cursus/steps/specs/`)
- `TABULAR_PREPROCESS_CONTRACT` (in `src/cursus/steps/contracts/`)

### Required Script
- `tabular_preprocess.py` (in `src/cursus/steps/scripts/`)

### Validation Infrastructure
- `UniversalStepBuilderTest` (in `src/cursus/validation/builders/`)
- `ProcessingStepBuilderTest` (in `src/cursus/validation/builders/variants/`)

## Expected Results

The test suite validates:

- ✅ **Universal Test Compliance**: All core architectural requirements
- ✅ **Processing Step Validation**: SKLearnProcessor creation and configuration
- ✅ **Configuration Validation**: 3-tier design compliance
- ✅ **Script Integration**: Proper script path resolution and contract alignment
- ✅ **Environment Variables**: Required and optional env var setup
- ✅ **Job Arguments**: Proper job_type argument passing
- ✅ **Multiple Job Types**: Support for training, validation, testing, calibration
- ✅ **Realistic Integration**: End-to-end pipeline simulation

## Success Criteria

The test expects:
- **Pass Rate**: ≥70% of all tests should pass
- **Critical Tests**: All inheritance, methods, specification, and contract tests must pass
- **Processing Tests**: All Processing-specific validations must pass
- **Integration Tests**: Realistic pipeline scenarios must work correctly

## Test Output Example

```
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀
TABULAR PREPROCESSING STEP BUILDER VALIDATION
Using Existing Cursus Validation Infrastructure
🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀

================================================================================
TESTING WITH UNIVERSAL STEP BUILDER TEST
================================================================================

Universal Test Results: 15/18 tests passed (83.3%)

================================================================================
TESTING WITH PROCESSING STEP BUILDER TEST
================================================================================

Processing Test Results: 12/15 tests passed (80.0%)

🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯
FINAL TEST SUMMARY
🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯🎯
Tests run: 4
Failures: 0
Errors: 0
Success rate: 100.0%

✅ ALL TESTS PASSED!
The TabularPreprocessingStepBuilder is ready for use.
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the `src` directory is in the Python path
2. **Missing Dependencies**: Install required packages (sagemaker, pydantic, etc.)
3. **Mock Failures**: Check that SageMaker components are properly mocked

### Debug Mode

For detailed debugging, the tests include verbose output options and detailed error reporting.

## Integration with Existing Infrastructure

This test suite integrates seamlessly with the existing validation infrastructure:

- Uses the same Universal Test framework as other step builders
- Follows the same patterns as `test_real_builders.py`
- Compatible with the existing mock factory system
- Supports the same validation patterns used throughout the codebase

## Advantages of Using Existing Validators

- **Consistency**: Same validation patterns across all step builders
- **Maintenance**: Centralized validation logic reduces duplication
- **Reliability**: Well-tested validation infrastructure
- **Extensibility**: Easy to add new validation patterns
- **Standards Compliance**: Ensures adherence to architectural standards

## Related Documentation

- [Universal Step Builder Test Design](../../../slipbox/1_design/universal_step_builder_test.md)
- [Processing Step Builder Patterns](../../../slipbox/1_design/processing_step_builder_patterns.md)
- [Validation Infrastructure](../../../src/cursus/validation/builders/README_ENHANCED_SYSTEM.md)
- [Tabular Preprocessing Step Builder](../../../src/cursus/steps/builders/builder_tabular_preprocessing_step.py)
- [Tabular Preprocessing Config](../../../src/cursus/steps/configs/config_tabular_preprocessing_step.py)
- [Preprocessing Training Spec](../../../src/cursus/steps/specs/preprocessing_training_spec.py)
- [Tabular Preprocess Contract](../../../src/cursus/steps/contracts/tabular_preprocess_contract.py)
