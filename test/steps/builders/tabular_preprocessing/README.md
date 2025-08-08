# Tabular Preprocessing Step Builder Tests - Enhanced 4-Level Processing Tester

This directory contains comprehensive tests for the `TabularPreprocessingStepBuilder` using the **Enhanced 4-Level Processing Tester** from `src/cursus/validation/builders/variants/processing_test.py`.

## Overview

The tests have been refactored to leverage the enhanced 4-level Processing tester that validates Processing step builders against all patterns identified in the Processing Step Builder Patterns analysis. This provides comprehensive coverage of:

- **Level 1: Interface Tests** - Method signatures, processor creation, configuration attributes
- **Level 2: Specification Tests** - Job types, environment variables, arguments, contract mapping  
- **Level 3: Path Mapping Tests** - Input/output creation, special patterns, S3 handling
- **Level 4: Integration Tests** - End-to-end step creation, dependencies, caching

## Files

### Core Test Files

- **`test_tabular_preprocessing.py`** - Main test suite using the 4-level Processing tester
- **`test_scoring.py`** - Comprehensive scoring and evaluation system
- **`run_test.py`** - Test runner script with 4-level validation
- **`run_scoring_test.py`** - Scoring test runner with pattern analysis

### Test Structure

```
test_tabular_preprocessing.py
├── TestTabularPreprocessingWith4LevelTester
│   ├── test_4_level_processing_validation()      # Complete 4-level validation
│   ├── test_level1_interface_tests()             # Level 1: Interface compliance
│   ├── test_level2_specification_tests()         # Level 2: Specification compliance
│   ├── test_level3_path_mapping_tests()          # Level 3: Path mapping compliance
│   ├── test_level4_integration_tests()           # Level 4: Integration compliance
│   └── test_legacy_compatibility()               # Backward compatibility
└── TestTabularPreprocessingMultipleJobTypes
    └── test_all_job_types_with_4_level_tester()  # Multi job-type validation
```

## Usage

### 1. Run All 4-Level Tests

```bash
python run_test.py
```

This runs the complete 4-level Processing validation suite and provides:
- Pattern compliance analysis
- Level-by-level results
- Processing-specific pattern validation
- Troubleshooting guidance

### 2. Run Specific Level Tests

```bash
# Level 1: Interface Tests
python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level1_interface_tests

# Level 2: Specification Tests  
python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level2_specification_tests

# Level 3: Path Mapping Tests
python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level3_path_mapping_tests

# Level 4: Integration Tests
python test_tabular_preprocessing.py TestTabularPreprocessingWith4LevelTester.test_level4_integration_tests
```

### 3. Run Comprehensive Scoring

```bash
python run_scoring_test.py
```

This generates detailed scoring reports with:
- Overall Processing step compliance score
- Level-by-level scoring breakdown
- Pattern-specific analysis
- Recommendations for improvement

### 4. Quick Validation

```bash
# Quick test validation
python run_test.py --quick

# Quick scoring validation
python run_scoring_test.py --quick
```

### 5. Pattern Analysis

```bash
python run_scoring_test.py --patterns
```

This analyzes compliance with specific Processing patterns:
- Processor Creation Patterns (SKLearn vs XGBoost)
- Step Creation Patterns (Pattern A vs Pattern B)
- Environment Variable Patterns (Basic, JSON-serialized, Step-specific)
- Input/Output Handling Patterns
- Special Input Patterns (Local path override, File upload)

## 4-Level Processing Test Coverage

### Level 1: Interface Tests
- ✅ `test_processor_creation_method` - Validates `_create_processor()` implementation
- ✅ `test_processing_configuration_attributes` - Checks required config attributes
- ✅ `test_framework_specific_methods` - Tests SKLearn vs XGBoost specifics
- ✅ `test_step_creation_pattern_compliance` - Validates Pattern A vs Pattern B
- ✅ `test_processing_input_output_methods` - Tests `_get_inputs()` and `_get_outputs()`
- ✅ `test_environment_variables_method` - Validates `_get_environment_variables()`
- ✅ `test_job_arguments_method` - Tests `_get_job_arguments()`

### Level 2: Specification Tests
- ✅ `test_job_type_specification_loading` - Multi job-type support
- ✅ `test_environment_variable_patterns` - All 3 env var patterns
- ✅ `test_job_arguments_patterns` - All 3 job argument patterns
- ✅ `test_specification_driven_inputs` - Spec-driven input handling
- ✅ `test_specification_driven_outputs` - Spec-driven output handling
- ✅ `test_contract_path_mapping` - Contract-based path mapping
- ✅ `test_multi_job_type_support` - Job type variations
- ✅ `test_framework_specific_specifications` - Framework-specific specs

### Level 3: Path Mapping Tests
- ✅ `test_processing_input_creation` - ProcessingInput object creation
- ✅ `test_processing_output_creation` - ProcessingOutput object creation
- ✅ `test_container_path_mapping` - Script contract path mapping
- ✅ `test_special_input_handling` - Special input patterns
- ✅ `test_s3_path_normalization` - S3 URI handling
- ✅ `test_file_upload_patterns` - File upload pattern
- ✅ `test_local_path_override_patterns` - Local path override pattern
- ✅ `test_dependency_input_extraction` - Dependency resolution

### Level 4: Integration Tests
- ✅ `test_step_creation_pattern_execution` - Pattern A vs B execution
- ✅ `test_framework_specific_step_creation` - Framework-specific creation
- ✅ `test_processing_dependency_resolution` - Dependency handling
- ✅ `test_step_name_generation` - Standardized naming
- ✅ `test_cache_configuration` - Caching patterns
- ✅ `test_step_dependencies_handling` - Dependency management
- ✅ `test_end_to_end_step_creation` - Complete step creation
- ✅ `test_specification_attachment` - Spec attachment to steps

## Processing Pattern Compliance

The 4-level tester validates compliance with all Processing step patterns identified in the analysis:

### ✅ Processor Creation Patterns
- **SKLearn Pattern**: Uses `SKLearnProcessor` with standard configuration
- **XGBoost Pattern**: Uses `XGBoostProcessor` with framework-specific config

### ✅ Step Creation Patterns
- **Pattern A**: Direct `ProcessingStep` instantiation (SKLearn)
- **Pattern B**: `processor.run()` + `step_args` (XGBoost)

### ✅ Environment Variable Patterns
- **Basic Pattern**: Standard SageMaker variables (`SAGEMAKER_PROGRAM`, `SAGEMAKER_SUBMIT_DIRECTORY`)
- **JSON-Serialized Pattern**: Complex configurations as JSON strings
- **Step-Specific Pattern**: Custom variables (`LABEL_FIELD`, `CURRENCY_CONVERSION_DICT`)

### ✅ Job Arguments Patterns
- **Simple Pattern**: Basic job type arguments
- **Complex Pattern**: Arguments with optional parameters
- **No Arguments Pattern**: Steps without job arguments

### ✅ Special Input Handling Patterns
- **Local Path Override Pattern**: Override dependency inputs with local paths
- **File Upload Pattern**: Automatic file upload and S3 handling
- **S3 Path Handling**: S3 URI normalization and validation

## Scoring System

The scoring system provides detailed analysis:

### Score Ranges
- **90-100%**: 🟢 Excellent - Fully compliant with Processing patterns
- **80-89%**: 🟡 Good - Minor pattern deviations
- **70-79%**: 🟠 Satisfactory - Some pattern issues
- **60-69%**: 🔴 Needs Work - Significant pattern problems
- **<60%**: ⚫ Poor - Major pattern violations

### Generated Reports
- **Overall Score**: Weighted average across all levels
- **Level Scores**: Individual scores for each of the 4 levels
- **Pattern Analysis**: Compliance with specific Processing patterns
- **Recommendations**: Targeted improvement suggestions

## Dependencies

The tests require the following components:

### Core Components
- `TabularPreprocessingStepBuilder` - The step builder being tested
- `TabularPreprocessingConfig` - Configuration class
- `PREPROCESSING_TRAINING_SPEC` - Step specification
- `TABULAR_PREPROCESS_CONTRACT` - Script contract

### Testing Framework
- `ProcessingStepBuilderTest` - Enhanced 4-level Processing tester
- `StepBuilderScorer` - Scoring and analysis system
- `score_builder_results` - Report generation function

### Scripts
- `tabular_preprocess.py` - Processing script in `src/cursus/steps/scripts/`

## Troubleshooting

### Common Issues

#### Level 1 Failures
- **Processor Creation**: Check `_create_processor()` method implementation
- **Configuration**: Verify all required processing config attributes exist
- **Framework Methods**: Ensure framework-specific methods are implemented

#### Level 2 Failures
- **Specification Loading**: Check job type-based specification loading
- **Environment Variables**: Verify all 3 environment variable patterns
- **Contract Mapping**: Ensure contract-based path mapping works

#### Level 3 Failures
- **Input/Output Creation**: Check ProcessingInput/ProcessingOutput creation
- **Special Patterns**: Verify special input handling patterns
- **Path Mapping**: Ensure container path mapping is correct

#### Level 4 Failures
- **Step Creation**: Debug end-to-end step creation process
- **Dependencies**: Check dependency resolution and handling
- **Integration**: Verify complete system integration

### Debug Commands

```bash
# Verbose test output
python test_tabular_preprocessing.py -v

# Quick validation check
python run_test.py --quick

# Pattern-specific analysis
python run_scoring_test.py --patterns

# Help and usage examples
python run_test.py --help
python run_scoring_test.py --help
```

## Migration from Legacy Tests

The tests have been fully refactored to use the 4-level Processing tester while maintaining backward compatibility:

### What Changed
- ✅ **Enhanced Coverage**: Now tests all Processing patterns identified in the analysis
- ✅ **4-Level Structure**: Organized into Interface, Specification, Path Mapping, and Integration levels
- ✅ **Pattern Validation**: Specific validation for Processing step patterns
- ✅ **Comprehensive Scoring**: Detailed scoring with pattern-specific analysis
- ✅ **Better Reporting**: Level-by-level results and recommendations

### What Stayed the Same
- ✅ **Test Interface**: Same test runner commands and basic usage
- ✅ **Configuration**: Same test configuration approach
- ✅ **Compatibility**: Legacy test methods still work via compatibility layer

## Benefits of 4-Level Testing

1. **Comprehensive Coverage**: Tests all identified Processing step patterns
2. **Structured Validation**: Clear separation of concerns across 4 levels
3. **Pattern Compliance**: Specific validation against Processing patterns
4. **Detailed Analysis**: Level-by-level scoring and recommendations
5. **Maintainability**: Easier to understand and maintain test structure
6. **Extensibility**: Easy to add new Processing pattern tests

This enhanced testing approach ensures that the `TabularPreprocessingStepBuilder` fully complies with all Processing step patterns and provides a robust foundation for Processing step validation.
