---
tags:
  - code
  - cli
  - builder
  - testing
  - validation
  - step-builders
keywords:
  - builder-test
  - all
  - level
  - variant
  - test-by-type
  - registry-report
  - validate-builder
  - list-builders
  - UniversalStepBuilderTest
  - StepBuilderScorer
  - RegistryStepDiscovery
topics:
  - step builder testing
  - builder validation
  - CLI tools
  - testing framework
language: python
date of note: 2024-12-07
---

# Builder Test CLI

Command-line interface for the Universal Step Builder Test System providing comprehensive testing and validation of step builder implementations with enhanced scoring and registry features.

## Overview

The Builder Test CLI provides comprehensive testing tools for cursus step builders, supporting validation across four critical levels of builder implementation: Interface Tests, Specification Tests, Step Creation Tests, and Integration Tests. The CLI offers both individual builder testing and batch testing capabilities with detailed scoring, reporting, and visualization features.

The module supports multiple testing modes including universal test suites, level-specific testing, variant-specific testing, and SageMaker step type testing. It features comprehensive scoring and quality rating systems, registry discovery and validation capabilities, JSON export functionality for integration with other tools, and visualization chart generation for test results analysis.

## Classes and Methods

### Commands
- [`all`](#all) - Run all tests (universal test suite)
- [`level`](#level) - Run tests for a specific level
- [`variant`](#variant) - Run tests for a specific variant
- [`test-by-type`](#test-by-type) - Test all builders for a specific SageMaker step type
- [`registry-report`](#registry-report) - Generate registry discovery report
- [`validate-builder`](#validate-builder) - Validate that a step builder is available and can be loaded
- [`list-builders`](#list-builders) - List available step builder classes

### Functions
- [`print_test_results`](#print_test_results) - Print test results in formatted way with optional scoring display
- [`print_enhanced_results`](#print_enhanced_results) - Print enhanced results with scoring and structured reporting
- [`run_all_tests_with_scoring`](#run_all_tests_with_scoring) - Run all tests with scoring enabled
- [`run_registry_discovery_report`](#run_registry_discovery_report) - Generate and display registry discovery report
- [`run_test_by_sagemaker_type`](#run_test_by_sagemaker_type) - Test all builders for a specific SageMaker step type
- [`validate_builder_availability`](#validate_builder_availability) - Validate that a step builder is available and can be loaded
- [`export_results_to_json`](#export_results_to_json) - Export test results to JSON file
- [`generate_score_chart`](#generate_score_chart) - Generate score visualization chart
- [`import_builder_class`](#import_builder_class) - Import a builder class from a module path
- [`run_level_tests`](#run_level_tests) - Run tests for a specific level
- [`run_variant_tests`](#run_variant_tests) - Run tests for a specific variant
- [`run_all_tests`](#run_all_tests) - Run all tests (universal test suite)
- [`list_available_builders`](#list_available_builders) - List available step builder classes by scanning the builders directory
- [`main`](#main) - Main CLI entry point

## API Reference

### all

all(_builder_class_, _--verbose_, _--scoring_, _--export-json_, _--export-chart_, _--output-dir_)

Runs all tests (universal test suite) for a specified step builder class with comprehensive validation.

**Parameters:**
- **builder_class** (_str_) – Full path to the step builder class (e.g., src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder).
- **--verbose** (_Flag_) – Show detailed output including test details and logs.
- **--scoring** (_Flag_) – Enable quality scoring and enhanced reporting.
- **--export-json** (_str_) – Export test results to JSON file at specified path.
- **--export-chart** (_Flag_) – Generate score visualization chart (requires matplotlib).
- **--output-dir** (_str_) – Output directory for exports (default: test_reports).

```bash
# Basic all tests
python -m cursus.cli builder-test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder

# All tests with scoring and exports
python -m cursus.cli builder-test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --scoring --export-json results.json --export-chart

# Verbose testing with custom output directory
python -m cursus.cli builder-test all src.cursus.steps.builders.builder_processing_step.ProcessingStepBuilder --verbose --output-dir ./test_reports
```

### level

level(_level_number_, _builder_class_, _--verbose_, _--scoring_, _--export-json_, _--export-chart_, _--output-dir_)

Runs tests for a specific validation level with focused testing on particular aspects of builder implementation.

**Parameters:**
- **level_number** (_int_) – Test level to run: 1=Interface, 2=Specification, 3=Step Creation, 4=Integration.
- **builder_class** (_str_) – Full path to the step builder class.
- **--verbose** (_Flag_) – Show detailed output including test details and logs.
- **--scoring** (_Flag_) – Enable quality scoring and enhanced reporting.
- **--export-json** (_str_) – Export test results to JSON file at specified path.
- **--export-chart** (_Flag_) – Generate score visualization chart.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# Test interface level
python -m cursus.cli builder-test level 1 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --verbose

# Test specification level with scoring
python -m cursus.cli builder-test level 2 src.cursus.steps.builders.builder_processing_step.ProcessingStepBuilder --scoring

# Test integration level with exports
python -m cursus.cli builder-test level 4 src.cursus.steps.builders.builder_transform_step.TransformStepBuilder --export-json level4_results.json
```

### variant

variant(_variant_name_, _builder_class_, _--verbose_, _--scoring_, _--export-json_, _--export-chart_, _--output-dir_)

Runs tests for a specific variant with specialized testing for particular step types.

**Parameters:**
- **variant_name** (_str_) – Test variant to run (currently supports: processing).
- **builder_class** (_str_) – Full path to the step builder class.
- **--verbose** (_Flag_) – Show detailed output including test details and logs.
- **--scoring** (_Flag_) – Enable quality scoring and enhanced reporting.
- **--export-json** (_str_) – Export test results to JSON file.
- **--export-chart** (_Flag_) – Generate score visualization chart.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# Test processing variant
python -m cursus.cli builder-test variant processing src.cursus.steps.builders.builder_processing_step.ProcessingStepBuilder --verbose

# Processing variant with scoring
python -m cursus.cli builder-test variant processing src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder --scoring --export-json processing_results.json
```

### test-by-type

test-by-type(_sagemaker_type_, _--verbose_, _--scoring_, _--export-json_, _--export-chart_, _--output-dir_)

Tests all builders for a specific SageMaker step type with batch testing capabilities.

**Parameters:**
- **sagemaker_type** (_str_) – SageMaker step type to test (Training, Transform, Processing, CreateModel, RegisterModel).
- **--verbose** (_Flag_) – Show detailed output for all tested builders.
- **--scoring** (_Flag_) – Enable quality scoring for all builders.
- **--export-json** (_str_) – Export batch test results to JSON file.
- **--export-chart** (_Flag_) – Generate score visualization charts for all builders.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# Test all Training step builders
python -m cursus.cli builder-test test-by-type Training --verbose

# Test all Processing builders with scoring
python -m cursus.cli builder-test test-by-type Processing --scoring --export-json processing_batch_results.json

# Comprehensive testing of Transform builders
python -m cursus.cli builder-test test-by-type Transform --verbose --scoring --export-chart --output-dir ./transform_reports
```

### registry-report

registry-report(_--verbose_, _--export-json_, _--output-dir_)

Generates registry discovery report showing available builders and their status.

**Parameters:**
- **--verbose** (_Flag_) – Show detailed output including error details.
- **--export-json** (_str_) – Export registry report to JSON file.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# Generate basic registry report
python -m cursus.cli builder-test registry-report

# Detailed registry report with export
python -m cursus.cli builder-test registry-report --verbose --export-json registry_report.json
```

### validate-builder

validate-builder(_step_name_, _--verbose_, _--export-json_, _--output-dir_)

Validates that a step builder is available and can be loaded from the registry.

**Parameters:**
- **step_name** (_str_) – Step name from registry to validate.
- **--verbose** (_Flag_) – Show detailed validation information.
- **--export-json** (_str_) – Export validation results to JSON file.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# Validate specific builder
python -m cursus.cli builder-test validate-builder XGBoostTraining

# Detailed validation with export
python -m cursus.cli builder-test validate-builder ProcessingStep --verbose --export-json validation_result.json
```

### list-builders

list-builders(_--verbose_, _--export-json_, _--output-dir_)

Lists available step builder classes by scanning the builders directory.

**Parameters:**
- **--verbose** (_Flag_) – Show additional information about builders.
- **--export-json** (_str_) – Export builder list to JSON file.
- **--output-dir** (_str_) – Output directory for exports.

```bash
# List available builders
python -m cursus.cli builder-test list-builders

# List builders with export
python -m cursus.cli builder-test list-builders --export-json builders_list.json
```

### print_test_results

print_test_results(_results_, _verbose=False_, _show_scoring=False_)

Prints test results in a formatted way with optional scoring display and color-coded output.

**Parameters:**
- **results** (_Dict[str, Any]_) – Test results dictionary from test execution.
- **verbose** (_bool_) – Show detailed output including test details and recommendations.
- **show_scoring** (_bool_) – Show scoring information and quality ratings.

```python
from cursus.cli.builder_test_cli import print_test_results

# Print basic results
print_test_results(results)

# Print detailed results with scoring
print_test_results(results, verbose=True, show_scoring=True)
```

### print_enhanced_results

print_enhanced_results(_results_, _verbose=False_)

Prints enhanced results with scoring and structured reporting for comprehensive analysis.

**Parameters:**
- **results** (_Dict[str, Any]_) – Enhanced test results with scoring data.
- **verbose** (_bool_) – Show detailed scoring breakdown and failed tests summary.

```python
from cursus.cli.builder_test_cli import print_enhanced_results

# Print enhanced results
print_enhanced_results(results, verbose=True)
```

### run_all_tests_with_scoring

run_all_tests_with_scoring(_builder_class_, _verbose=False_, _enable_structured_reporting=False_)

Runs all tests with scoring enabled for comprehensive quality assessment.

**Parameters:**
- **builder_class** (_Type_) – Builder class to test.
- **verbose** (_bool_) – Enable verbose output during testing.
- **enable_structured_reporting** (_bool_) – Enable structured reporting for exports.

**Returns:**
- **Dict[str, Any]** – Test results with scoring data and structured reports.

```python
from cursus.cli.builder_test_cli import run_all_tests_with_scoring, import_builder_class

# Import and test builder with scoring
builder_class = import_builder_class("src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder")
results = run_all_tests_with_scoring(builder_class, verbose=True, enable_structured_reporting=True)
```

### run_registry_discovery_report

run_registry_discovery_report()

Generates and displays registry discovery report with comprehensive builder analysis.

**Returns:**
- **Dict[str, Any]** – Registry discovery report with availability statistics.

```python
from cursus.cli.builder_test_cli import run_registry_discovery_report

# Generate registry report
report = run_registry_discovery_report()
print(f"Total steps: {report['total_steps']}")
```

### run_test_by_sagemaker_type

run_test_by_sagemaker_type(_sagemaker_step_type_, _verbose=False_, _enable_scoring=True_)

Tests all builders for a specific SageMaker step type with batch processing capabilities.

**Parameters:**
- **sagemaker_step_type** (_str_) – SageMaker step type to test.
- **verbose** (_bool_) – Enable verbose output for all builders.
- **enable_scoring** (_bool_) – Enable scoring for all tested builders.

**Returns:**
- **Dict[str, Any]** – Batch test results for all builders of the specified type.

```python
from cursus.cli.builder_test_cli import run_test_by_sagemaker_type

# Test all Training builders
results = run_test_by_sagemaker_type("Training", verbose=True, enable_scoring=True)
```

### validate_builder_availability

validate_builder_availability(_step_name_)

Validates that a step builder is available and can be loaded from the registry.

**Parameters:**
- **step_name** (_str_) – Step name from registry to validate.

**Returns:**
- **Dict[str, Any]** – Validation results with availability status and error information.

```python
from cursus.cli.builder_test_cli import validate_builder_availability

# Validate builder availability
validation = validate_builder_availability("XGBoostTraining")
print(f"Loadable: {validation['loadable']}")
```

### export_results_to_json

export_results_to_json(_results_, _output_path_)

Exports test results to JSON file with proper directory creation and error handling.

**Parameters:**
- **results** (_Dict[str, Any]_) – Test results to export.
- **output_path** (_str_) – Path for the output JSON file.

```python
from cursus.cli.builder_test_cli import export_results_to_json

# Export results to JSON
export_results_to_json(results, "test_results.json")
```

### generate_score_chart

generate_score_chart(_results_, _builder_name_, _output_dir_)

Generates score visualization chart for test results analysis.

**Parameters:**
- **results** (_Dict[str, Any]_) – Test results with scoring data.
- **builder_name** (_str_) – Name of the builder for chart title.
- **output_dir** (_str_) – Output directory for chart file.

**Returns:**
- **Optional[str]** – Path to generated chart file, or None if generation failed.

```python
from cursus.cli.builder_test_cli import generate_score_chart

# Generate score chart
chart_path = generate_score_chart(results, "XGBoostTrainingStepBuilder", "./charts")
if chart_path:
    print(f"Chart saved: {chart_path}")
```

### import_builder_class

import_builder_class(_class_path_)

Imports a builder class from a module path with proper error handling and path resolution.

**Parameters:**
- **class_path** (_str_) – Full path to the builder class.

**Returns:**
- **Type** – Imported builder class.

**Raises:**
- **ImportError** – If module cannot be imported.
- **AttributeError** – If class cannot be found in module.

```python
from cursus.cli.builder_test_cli import import_builder_class

# Import builder class
builder_class = import_builder_class("src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder")
print(f"Imported: {builder_class.__name__}")
```

### run_level_tests

run_level_tests(_builder_class_, _level_, _verbose=False_)

Runs tests for a specific level with focused validation on particular aspects.

**Parameters:**
- **builder_class** (_Type_) – Builder class to test.
- **level** (_int_) – Test level (1-4).
- **verbose** (_bool_) – Enable verbose output.

**Returns:**
- **Dict[str, Dict[str, Any]]** – Level-specific test results.

**Raises:**
- **ValueError** – If invalid test level is specified.

```python
from cursus.cli.builder_test_cli import run_level_tests, import_builder_class

# Run Level 1 tests
builder_class = import_builder_class("src.cursus.steps.builders.builder_processing_step.ProcessingStepBuilder")
results = run_level_tests(builder_class, 1, verbose=True)
```

### run_variant_tests

run_variant_tests(_builder_class_, _variant_, _verbose=False_)

Runs tests for a specific variant with specialized testing for particular step types.

**Parameters:**
- **builder_class** (_Type_) – Builder class to test.
- **variant** (_str_) – Test variant name.
- **verbose** (_bool_) – Enable verbose output.

**Returns:**
- **Dict[str, Dict[str, Any]]** – Variant-specific test results.

**Raises:**
- **ValueError** – If invalid variant is specified.

```python
from cursus.cli.builder_test_cli import run_variant_tests, import_builder_class

# Run processing variant tests
builder_class = import_builder_class("src.cursus.steps.builders.builder_processing_step.ProcessingStepBuilder")
results = run_variant_tests(builder_class, "processing", verbose=True)
```

### run_all_tests

run_all_tests(_builder_class_, _verbose=False_, _enable_scoring=False_)

Runs all tests (universal test suite) with optional scoring capabilities.

**Parameters:**
- **builder_class** (_Type_) – Builder class to test.
- **verbose** (_bool_) – Enable verbose output.
- **enable_scoring** (_bool_) – Enable quality scoring.

**Returns:**
- **Dict[str, Any]** – Complete test results.

```python
from cursus.cli.builder_test_cli import run_all_tests, import_builder_class

# Run all tests with scoring
builder_class = import_builder_class("src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder")
results = run_all_tests(builder_class, verbose=True, enable_scoring=True)
```

### list_available_builders

list_available_builders()

Lists available step builder classes by scanning the builders directory with dependency checking.

**Returns:**
- **List[str]** – List of available builder class paths.

```python
from cursus.cli.builder_test_cli import list_available_builders

# Get available builders
builders = list_available_builders()
for builder in builders:
    print(f"Available: {builder}")
```

### main

main()

Main CLI entry point with comprehensive argument parsing and command routing.

**Returns:**
- **int** – Exit code (0 for success, 1 for error).

```python
from cursus.cli.builder_test_cli import main

# Run CLI
exit_code = main()
```

## Testing Levels

The builder test CLI validates four critical levels of builder implementation:

### Level 1: Interface Tests
- **Inheritance**: Validates proper inheritance from base classes
- **Naming Conventions**: Checks class and method naming standards
- **Required Methods**: Ensures all required methods are implemented
- **Registry Integration**: Validates integration with step registry
- **Documentation Standards**: Checks docstring and documentation quality
- **Type Hints**: Validates proper type annotation usage
- **Error Handling**: Tests exception handling and error reporting
- **Method Return Types**: Validates return type consistency
- **Configuration Validation**: Tests configuration parameter validation
- **Generic Step Creation**: Tests generic step creation capabilities

### Level 2: Specification Tests
- **Specification Usage**: Validates proper specification implementation
- **Contract Alignment**: Checks alignment with step contracts
- **Environment Variable Handling**: Tests environment variable processing
- **Job Arguments**: Validates job argument handling
- **Environment Variables Processing**: Tests environment setup
- **Property Files Configuration**: Validates property file handling

### Level 3: Step Creation Tests
- **Step Instantiation**: Tests step object creation
- **Step Configuration Validity**: Validates configuration parameter usage
- **Step Dependencies Attachment**: Tests dependency injection
- **Step Name Generation**: Validates step naming logic
- **Input Path Mapping**: Tests input path configuration
- **Output Path Mapping**: Tests output path configuration
- **Property Path Validity**: Validates property path generation
- **Processing Inputs Outputs**: Tests processing step I/O handling
- **Processing Code Handling**: Validates code injection and execution

### Level 4: Integration Tests
- **Dependency Resolution**: Tests dependency resolution logic
- **Step Creation**: Validates end-to-end step creation
- **Step Name**: Tests step name generation in context
- **Generic Dependency Handling**: Tests generic dependency patterns
- **Processing Step Dependencies**: Validates processing-specific dependencies

## Variant Testing

### Processing Variant
- **Processor Creation**: Tests processing step creation
- **Estimator Methods**: Validates estimator interface implementation
- **Transformer Methods**: Tests transformer interface implementation
- **Step Type Validation**: Ensures proper step type classification

## Scoring System

The builder test CLI includes a comprehensive scoring system:

### Overall Score
- Calculated from all test levels and variants
- Weighted scoring based on test importance and failure severity
- Range: 0-100 with quality rating categories

### Quality Ratings
- **Excellent** (90-100): Outstanding implementation with minimal issues
- **Good** (80-89): Strong implementation with minor issues
- **Satisfactory** (70-79): Acceptable implementation with some issues
- **Needs Work** (60-69): Significant issues requiring attention
- **Poor** (0-59): Major implementation problems requiring immediate attention

### Level Scores
- Individual scores for each test level
- Detailed breakdown of test failures and issues
- Recommendations for improvement at each level

## Registry Integration

### Discovery Features
- **Automatic Discovery**: Scans registry for available builders
- **Availability Validation**: Checks if builders can be loaded
- **Dependency Analysis**: Identifies missing dependencies
- **Error Reporting**: Provides detailed error information for failed imports

### Batch Testing
- **Type-based Testing**: Test all builders of a specific SageMaker step type
- **Comprehensive Coverage**: Ensures all registered builders are testable
- **Performance Analysis**: Tracks testing performance across builders

## Export Capabilities

### JSON Export
- **Complete Results**: Exports all test results and scoring data
- **Structured Format**: Machine-readable format for integration
- **Metadata Inclusion**: Includes timestamps and test configuration
- **Error Handling**: Robust serialization with fallback options

### Chart Generation
- **Score Visualization**: High-resolution charts showing score breakdowns
- **Level Analysis**: Visual representation of level-specific scores
- **Trend Analysis**: Historical comparison capabilities
- **Export Formats**: PNG charts with customizable styling

## Usage Patterns

### Development Workflow
```bash
# 1. List available builders
python -m cursus.cli builder-test list-builders

# 2. Validate specific builder availability
python -m cursus.cli builder-test validate-builder XGBoostTraining

# 3. Run comprehensive tests with scoring
python -m cursus.cli builder-test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --scoring --verbose

# 4. Focus on specific level if issues found
python -m cursus.cli builder-test level 2 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --verbose

# 5. Generate reports and charts
python -m cursus.cli builder-test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --scoring --export-json results.json --export-chart
```

### Quality Assurance
```bash
# Test all builders of specific type
python -m cursus.cli builder-test test-by-type Training --scoring --export-json training_builders_report.json

# Generate registry discovery report
python -m cursus.cli builder-test registry-report --verbose --export-json registry_status.json

# Comprehensive testing with exports
python -m cursus.cli builder-test all MyBuilder --scoring --export-json --export-chart --output-dir ./qa-reports
```

### CI/CD Integration
```bash
# Automated testing in CI pipeline
python -m cursus.cli builder-test all $BUILDER_CLASS --export-json ci-test-results.json

# Batch testing for release validation
python -m cursus.cli builder-test test-by-type Processing --scoring --export-json processing-validation.json
```

## Error Handling

The builder test CLI provides comprehensive error handling:

- **Import Errors**: Graceful handling of missing modules or classes
- **Test Execution Errors**: Detailed error reporting for test failures
- **Registry Errors**: Comprehensive validation of registry availability
- **Export Errors**: Robust file handling with proper error messages
- **Dependency Errors**: Clear reporting of missing dependencies

## Integration Points

- **UniversalStepBuilderTest**: Core testing engine for comprehensive validation
- **StepBuilderScorer**: Scoring system for quantitative quality assessment
- **RegistryStepDiscovery**: Registry integration for builder discovery
- **Validation Framework**: Integration with cursus validation infrastructure
- **Export Systems**: Multiple output formats for different use cases

## Related Documentation

- [CLI Module](__init__.md) - Main CLI dispatcher and command routing
- [Alignment CLI](alignment_cli.md) - Comprehensive alignment validation tools
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
- [Universal Step Builder Test](../../validation/builders/universal_test.md) - Core testing engine
- [Step Builder Scorer](../../validation/builders/scoring.md) - Quality scoring system
- [Registry Step Discovery](../../validation/builders/registry_discovery.md) - Builder discovery system
