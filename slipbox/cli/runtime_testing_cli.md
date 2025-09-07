---
tags:
  - code
  - cli
  - runtime
  - testing
  - pipeline
  - scripts
keywords:
  - runtime
  - test_script
  - test_pipeline
  - test_compatibility
  - RuntimeTester
  - pipeline testing
  - script validation
  - data compatibility
topics:
  - runtime testing
  - pipeline validation
  - script testing
  - CLI tools
language: python
date of note: 2024-12-07
---

# Runtime Testing CLI

Simplified command-line interface for pipeline runtime testing providing comprehensive tools for testing individual scripts and complete pipelines for functionality and data flow compatibility.

## Overview

The Runtime Testing CLI provides comprehensive testing tools for cursus pipeline scripts and complete pipeline flows, supporting individual script functionality testing with execution validation, complete pipeline flow testing with data compatibility checks, and data compatibility testing between script pairs with detailed issue reporting. The CLI offers both text and JSON output formats for integration with different workflows.

The module supports workspace-based testing with configurable test environments, comprehensive error reporting with detailed diagnostics, and flexible output formats for both human-readable and machine-processable results. All commands provide detailed help documentation and examples for effective usage.

## Classes and Methods

### Commands
- [`runtime`](#runtime) - Main command group for runtime testing tools
- [`test_script`](#test_script) - Test a single script functionality
- [`test_pipeline`](#test_pipeline) - Test complete pipeline flow
- [`test_compatibility`](#test_compatibility) - Test data compatibility between two scripts

### Functions
- [`main`](#main) - Main entry point for runtime testing CLI

## API Reference

### runtime

@click.group()

Main command group for pipeline runtime testing tools with version information.

```bash
# Access runtime testing commands
python -m cursus.cli runtime-testing --help

# Show version information
python -m cursus.cli runtime-testing --version
```

### test_script

test_script(_script_name_, _--workspace-dir_, _--output-format_)

Tests a single script functionality with execution validation and performance metrics.

**Parameters:**
- **script_name** (_str_) – Name of the script to test.
- **--workspace-dir** (_str_) – Workspace directory for test execution (default: ./test_workspace).
- **--output-format** (_Choice_) – Output format for results: text or json (default: text).

```bash
# Basic script testing
python -m cursus.cli runtime-testing test_script my_script

# Test with custom workspace
python -m cursus.cli runtime-testing test_script data_processing --workspace-dir ./custom_workspace

# JSON output for automation
python -m cursus.cli runtime-testing test_script model_training --output-format json

# Test script in specific workspace
python -m cursus.cli runtime-testing test_script xgboost_training --workspace-dir ./ml_workspace --output-format text
```

### test_pipeline

test_pipeline(_pipeline_config_, _--workspace-dir_, _--output-format_)

Tests complete pipeline flow with comprehensive validation of all components and data flows.

**Parameters:**
- **pipeline_config** (_str_) – Path to pipeline configuration file (JSON format).
- **--workspace-dir** (_str_) – Workspace directory for test execution (default: ./test_workspace).
- **--output-format** (_Choice_) – Output format for results: text or json (default: text).

```bash
# Basic pipeline testing
python -m cursus.cli runtime-testing test_pipeline pipeline_config.json

# Test with custom workspace
python -m cursus.cli runtime-testing test_pipeline ml_pipeline.json --workspace-dir ./ml_workspace

# JSON output for CI/CD integration
python -m cursus.cli runtime-testing test_pipeline training_pipeline.json --output-format json

# Comprehensive pipeline validation
python -m cursus.cli runtime-testing test_pipeline complex_pipeline.json --workspace-dir ./test_env --output-format text
```

### test_compatibility

test_compatibility(_script_a_, _script_b_, _--workspace-dir_, _--output-format_)

Tests data compatibility between two scripts with detailed compatibility analysis.

**Parameters:**
- **script_a** (_str_) – First script name (data producer).
- **script_b** (_str_) – Second script name (data consumer).
- **--workspace-dir** (_str_) – Workspace directory for test execution (default: ./test_workspace).
- **--output-format** (_Choice_) – Output format for results: text or json (default: text).

```bash
# Basic compatibility testing
python -m cursus.cli runtime-testing test_compatibility preprocessing training

# Test data flow compatibility
python -m cursus.cli runtime-testing test_compatibility data_loader model_trainer --workspace-dir ./ml_workspace

# JSON output for detailed analysis
python -m cursus.cli runtime-testing test_compatibility feature_engineering model_evaluation --output-format json

# Cross-script compatibility validation
python -m cursus.cli runtime-testing test_compatibility transform_data train_model --workspace-dir ./pipeline_test --output-format text
```

### main

main()

Main entry point for runtime testing CLI with command group initialization.

```python
from cursus.cli.runtime_testing_cli import main

# Run runtime testing CLI
main()
```

## Testing Features

### Script Testing
The CLI provides comprehensive script testing capabilities:

#### Functionality Validation
- **Execution Testing**: Validates that scripts can execute without errors
- **Main Function Detection**: Checks for proper main function implementation
- **Performance Metrics**: Measures execution time and resource usage
- **Error Reporting**: Provides detailed error messages and stack traces

#### Test Results
- **Success Status**: Boolean indicator of test success/failure
- **Execution Time**: Precise timing measurements in seconds
- **Main Function Status**: Verification of main function presence
- **Error Messages**: Detailed error information when tests fail

### Pipeline Testing
The CLI supports complete pipeline flow testing:

#### Pipeline Configuration
Pipeline configuration files use JSON format:

```json
{
  "pipeline_name": "ml_training_pipeline",
  "scripts": [
    {
      "name": "data_preprocessing",
      "dependencies": []
    },
    {
      "name": "feature_engineering", 
      "dependencies": ["data_preprocessing"]
    },
    {
      "name": "model_training",
      "dependencies": ["feature_engineering"]
    }
  ],
  "data_flows": [
    {
      "from": "data_preprocessing",
      "to": "feature_engineering",
      "data_type": "pandas.DataFrame"
    },
    {
      "from": "feature_engineering", 
      "to": "model_training",
      "data_type": "numpy.ndarray"
    }
  ]
}
```

#### Pipeline Validation
- **Script Execution**: Tests all scripts in the pipeline
- **Data Flow Validation**: Verifies data compatibility between steps
- **Dependency Resolution**: Validates script dependencies
- **Error Aggregation**: Collects and reports all pipeline errors

### Compatibility Testing
The CLI provides detailed data compatibility testing:

#### Compatibility Analysis
- **Data Type Matching**: Validates data type compatibility
- **Schema Validation**: Checks data structure compatibility
- **Format Verification**: Ensures data format consistency
- **Issue Reporting**: Provides detailed compatibility issue descriptions

#### Sample Data Generation
- **Automatic Generation**: Creates sample data for testing
- **Type-aware Testing**: Generates appropriate data types
- **Realistic Scenarios**: Uses realistic data patterns
- **Configurable Parameters**: Supports custom data generation

## Output Formats

### Text Output
Human-readable format with color-coded status indicators:

```
Script: data_preprocessing
Status: PASS
Execution time: 0.245s
Has main function: Yes
```

```
Pipeline: ml_pipeline.json
Status: PASS

Script Results:
  data_preprocessing: PASS
  feature_engineering: PASS
  model_training: PASS

Data Flow Results:
  preprocessing->engineering: PASS
  engineering->training: PASS
```

### JSON Output
Machine-readable format for automation and integration:

```json
{
  "success": true,
  "execution_time": 0.245,
  "has_main_function": true,
  "error_message": null
}
```

```json
{
  "pipeline_success": true,
  "errors": [],
  "script_results": {
    "data_preprocessing": {
      "success": true,
      "execution_time": 0.245,
      "has_main_function": true,
      "error_message": null
    }
  },
  "data_flow_results": {
    "preprocessing->engineering": {
      "compatible": true,
      "compatibility_issues": []
    }
  }
}
```

## Workspace Management

### Workspace Structure
The CLI uses workspace directories for test execution:

```
test_workspace/
├── scripts/              # Script files for testing
├── data/                 # Test data files
├── config/               # Configuration files
├── temp/                 # Temporary execution files
└── results/              # Test result outputs
```

### Workspace Configuration
- **Isolated Execution**: Each test runs in isolated workspace
- **Configurable Paths**: Custom workspace directory support
- **Resource Management**: Automatic cleanup of temporary files
- **Data Persistence**: Optional result persistence for analysis

## Usage Patterns

### Development Workflow
```bash
# 1. Test individual script during development
python -m cursus.cli runtime-testing test_script my_new_script --workspace-dir ./dev_workspace

# 2. Test data compatibility between scripts
python -m cursus.cli runtime-testing test_compatibility data_loader feature_extractor --workspace-dir ./dev_workspace

# 3. Test complete pipeline before deployment
python -m cursus.cli runtime-testing test_pipeline production_pipeline.json --workspace-dir ./staging_workspace

# 4. Generate JSON reports for CI/CD
python -m cursus.cli runtime-testing test_pipeline pipeline.json --output-format json > test_results.json
```

### CI/CD Integration
```bash
# Automated testing in CI pipeline
python -m cursus.cli runtime-testing test_pipeline pipeline.json --output-format json --workspace-dir ./ci_workspace

# Exit code handling for build systems
if python -m cursus.cli runtime-testing test_script critical_script --workspace-dir ./ci_workspace; then
    echo "Script test passed"
else
    echo "Script test failed"
    exit 1
fi
```

### Quality Assurance
```bash
# Comprehensive pipeline validation
python -m cursus.cli runtime-testing test_pipeline full_pipeline.json --workspace-dir ./qa_workspace --output-format text

# Batch compatibility testing
for script_pair in "a b" "b c" "c d"; do
    python -m cursus.cli runtime-testing test_compatibility $script_pair --workspace-dir ./qa_workspace
done
```

## Error Handling

The runtime testing CLI provides comprehensive error handling:

- **Script Execution Errors**: Captures and reports script runtime errors
- **Configuration Errors**: Validates pipeline configuration files
- **Workspace Errors**: Handles workspace setup and access issues
- **Data Compatibility Errors**: Reports detailed compatibility issues
- **System Errors**: Graceful handling of system-level failures

### Exit Codes
- **0**: All tests passed successfully
- **1**: One or more tests failed or errors occurred

## Integration Points

- **RuntimeTester**: Core testing engine for script and pipeline validation
- **RuntimeTestingConfiguration**: Configuration management for test execution
- **PipelineTestingSpecBuilder**: Pipeline specification building and validation
- **PipelineDAG**: Pipeline dependency analysis and execution planning
- **Validation Framework**: Integration with cursus validation infrastructure

## Performance Considerations

- **Isolated Execution**: Each test runs in isolated environment
- **Resource Monitoring**: Tracks execution time and resource usage
- **Parallel Testing**: Supports concurrent test execution where possible
- **Memory Management**: Efficient memory usage for large pipelines
- **Cleanup Operations**: Automatic cleanup of temporary resources

## Related Documentation

- [CLI Module](__init__.md) - Main CLI dispatcher and command routing
- [Alignment CLI](alignment_cli.md) - Comprehensive alignment validation tools
- [Builder Test CLI](builder_test_cli.md) - Step builder testing and validation
- [Validation CLI](validation_cli.md) - Naming and interface validation tools
- [Registry CLI](registry_cli.md) - Registry management and workspace tools
- [Runtime Testing Framework](../../validation/runtime/runtime_testing.md) - Core runtime testing engine
- [Runtime Models](../../validation/runtime/runtime_models.md) - Data models for runtime testing
