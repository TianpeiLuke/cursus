# Runtime Testing Module Unit Tests

This directory contains comprehensive unit tests for the Pipeline Runtime Testing System components located in `src/cursus/validation/runtime/`.

## Overview

The Pipeline Runtime Testing System provides functionality for testing pipeline scripts in isolation and end-to-end, with support for synthetic and real data sources. These unit tests ensure the reliability and correctness of all core components.

## Test Structure

The test directory mirrors the structure of the source code:

```
test/validation/runtime/
├── __init__.py
├── README.md
├── run_all_tests.py          # Test runner script
├── core/                     # Tests for core execution components
│   ├── __init__.py
│   ├── test_pipeline_script_executor.py
│   ├── test_script_import_manager.py
│   └── test_data_flow_manager.py
└── utils/                    # Tests for utility components
    ├── __init__.py
    ├── test_result_models.py
    ├── test_execution_context.py
    └── test_error_handling.py
```

## Test Coverage

### Core Components (`core/`)

#### `test_pipeline_script_executor.py`
Tests for the main orchestrator class `PipelineScriptExecutor`:
- **Initialization**: Workspace creation, manager initialization
- **Script Discovery**: Path resolution, error handling for missing scripts
- **Execution Context**: Synthetic and local data source preparation
- **Isolation Testing**: Success and failure scenarios
- **Error Handling**: Comprehensive error categorization and recommendations
- **Performance Monitoring**: Execution time and memory usage tracking

**Key Test Cases:**
- ✅ Successful script execution with synthetic data
- ✅ Script import errors with specific recommendations
- ✅ Script execution errors with proper error categorization
- ✅ Configuration errors with helpful guidance
- ✅ File not found errors with clear messaging
- ✅ Performance recommendations for slow/memory-intensive scripts

#### `test_script_import_manager.py`
Tests for dynamic script importing and execution:
- **Dynamic Import**: Module loading, caching, error handling
- **Script Execution**: Function calling, parameter passing, result capture
- **Memory Monitoring**: psutil integration, fallback handling
- **Error Conversion**: Generic exceptions to specific error types
- **Performance Tracking**: Execution time and memory usage measurement

**Key Test Cases:**
- ✅ Successful script import and caching
- ✅ Import failures (missing files, syntax errors, no main function)
- ✅ Script execution with proper parameter passing
- ✅ Memory usage calculation with and without psutil
- ✅ Error chaining and cause preservation

#### `test_data_flow_manager.py`
Tests for data flow management between pipeline steps:
- **Directory Management**: Workspace setup, path creation
- **Input Setup**: Upstream output mapping, path validation
- **Output Capture**: Result validation, metadata generation
- **Data Lineage**: Tracking, persistence, error handling
- **File Operations**: JSON serialization, error recovery

**Key Test Cases:**
- ✅ Workspace directory creation and management
- ✅ Input path mapping from upstream outputs
- ✅ Output capture with metadata generation
- ✅ Data lineage tracking and persistence
- ✅ Error handling for missing files and I/O failures

### Utility Components (`utils/`)

#### `test_result_models.py`
Tests for Pydantic models used throughout the system:
- **ExecutionResult**: Script execution outcomes, serialization
- **TestResult**: Test outcomes, success determination, recommendations
- **Model Validation**: Field validation, edge cases
- **Serialization**: Pydantic v2 model_dump() functionality

**Key Test Cases:**
- ✅ Model creation with minimal and full field sets
- ✅ Success/failure status handling
- ✅ Timestamp generation and validation
- ✅ Recommendation list management
- ✅ Pydantic v2 serialization compatibility

#### `test_execution_context.py`
Tests for script execution context management:
- **Context Creation**: Path dictionaries, environment variables
- **Job Arguments**: argparse.Namespace handling, complex parameters
- **Path Validation**: Absolute/relative paths, mixed formats
- **Serialization**: Model serialization with argparse.Namespace
- **Edge Cases**: Empty contexts, None values, complex data types

**Key Test Cases:**
- ✅ Context creation with various path configurations
- ✅ Complex job arguments with nested data structures
- ✅ Environment variable type validation (all strings)
- ✅ Serialization with and without job_args
- ✅ Path format validation and edge cases

#### `test_error_handling.py`
Tests for custom exception classes:
- **Error Creation**: All custom error types
- **Inheritance**: Proper Exception inheritance
- **Error Chaining**: Cause preservation, context handling
- **Specificity**: Individual error type catching
- **Complex Messages**: Multi-line, detailed error messages

**Key Test Cases:**
- ✅ All custom error types (ScriptExecutionError, ScriptImportError, etc.)
- ✅ Error inheritance and exception hierarchy
- ✅ Error chaining with __cause__ and __context__
- ✅ Specific error type catching and handling
- ✅ Complex error messages and details

## Running Tests

### Run All Tests
```bash
# From the project root
python test/validation/runtime/run_all_tests.py
```

### Run Specific Module Tests
```bash
# Core components
python test/validation/runtime/run_all_tests.py core

# Utility components  
python test/validation/runtime/run_all_tests.py utils

# Specific component tests
python test/validation/runtime/run_all_tests.py pipeline_script_executor
python test/validation/runtime/run_all_tests.py script_import_manager
python test/validation/runtime/run_all_tests.py data_flow_manager
python test/validation/runtime/run_all_tests.py result_models
python test/validation/runtime/run_all_tests.py execution_context
python test/validation/runtime/run_all_tests.py error_handling
```

### Run Individual Test Files
```bash
# From the project root
python -m unittest test.validation.runtime.core.test_pipeline_script_executor
python -m unittest test.validation.runtime.utils.test_result_models
```

## Test Design Principles

### Comprehensive Coverage
- **Happy Path**: All successful execution scenarios
- **Error Paths**: All failure modes and edge cases
- **Boundary Conditions**: Empty inputs, None values, extreme values
- **Integration Points**: Component interactions and data flow

### Isolation and Mocking
- **Unit Isolation**: Each test focuses on a single component
- **Mock Dependencies**: External dependencies are mocked appropriately
- **Temporary Resources**: Tests use temporary directories and cleanup properly
- **No Side Effects**: Tests don't affect the file system or global state

### Realistic Scenarios
- **Real-World Data**: Tests use realistic file paths, error messages, and data structures
- **Error Simulation**: Tests simulate actual error conditions that could occur in production
- **Performance Considerations**: Tests validate performance monitoring and recommendations
- **Configuration Variety**: Tests cover different configuration scenarios

### Maintainability
- **Clear Naming**: Test methods have descriptive names indicating what they test
- **Good Documentation**: Each test class and method is well-documented
- **Logical Organization**: Tests are grouped by functionality and component
- **Easy Debugging**: Test failures provide clear information about what went wrong

## Dependencies

The tests require the following packages:
- `unittest` (Python standard library)
- `unittest.mock` (Python standard library)
- `tempfile` (Python standard library)
- `pathlib` (Python standard library)
- `pydantic` (for model testing)
- `argparse` (Python standard library)

Optional dependencies:
- `psutil` (for memory monitoring tests - gracefully handles absence)

## Integration with CI/CD

These tests are designed to be run in continuous integration environments:

- **No External Dependencies**: Tests don't require external services or files
- **Deterministic**: Tests produce consistent results across runs
- **Fast Execution**: Tests complete quickly for rapid feedback
- **Clear Reporting**: Test failures provide actionable information

## Future Enhancements

As the Pipeline Runtime Testing System evolves, additional test coverage will be added for:

- **Data Management Components**: Synthetic data generation, S3 integration
- **Testing Modes**: Pipeline end-to-end testing, deep dive analysis
- **Jupyter Integration**: Notebook interface, visualization components
- **Production Features**: Deployment validation, health checking

## Contributing

When adding new components to the runtime testing system:

1. **Mirror Structure**: Create corresponding test files in the same directory structure
2. **Comprehensive Coverage**: Include tests for all public methods and error conditions
3. **Follow Patterns**: Use the same testing patterns and conventions as existing tests
4. **Update Runner**: Add new test modules to the test runner configuration
5. **Document Tests**: Update this README with information about new test coverage

## Related Documentation

- **[Pipeline Runtime Testing Master Design](../../../slipbox/1_design/pipeline_runtime_testing_master_design.md)**: Overall system design
- **[Core Engine Design](../../../slipbox/1_design/pipeline_runtime_core_engine_design.md)**: Core component specifications
- **[Implementation Guide](../../../src/cursus/validation/runtime/README.md)**: Implementation documentation
