# Base Classes Test Suite

This directory contains comprehensive unit tests for all base classes in `cursus.core.base`.

## Overview

The base classes form the foundation of the cursus pipeline system, providing:
- Configuration management (`config_base.py`)
- Step building abstractions (`builder_base.py`) 
- Pipeline specifications (`specification_base.py`)
- Script contracts (`contract_base.py`)
- Hyperparameter management (`hyperparameters_base.py`)
- Shared enumerations (`enums.py`)

## Test Files

### Individual Test Modules

| Test File | Target Module | Description |
|-----------|---------------|-------------|
| `test_config_base.py` | `config_base.py` | Tests for `BasePipelineConfig` class |
| `test_builder_base.py` | `builder_base.py` | Tests for `StepBuilderBase` abstract class |
| `test_specification_base.py` | `specification_base.py` | Tests for specification classes (`StepSpecification`, `OutputSpec`, `DependencySpec`) |
| `test_contract_base.py` | `contract_base.py` | Tests for `ScriptContract` and `ScriptAnalyzer` classes |
| `test_hyperparameters_base.py` | `hyperparameters_base.py` | Tests for `ModelHyperparameters` class |
| `test_enums.py` | `enums.py` | Tests for `DependencyType` and `NodeType` enums |

### Test Runners

| File | Purpose |
|------|---------|
| `test_all_base.py` | Comprehensive test runner for all base classes |
| `__init__.py` | Package initialization |

## Running Tests

### Run All Tests

```bash
# From project root
python -m test.base.test_all_base

# Or with summary (default)
python -m test.base.test_all_base --summary
```

### Run Individual Modules

```bash
# Run each module separately
python -m test.base.test_all_base --individual

# Or run specific modules
python -m test.base.test_config_base
python -m test.base.test_builder_base
python -m test.base.test_specification_base
python -m test.base.test_contract_base
python -m test.base.test_hyperparameters_base
python -m test.base.test_enums
```

### Using unittest directly

```bash
# Run all tests in the directory
python -m unittest discover test.base

# Run specific test file
python -m unittest test.base.test_config_base

# Run specific test class
python -m unittest test.base.test_config_base.TestBasePipelineConfig

# Run specific test method
python -m unittest test.base.test_config_base.TestBasePipelineConfig.test_init_with_required_fields
```

## Test Coverage

### BasePipelineConfig Tests (`test_config_base.py`)

- ✅ Initialization with required/optional fields
- ✅ Derived property calculations (aws_region, pipeline_name, etc.)
- ✅ Region validation and mapping
- ✅ Source directory validation
- ✅ Field categorization (essential, system, derived)
- ✅ Configuration inheritance (`from_base_config`)
- ✅ String representation and printing
- ✅ Model serialization

### StepBuilderBase Tests (`test_builder_base.py`)

- ✅ Abstract class implementation requirements
- ✅ Initialization with required/optional parameters
- ✅ Name sanitization for SageMaker
- ✅ Step name generation
- ✅ Job name generation
- ✅ Property path handling
- ✅ Dependency management
- ✅ Environment variable handling
- ✅ Job argument processing
- ✅ Safe logging methods

### Specification Tests (`test_specification_base.py`)

- ✅ `OutputSpec` creation and matching
- ✅ `DependencySpec` creation and matching
- ✅ `ValidationResult` and `AlignmentResult` handling
- ✅ `StepSpecification` validation
- ✅ Contract alignment validation
- ✅ Dependency categorization (required/optional)
- ✅ Output/dependency name resolution with aliases

### Contract Tests (`test_contract_base.py`)

- ✅ `ScriptContract` validation and creation
- ✅ Path validation (input/output paths)
- ✅ Argument validation (CLI conventions)
- ✅ Implementation validation against contracts
- ✅ `ScriptAnalyzer` AST parsing
- ✅ Environment variable extraction
- ✅ Argument usage detection
- ✅ Input/output path detection

### Hyperparameters Tests (`test_hyperparameters_base.py`)

- ✅ Three-tier field categorization
- ✅ Derived property calculations
- ✅ Binary/multiclass classification detection
- ✅ Class weights handling
- ✅ Validation constraints (batch_size, max_epochs)
- ✅ Configuration inheritance
- ✅ Serialization for SageMaker
- ✅ Field caching behavior

### Enum Tests (`test_enums.py`)

- ✅ `DependencyType` enum values and members
- ✅ `NodeType` enum values and members
- ✅ Equality comparisons
- ✅ Hashability for dictionary keys and sets
- ✅ String representations
- ✅ Iteration over enum members
- ✅ Type checking and edge cases

## Test Patterns

### Common Test Patterns Used

1. **Setup/Teardown**: Using `setUp()` method for test fixtures
2. **Mocking**: Using `unittest.mock` for external dependencies
3. **Exception Testing**: Using `assertRaises()` for validation testing
4. **Property Testing**: Verifying derived properties and caching
5. **Edge Case Testing**: Testing boundary conditions and error cases
6. **Integration Testing**: Testing interactions between classes

### Mock Usage

Tests extensively use mocking for:
- SageMaker sessions
- File system operations
- AST parsing
- Registry managers
- Dependency resolvers

### Temporary Files

Some tests (especially `test_contract_base.py`) create temporary files for testing script analysis functionality.

## Test Statistics

The test suite includes:
- **200+** individual test methods
- **6** test modules covering all base classes
- **Comprehensive coverage** of public APIs
- **Edge case testing** for error conditions
- **Integration testing** between related classes

## Dependencies

Test dependencies include:
- `unittest` (standard library)
- `unittest.mock` (standard library)
- `tempfile` (for temporary file testing)
- `pathlib` (for path handling)
- `json` (for serialization testing)

## Contributing

When adding new tests:

1. Follow the existing naming conventions (`test_<functionality>`)
2. Include docstrings describing what each test validates
3. Use appropriate mocking for external dependencies
4. Test both success and failure cases
5. Update this README if adding new test files
6. Ensure tests are deterministic and don't depend on external state

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the project root
2. **Path Issues**: The tests add the project root to `sys.path` automatically
3. **Mock Issues**: Verify mock patches target the correct module paths
4. **Temporary Files**: Tests clean up temporary files automatically

### Debug Mode

Run tests with verbose output:
```bash
python -m unittest test.base.test_config_base -v
```

### Specific Test Debugging

To debug a specific failing test:
```bash
python -m unittest test.base.test_config_base.TestBasePipelineConfig.test_specific_method -v
