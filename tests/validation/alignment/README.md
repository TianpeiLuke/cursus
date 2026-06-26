# Alignment Validation Test Suite

This directory contains comprehensive tests for the alignment validation system, which ensures consistency between scripts, contracts, specifications, and builder configurations.

## Test Structure

The tests have been organized into focused, single-responsibility test files for better maintainability:

### Utils Tests (`utils/`)
- `test_severity_level.py` - Tests for SeverityLevel enum
- `test_alignment_level.py` - Tests for AlignmentLevel enum  
- `test_alignment_issue.py` - Tests for AlignmentIssue model
- `test_path_reference.py` - Tests for PathReference model
- `test_utility_functions.py` - Tests for utility functions

### Reporter Tests (`reporter/`)
- `test_validation_result.py` - Tests for ValidationResult model
- `test_alignment_report.py` - Tests for AlignmentReport class

### Script-Contract Tests (`script_contract/`)
- `test_path_validation.py` - Tests for path usage validation
- `test_argument_validation.py` - Tests for argument parsing validation

### Unified Tester Tests (`unified_tester/`)
- `test_level_validation.py` - Tests for individual level validation
- `test_full_validation.py` - Tests for full validation orchestration

## Running Tests

### Run All Tests
```bash
python test/validation/alignment/run_all_alignment_tests.py
```

### Run Specific Categories
```bash
# Run only utils tests
python test/validation/alignment/run_all_alignment_tests.py --module utils

# Run only reporter tests  
python test/validation/alignment/run_all_alignment_tests.py --module reporter

# Run only script-contract tests
python test/validation/alignment/run_all_alignment_tests.py --module script_contract

# Run only unified tester tests
python test/validation/alignment/run_all_alignment_tests.py --module unified
```

### Run Specific Test Files
```bash
# Run a specific test file
python -m unittest test.validation.alignment.utils.test_severity_level

# Run a specific test class
python -m unittest test.validation.alignment.utils.test_severity_level.TestSeverityLevel

# Run a specific test method
python -m unittest test.validation.alignment.utils.test_severity_level.TestSeverityLevel.test_severity_levels_exist
```

### Test Options
```bash
# Run with minimal output
python test/validation/alignment/run_all_alignment_tests.py -v 1

# Show coverage report
python test/validation/alignment/run_all_alignment_tests.py --coverage

# Run specific test class
python test/validation/alignment/run_all_alignment_tests.py --test TestSeverityLevel
```

## Test Coverage

The test suite provides comprehensive coverage for:

### Alignment Utilities (8 components)
- ✅ SeverityLevel enum
- ✅ AlignmentLevel enum
- ✅ AlignmentIssue model
- ✅ PathReference model
- ✅ EnvVarAccess model
- ✅ ImportStatement model
- ✅ ArgumentDefinition model
- ✅ Utility functions

### Alignment Reporter (7 components)
- ✅ ValidationResult model
- ✅ AlignmentSummary model
- ✅ AlignmentRecommendation model
- ✅ AlignmentReport class
- ✅ JSON export
- ✅ HTML export
- ✅ Recommendation generation

### Script-Contract Alignment (6 components)
- ✅ Path usage validation
- ✅ Environment variable validation
- ✅ Argument parsing validation
- ✅ Import validation
- ✅ Script analysis
- ✅ Contract validation

### Unified Alignment Tester (7 components)
- ✅ Level 1 validation
- ✅ Level 2 validation
- ✅ Level 3 validation
- ✅ Level 4 validation
- ✅ Full validation orchestration
- ✅ Report generation
- ✅ Error handling

**Total: 28 components tested across 11 test files**

## Test Design Principles

### Single Responsibility
Each test file focuses on testing a single class or component, making tests easier to understand and maintain.

### Comprehensive Coverage
Tests cover:
- ✅ Happy path scenarios
- ✅ Edge cases and error conditions
- ✅ Input validation
- ✅ Exception handling
- ✅ Integration scenarios

### Realistic Test Data
Tests use realistic configurations and scenarios that mirror actual usage patterns.

### Isolation and Mocking
Tests use proper mocking to isolate units under test and avoid external dependencies.

### Clear Assertions
Test assertions are specific and provide meaningful error messages when they fail.

## Benefits of the New Structure

1. **Maintainability**: Smaller, focused test files are easier to understand and modify
2. **Parallel Execution**: Tests can be run in parallel more effectively
3. **Selective Testing**: Easy to run tests for specific components during development
4. **Clear Organization**: Logical grouping makes it easy to find relevant tests
5. **Reduced Complexity**: Each test file has a single, clear purpose
6. **Better Debugging**: Failures are easier to isolate and debug

## Integration with CI/CD

The test runner provides:
- ✅ Exit codes for CI/CD integration
- ✅ Detailed reporting with statistics
- ✅ Configurable verbosity levels
- ✅ JSON and HTML report generation
- ✅ Coverage analysis
- ✅ Failure categorization

This structure supports automated testing pipelines and provides comprehensive validation of the alignment system.
