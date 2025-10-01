# Dynamic Universal Builder Testing

## Overview

This directory contains dynamic universal builder testing that automatically discovers and tests all builders in `src/cursus/steps/builders` without requiring hard-coded maintenance.

## New Dynamic Testing Approach

### CLI Commands
```bash
# Test all builders
cursus builder-test test-all-discovered

# Test specific step type
cursus builder-test test-all-discovered --step-type Processing --verbose --scoring

# Test individual builder
cursus builder-test test-single TabularPreprocessing --scoring

# List available builders
cursus builder-test list-discovered --step-type Training
```

### Pytest Integration
```bash
# Run all dynamic tests
pytest test/steps/builders/test_dynamic_universal.py

# Run specific test categories
pytest test/steps/builders/test_dynamic_universal.py::TestDynamicUniversalBuilders::test_step_type_filtering

# Run parametrized individual builder tests
pytest test/steps/builders/test_dynamic_universal.py::TestDynamicUniversalBuilders::test_individual_builder_compliance

# Run step-type specific tests
pytest test/steps/builders/test_processing_step_builders.py
pytest test/steps/builders/test_training_step_builders.py
pytest test/steps/builders/test_createmodel_step_builders.py
pytest test/steps/builders/test_transform_step_builders.py
```

## File Structure

```
test/steps/builders/
├── test_dynamic_universal.py        # NEW - Dynamic universal testing (comprehensive)
├── test_createmodel_step_builders.py # REWRITTEN - Dynamic CreateModel testing
├── test_training_step_builders.py   # REWRITTEN - Dynamic Training testing
├── test_processing_step_builders.py # REWRITTEN - Dynamic Processing testing
├── test_transform_step_builders.py  # REWRITTEN - Dynamic Transform testing
├── README.md                        # NEW - This documentation
├── results/                         # NEW - Test results storage
│   ├── .gitignore
│   ├── all_builders/               # Results from CLI 'test-all-discovered'
│   └── individual/                 # Results from CLI 'test-single'
└── legacy/                         # ARCHIVED - Manual runner scripts
    ├── run_createmodel_tests.py    # Manual test runners (replaced by CLI)
    ├── run_processing_tests.py
    ├── run_training_tests.py
    └── run_transform_tests.py
```

## Key Features

### ✅ Zero Maintenance
- **No Hard-Coded Lists**: Builders are discovered automatically via step catalog
- **Automatic Inclusion**: New builders are automatically included in tests
- **Registry-Driven**: Uses existing registry and step catalog as single source of truth
- **Future-Proof**: Adapts to changes in registry and step catalog systems

### ✅ Comprehensive Testing
- **Universal Test Framework**: All builders tested with same comprehensive standards
- **Step Type Variants**: Automatic application of step-type-specific test patterns
- **Registry Validation**: Tests use actual registry data instead of guessing from names
- **Quality Scoring**: Optional scoring and enhanced reporting capabilities

### ✅ Enhanced Reporting (NEW)
- **Visual Score Charts**: Pie charts and bar charts showing test results (requires matplotlib)
- **Comprehensive Metadata**: Detailed information about test execution and builder properties
- **Summary Statistics**: Pass rates, test counts, and status indicators across all builders
- **Step Type Breakdown**: Organized reporting by SageMaker step type classification
- **Structured Directory Organization**: Organized folder structures with canonical names
- **Legacy Report Features**: Incorporates valuable features from legacy report generators

### ✅ Simple User Experience
- **Two Primary Commands**: `test-all-discovered` and `test-single` cover all scenarios
- **Familiar Interface**: Uses existing CLI patterns and conventions
- **Clear Feedback**: Comprehensive reporting with progress indicators
- **Automatic Results**: Auto-saves to organized directory structure

## Migration from Legacy Tests

### What Changed
- ❌ **Removed**: Hard-coded builder lists requiring manual maintenance
- ❌ **Archived**: Manual test runner scripts (moved to legacy/)
- ✅ **Enhanced**: Step-type-specific test files now use dynamic discovery
- ✅ **Added**: Dynamic pytest integration with parametrized tests
- ✅ **Added**: Comprehensive test results storage
- ✅ **Maintained**: Existing test file structure for backward compatibility

### Benefits of New Approach
- ✅ **Zero Maintenance**: New builders automatically included in tests
- ✅ **Backward Compatibility**: Existing test files preserved but enhanced
- ✅ **Step Type Focus**: Each test file focuses on specific SageMaker step type
- ✅ **Comprehensive Coverage**: All builders tested with same standards
- ✅ **Better Reporting**: Structured results with scoring and analytics
- ✅ **CI/CD Ready**: Standard pytest integration for automated testing

## Test Categories

### 1. Dynamic Universal Tests (`test_dynamic_universal.py`)
**Comprehensive testing for all builders discovered via step catalog**

- **Builder Discovery**: Tests that all builders are found and valid
- **Step Type Filtering**: Tests step type classification functionality
- **Individual Compliance**: Parametrized tests for each builder
- **Comprehensive Testing**: Full test suite with results storage
- **Step Catalog Integration**: Tests step catalog functionality
- **Results Storage**: Tests results storage and directory structure

### 2. Step-Type Specific Tests
**Enhanced tests focusing on specific SageMaker step types**

#### Processing Steps (`test_processing_step_builders.py`)
- Dynamic discovery of Processing builders
- Processing-specific requirements (processor creation, I/O methods, code handling)
- Registry data validation
- Step type classification verification

#### Training Steps (`test_training_step_builders.py`)
- Dynamic discovery of Training builders
- Training-specific requirements (estimator creation, hyperparameters)
- Registry data validation with framework information
- Step type classification verification

#### CreateModel Steps (`test_createmodel_step_builders.py`)
- Dynamic discovery of CreateModel builders
- CreateModel-specific requirements (model creation methods)
- Registry data validation
- Step type classification verification

#### Transform Steps (`test_transform_step_builders.py`)
- Dynamic discovery of Transform builders (optional)
- Transform-specific requirements (transformer creation, model data handling)
- Registry data validation
- Step type classification verification

## Usage Examples

### CLI Usage
```bash
# Test all builders with scoring and export
cursus builder-test test-all-discovered --scoring --export-json all_results.json

# Test Processing builders only
cursus builder-test test-all-discovered --step-type Processing --verbose

# Test specific builder with detailed output
cursus builder-test test-single TabularPreprocessing --verbose --scoring

# List builders by step type
cursus builder-test list-discovered --step-type Training
```

### Pytest Usage
```bash
# Run comprehensive dynamic tests
pytest test/steps/builders/test_dynamic_universal.py -v

# Run specific step type tests
pytest test/steps/builders/test_processing_step_builders.py -v

# Run parametrized tests for individual builders
pytest test/steps/builders/test_dynamic_universal.py::TestDynamicUniversalBuilders::test_individual_builder_compliance -v

# Run with coverage
pytest test/steps/builders/ --cov=cursus.steps.builders
```

## Results Storage

### Automatic Results Storage
- **All Builders**: `results/all_builders/all_builders_YYYYMMDD_HHMMSS.json`
- **Single Builder**: `results/individual/{canonical_name}_YYYYMMDD_HHMMSS.json`
- **Custom Export**: `--export-json custom_path.json` option available

### Results Structure
```json
{
  "BuilderName": {
    "test_results": {
      "test_inheritance": {"passed": true},
      "test_required_methods": {"passed": false, "error": "Missing method"}
    },
    "scoring": {
      "overall": {"score": 85.5, "rating": "Good"},
      "levels": {
        "level1_interface": {"score": 90, "passed": 9, "total": 10}
      }
    },
    "structured_report": {...}
  }
}
```

## Integration with Existing Systems

### Step Catalog Integration
- Uses `StepCatalog.get_all_builders()` for comprehensive discovery
- Uses `StepCatalog.get_builders_by_step_type()` for filtered discovery
- Uses `StepCatalog.load_builder_class()` for individual builder loading
- Uses `StepCatalog.get_step_info()` for registry data validation

### Registry Integration
- Validates step type classification via registry data
- Tests registry data completeness and accuracy
- Uses registry information instead of name-based guessing
- Ensures consistency between registry and actual builder classes

### Universal Test Framework Integration
- Uses existing `UniversalStepBuilderTest` for comprehensive testing
- Supports scoring and structured reporting
- Integrates with step catalog discovery
- Maintains compatibility with existing test patterns

## Development Guidelines

### Adding New Builders
1. **No Action Required**: New builders are automatically discovered
2. **Registry Entry**: Ensure builder has proper registry entry with correct step type
3. **Standard Compliance**: Follow existing builder patterns and interfaces
4. **Verification**: Run `cursus builder-test test-single YourBuilder` to verify

### Modifying Tests
1. **Dynamic Tests**: Modify `test_dynamic_universal.py` for universal changes
2. **Step-Type Tests**: Modify specific test files for step-type-specific requirements
3. **Registry Tests**: Use registry data instead of hard-coded patterns
4. **Results Storage**: Tests automatically save results for analysis

### Best Practices
- ✅ **Use Registry Data**: Always use registry information instead of name guessing
- ✅ **Dynamic Discovery**: Never hard-code builder lists
- ✅ **Step Type Focus**: Keep step-type-specific tests focused on their domain
- ✅ **Comprehensive Coverage**: Ensure all builders are tested with same standards
- ✅ **Clear Assertions**: Provide clear error messages for failed tests

## Troubleshooting

### Common Issues

#### Builder Not Found
```bash
# Check if builder is in registry
cursus builder-test list-discovered

# Validate specific builder
cursus builder-test validate-builder YourBuilderName
```

#### Test Failures
```bash
# Run with verbose output
cursus builder-test test-single YourBuilder --verbose

# Check registry data
cursus builder-test registry-report --verbose
```

#### Import Errors
- Ensure builder class is properly importable
- Check for missing dependencies
- Verify module structure and naming

### Getting Help
- Check CLI help: `cursus builder-test --help`
- Review test results in `results/` directory
- Examine registry report: `cursus builder-test registry-report`
- Run individual pytest files for detailed output

## Future Enhancements

### Planned Features
- **Performance Testing**: Add performance benchmarks for builders
- **Integration Testing**: Add end-to-end pipeline testing
- **Documentation Testing**: Validate builder documentation completeness
- **Dependency Analysis**: Analyze and test builder dependencies

### Extensibility
- **Custom Test Variants**: Easy addition of new step-type-specific tests
- **Plugin System**: Support for custom test plugins
- **Reporting Enhancements**: Additional reporting formats and visualizations
- **CI/CD Integration**: Enhanced integration with continuous integration systems

## Conclusion

The Dynamic Universal Builder Testing system provides a comprehensive, maintainable, and user-friendly approach to testing all step builders in the cursus framework. By eliminating hard-coded maintenance requirements and leveraging existing infrastructure, it ensures consistent quality while reducing development overhead.

The system automatically adapts to changes in the codebase, provides clear feedback on test results, and maintains backward compatibility with existing testing patterns. This makes it an ideal solution for both development and production environments.
