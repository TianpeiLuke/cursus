# Cursus CLI - Universal Step Builder Test System

The Cursus CLI provides comprehensive testing capabilities for step builders using the UniversalStepBuilderTestBase architecture. It allows you to run tests at different levels and with different variants to ensure your step builders comply with the system requirements.

## Installation

Make sure you have cursus installed:

```bash
pip install cursus
```

## Usage

The CLI is accessible via the `cursus` command with the `test` subcommand group.

### Basic Commands

#### List Available Builders

```bash
cursus test list-builders
```

Shows common step builder classes that can be tested.

#### Run All Tests (Universal Test Suite)

```bash
cursus test all <builder_class_path>
```

Runs the complete universal test suite including all 4 levels and step-type specific tests.

**Example:**
```bash
cursus test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
```

### Level-Specific Tests

Run tests for specific architectural levels:

#### Level 1: Interface Tests
```bash
cursus test level 1 <builder_class_path>
```

Tests basic interface compliance:
- Class inheritance and naming conventions
- Required method implementation and signatures
- Registry integration and decorator usage
- Basic error handling and validation
- Documentation standards compliance

#### Level 2: Specification Tests
```bash
cursus test level 2 <builder_class_path>
```

Tests specification and contract compliance:
- Step specification usage and alignment
- Script contract integration
- Environment variable handling
- Job arguments validation

#### Level 3: Path Mapping Tests
```bash
cursus test level 3 <builder_class_path>
```

Tests path mapping and property path validation:
- Input path mapping correctness
- Output path mapping correctness
- Property path validity and resolution

#### Level 4: Integration Tests
```bash
cursus test level 4 <builder_class_path>
```

Tests system integration and end-to-end functionality:
- Dependency resolution correctness
- Step creation and configuration
- Step name generation and consistency

### Variant-Specific Tests

Run tests for specific step type variants:

#### Processing Variant Tests
```bash
cursus test variant processing <builder_class_path>
```

Runs specialized tests for Processing step builders:
- Processor creation patterns
- Processing inputs/outputs handling
- Processing job arguments
- Environment variables for processing
- Property files configuration
- Processing code handling

**Example:**
```bash
cursus test variant processing src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder
```

## Examples

### Complete Testing Workflow

1. **List available builders:**
   ```bash
   cursus test list-builders
   ```

2. **Run all tests for a builder:**
   ```bash
   cursus test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
   ```

3. **Run specific level tests:**
   ```bash
   # Test interface compliance
   cursus test level 1 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
   
   # Test path mapping
   cursus test level 3 src.cursus.steps.builders.builder_model_eval_step.ModelEvalStepBuilder
   ```

4. **Run variant-specific tests:**
   ```bash
   # Test processing-specific functionality
   cursus test variant processing src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder
   ```

### Verbose Output

Add the `--verbose` or `-v` flag to any command for detailed output:

```bash
cursus test all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --verbose
```

Or use the global verbose flag:

```bash
cursus --verbose test level 1 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
```

## Test Results

The CLI provides comprehensive test results with:

- **Summary statistics**: Pass/fail counts and percentages
- **Grouped results**: Tests organized by level and type
- **Detailed error messages**: Specific failure reasons
- **Color-coded output**: Easy visual identification of results
- **Exit codes**: 0 for success, 1 for failures

### Sample Output

```
📊 Test Results Summary: 12/15 tests passed (80.0%)
============================================================

📁 Level 1 (Interface): 8/9 passed (88.9%)
  ✅ test_inheritance
  ✅ test_naming_conventions
  ✅ test_required_methods
  ❌ test_registry_integration
    💬 Builder may not be registered with @register_builder() decorator
  ✅ test_documentation_standards
  ✅ test_type_hints
  ✅ test_error_handling
  ✅ test_method_return_types
  ✅ test_configuration_validation

📁 Step Type Specific: 4/6 passed (66.7%)
  ✅ test_step_type_detection
  ✅ test_step_type_classification
  ❌ test_processing_processor_methods
    💬 No processor creation methods found
  ❌ test_processing_io_methods
    💬 Missing _get_inputs or _get_outputs methods
  ✅ test_processing_job_arguments
  ✅ test_processing_code_handling

============================================================

⚠️  3 test(s) failed. Please review and fix the issues.
```

## Builder Class Paths

When specifying builder classes, use the full import path:

- **Format**: `module.path.ClassName`
- **Example**: `src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder`

### Common Builder Classes

- `src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder`
- `src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder`
- `src.cursus.steps.builders.builder_model_eval_step.ModelEvalStepBuilder`

## Architecture

The test system is built on the UniversalStepBuilderTestBase architecture with:

### 4 Test Levels
1. **Interface Tests**: Basic compliance and interface requirements
2. **Specification Tests**: Specification and contract alignment
3. **Path Mapping Tests**: Input/output path mapping validation
4. **Integration Tests**: System integration and end-to-end functionality

### Test Variants
- **Processing Variant**: Specialized tests for Processing step builders
- **Training Variant**: (Future) Specialized tests for Training step builders
- **Transform Variant**: (Future) Specialized tests for Transform step builders

### Extensibility

The system is designed to be extensible:
- New test levels can be added by creating classes that inherit from `UniversalStepBuilderTestBase`
- New variants can be added in the `variants/` directory
- The CLI automatically discovers and integrates new test types

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the builder class path is correct and the module is importable
2. **Missing Dependencies**: Some tests may require specific dependencies to be installed
3. **Configuration Issues**: Some builders may require specific configuration to initialize properly

### Debug Mode

Use verbose mode to get detailed information about test execution:

```bash
cursus --verbose test all <builder_class_path>
```

This will show:
- Import process details
- Test execution logs
- Detailed error information
- Stack traces for failures

## Contributing

To add new test variants or levels:

1. Create new test classes in the appropriate directory
2. Inherit from `UniversalStepBuilderTestBase`
3. Implement the required abstract methods
4. Update the CLI to include the new tests
5. Add documentation and examples

For more information, see the developer guide in the main documentation.
