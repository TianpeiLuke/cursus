# Processing Step Builder Tests

This directory contains comprehensive tests for all Processing step builders in the cursus pipeline system.

## Overview

The Processing step builder tests validate that all Processing-type step builders comply with:

1. **Universal Step Builder Requirements** - Interface compliance, specification alignment, dependency resolution
2. **Processing-Specific Requirements** - Processor creation, input/output handling, environment variables
3. **Framework-Specific Requirements** - SKLearn vs XGBoost processor validation
4. **SageMaker Integration** - ProcessingStep creation and configuration

## Test Coverage

### Processing Step Builders Tested

The test suite covers all 8 Processing step builders:

1. **TabularPreprocessing** - Data preprocessing with SKLearnProcessor
2. **RiskTableMapping** - Categorical feature mapping with SKLearnProcessor  
3. **CurrencyConversion** - Currency conversion processing with SKLearnProcessor
4. **DummyTraining** - Pretrained model handling with SKLearnProcessor
5. **XGBoostModelEval** - Model evaluation with XGBoostProcessor
6. **ModelCalibration** - Model calibration with SKLearnProcessor
7. **Package** - Model packaging with SKLearnProcessor
8. **Payload** - Payload testing with SKLearnProcessor

### Test Categories

#### Universal Tests (from UniversalStepBuilderTest)
- **Interface Compliance** - Inheritance, required methods, documentation
- **Specification Alignment** - Step specifications, script contracts
- **Path Mapping** - Input/output path validation
- **Integration** - Step creation, dependency resolution
- **SageMaker Step Type** - Step type detection and compliance

#### Processing-Specific Tests
- **Processor Creation Method** - `_create_processor()` method validation
- **Expected Processor Type** - SKLearnProcessor vs XGBoostProcessor validation
- **Processing I/O Methods** - `_get_inputs()` and `_get_outputs()` validation
- **Environment Variables** - `_get_environment_variables()` method validation
- **Job Arguments Handling** - `_get_job_arguments()` method validation (optional)
- **Framework-Specific** - Framework-specific validation (sklearn vs xgboost)
- **Processing Step Creation** - `create_step()` method validation

## Running the Tests

### Option 1: Run All Processing Tests

```bash
# From the cursus root directory
cd test/steps/builders
python run_processing_tests.py
```

Or using the executable:

```bash
./test/steps/builders/run_processing_tests.py
```

### Option 2: Run Tests for Specific Builder

```bash
# Test a specific Processing step builder
python run_processing_tests.py TabularPreprocessing
python run_processing_tests.py XGBoostModelEval
python run_processing_tests.py Package
```

### Option 3: Run with pytest

```bash
# Run all Processing tests with pytest
cd test/steps/builders
pytest test_processing_step_builders.py -v

# Run tests for specific builder with pytest
pytest test_processing_step_builders.py::test_individual_processing_builder_universal_compliance -k "TabularPreprocessing"
```

### Option 4: Run with unittest

```bash
# Run with unittest directly
cd test/steps/builders
python -m unittest test_processing_step_builders.TestProcessingStepBuilders -v
```

## Test Output

### Successful Test Run Example

```
================================================================================
PROCESSING STEP BUILDERS TEST SUITE
================================================================================

Found 8 Processing step builders to test:
  ✓ TabularPreprocessing (TabularPreprocessingStepBuilder)
  ✓ RiskTableMapping (RiskTableMappingStepBuilder)
  ✓ CurrencyConversion (CurrencyConversionStepBuilder)
  ✓ DummyTraining (DummyTrainingStepBuilder)
  ✓ XGBoostModelEval (XGBoostModelEvalStepBuilder)
  ✓ ModelCalibration (ModelCalibrationStepBuilder)
  ✓ Package (PackageStepBuilder)
  ✓ Payload (PayloadStepBuilder)

============================================================
Testing TabularPreprocessing (TabularPreprocessingStepBuilder)
============================================================

TabularPreprocessing Results: 15/15 tests passed (100.0%)
✅ All tests passed!

============================================================
Testing XGBoostModelEval (XGBoostModelEvalStepBuilder)
============================================================

XGBoostModelEval Results: 15/15 tests passed (100.0%)
✅ All tests passed!

================================================================================
OVERALL PROCESSING STEP BUILDERS TEST SUMMARY
================================================================================

Overall Statistics:
  Builders tested: 8
  Builders with all tests passing: 8 (100.0%)
  Total tests run: 120
  Total tests passed: 120 (100.0%)
```

### Failed Test Example

```
TabularPreprocessing Results: 14/15 tests passed (93.3%)
Failed Tests:
  ❌ test_processor_creation_method: No processor creation methods found
```

## Test Architecture

### ProcessingStepBuilderTestSuite Class

The main test suite class that provides:

- **Dynamic Builder Loading** - Automatically discovers and loads Processing step builders
- **Processing-Specific Validation** - Specialized tests for Processing requirements
- **Framework Detection** - Identifies sklearn vs xgboost frameworks
- **Comprehensive Reporting** - Detailed test results and statistics

### TestProcessingStepBuilders Class

The unittest test case class that provides:

- **Batch Testing** - Tests all Processing builders in one run
- **Individual Testing** - Tests each builder separately
- **Result Aggregation** - Combines universal and Processing-specific results
- **Critical Test Validation** - Ensures critical tests pass for all builders

### Pytest Integration

Parametrized tests for individual builder testing:

- `test_individual_processing_builder_universal_compliance` - Universal tests per builder
- `test_individual_processing_builder_processing_specific` - Processing tests per builder

## Expected Processor Types

The test suite validates that each Processing step builder uses the correct processor type:

| Step Builder | Expected Processor | Framework |
|--------------|-------------------|-----------|
| TabularPreprocessing | SKLearnProcessor | sklearn |
| RiskTableMapping | SKLearnProcessor | sklearn |
| CurrencyConversion | SKLearnProcessor | sklearn |
| DummyTraining | SKLearnProcessor | sklearn |
| XGBoostModelEval | XGBoostProcessor | xgboost |
| ModelCalibration | SKLearnProcessor | sklearn |
| Package | SKLearnProcessor | sklearn |
| Payload | SKLearnProcessor | sklearn |

## Critical Tests

The following tests are considered critical and must pass for all Processing step builders:

1. **test_inheritance** - Must inherit from StepBuilderBase
2. **test_required_methods** - Must implement required methods
3. **test_processor_creation_method** - Must have processor creation method
4. **test_processing_io_methods** - Must have input/output methods

If any critical test fails, the entire test suite will fail.

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ❌ Import error: No module named 'cursus.steps.builders.builder_xxx'
   ```
   - Ensure you're running from the cursus root directory
   - Check that the builder file exists in `src/cursus/steps/builders/`

2. **No Builders Found**
   ```
   ❌ No Processing step builders found. Check your imports and paths.
   ```
   - Verify the BUILDER_CLASS_MAP in the test suite is correct
   - Check that builder files are properly named and located

3. **Configuration Errors**
   ```
   ❌ Error testing XGBoostModelEval: Missing required attribute: processing_entry_point
   ```
   - This indicates the mock configuration is incomplete
   - The test uses mock configurations, so this shouldn't happen in normal operation

### Debug Mode

To run tests with more detailed output:

```bash
# Run with verbose output
python run_processing_tests.py TabularPreprocessing

# Run with pytest verbose mode
pytest test_processing_step_builders.py -v -s
```

## Extending the Tests

### Adding New Processing Step Builders

1. Add the new step to `PROCESSING_STEPS` list
2. Add the builder class mapping to `BUILDER_CLASS_MAP`
3. Add expected processor type to `EXPECTED_PROCESSORS`
4. Add expected framework to `EXPECTED_FRAMEWORKS`

### Adding New Test Cases

1. Add new test method to `ProcessingStepBuilderTestSuite`
2. Call the new test method in `run_processing_specific_tests()`
3. Update critical tests list if the new test is critical

### Framework-Specific Tests

For new frameworks (e.g., PyTorch, TensorFlow):

1. Add framework detection logic in `_test_framework_specific()`
2. Add framework-specific validation requirements
3. Update `EXPECTED_FRAMEWORKS` mapping

## Integration with CI/CD

These tests can be integrated into CI/CD pipelines:

```bash
# In CI/CD script
cd test/steps/builders
python run_processing_tests.py
if [ $? -ne 0 ]; then
    echo "Processing step builder tests failed"
    exit 1
fi
```

## Test Reports

### Generating Comprehensive Reports

The test suite includes a comprehensive reporting system that generates detailed JSON reports similar to the alignment validation reports:

```bash
# Generate reports for all Processing step builders
python generate_processing_reports.py

# Or use the builder reporter directly
python -m cursus.validation.builders.builder_reporter --step-type Processing
```

### Report Structure

Reports are generated in the same format as alignment validation reports with:

- **Individual Reports** - Detailed JSON reports for each step builder
- **Summary Reports** - Aggregated statistics across all Processing builders
- **Multi-Level Analysis** - Results organized by test levels (Interface, Specification, Path Mapping, Integration)
- **Issue Classification** - Issues categorized by severity (INFO, WARNING, ERROR, CRITICAL)
- **Actionable Recommendations** - Specific steps to fix identified issues

### Report Locations

```
test/steps/builders/reports/
├── individual/                    # Individual builder reports
│   ├── tabularpreprocessing_builder_test_report.json
│   ├── xgboostmodeleval_builder_test_report.json
│   └── ...
├── json/                         # Summary reports
│   └── processing_builder_test_summary.json
└── html/                         # Future HTML reports
```

### Report Format Example

Each individual report contains:

```json
{
  "builder_name": "TabularPreprocessing",
  "builder_class": "TabularPreprocessingStepBuilder", 
  "sagemaker_step_type": "Processing",
  "level1_interface": {
    "passed": true,
    "issues": [...],
    "test_results": {...}
  },
  "level2_specification": {...},
  "level3_path_mapping": {...},
  "level4_integration": {...},
  "step_type_specific": {...},
  "overall_status": "MOSTLY_PASSING",
  "summary": {
    "total_tests": 31,
    "passed_tests": 28,
    "pass_rate": 90.3,
    "total_issues": 15,
    "critical_issues": 0,
    "error_issues": 3,
    "warning_issues": 2,
    "info_issues": 10
  },
  "recommendations": [...],
  "metadata": {...}
}
```

## Related Documentation

- [Universal Step Builder Test Design](../../../slipbox/1_design/universal_step_builder_test.md)
- [SageMaker Step Type Universal Builder Tester Design](../../../slipbox/1_design/sagemaker_step_type_universal_builder_tester_design.md)
- [Processing Step Builder Patterns](../../../slipbox/1_design/processing_step_builder_patterns.md)
- [Step Builder Registry Guide](../../../slipbox/0_developer_guide/step_builder_registry_guide.md)
- [Alignment Validation Reports](../../steps/scripts/alignment_validation/reports/) - Similar report format
