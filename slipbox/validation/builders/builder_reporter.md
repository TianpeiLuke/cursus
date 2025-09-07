---
tags:
  - code
  - validation
  - builders
  - reporting
  - structured_reports
keywords:
  - builder test reporting
  - structured reports
  - issue tracking
  - recommendations
  - validation results
  - report generation
topics:
  - builder validation reporting
  - structured report generation
  - issue analysis
  - recommendation system
language: python
date of note: 2025-08-15
---

# Builder Test Reporting System API Reference

## Overview

The `builder_reporter.py` module provides comprehensive reporting capabilities for step builder test results, including summary generation, issue analysis, and export functionality. It follows the same structural patterns as the alignment validation reporting system, providing consistent and detailed analysis of builder validation results.

## Classes and Methods

- **BuilderTestIssue**: Individual issue representation with severity and categorization
- **BuilderTestResult**: Single test result with issues and metadata  
- **BuilderTestSummary**: Executive summary with statistics and status
- **BuilderTestRecommendation**: Actionable recommendations for fixing issues
- **BuilderTestReport**: Comprehensive report container with all components
- **BuilderTestReporter**: Main orchestrator for report generation

## API Reference

### _class_ cursus.validation.builders.builder_reporter.BuilderTestIssue

Issue found during step builder testing, similar to AlignmentIssue but specific to builder testing.

**Attributes:**
- *severity* (*str*): Issue severity level (INFO, WARNING, ERROR, CRITICAL)
- *category* (*str*): Issue category (interface, specification, path_mapping, integration, step_type_specific)
- *message* (*str*): Human-readable issue description
- *details* (*Dict[str, Any]*): Additional context information
- *recommendation* (*Optional[str]*): Suggested fix for the issue
- *test_name* (*str*): Associated test name that generated this issue

**Methods:**

#### to_dict()

Convert issue to dictionary for serialization.

**Returns:**
- *Dict[str, Any]*: Dictionary representation of the issue

```python
issue = BuilderTestIssue(
    severity="ERROR",
    category="interface",
    message="Builder does not inherit from StepBuilderBase",
    test_name="test_inheritance"
)
issue_dict = issue.to_dict()
```

### _class_ cursus.validation.builders.builder_reporter.BuilderTestResult

Result of a single builder test containing detailed information about what was tested, whether it passed, and specific issues found.

**Attributes:**
- *test_name* (*str*): Name of the test that was executed
- *passed* (*bool*): Whether the test passed or failed
- *issues* (*List[BuilderTestIssue]*): List of issues found during testing
- *details* (*Dict[str, Any]*): Additional test execution details
- *timestamp* (*datetime*): When the test was executed
- *test_level* (*str*): Test level (interface, specification, path_mapping, integration, step_type_specific)

**Methods:**

#### add_issue(issue)

Add a builder test issue to this result and update pass status based on severity.

**Parameters:**
- *issue* (*BuilderTestIssue*): Issue to add to this test result

```python
result = BuilderTestResult(test_name="test_inheritance", passed=True)
issue = BuilderTestIssue(severity="ERROR", category="interface", message="Failed", test_name="test_inheritance")
result.add_issue(issue)  # This will set passed=False due to ERROR severity
```

#### get_highest_severity()

Get the highest severity level among all issues in this result.

**Returns:**
- *Optional[str]*: Highest severity level or None if no issues

#### has_critical_issues()

Check if this result has any critical issues.

**Returns:**
- *bool*: True if critical issues exist

#### has_errors()

Check if this result has any error-level issues.

**Returns:**
- *bool*: True if error-level issues exist

#### to_dict()

Convert result to dictionary for serialization.

**Returns:**
- *Dict[str, Any]*: Dictionary representation including highest severity

### _class_ cursus.validation.builders.builder_reporter.BuilderTestSummary

Executive summary of builder test results providing high-level statistics and key findings from the testing.

**Attributes:**
- *builder_name* (*str*): Name of the builder being tested
- *builder_class* (*str*): Class name of the builder
- *sagemaker_step_type* (*str*): SageMaker step type (Training, Processing, etc.)
- *total_tests* (*int*): Total number of tests executed
- *passed_tests* (*int*): Number of tests that passed
- *failed_tests* (*int*): Number of tests that failed
- *pass_rate* (*float*): Percentage of tests that passed
- *total_issues* (*int*): Total number of issues found
- *critical_issues* (*int*): Number of critical severity issues
- *error_issues* (*int*): Number of error severity issues
- *warning_issues* (*int*): Number of warning severity issues
- *info_issues* (*int*): Number of info severity issues
- *highest_severity* (*Optional[str]*): Highest severity level found
- *overall_status* (*str*): Overall status (PASSING, MOSTLY_PASSING, PARTIALLY_PASSING, FAILING)
- *validation_timestamp* (*datetime*): When the validation was performed
- *metadata* (*Dict[str, Any]*): Additional metadata including scoring information

**Class Methods:**

#### from_results(builder_name, builder_class, sagemaker_step_type, results)

Create BuilderTestSummary from test results with automatic status determination.

**Parameters:**
- *builder_name* (*str*): Name of the builder
- *builder_class* (*str*): Builder class name
- *sagemaker_step_type* (*str*): SageMaker step type
- *results* (*Dict[str, BuilderTestResult]*): Dictionary of test results

**Returns:**
- *BuilderTestSummary*: Generated summary with calculated statistics

```python
summary = BuilderTestSummary.from_results(
    "XGBoostTraining", 
    "XGBoostTrainingStepBuilder", 
    "Training", 
    test_results
)
```

**Methods:**

#### is_passing()

Check if the overall testing is passing (no critical or error issues).

**Returns:**
- *bool*: True if no critical or error issues exist

#### to_dict()

Convert summary to dictionary for serialization.

**Returns:**
- *Dict[str, Any]*: Dictionary representation including calculated fields

### _class_ cursus.validation.builders.builder_reporter.BuilderTestRecommendation

Actionable recommendation for fixing builder test issues.

**Attributes:**
- *category* (*str*): Category of the recommendation
- *priority* (*str*): Priority level (HIGH, MEDIUM, LOW)
- *title* (*str*): Short title of the recommendation
- *description* (*str*): Detailed description of the issue
- *affected_components* (*List[str]*): List of components this affects
- *steps* (*List[str]*): Step-by-step instructions for implementing the fix

**Methods:**

#### to_dict()

Convert recommendation to dictionary for serialization.

**Returns:**
- *Dict[str, Any]*: Dictionary representation of the recommendation

### _class_ cursus.validation.builders.builder_reporter.BuilderTestReport

Comprehensive report of step builder test results containing results from all test levels with detailed analysis and actionable recommendations.

**Attributes:**
- *builder_name* (*str*): Name of the builder being tested
- *builder_class* (*str*): Builder class name
- *sagemaker_step_type* (*str*): SageMaker step type
- *level1_interface* (*Dict[str, BuilderTestResult]*): Level 1 interface test results
- *level2_specification* (*Dict[str, BuilderTestResult]*): Level 2 specification test results
- *level3_step_creation* (*Dict[str, BuilderTestResult]*): Level 3 step creation test results
- *level4_integration* (*Dict[str, BuilderTestResult]*): Level 4 integration test results
- *step_type_specific* (*Dict[str, BuilderTestResult]*): Step type-specific test results
- *summary* (*Optional[BuilderTestSummary]*): Executive summary of results
- *recommendations* (*List[BuilderTestRecommendation]*): List of actionable recommendations
- *metadata* (*Dict[str, Any]*): Additional metadata including scoring data

**Methods:**

#### add_level1_result(test_name, result)

Add a Level 1 (Interface) test result to the report.

**Parameters:**
- *test_name* (*str*): Name of the test
- *result* (*BuilderTestResult*): Test result to add

#### add_level2_result(test_name, result)

Add a Level 2 (Specification) test result to the report.

**Parameters:**
- *test_name* (*str*): Name of the test
- *result* (*BuilderTestResult*): Test result to add

#### add_level3_result(test_name, result)

Add a Level 3 (Step Creation) test result to the report.

**Parameters:**
- *test_name* (*str*): Name of the test
- *result* (*BuilderTestResult*): Test result to add

#### add_level4_result(test_name, result)

Add a Level 4 (Integration) test result to the report.

**Parameters:**
- *test_name* (*str*): Name of the test
- *result* (*BuilderTestResult*): Test result to add

#### add_step_type_result(test_name, result)

Add a step type-specific test result to the report.

**Parameters:**
- *test_name* (*str*): Name of the test
- *result* (*BuilderTestResult*): Test result to add

#### get_all_results()

Get all test results across all levels.

**Returns:**
- *Dict[str, BuilderTestResult]*: Combined dictionary of all test results

#### generate_summary()

Generate executive summary of test status and store it in the report.

**Returns:**
- *BuilderTestSummary*: Generated summary with statistics and status

#### get_critical_issues()

Get all critical test issues requiring immediate attention.

**Returns:**
- *List[BuilderTestIssue]*: List of all critical severity issues

#### get_error_issues()

Get all error-level test issues.

**Returns:**
- *List[BuilderTestIssue]*: List of all error severity issues

#### has_critical_issues()

Check if the report has any critical issues.

**Returns:**
- *bool*: True if critical issues exist

#### has_errors()

Check if the report has any error-level issues.

**Returns:**
- *bool*: True if error-level issues exist

#### is_passing()

Check if the overall test validation is passing.

**Returns:**
- *bool*: True if no critical or error issues exist

#### get_recommendations()

Get actionable recommendations for fixing test issues. Generates recommendations automatically if not already created.

**Returns:**
- *List[BuilderTestRecommendation]*: List of actionable recommendations

#### export_to_json()

Export report to JSON format similar to alignment validation reports.

**Returns:**
- *str*: JSON string representation of the complete report

#### save_to_file(output_path)

Save report to JSON file.

**Parameters:**
- *output_path* (*Path*): Path where the report should be saved

#### print_summary()

Print a formatted summary to console with color-coded status indicators.

```python
report = BuilderTestReport("XGBoostTraining", "XGBoostTrainingStepBuilder", "Training")
# ... add test results ...
report.generate_summary()
report.print_summary()
```

### _class_ cursus.validation.builders.builder_reporter.BuilderTestReporter

Main reporter class for generating step builder test reports. Provides methods to test builders and generate comprehensive reports in the same format as alignment validation reports.

**Attributes:**
- *output_dir* (*Path*): Directory for saving reports with subdirectories for different formats

**Methods:**

#### __init__(output_dir=None)

Initialize the reporter with output directory structure.

**Parameters:**
- *output_dir* (*Optional[Path]*): Output directory (defaults to test/steps/builders/reports)

#### test_and_report_builder(builder_class, step_name=None)

Test a step builder and generate a comprehensive report.

**Parameters:**
- *builder_class* (*Type[StepBuilderBase]*): The step builder class to test
- *step_name* (*Optional[str]*): Step name (will be inferred if not provided)

**Returns:**
- *BuilderTestReport*: Complete test report with all results and analysis

```python
reporter = BuilderTestReporter()
report = reporter.test_and_report_builder(XGBoostTrainingStepBuilder)
print(f"Overall Status: {report.summary.overall_status}")
```

#### test_and_save_builder_report(builder_class, step_name=None)

Test a builder and save the report to file.

**Parameters:**
- *builder_class* (*Type[StepBuilderBase]*): The step builder class to test
- *step_name* (*Optional[str]*): Step name (will be inferred if not provided)

**Returns:**
- *BuilderTestReport*: Complete test report (also saved to file)

#### test_step_type_builders(sagemaker_step_type)

Test all builders of a specific SageMaker step type and generate individual reports.

**Parameters:**
- *sagemaker_step_type* (*str*): SageMaker step type (Training, Processing, etc.)

**Returns:**
- *Dict[str, BuilderTestReport]*: Dictionary mapping step names to their reports

```python
reporter = BuilderTestReporter()
reports = reporter.test_step_type_builders("Processing")
for step_name, report in reports.items():
    print(f"{step_name}: {report.summary.overall_status}")
```

## Report Generation Process

The system follows a structured 5-step process:

1. **Test Execution**: Run UniversalStepBuilderTest with scoring enabled
2. **Result Conversion**: Convert raw results to BuilderTestResult objects
3. **Issue Analysis**: Convert errors to categorized issues with recommendations
4. **Summary Generation**: Create executive summary with statistics and status
5. **Recommendation Generation**: Generate actionable recommendations based on issue patterns

## Recommendation System

The system automatically generates recommendations based on issue patterns:

- **Interface Issues**: Fix builder interface compliance and inheritance
- **Specification Issues**: Fix specification alignment and contract usage
- **Path Mapping Issues**: Fix input/output path mapping and property paths
- **Integration Issues**: Fix dependency resolution and step creation
- **Step Type Issues**: Fix step type-specific requirements

## Export Formats

### JSON Export Structure
```json
{
  "builder_name": "XGBoostTraining",
  "builder_class": "XGBoostTrainingStepBuilder", 
  "sagemaker_step_type": "Training",
  "level1_interface": {
    "passed": true,
    "issues": [],
    "test_results": {...}
  },
  "summary": {
    "total_tests": 20,
    "passed_tests": 18,
    "pass_rate": 90.0,
    "overall_status": "MOSTLY_PASSING"
  },
  "recommendations": [...],
  "quality_score": 85.5,
  "quality_rating": "Good"
}
```

### Console Output Format
```
================================================================================
STEP BUILDER TEST REPORT: XGBoostTraining
================================================================================

Builder: XGBoostTrainingStepBuilder
SageMaker Step Type: Training
Overall Status: ✅ MOSTLY_PASSING
Pass Rate: 90.0% (18/20)
📊 Quality Score: 85.5/100 - Good
Total Issues: 2

Level 1 (Interface): 5/5 tests passed (100.0%) - Score: 95.0/100
Level 2 (Specification): 3/4 tests passed (75.0%) - Score: 80.0/100
Level 3 (Step Creation): 4/5 tests passed (80.0%) - Score: 85.0/100
Level 4 (Integration): 5/6 tests passed (83.3%) - Score: 88.0/100
```

## Integration Features

- **Scoring Integration**: Supports quality scoring from UniversalStepBuilderTest
- **Structured Reporting**: Consistent format with alignment validation reports
- **Batch Processing**: Can test multiple builders by step type
- **Error Handling**: Graceful degradation with error reporting
- **Performance Optimization**: Lazy loading and memory management

## Related Components

- **[universal_test.md](universal_test.md)**: Test execution engine that provides the raw test results
- **[interface_tests.md](interface_tests.md)**: Level 1 interface compliance tests
- **[specification_tests.md](specification_tests.md)**: Level 2 specification alignment tests  
- **[integration_tests.md](integration_tests.md)**: Level 4 integration and step creation tests
- **[scoring.md](scoring.md)**: Quality scoring system for enhanced reporting
