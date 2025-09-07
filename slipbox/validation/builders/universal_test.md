---
tags:
  - code
  - validation
  - builders
  - universal_test
  - comprehensive_testing
keywords:
  - UniversalStepBuilderTest
  - TestUniversalStepBuilder
  - step builder validation
  - comprehensive testing
  - scoring integration
  - structured reporting
topics:
  - builder validation
  - comprehensive testing
  - test orchestration
language: python
date of note: 2025-09-07
---

# Universal Step Builder Test

Universal test suite for validating step builder implementation compliance across all architectural levels.

## Overview

The Universal Step Builder Test combines all test levels into a single comprehensive test suite that evaluates step builders against all architectural requirements with integrated scoring and reporting capabilities. It orchestrates Level 1 (Interface), Level 2 (Specification), Level 3 (Step Creation), and Level 4 (Integration) tests along with SageMaker step type-specific validation.

The test suite provides flexible configuration options including verbose output, quality scoring, structured reporting, and step type-specific test variants. It supports both individual builder testing and batch testing by SageMaker step type.

## Classes and Methods

### Classes
- [`UniversalStepBuilderTest`](#universalstepbuildertest) - Universal test suite for validating step builder implementation compliance
- [`TestUniversalStepBuilder`](#testuniversalstepbuilder) - Test cases for the UniversalStepBuilderTest class itself

## API Reference

### UniversalStepBuilderTest

_class_ cursus.validation.builders.universal_test.UniversalStepBuilderTest(_builder_class_, _config=None_, _spec=None_, _contract=None_, _step_name=None_, _verbose=False_, _enable_scoring=True_, _enable_structured_reporting=False_)

Universal test suite for validating step builder implementation compliance across all architectural levels.

**Parameters:**
- **builder_class** (_Type[StepBuilderBase]_) – The step builder class to test
- **config** (_Optional[ConfigBase]_) – Optional config to use (will create mock if not provided)
- **spec** (_Optional[StepSpecification]_) – Optional step specification (will extract from builder if not provided)
- **contract** (_Optional[ScriptContract]_) – Optional script contract (will extract from builder if not provided)
- **step_name** (_Optional[Union[str, StepName]]_) – Optional step name (will extract from class name if not provided)
- **verbose** (_bool_) – Whether to print verbose output
- **enable_scoring** (_bool_) – Whether to calculate and include quality scores
- **enable_structured_reporting** (_bool_) – Whether to generate structured reports

```python
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.steps.builders.processing_step_builder import ProcessingStepBuilder

tester = UniversalStepBuilderTest(ProcessingStepBuilder, verbose=True, enable_scoring=True)
```

#### run_all_tests

run_all_tests(_include_scoring=None_, _include_structured_report=None_)

Run all tests across all levels with optional scoring and structured reporting.

**Parameters:**
- **include_scoring** (_Optional[bool]_) – Whether to calculate and include quality scores (overrides instance setting)
- **include_structured_report** (_Optional[bool]_) – Whether to generate structured report (overrides instance setting)

**Returns:**
- **Dict[str, Any]** – Dictionary containing test results and optional scoring/reporting data

```python
# Run with default settings
results = tester.run_all_tests()

# Override settings for this run
results = tester.run_all_tests(include_scoring=True, include_structured_report=True)
```

#### run_step_type_specific_tests

run_step_type_specific_tests()

Run tests specific to the SageMaker step type including step type detection, classification, and compliance validation.

**Returns:**
- **Dict[str, Any]** – Dictionary containing step type-specific test results

```python
step_type_results = tester.run_step_type_specific_tests()
```

#### run_all_tests_legacy

run_all_tests_legacy()

Legacy method that returns raw results for backward compatibility, maintaining the original behavior before scoring and structured reporting enhancements.

**Returns:**
- **Dict[str, Dict[str, Any]]** – Raw test results dictionary

```python
legacy_results = tester.run_all_tests_legacy()
```

#### run_all_tests_with_scoring

run_all_tests_with_scoring()

Convenience method to run tests with scoring enabled.

**Returns:**
- **Dict[str, Any]** – Dictionary containing test results and scoring data

```python
scoring_results = tester.run_all_tests_with_scoring()
overall_score = scoring_results['scoring']['overall']['score']
```

#### run_all_tests_with_full_report

run_all_tests_with_full_report()

Convenience method to run tests with both scoring and structured reporting enabled.

**Returns:**
- **Dict[str, Any]** – Dictionary containing test results, scoring, and structured report

```python
full_results = tester.run_all_tests_with_full_report()
structured_report = full_results['structured_report']
```

#### export_results_to_json

export_results_to_json(_output_path=None_)

Export test results with scoring to JSON format.

**Parameters:**
- **output_path** (_Optional[str]_) – Optional path to save the JSON file

**Returns:**
- **str** – JSON string of the results

```python
# Export to string
json_content = tester.export_results_to_json()

# Export to file
tester.export_results_to_json('test_results.json')
```

#### test_all_builders_by_type

_classmethod_ test_all_builders_by_type(_sagemaker_step_type_, _verbose=False_, _enable_scoring=True_)

Test all builders for a specific SageMaker step type using registry discovery.

**Parameters:**
- **sagemaker_step_type** (_str_) – The SageMaker step type to test (e.g., 'Training', 'Transform')
- **verbose** (_bool_) – Whether to print verbose output
- **enable_scoring** (_bool_) – Whether to calculate and include quality scores

**Returns:**
- **Dict[str, Any]** – Dictionary containing test results for all builders of the specified type

```python
# Test all Training step builders
training_results = UniversalStepBuilderTest.test_all_builders_by_type(
    'Training', 
    verbose=True, 
    enable_scoring=True
)
```

#### generate_registry_discovery_report

_classmethod_ generate_registry_discovery_report()

Generate a comprehensive report of step builder discovery status.

**Returns:**
- **Dict[str, Any]** – Dictionary containing discovery report

```python
discovery_report = UniversalStepBuilderTest.generate_registry_discovery_report()
```

#### validate_builder_availability

_classmethod_ validate_builder_availability(_step_name_)

Validate that a step builder is available and can be loaded.

**Parameters:**
- **step_name** (_str_) – The step name to validate

**Returns:**
- **Dict[str, Any]** – Dictionary containing validation results

```python
availability = UniversalStepBuilderTest.validate_builder_availability('tabular_preprocessing')
```

### TestUniversalStepBuilder

_class_ cursus.validation.builders.universal_test.TestUniversalStepBuilder()

Test cases for the UniversalStepBuilderTest class itself, verifying that the universal test suite works correctly.

```python
import unittest

class TestUniversalStepBuilder(unittest.TestCase):
    def test_with_xgboost_training_builder(self):
        # Test implementation
        pass
```

#### test_with_xgboost_training_builder

test_with_xgboost_training_builder()

Test UniversalStepBuilderTest with XGBoostTrainingStepBuilder to verify basic functionality.

#### test_with_tabular_preprocessing_builder

test_with_tabular_preprocessing_builder()

Test UniversalStepBuilderTest with TabularPreprocessingStepBuilder to verify basic functionality.

#### test_with_explicit_components

test_with_explicit_components()

Test UniversalStepBuilderTest with explicitly provided components including custom config and specification.

#### test_scoring_integration

test_scoring_integration()

Test the scoring integration functionality to verify enhanced result structure and scoring data.

#### test_structured_reporting

test_structured_reporting()

Test the structured reporting functionality to verify comprehensive report generation.

#### test_convenience_methods

test_convenience_methods()

Test the convenience methods for different testing modes including legacy, scoring, and full report methods.

## Related Documentation

- [Interface Tests](interface_tests.md) - Level 1 interface compliance tests
- [Specification Tests](specification_tests.md) - Level 2 specification alignment tests
- [Step Creation Tests](step_creation_tests.md) - Level 3 step creation tests
- [Integration Tests](integration_tests.md) - Level 4 integration tests
- [Scoring System](scoring.md) - Quality scoring and rating system
- [SageMaker Step Type Validator](sagemaker_step_type_validator.md) - Step type-specific validation
