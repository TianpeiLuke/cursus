---
tags:
  - entry_point
  - test
  - validation
  - builders
  - standardization
keywords:
  - step builder validation
  - universal test suite
  - standardization testing
  - builder compliance
  - interface testing
topics:
  - builder validation
  - standardization testing
  - test framework
  - compliance validation
language: python
date of note: 2025-08-18
---

# Step Builder Validation System

The Step Builder Validation System provides comprehensive standardization testing for step builder implementations within the Cursus framework. It ensures that all step builders follow correct patterns, implement required interfaces, and maintain consistency across the framework.

## Overview

The Step Builder Validation System is the **Standardization Tester** component of the Cursus validation framework. It validates step builder implementations against architectural requirements through a comprehensive 4-level test suite with integrated scoring and reporting capabilities.

## Universal Step Builder Test

The `UniversalStepBuilderTest` is the main orchestrator that combines all test levels into a single comprehensive test suite. It evaluates step builders against all architectural requirements with integrated scoring and reporting.

### Test Levels

#### Level 1: Interface Tests
**Purpose**: Validates basic interface compliance and inheritance

**Key Validations**:
- Proper inheritance from `StepBuilderBase`
- Required method implementation
- Method signature compliance
- Error handling patterns

**Test Class**: `InterfaceTests`

#### Level 2: Specification Tests
**Purpose**: Validates alignment with step specifications

**Key Validations**:
- Specification loading and parsing
- Contract alignment
- Environment variable handling
- Job type compatibility

**Test Class**: `SpecificationTests`

#### Level 3: Step Creation Tests
**Purpose**: Validates step creation and configuration

**Key Validations**:
- Input/output path handling
- Property path resolution
- Configuration field usage
- Step instantiation

**Test Class**: `StepCreationTests` (with Processing-specific variant)

#### Level 4: Integration Tests
**Purpose**: Validates integration with the broader framework

**Key Validations**:
- Dependency resolution
- Registry integration
- Cross-component compatibility
- End-to-end functionality

**Test Class**: `IntegrationTests`

### Step Type-Specific Testing

The universal test includes SageMaker step type-specific validation:

- **Processing Steps**: Processor creation, input/output methods
- **Training Steps**: Estimator creation, hyperparameter handling
- **Transform Steps**: Transformer creation methods
- **CreateModel Steps**: Model creation methods
- **RegisterModel Steps**: Model package methods

## Key Features

### Comprehensive Coverage
- **4-Level Architecture**: Complete validation across all architectural levels
- **Step Type Awareness**: Enhanced validation based on SageMaker step types
- **Framework Detection**: Automatic detection of ML frameworks
- **Quality Scoring**: Integrated scoring system with quality ratings

### Enhanced Reporting
- **Structured Reports**: Detailed validation reports with actionable recommendations
- **Console Output**: Clear, formatted console reporting with scoring
- **Export Formats**: JSON export capabilities
- **Visual Charts**: Integration with chart utilities for visualization

### Performance Optimization
- **Lazy Loading**: Components loaded only when needed
- **Result Caching**: Validation results cached for performance
- **Batch Processing**: Efficient validation of multiple builders
- **Error Recovery**: Graceful handling of validation errors

## Usage Examples

### Basic Builder Validation

```python
from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Create tester
tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder)

# Run all tests (returns raw results by default)
results = tester.run_all_tests()

# Check key tests
if results["test_inheritance"]["passed"]:
    print("✅ Inheritance test passed")
```

### Enhanced Testing with Scoring

```python
# Enable scoring for quality metrics
tester = UniversalStepBuilderTest(
    TabularPreprocessingStepBuilder,
    enable_scoring=True,
    verbose=True
)

# Run tests with scoring
results = tester.run_all_tests()

# Access scoring information
scoring = results['scoring']
overall_score = scoring['overall']['score']
overall_rating = scoring['overall']['rating']

print(f"Quality Score: {overall_score:.1f}/100 ({overall_rating})")
```

### Full Reporting

```python
# Enable both scoring and structured reporting
tester = UniversalStepBuilderTest(
    TabularPreprocessingStepBuilder,
    enable_scoring=True,
    enable_structured_reporting=True,
    verbose=True
)

# Run tests with full reporting
results = tester.run_all_tests()

# Access all components
test_results = results['test_results']
scoring = results['scoring']
structured_report = results['structured_report']
```

### Convenience Methods

```python
# Legacy method (raw results only)
raw_results = tester.run_all_tests_legacy()

# Scoring method
scoring_results = tester.run_all_tests_with_scoring()

# Full report method
full_results = tester.run_all_tests_with_full_report()

# Export to JSON
json_content = tester.export_results_to_json('results.json')
```

## Class Methods and Utilities

### Batch Testing by Step Type

```python
# Test all builders of a specific SageMaker step type
results = UniversalStepBuilderTest.test_all_builders_by_type(
    sagemaker_step_type='Training',
    verbose=True,
    enable_scoring=True
)

# Results contain test data for all Training step builders
for step_name, result in results.items():
    if 'scoring' in result:
        score = result['scoring']['overall']['score']
        print(f"{step_name}: {score:.1f}/100")
```

### Registry Discovery

```python
# Generate discovery report
discovery_report = UniversalStepBuilderTest.generate_registry_discovery_report()

# Validate builder availability
availability = UniversalStepBuilderTest.validate_builder_availability('tabular_preprocessing')
```

## Test Variants

### Processing Step Variant

Processing step builders use a specialized test variant:

```python
# Automatically detected for Processing steps
if self._is_processing_step_builder():
    from .variants.processing_step_creation_tests import ProcessingStepCreationTests
    self.step_creation_tests = ProcessingStepCreationTests(...)
else:
    self.step_creation_tests = StepCreationTests(...)
```

**Processing-Specific Tests**:
- Processor creation methods (`_create_processor`, `_get_processor`)
- Input/output method validation (`_get_inputs`, `_get_outputs`)
- Processing-specific configuration handling

### Other Step Type Variants

The system includes variants for different step types:

- **CreateModel Variants**: Model creation and configuration tests
- **Training Variants**: Estimator and hyperparameter tests
- **Transform Variants**: Transformer creation and configuration tests

## Scoring System Integration

### Quality Scoring

The universal test integrates with the scoring system:

```python
from .scoring import StepBuilderScorer, LEVEL_WEIGHTS, RATING_LEVELS

# Calculate scores
scorer = StepBuilderScorer(raw_results)
score_report = scorer.generate_report()

# Score structure
{
    'overall': {
        'score': 85.5,
        'rating': 'Good',
        'passed': 15,
        'total': 18
    },
    'levels': {
        'level1_interface': {'score': 90.0, 'passed': 4, 'total': 4},
        'level2_specification': {'score': 85.0, 'passed': 3, 'total': 4},
        'level3_step_creation': {'score': 80.0, 'passed': 4, 'total': 5},
        'level4_integration': {'score': 88.0, 'passed': 4, 'total': 5}
    }
}
```

### Rating Levels

Quality ratings based on scores:

- **Excellent**: 90-100 points
- **Good**: 75-89 points
- **Fair**: 60-74 points
- **Poor**: Below 60 points

### Level Weights

Different test levels have different weights in overall scoring:

```python
LEVEL_WEIGHTS = {
    'level1_interface': 1.5,        # Higher weight for interface compliance
    'level2_specification': 1.2,   # Important for specification alignment
    'level3_step_creation': 1.0,   # Standard weight
    'level4_integration': 1.3      # Higher weight for integration
}
```

## Configuration Options

### Test Configuration

```python
# Basic configuration
tester = UniversalStepBuilderTest(
    builder_class=BuilderClass,
    config=custom_config,           # Optional custom config
    spec=custom_spec,               # Optional custom specification
    contract=custom_contract,       # Optional custom contract
    step_name=custom_step_name,     # Optional custom step name
    verbose=True                    # Enable verbose output
)
```

### Feature Flags

```python
# Enable/disable features
tester = UniversalStepBuilderTest(
    builder_class=BuilderClass,
    enable_scoring=True,            # Enable quality scoring
    enable_structured_reporting=True,  # Enable structured reports
    verbose=True                    # Enable verbose console output
)
```

### Method-Level Configuration

```python
# Override instance settings per method call
results = tester.run_all_tests(
    include_scoring=True,           # Override instance setting
    include_structured_report=False # Override instance setting
)
```

## Integration with Validation Framework

### Simple Integration API

```python
# Universal test is used internally by the simple integration API
from cursus.validation import validate_development

# This internally uses UniversalStepBuilderTest
results = validate_development(TabularPreprocessingStepBuilder)
```

### Alignment System Integration

The builder validation system integrates with the alignment validation system for comprehensive validation coverage.

## Best Practices

### For Developers

1. **Run Tests Early**: Use universal tests during development
2. **Address Issues Sequentially**: Fix Level 1 issues before proceeding to Level 2
3. **Monitor Scores**: Use scoring to track improvement over time
4. **Use Verbose Mode**: Enable verbose output for detailed feedback
5. **Export Results**: Export results for documentation and tracking

### For Framework Maintainers

1. **Keep Tests Updated**: Regularly update test suites for new requirements
2. **Monitor Performance**: Track test execution performance
3. **Enhance Coverage**: Continuously improve test coverage
4. **Document Changes**: Document test changes and new requirements
5. **Version Compatibility**: Maintain backward compatibility when possible

## Directory Structure

```
slipbox/validation/builders/
├── README.md                    # This overview document
├── universal_test.md            # Universal test suite documentation
├── interface_tests.md           # Level 1 interface tests
├── specification_tests.md       # Level 2 specification tests
├── step_creation_tests.md       # Level 3 step creation tests
├── integration_tests.md         # Level 4 integration tests
├── scoring.md                   # Quality scoring system
├── builder_reporter.md          # Reporting system
└── variants/                    # Step type-specific test variants
    ├── processing_tests.md
    ├── training_tests.md
    ├── transform_tests.md
    └── createmodel_tests.md
```

## Related Documentation

- **Core API**: See `../simple_integration.md` for the main validation API
- **Alignment Testing**: See `../alignment/` for alignment validation
- **Design Documents**: See `slipbox/1_design/` for architectural designs
- **Implementation**: See `src/cursus/validation/builders/` for source code

The Step Builder Validation System provides comprehensive, efficient, and user-friendly standardization testing that ensures all step builders meet the high-quality standards required by the Cursus framework.
