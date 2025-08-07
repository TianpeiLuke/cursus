# Universal Processing Builder Test Framework

This directory contains the Universal Processing Builder Test framework, a comprehensive testing system specifically designed for Processing step builders in the cursus framework. The framework enforces standardization rules, alignment rules, and provides LLM-powered feedback and scoring.

## Overview

The Universal Processing Builder Test framework extends the existing `universal_builder_test` system with Processing-specific validations and enhanced LLM integration for intelligent feedback. It provides:

1. **Standardization Rule Enforcement** - Validates naming conventions, interface compliance, documentation standards, error handling, and testing standards
2. **Alignment Rule Enforcement** - Validates alignment between scripts, contracts, specifications, and builders
3. **LLM-Powered Feedback** - Provides intelligent analysis, scoring, and recommendations for step builder implementations

## Key Components

### 1. ProcessingStepBuilderValidator

The core validator that extends `UniversalStepBuilderTestBase` with Processing-specific validations:

- **Standardization Validation**: Checks naming conventions, interface compliance, documentation, error handling, and testing standards
- **Alignment Validation**: Validates Script ↔ Contract, Contract ↔ Specification, Specification ↔ Dependencies, and Builder ↔ Configuration alignment
- **Processing-Specific Tests**: Tests ProcessingStep creation, processor configuration, input/output handling, environment variables, and job arguments

### 2. ProcessingStepBuilderLLMAnalyzer

LLM-powered analyzer that provides intelligent feedback:

- **Scoring System**: Calculates code quality, architecture compliance, and maintainability scores
- **Feedback Generation**: Provides strengths, weaknesses, and actionable recommendations
- **Detailed Analysis**: Generates comprehensive analysis reports

### 3. UniversalProcessingBuilderTest

Main orchestrator that combines all validation levels:

- **Comprehensive Testing**: Runs universal tests, Processing-specific tests, standardization validation, and alignment validation
- **Report Generation**: Creates detailed JSON reports and analysis files
- **Scoring Integration**: Provides overall scores and ratings

## Usage

### Basic Usage

```python
from test.steps.builders.universal_processing_builder_test import test_processing_builder
from src.cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Test a Processing step builder
results = test_processing_builder(
    builder_class=TabularPreprocessingStepBuilder,
    verbose=True,
    save_reports=True
)

print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
print(f"Rating: {results['summary']['overall_rating']}")
```

### Advanced Usage

```python
from test.steps.builders.universal_processing_builder_test import UniversalProcessingBuilderTest
from src.cursus.steps.builders.builder_payload_step import PayloadStepBuilder
from src.cursus.steps.configs.config_payload_step import PayloadConfig

# Create custom config
config = PayloadConfig(
    region="NA",
    pipeline_name="test-pipeline",
    bucket="test-bucket",
    # ... other config parameters
)

# Create comprehensive tester
tester = UniversalProcessingBuilderTest(
    builder_class=PayloadStepBuilder,
    config=config,
    verbose=True,
    save_reports=True,
    output_dir="custom_reports"
)

# Run comprehensive test
results = tester.run_comprehensive_test()
```

### Testing Individual Components

```python
from test.steps.builders.universal_processing_builder_test import ProcessingStepBuilderValidator

# Create validator
validator = ProcessingStepBuilderValidator(
    builder_class=YourStepBuilder,
    verbose=True
)

# Test standardization rules
standardization_violations = validator.validate_standardization_rules()
for violation in standardization_violations:
    print(f"[{violation.severity}] {violation.message}")
    print(f"Suggestion: {violation.suggestion}")

# Test alignment rules
alignment_violations = validator.validate_alignment_rules()
for violation in alignment_violations:
    print(f"{violation.component_a} ↔ {violation.component_b}: {violation.message}")
```

## Supported Processing Step Types

The framework automatically detects and supports the following Processing step types:

- **TabularPreprocessing** - Data preprocessing steps
- **PayloadGeneration** - MIMS payload generation steps
- **ModelEvaluation** - Model evaluation steps
- **PackageCreation** - Model packaging steps
- **DataLoading** - Data loading steps (including CradleData)
- **RiskTableMapping** - Risk table mapping steps
- **CurrencyConversion** - Currency conversion steps
- **ModelCalibration** - Model calibration steps
- **BatchTransform** - Batch transformation steps

## Validation Rules

### Standardization Rules

1. **Naming Conventions (NAMING_001-003)**
   - Class names must follow `XXXStepBuilder` pattern
   - Step types must be in PascalCase
   - Public methods should be in snake_case

2. **Interface Standardization (INTERFACE_001-004)**
   - Must inherit from `StepBuilderBase`
   - Must implement required methods: `validate_configuration`, `_get_inputs`, `_get_outputs`, `create_step`
   - Should use `@register_builder()` decorator

3. **Documentation Standards (DOC_001-003)**
   - Classes must have comprehensive docstrings
   - Methods should have proper documentation
   - Documentation should include purpose, parameters, returns, and exceptions

4. **Error Handling Standards (ERROR_001-002)**
   - `validate_configuration()` should raise `ValueError` for invalid configs
   - Proper exception handling throughout the implementation

5. **Testing Standards (TEST_001)**
   - Comprehensive unit tests should exist
   - Integration tests and error handling tests recommended

### Alignment Rules

1. **Script ↔ Contract Alignment**
   - Environment variables must match contract requirements
   - Script paths must align with contract definitions

2. **Contract ↔ Specification Alignment**
   - Input paths in contract must match specification dependencies
   - Output paths in contract must match specification outputs

3. **Specification ↔ Dependencies Alignment**
   - Required dependencies must have compatible sources defined
   - Dependency logical names must be consistent

4. **Builder ↔ Configuration Alignment**
   - Builder must use configuration parameters correctly
   - Instance type selection must respect configuration flags

## Output Reports

The framework generates several types of reports:

### 1. Comprehensive JSON Report
```
{
  "builder_class": "TabularPreprocessingStepBuilder",
  "module": "src.cursus.steps.builders.builder_tabular_preprocessing_step",
  "processing_step_type": "TabularPreprocessing",
  "test_results": { ... },
  "standardization_violations": [ ... ],
  "alignment_violations": [ ... ],
  "llm_feedback": { ... },
  "summary": {
    "total_tests": 15,
    "passed_tests": 13,
    "pass_rate": 86.7,
    "overall_score": 82.5,
    "overall_rating": "Good"
  }
}
```

### 2. Detailed Analysis Text File
Contains comprehensive analysis including:
- Test results summary
- Failed test details
- Standardization violations
- Alignment violations
- Recommendations for improvement

## Scoring System

The framework uses a weighted scoring system:

- **Test Pass Rate** (40%): Percentage of tests that pass
- **Code Quality Score** (30%): Based on standardization violations
- **Architecture Compliance Score** (20%): Based on alignment violations
- **Maintainability Score** (10%): Combination of pass rate and code quality

### Rating Levels
- **90-100**: Excellent
- **80-89**: Good
- **70-79**: Satisfactory
- **60-69**: Needs Work
- **0-59**: Poor

## Integration with CI/CD

The framework can be integrated into CI/CD pipelines:

```python
# In your test suite
import pytest
from test.steps.builders.universal_processing_builder_test import test_processing_builder

@pytest.mark.parametrize("builder_class", [
    TabularPreprocessingStepBuilder,
    PayloadStepBuilder,
    PackageStepBuilder,
    XGBoostModelEvalStepBuilder
])
def test_processing_builder_compliance(builder_class):
    """Test Processing builder compliance with standards."""
    results = test_processing_builder(
        builder_class=builder_class,
        verbose=False,
        save_reports=True
    )
    
    # Assert minimum quality standards
    assert results['summary']['overall_score'] >= 70, f"Builder {builder_class.__name__} scored {results['summary']['overall_score']:.1f}/100"
    assert results['summary']['standardization_errors'] == 0, f"Builder {builder_class.__name__} has standardization errors"
    assert results['summary']['alignment_errors'] == 0, f"Builder {builder_class.__name__} has alignment errors"
```

## Running the Tests

### Command Line Usage

```bash
# Run all Processing builder tests
cd /path/to/cursus
python -m test.steps.builders.universal_processing_builder_test

# Run specific builder test
python -c "
from test.steps.builders.universal_processing_builder_test import test_processing_builder
from src.cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
test_processing_builder(TabularPreprocessingStepBuilder, verbose=True)
"
```

### Programmatic Usage

```python
# Test all available Processing builders
from test.steps.builders.universal_processing_builder_test import test_processing_builder

builders_to_test = [
    'TabularPreprocessingStepBuilder',
    'PayloadStepBuilder', 
    'PackageStepBuilder',
    'XGBoostModelEvalStepBuilder'
]

for builder_name in builders_to_test:
    try:
        module_path = f"src.cursus.steps.builders.builder_{builder_name.lower().replace('stepbuilder', '_step')}"
        module = __import__(module_path, fromlist=[builder_name])
        builder_class = getattr(module, builder_name)
        
        print(f"\nTesting {builder_name}...")
        results = test_processing_builder(builder_class, verbose=True)
        print(f"Score: {results['summary']['overall_score']:.1f}/100 ({results['summary']['overall_rating']})")
        
    except ImportError as e:
        print(f"Could not test {builder_name}: {e}")
```

## Best Practices

1. **Run Tests Early**: Use the framework during development to catch issues early
2. **Address Violations**: Fix ERROR-level violations before committing code
3. **Improve Documentation**: Address documentation warnings to improve maintainability
4. **Monitor Scores**: Aim for scores above 80 for production code
5. **Use Custom Configs**: Provide realistic configurations for more accurate testing
6. **Review Reports**: Use the detailed analysis to understand improvement areas

## Extending the Framework

To add support for new Processing step types:

1. Add the new type to `ProcessingStepType` enum
2. Update `_detect_processing_step_type()` method
3. Add type-specific configuration in `_add_builder_specific_config()`
4. Add any type-specific validation rules as needed

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Configuration Errors**: Provide valid configuration objects for testing
3. **Missing Specifications**: Some tests require step specifications to be available
4. **Mock Failures**: Check that mock objects are properly configured

### Debug Mode

Enable verbose output for detailed debugging:

```python
results = test_processing_builder(
    builder_class=YourBuilder,
    verbose=True  # Enable detailed output
)
```

## Contributing

When contributing to the framework:

1. Follow the existing code patterns and documentation standards
2. Add tests for new functionality
3. Update this README for any new features
4. Ensure backward compatibility with existing builders

## See Also

- [Standardization Rules](../../../slipbox/0_developer_guide/standardization_rules.md)
- [Alignment Rules](../../../slipbox/0_developer_guide/alignment_rules.md)
- [Universal Builder Test Documentation](../universal_builder_test/README.md)
- [Step Builder Development Guide](../../../slipbox/0_developer_guide/step_builder.md)
