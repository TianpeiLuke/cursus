# Universal Step Builder Test

This directory contains the Universal Step Builder Test, a standardized test suite for validating that step builder classes meet architectural requirements.

The test suite is organized into multiple levels of increasing architectural complexity, allowing developers to focus on specific aspects of conformance. It also includes a scoring system to evaluate and compare the quality of different step builders.

## Purpose

The Universal Step Builder Test ensures that all step builders:

1. Implement the required interface methods
2. Use specifications and script contracts correctly
3. Handle environment variables and dependencies properly
4. Create valid steps with proper attributes
5. Handle errors appropriately
6. Map specification dependencies to contract paths
7. Map specification outputs to contract paths
8. Process property paths correctly

## Usage

### Running Tests for All Builders

To test all step builders in the project:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py
```

This will:
- Discover all step builder classes in src/pipeline_steps
- Run the universal test suite against each builder
- Provide a summary of test results

### Running Tests for a Specific Builder

To test a specific builder:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --builder XGBoostTrainingStepBuilder
```

### Using Explicit Components

You can provide explicit components to test with:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --builder XGBoostTrainingStepBuilder \
    --spec src.pipeline_step_specs.xgboost_training_spec.XGBOOST_TRAINING_SPEC \
    --contract src.pipeline_script_contracts.xgboost_train_contract.XGBOOST_TRAIN_CONTRACT \
    --step-name CustomXGBoostTrainingStep
```

### Viewing Detailed Results

For detailed test results, use the `--verbose` flag:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --verbose
```

### Generating Quality Scores

To generate quality scores and reports:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --score
```

You can specify an output directory for reports:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --score --output-dir ./reports
```

To disable chart generation (if matplotlib is not available):

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --score --no-chart
```

### Using in Unit Tests

You can also use the Universal Step Builder Test in your unit tests:

```python
import unittest
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from test.pipeline_steps.test_universal_step_builder import UniversalStepBuilderTest

class TestXGBoostTrainingStepBuilder(unittest.TestCase):
    def test_universal_compliance(self):
        # Test compliance with universal requirements
        tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
        results = tester.run_all_tests()
        
        # Assert all tests passed
        for test_name, result in results.items():
            self.assertTrue(result["passed"], f"{test_name} failed: {result.get('error', '')}")
            
    def test_with_explicit_components(self):
        # Import specification and contract
        from src.pipeline_step_specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
        from src.pipeline_script_contracts.xgboost_train_contract import XGBOOST_TRAIN_CONTRACT
        
        # Test compliance with custom components
        tester = UniversalStepBuilderTest(
            XGBoostTrainingStepBuilder,
            spec=XGBOOST_TRAINING_SPEC,
            contract=XGBOOST_TRAIN_CONTRACT,
            step_name="CustomXGBoostTrainingStep"
        )
        results = tester.run_all_tests()
        
        # Assert all tests passed
        for test_name, result in results.items():
            self.assertTrue(result["passed"], f"{test_name} failed: {result.get('error', '')}")
```

## Test Organization

The Universal Step Builder Test is organized into four levels of increasing complexity:

### Level 1: Interface Tests
- **Interface Compliance** - Proper inheritance and method implementation
- **Error Handling** - Appropriate handling of invalid configurations
- *Knowledge Required*: Basic understanding of the StepBuilderBase class

### Level 2: Specification Tests
- **Specification Integration** - Valid use of step specifications
- **Contract Alignment** - Proper alignment between specifications and script contracts
- **Environment Variables** - Correct handling of environment variables from contracts
- **Job Arguments** - Proper generation of command-line arguments from script contracts
- *Knowledge Required*: Understanding of specifications and script contracts

### Level 3: Path Mapping Tests
- **Input Path Mapping** - Correct mapping of specification dependencies to contract paths
- **Output Path Mapping** - Correct mapping of specification outputs to contract paths
- **Property Path Validity** - Validation of output specification property paths
- *Knowledge Required*: Deeper understanding of path mapping between specifications and contracts

### Level 4: Integration Tests
- **Dependency Resolution** - Proper extraction of inputs from dependencies
- **Step Creation** - Creation of valid steps with required attributes
- **Step Name** - Correct generation and usage of step names in created steps
- *Knowledge Required*: Complete understanding of the pipeline system integration

## Extending the Tests

The test suite is organized into separate modules for each level of complexity:

- `universal_step_builder_test/base_test.py` - Common test utilities
- `universal_step_builder_test/level1_interface_tests.py` - Basic interface tests
- `universal_step_builder_test/level2_specification_tests.py` - Specification and contract tests
- `universal_step_builder_test/level3_path_mapping_tests.py` - Path mapping tests
- `universal_step_builder_test/level4_integration_tests.py` - System integration tests
- `universal_step_builder_test/universal_tests.py` - Combined test suite
- `universal_step_builder_test/scoring.py` - Scoring system for evaluating builder quality

To extend the tests:

1. Choose the appropriate level for your new test based on complexity
2. Add your test method to the corresponding class
3. The test will be automatically included when running the full suite

## Quality Scoring System

The Universal Step Builder Test includes a scoring system to evaluate the quality and compliance of step builders. The scoring system:

1. **Assigns weights to test levels**:
   - Level 1 (Interface): Weight 1.0
   - Level 2 (Specification): Weight 1.5
   - Level 3 (Path Mapping): Weight 2.0
   - Level 4 (Integration): Weight 2.5

2. **Considers test importance**:
   - Critical tests like step creation have higher weights
   - Basic tests like inheritance have lower weights

3. **Generates detailed reports**:
   - Overall score (0-100) and rating (Poor to Excellent)
   - Per-level scores to identify specific areas for improvement
   - List of failed tests with detailed error messages

4. **Creates visual charts** (if matplotlib is available):
   - Bar charts showing scores for each test level
   - Overall score line for quick comparison
   - Color coding based on score ranges

### Scoring Ratings

The scoring system uses the following rating scale:

- **90-100**: Excellent - Fully compliant with architectural requirements
- **80-89**: Good - Mostly compliant with minor issues
- **70-79**: Satisfactory - Functional but has room for improvement
- **60-69**: Needs Work - Significant compliance issues
- **0-59**: Poor - Major architectural problems

### Using Scores for Quality Control

The quality scores can be used for:

1. **Setting quality gates** in CI/CD pipelines
2. **Tracking improvement** over time
3. **Identifying common issues** across multiple builders
4. **Prioritizing fixes** based on level weights and test importance
5. **Standardizing** step builder implementations across teams
