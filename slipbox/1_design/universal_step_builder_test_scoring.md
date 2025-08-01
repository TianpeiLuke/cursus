# Universal Step Builder Test Scoring System

This document outlines the design and implementation of a quality scoring system for the Universal Step Builder Test. This scoring system provides quantitative metrics to evaluate the quality and architectural compliance of step builder implementations.

> **Related Documentation**  
> This document describes the scoring system that extends the Universal Step Builder Test.  
> For the core design of the test framework itself, see [universal_step_builder_test.md](./universal_step_builder_test.md).

## Purpose

The Universal Step Builder Test Scoring System:

1. **Quantifies Quality** - Translates test results into measurable metrics
2. **Evaluates Compliance** - Assesses adherence to architectural standards
3. **Enables Comparisons** - Allows for comparison between different builder implementations
4. **Identifies Weaknesses** - Pinpoints specific areas needing improvement
5. **Establishes Baselines** - Creates quality thresholds for acceptance

## Design Principles

The scoring system follows these key design principles:

1. **Weighted Assessment** - Different aspects of compliance have different importance
2. **Multi-Level Evaluation** - Scoring is broken down by architectural levels
3. **Objective Metrics** - Scores are based on concrete test results, not subjective judgment
4. **Visual Reporting** - Results are presented in easy-to-understand visual formats
5. **Actionable Feedback** - Reports identify specific areas for improvement

## Scoring Architecture

The scoring system classifies tests into four architectural levels with increasing weights:

| Level | Name | Weight | Knowledge Required | Description |
|-------|------|--------|-------------------|-------------|
| 1 | Interface | 1.0 | Basic | Fundamental interface compliance |
| 2 | Specification | 1.5 | Moderate | Specification and contract usage |
| 3 | Path Mapping | 2.0 | Advanced | Path and property mapping |
| 4 | Integration | 2.5 | Expert | System integration and dependency handling |

This weighted approach reflects that failures in higher levels (e.g., integration) have more significant impacts on system reliability than failures in lower levels (e.g., basic interface).

## Test Importance Weighting

In addition to level weights, individual tests have importance weights based on their criticality:

| Test | Weight | Rationale |
|------|--------|-----------|
| test_inheritance | 1.0 | Basic requirement |
| test_required_methods | 1.2 | Slightly more important than basic inheritance |
| test_specification_usage | 1.2 | Core specification compliance |
| test_contract_alignment | 1.3 | Critical for script contract integration |
| test_property_path_validity | 1.3 | Essential for runtime property access |
| test_dependency_resolution | 1.4 | Critical for pipeline integration |
| test_step_creation | 1.5 | Most critical test - final output |

All other tests default to weight 1.0 unless explicitly overridden.

## Scoring Algorithm

The scoring algorithm calculates:

1. **Level Scores** - For each architectural level:
   ```
   level_score = (sum(test_importance * test_passed) / sum(test_importance)) * 100
   ```
   Where `test_passed` is 1 if the test passed, 0 if it failed.

2. **Overall Score** - Weighted average of level scores:
   ```
   overall_score = (sum(level_score * level_weight) / sum(level_weight))
   ```

3. **Rating** - Categorical rating based on overall score:

   | Score Range | Rating |
   |-------------|--------|
   | 90-100 | Excellent |
   | 80-89 | Good |
   | 70-79 | Satisfactory |
   | 60-69 | Needs Work |
   | 0-59 | Poor |

## Implementation

The scoring system is implemented in the `test/pipeline_steps/universal_step_builder_test/scoring.py` module, with these key components:

### 1. Test-to-Level Mapping

```python
# Define test to level mapping
TEST_LEVEL_MAP = {
    "test_inheritance": "level1_interface",
    "test_required_methods": "level1_interface",
    "test_error_handling": "level1_interface",
    
    "test_specification_usage": "level2_specification",
    "test_contract_alignment": "level2_specification",
    "test_environment_variable_handling": "level2_specification",
    "test_job_arguments": "level2_specification",
    
    "test_input_path_mapping": "level3_path_mapping",
    "test_output_path_mapping": "level3_path_mapping",
    "test_property_path_validity": "level3_path_mapping",
    
    "test_dependency_resolution": "level4_integration",
    "test_step_creation": "level4_integration",
    "test_step_name": "level4_integration",
}
```

### 2. Level and Test Weighting

```python
# Define weights for each test level
LEVEL_WEIGHTS = {
    "level1_interface": 1.0,    # Basic interface compliance
    "level2_specification": 1.5, # Specification and contract compliance
    "level3_path_mapping": 2.0,  # Path mapping and property paths
    "level4_integration": 2.5,   # System integration
}

# Define importance weights for specific tests
TEST_IMPORTANCE = {
    # All tests default to 1.0, override specific tests if needed
    "test_inheritance": 1.0,
    "test_required_methods": 1.2,
    "test_specification_usage": 1.2,
    "test_contract_alignment": 1.3,
    "test_property_path_validity": 1.3,
    "test_dependency_resolution": 1.4,
    "test_step_creation": 1.5,
}
```

### 3. StepBuilderScorer Class

The `StepBuilderScorer` class encapsulates the scoring logic:

```python
class StepBuilderScorer:
    """
    A scorer for evaluating step builder quality based on test results.
    """
    
    def __init__(self, results: Dict[str, Dict[str, Any]]):
        """
        Initialize with test results.
        
        Args:
            results: Dictionary mapping test names to their results
        """
        self.results = results
        self.level_results = self._group_by_level()
        
    def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
        """
        Calculate score for a specific level.
        
        Args:
            level: Name of the level to score
            
        Returns:
            Tuple containing (score, passed_tests, total_tests)
        """
        # Implementation...
        
    def calculate_overall_score(self) -> float:
        """
        Calculate overall score across all levels.
        
        Returns:
            Overall score (0-100)
        """
        # Implementation...
        
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive score report.
        
        Returns:
            Dictionary containing the full score report
        """
        # Implementation...
        
    def print_report(self) -> None:
        """Print a formatted score report to the console."""
        # Implementation...
        
    def generate_chart(self, builder_name: str, output_dir: str = "test_reports") -> Optional[str]:
        """
        Generate a chart visualization of the score report.
        
        Args:
            builder_name: Name of the builder
            output_dir: Directory to save the chart in
            
        Returns:
            Path to the saved chart or None if matplotlib is not available
        """
        # Implementation...
```

### 4. Report Structure

The score report has this structure:

```json
{
  "overall": {
    "score": 85.5,
    "rating": "Good",
    "passed": 12,
    "total": 13,
    "pass_rate": 92.3
  },
  "levels": {
    "level1_interface": {
      "score": 100.0,
      "passed": 3,
      "total": 3,
      "tests": {
        "test_inheritance": true,
        "test_required_methods": true,
        "test_error_handling": true
      }
    },
    "level2_specification": {
      "score": 75.0,
      "passed": 3,
      "total": 4,
      "tests": {
        "test_specification_usage": true,
        "test_contract_alignment": true,
        "test_environment_variable_handling": true,
        "test_job_arguments": false
      }
    },
    "level3_path_mapping": { /* Similar structure */ },
    "level4_integration": { /* Similar structure */ }
  },
  "failed_tests": [
    {
      "name": "test_job_arguments",
      "error": "Job arguments did not match expected format"
    }
  ]
}
```

### 5. Visual Reporting

The scoring system generates visual charts using matplotlib:

- Bar chart showing scores for each level
- Overall score line for comparison
- Color coding based on score ranges:
  - Green: ≥ 90 (Excellent)
  - Light green: ≥ 80 (Good)
  - Orange: ≥ 70 (Satisfactory)
  - Salmon: ≥ 60 (Needs Work)
  - Red: < 60 (Poor)

## Integration with CLI

The scoring system is integrated with the test runner command-line interface:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --score
```

Additional options:

```bash
python test/pipeline_steps/run_universal_step_builder_test.py --score --output-dir ./reports --no-chart
```

## Quality Gates

The scoring system enables the establishment of quality gates for CI/CD pipelines:

1. **Minimum Overall Score** - E.g., require at least 80% overall score
2. **Level-Specific Requirements** - E.g., require at least 90% for level1 and level2
3. **Critical Test Requirements** - E.g., require all tests in level4_integration to pass

Example quality gate implementation:

```python
def check_quality_gate(report):
    """Check if report passes quality gates."""
    # Check overall score
    if report["overall"]["score"] < 80:
        return False, "Overall score below 80%"
    
    # Check critical levels
    if report["levels"]["level1_interface"]["score"] < 90:
        return False, "Interface compliance below 90%"
    
    if report["levels"]["level2_specification"]["score"] < 90:
        return False, "Specification compliance below 90%"
    
    # Check critical tests
    for test in ["test_step_creation", "test_dependency_resolution"]:
        for test_result in report["failed_tests"]:
            if test_result["name"] == test:
                return False, f"Critical test failed: {test}"
    
    return True, "All quality gates passed"
```

## Usage Examples

### 1. Basic Score Generation

```python
from universal_step_builder_test.scoring import score_builder_results

# Generate score report
report = score_builder_results(
    results=test_results,
    builder_name="XGBoostTrainingStepBuilder",
    save_report=True,
    output_dir="reports",
    generate_chart=True
)

# Check quality gate
passed, message = check_quality_gate(report)
if not passed:
    print(f"Quality gate failed: {message}")
```

### 2. Comparing Multiple Builders

```python
# Test multiple builders
all_reports = {}
for builder_name, builder_class in builder_classes.items():
    tester = UniversalStepBuilderTest(builder_class)
    results = tester.run_all_tests()
    
    # Generate score report
    report = score_builder_results(
        results=results,
        builder_name=builder_name,
        save_report=True,
        output_dir="reports",
        generate_chart=True
    )
    
    all_reports[builder_name] = report

# Find best and worst builders
best_builder = max(all_reports.items(), key=lambda x: x[1]["overall"]["score"])
worst_builder = min(all_reports.items(), key=lambda x: x[1]["overall"]["score"])

print(f"Best builder: {best_builder[0]} with score {best_builder[1]['overall']['score']}")
print(f"Worst builder: {worst_builder[0]} with score {worst_builder[1]['overall']['score']}")
```

## Benefits and Applications

The Universal Step Builder Test Scoring System provides several key benefits:

1. **Objective Quality Measurement** - Provides concrete metrics rather than subjective evaluations
2. **Targeted Improvement** - Identifies specific areas needing attention
3. **Architectural Compliance** - Ensures adherence to design principles and best practices
4. **Standardization** - Promotes consistent implementation across different builders
5. **Progress Tracking** - Enables monitoring of improvement over time
6. **CI/CD Integration** - Provides automated quality gates for continuous integration
7. **Developer Guidance** - Helps developers understand architectural requirements

## Extension Points

The scoring system is designed to be extensible:

1. **Custom Weights** - Adjust level and test weights for specific project needs
2. **Additional Tests** - New tests can be added to the appropriate level
3. **Custom Quality Gates** - Define project-specific quality requirements
4. **Reporting Formats** - Add additional visualization or reporting formats
5. **Trend Analysis** - Add historical tracking of scores over time

## Conclusion

The Universal Step Builder Test Scoring System transforms test results into actionable quality metrics. By providing quantitative measurements and clear visualizations, it enables teams to objectively assess step builder quality, identify areas for improvement, and enforce architectural standards through automated quality gates.
