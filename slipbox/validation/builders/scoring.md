---
tags:
  - code
  - validation
  - builders
  - scoring
  - quality_assessment
keywords:
  - step builder scoring
  - quality metrics
  - weighted scoring
  - test level weights
  - rating system
  - score calculation
topics:
  - builder quality assessment
  - scoring methodology
  - test result analysis
  - quality metrics
language: python
date of note: 2025-08-15
---

# Step Builder Scoring System

## Overview

The `scoring.py` module provides a comprehensive scoring mechanism to evaluate step builders based on their performance across four test levels. The scoring system assigns different weights to different test levels, reflecting their importance in the overall architecture and providing quantitative quality assessment.

## Architecture

### Scoring Methodology
The scoring system uses a weighted approach where different test levels have different importance:

```
Level 1 (Interface): 1.0x weight
Level 2 (Specification): 1.5x weight  
Level 3 (Path Mapping): 2.0x weight
Level 4 (Integration): 2.5x weight
```

### Key Components

1. **StepBuilderScorer**: Main scoring engine class
2. **Level Weights**: Configurable weights for each test level
3. **Test Detection**: Smart pattern recognition for test categorization
4. **Rating System**: Qualitative ratings based on numerical scores
5. **Report Generation**: Comprehensive scoring reports with recommendations

## Scoring Configuration

### Level Weights
```python
LEVEL_WEIGHTS = {
    "level1_interface": 1.0,      # Basic interface compliance
    "level2_specification": 1.5,  # Specification and contract compliance
    "level3_path_mapping": 2.0,   # Path mapping and property paths
    "level4_integration": 2.5,    # System integration
}
```

### Test Importance Weights
Individual tests can have additional importance multipliers:
```python
TEST_IMPORTANCE = {
    "test_inheritance": 1.0,
    "test_required_methods": 1.2,
    "test_specification_usage": 1.2,
    "test_contract_alignment": 1.3,
    "test_property_path_validity": 1.3,
    "test_dependency_resolution": 1.4,
    "test_step_creation": 1.5,
}
```

### Rating Levels
```python
RATING_LEVELS = {
    90: "Excellent",   # 90-100: Excellent
    80: "Good",        # 80-89: Good
    70: "Satisfactory",# 70-79: Satisfactory
    60: "Needs Work",  # 60-69: Needs Work
    0: "Poor"          # 0-59: Poor
}
```

## StepBuilderScorer Class

### Core Functionality

#### Initialization
```python
scorer = StepBuilderScorer(results)
# results: Dictionary mapping test names to their results
```

#### Score Calculation
```python
# Calculate score for a specific level
score, passed, total = scorer.calculate_level_score("level1_interface")

# Calculate overall score across all levels
overall_score = scorer.calculate_overall_score()

# Get qualitative rating
rating = scorer.get_rating(overall_score)
```

#### Report Generation
```python
# Generate comprehensive report
report = scorer.generate_report()

# Save report to file
scorer.save_report("builder_name", "output_dir")

# Print formatted report
scorer.print_report(show_test_detection=True)
```

### Test Level Detection

The scorer uses multiple strategies to determine test levels:

#### 1. Explicit Level Prefix (Preferred)
```python
# Tests with explicit level prefixes
"level1_test_inheritance"      → level1_interface
"level2_test_specification"    → level2_specification
"level3_test_path_mapping"     → level3_path_mapping
"level4_test_integration"      → level4_integration
```

#### 2. Keyword-Based Detection
```python
# Level 1 keywords: interface, methods, creation, inheritance
"test_inheritance" → level1_interface
"test_required_methods" → level1_interface

# Level 2 keywords: specification, contract, environment, arguments
"test_specification_usage" → level2_specification
"test_contract_alignment" → level2_specification

# Level 3 keywords: path, mapping, input, output
"test_input_path_mapping" → level3_path_mapping
"test_output_path_mapping" → level3_path_mapping

# Level 4 keywords: dependency, step_creation, integration
"test_dependency_resolution" → level4_integration
"test_step_creation" → level4_integration
```

#### 3. Fallback Mapping
```python
# Explicit mapping for edge cases
TEST_LEVEL_MAP = {
    "test_generic_step_creation": "level1_interface",
    "test_processing_job_arguments": "level2_specification",
    "test_processing_inputs_outputs": "level3_path_mapping",
    "test_generic_dependency_handling": "level4_integration",
}
```

## Score Calculation Algorithm

### Level Score Calculation
```python
def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
    level_tests = self.level_results[level]
    total_weight = 0.0
    weighted_score = 0.0
    
    for test_name, result in level_tests.items():
        importance = TEST_IMPORTANCE.get(test_name, 1.0)
        total_weight += importance
        
        if result.get("passed", False):
            weighted_score += importance
    
    score = (weighted_score / total_weight) * 100.0 if total_weight > 0 else 0.0
    return score, passed_count, total_count
```

### Overall Score Calculation
```python
def calculate_overall_score(self) -> float:
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for level, weight in LEVEL_WEIGHTS.items():
        level_score, _, _ = self.calculate_level_score(level)
        total_weighted_score += level_score * weight
        total_weight += weight
    
    overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
    return min(100.0, max(0.0, overall_score))
```

## Report Structure

### Comprehensive Report Format
```python
{
    "overall": {
        "score": 85.3,
        "rating": "Good",
        "passed": 18,
        "total": 20,
        "pass_rate": 90.0
    },
    "levels": {
        "level1_interface": {
            "score": 95.0,
            "passed": 5,
            "total": 5,
            "tests": {
                "test_inheritance": True,
                "test_required_methods": True,
                # ... other tests
            }
        },
        # ... other levels
    },
    "failed_tests": [
        {
            "name": "test_contract_alignment",
            "error": "Contract not found for step"
        }
    ]
}
```

### Console Output
```
================================================================================
STEP BUILDER QUALITY SCORE REPORT
================================================================================

Overall Score: 85.3/100 - Good
Pass Rate: 90.0% (18/20 tests)

Scores by Level:
  Level 1 Interface: 95.0/100 (5/5 tests)
  Level 2 Specification: 82.5/100 (3/4 tests)
  Level 3 Path Mapping: 78.0/100 (4/5 tests)
  Level 4 Integration: 88.0/100 (5/6 tests)

❌ Failed Tests (2):
  • test_contract_alignment: Contract not found for step
  • test_property_path_validity: Invalid property path detected
```

## Advanced Features

### Test Detection Analysis
```python
# Get detailed test detection statistics
detection_summary = scorer.get_detection_summary()

# Print detection details
scorer.print_report(show_test_detection=True)
```

Output:
```
Test Level Detection Details:
--------------------------------------------------
  Explicit prefix (level1_, level2_, etc.): 8 tests
  Keyword-based detection: 10 tests
  Fallback to TEST_LEVEL_MAP: 2 tests
  Undetected (no level assigned): 0 tests
```

### Chart Generation
```python
# Generate visual score chart (requires matplotlib)
chart_path = scorer.generate_chart("builder_name", "output_dir")
```

Creates a bar chart showing:
- Level scores with color coding
- Overall score line
- Score labels on bars
- Weight information

### Batch Scoring
```python
# Convenience function for quick scoring
report = score_builder_results(
    results=test_results,
    builder_name="XGBoostTraining",
    save_report=True,
    output_dir="test_reports",
    generate_chart=True
)
```

## Integration Points

### With UniversalStepBuilderTest
```python
# Automatic integration in enhanced universal test
tester = UniversalStepBuilderTest(BuilderClass, enable_scoring=True)
results = tester.run_all_tests()
scoring_data = results['scoring']  # Generated by StepBuilderScorer
```

### With BuilderTestReporter
```python
# Can be integrated into existing reporting system
scorer = StepBuilderScorer(raw_results)
score_report = scorer.generate_report()
# Merge with BuilderTestReport structure
```

## Configuration and Customization

### Custom Weights
```python
# Modify level weights for different priorities
LEVEL_WEIGHTS["level3_path_mapping"] = 3.0  # Increase path mapping importance
LEVEL_WEIGHTS["level1_interface"] = 0.5     # Decrease interface importance
```

### Custom Test Importance
```python
# Add importance weights for specific tests
TEST_IMPORTANCE["test_custom_validation"] = 2.0
TEST_IMPORTANCE["test_critical_feature"] = 1.8
```

### Custom Rating Thresholds
```python
# Adjust rating thresholds
RATING_LEVELS[95] = "Outstanding"
RATING_LEVELS[85] = "Very Good"
```

## Usage Examples

### Basic Scoring
```python
from cursus.validation.builders.scoring import StepBuilderScorer

# Score test results
scorer = StepBuilderScorer(test_results)
overall_score = scorer.calculate_overall_score()
rating = scorer.get_rating(overall_score)

print(f"Quality Score: {overall_score:.1f}/100 - {rating}")
```

### Detailed Analysis
```python
# Generate and analyze comprehensive report
report = scorer.generate_report()

# Check specific levels
for level_name, level_data in report['levels'].items():
    if level_data['score'] < 70:
        print(f"⚠️  {level_name} needs improvement: {level_data['score']:.1f}/100")
```

### Export and Visualization
```python
# Save detailed report
scorer.save_report("MyBuilder", "reports")

# Generate chart if matplotlib available
chart_path = scorer.generate_chart("MyBuilder", "reports")

# Print formatted report
scorer.print_report(show_test_detection=True)
```

## Performance Considerations

### Efficient Test Detection
- **Caching**: Test level detection results are cached
- **Pattern Matching**: Optimized keyword matching algorithms
- **Lazy Evaluation**: Reports generated only when requested

### Memory Usage
- **Streaming**: Large result sets processed incrementally
- **Cleanup**: Temporary data structures cleaned after use
- **Optimization**: Minimal memory footprint for scoring calculations

## Error Handling

### Graceful Degradation
```python
try:
    scorer = StepBuilderScorer(results)
    report = scorer.generate_report()
except Exception as e:
    print(f"⚠️  Scoring failed: {e}")
    # Fall back to basic pass/fail reporting
```

### Validation
- **Input Validation**: Test results format validation
- **Missing Data**: Handles missing or malformed test results
- **Edge Cases**: Handles empty test suites and edge cases

## Related Components

- **universal_test.py**: Main integration point for scoring
- **builder_reporter.py**: Alternative reporting system
- **interface_tests.py**: Level 1 test source
- **specification_tests.py**: Level 2 test source
- **path_mapping_tests.py**: Level 3 test source
- **integration_tests.py**: Level 4 test source

## Future Enhancements

### 1. Machine Learning Integration
```python
# Potential ML-based scoring adjustments
def adjust_score_with_ml(base_score, test_patterns, historical_data):
    # Use ML model to refine scoring based on patterns
    pass
```

### 2. Trend Analysis
```python
# Track scoring trends over time
def analyze_score_trends(builder_name, historical_scores):
    # Analyze improvement/degradation trends
    pass
```

### 3. Comparative Analysis
```python
# Compare scores across builders
def compare_builder_scores(builder_scores):
    # Generate comparative analysis and rankings
    pass
```

## Conclusion

The scoring system provides a robust, flexible, and comprehensive approach to quantifying step builder quality. By combining weighted scoring across multiple test levels with intelligent test detection and rich reporting capabilities, it enables developers to:

- **Quantify quality** with objective metrics
- **Identify improvement areas** through level-specific scoring
- **Track progress** over time with consistent measurements
- **Make informed decisions** based on comprehensive analysis

The system's integration with the universal test framework provides a seamless experience while maintaining flexibility for standalone usage and customization.
