---
tags:
  - code
  - validation
  - alignment
  - scoring
  - metrics
keywords:
  - alignment scoring
  - validation metrics
  - quality assessment
  - alignment levels
  - weighted scoring
  - validation quality
  - alignment rating
  - score calculation
topics:
  - validation framework
  - alignment scoring
  - quality metrics
  - validation assessment
language: python
date of note: 2025-08-18
---

# Alignment Scorer

## Overview

The Alignment Scorer provides quantitative assessment of alignment validation quality across all four alignment levels. It calculates weighted scores, generates quality ratings, and produces comprehensive reports with visualizations.

## Scoring Architecture

### Four-Level Weighted Scoring

The scorer evaluates alignment quality across four hierarchical levels with increasing importance weights:

```python
ALIGNMENT_LEVEL_WEIGHTS = {
    "level1_script_contract": 1.0,      # Basic script-contract alignment
    "level2_contract_spec": 1.5,        # Contract-specification alignment
    "level3_spec_dependencies": 2.0,    # Specification-dependencies alignment
    "level4_builder_config": 2.5,       # Builder-configuration alignment
}
```

**Weight Rationale:**
- **Level 1 (1.0x)**: Foundation level - script must match contract
- **Level 2 (1.5x)**: Interface consistency - contract must align with specification
- **Level 3 (2.0x)**: Dependency integrity - specifications must resolve dependencies
- **Level 4 (2.5x)**: Configuration correctness - builders must handle config properly

### Quality Rating System

Scores are mapped to quality ratings for intuitive assessment:

```python
ALIGNMENT_RATING_LEVELS = {
    90: "Excellent",     # 90-100: Excellent alignment
    80: "Good",          # 80-89: Good alignment
    70: "Satisfactory",  # 70-79: Satisfactory alignment
    60: "Needs Work",    # 60-69: Needs improvement
    0: "Poor"            # 0-59: Poor alignment
}
```

### Test Importance Weighting

Individual tests within each level have importance weights for fine-tuned scoring:

```python
ALIGNMENT_TEST_IMPORTANCE = {
    "script_contract_path_alignment": 1.5,        # Critical path usage
    "contract_spec_logical_names": 1.4,           # Name consistency
    "spec_dependency_resolution": 1.3,            # Dependency resolution
    "builder_config_environment_vars": 1.2,       # Environment setup
    "script_contract_environment_vars": 1.2,      # Variable usage
    "contract_spec_dependency_mapping": 1.3,      # Dependency mapping
    "spec_dependency_property_paths": 1.4,        # Property path consistency
    "builder_config_specification_alignment": 1.5 # Config-spec alignment
}
```

## AlignmentScorer Class

### Initialization and Result Processing

```python
scorer = AlignmentScorer(validation_results)
```

The scorer accepts validation results in multiple formats:
- **AlignmentReport format**: `level1_results`, `level2_results`, etc.
- **Direct level format**: `level1`, `level2`, etc.
- **Test collection format**: `tests` or `validations` dictionary

### Smart Level Detection

The scorer uses intelligent pattern detection to categorize tests by alignment level:

```python
def _detect_level_from_test_name(self, test_name: str) -> Optional[str]:
    """Detect alignment level from test name using pattern detection."""
```

**Level 1 Keywords**: `script_contract`, `script`, `contract`, `path_alignment`, `environment_vars`
**Level 2 Keywords**: `contract_spec`, `logical_names`, `specification`, `contract_alignment`
**Level 3 Keywords**: `spec_dependencies`, `dependency`, `property_paths`, `dependency_resolution`
**Level 4 Keywords**: `builder_config`, `configuration`, `builder`, `config_alignment`

### Score Calculation Methods

#### Level Score Calculation

```python
def calculate_level_score(self, level: str) -> Tuple[float, int, int]:
    """Calculate score for a specific alignment level."""
```

**Process:**
1. Collect all tests for the specified level
2. Apply importance weights to each test
3. Calculate weighted pass rate
4. Return (score, passed_count, total_count)

**Formula:**
```
Level Score = (Σ(test_passed × importance_weight) / Σ(importance_weight)) × 100
```

#### Overall Score Calculation

```python
def calculate_overall_score(self) -> float:
    """Calculate overall alignment score across all levels."""
```

**Process:**
1. Calculate individual level scores
2. Apply level weights to each score
3. Compute weighted average across all levels
4. Clamp result to 0-100 range

**Formula:**
```
Overall Score = Σ(level_score × level_weight) / Σ(level_weight)
```

### Test Result Interpretation

The scorer handles multiple test result formats:

```python
def _is_test_passed(self, result: Any) -> bool:
    """Determine if a test passed based on its result structure."""
```

**Supported Formats:**
- **Dictionary with 'passed' field**: `{"passed": True}`
- **Dictionary with 'success' field**: `{"success": True}`
- **Dictionary with 'status' field**: `{"status": "passed"}`
- **Dictionary with 'errors' field**: `{"errors": []}` (empty = passed)
- **Dictionary with 'issues' field**: No critical/error issues = passed
- **Boolean value**: Direct true/false
- **String value**: "passed", "success", "ok", "true"

## Comprehensive Reporting

### Report Generation

```python
def generate_report(self) -> Dict[str, Any]:
    """Generate a comprehensive alignment score report."""
```

**Report Structure:**
```json
{
    "overall": {
        "score": 85.2,
        "rating": "Good",
        "passed": 12,
        "total": 15,
        "pass_rate": 80.0
    },
    "levels": {
        "level1_script_contract": {
            "score": 90.0,
            "passed": 3,
            "total": 3,
            "tests": {"test1": true, "test2": true, "test3": true}
        }
    },
    "failed_tests": [
        {
            "name": "dependency_resolution_test",
            "level": "level3_spec_dependencies",
            "error": "2 alignment issues found"
        }
    ],
    "metadata": {
        "scoring_system": "alignment_validation",
        "level_weights": {...},
        "test_importance": {...}
    }
}
```

### Console Output

```python
def print_report(self) -> None:
    """Print a formatted alignment score report to the console."""
```

**Sample Output:**
```
================================================================================
ALIGNMENT VALIDATION QUALITY SCORE REPORT
================================================================================

Overall Score: 85.2/100 - Good
Pass Rate: 80.0% (12/15 tests)

Scores by Alignment Level:
  Level 1 (Script ↔ Contract): 90.0/100 (3/3 tests)
  Level 2 (Contract ↔ Specification): 85.0/100 (4/5 tests)
  Level 3 (Specification ↔ Dependencies): 80.0/100 (3/4 tests)
  Level 4 (Builder ↔ Configuration): 83.3/100 (2/3 tests)

Failed Tests:
  ❌ dependency_resolution_test (Level 3): 2 alignment issues found
  ❌ config_parameter_usage (Level 4): Environment variable mismatch
  ❌ logical_name_consistency (Level 2): Name mismatch found
================================================================================
```

## Visualization and Export

### Chart Generation

```python
def generate_chart(self, script_name: str, output_dir: str = "alignment_reports") -> Optional[str]:
    """Generate a chart visualization of the alignment score report."""
```

**Chart Features:**
- **Bar Chart**: Individual level scores with color coding
- **Overall Score Line**: Horizontal line showing overall score
- **Color Coding**: 
  - Green (Excellent): 90-100
  - Light Green (Good): 80-89
  - Orange (Satisfactory): 70-79
  - Salmon (Needs Work): 60-69
  - Red (Poor): 0-59
- **Score Labels**: Percentage values on each bar
- **Professional Styling**: Grid lines, proper spacing, rotated labels

### Report Persistence

```python
def save_report(self, script_name: str, output_dir: str = "alignment_reports") -> str:
    """Save the alignment score report to a JSON file."""
```

**Output Files:**
- **JSON Report**: `{script_name}_alignment_score_report.json`
- **Chart Image**: `{script_name}_alignment_score_chart.png`

## Integration Patterns

### Standalone Usage

```python
# Direct scoring of validation results
validation_results = {...}  # From alignment validation
scorer = AlignmentScorer(validation_results)

# Get scores
overall_score = scorer.calculate_overall_score()
level1_score, passed, total = scorer.calculate_level_score("level1_script_contract")

# Generate comprehensive report
report = scorer.generate_report()
scorer.print_report()
```

### Integration with AlignmentReport

```python
# Used by AlignmentReport for integrated scoring
class AlignmentReport:
    def get_scorer(self) -> AlignmentScorer:
        scorer_results = self._convert_to_scorer_format()
        return AlignmentScorer(scorer_results)
    
    def get_alignment_score(self) -> float:
        return self.get_scorer().calculate_overall_score()
```

### Convenience Function

```python
def score_alignment_results(validation_results: Dict[str, Any], 
                          script_name: str = "Unknown",
                          save_report: bool = True,
                          output_dir: str = "alignment_reports",
                          generate_chart: bool = True) -> Dict[str, Any]:
    """Score alignment validation results for a script."""
```

**Complete Workflow:**
1. Create AlignmentScorer instance
2. Generate comprehensive report
3. Print formatted console output
4. Save JSON report (optional)
5. Generate visualization chart (optional)
6. Return report dictionary

## Error Handling and Robustness

### Graceful Degradation

The scorer handles various edge cases:

- **Missing Levels**: Returns 0.0 score for missing alignment levels
- **Empty Results**: Handles empty test collections gracefully
- **Malformed Data**: Defensive parsing of test results
- **Missing Dependencies**: Chart generation fails gracefully without matplotlib

### Error Message Extraction

```python
def _extract_error_message(self, result: Any) -> str:
    """Extract error message from test result."""
```

**Extraction Priority:**
1. Explicit 'error' field
2. Generic 'message' field
3. Count of 'issues' array
4. Count of 'errors' array
5. Default "Test failed" message

## Configuration and Customization

### Weight Adjustment

Modify scoring weights for different validation priorities:

```python
# Emphasize higher-level alignment
ALIGNMENT_LEVEL_WEIGHTS = {
    "level1_script_contract": 0.5,
    "level2_contract_spec": 1.0,
    "level3_spec_dependencies": 1.5,
    "level4_builder_config": 3.0,  # Highest priority
}
```

### Test Importance Tuning

Adjust individual test importance:

```python
# Critical path alignment gets highest weight
ALIGNMENT_TEST_IMPORTANCE = {
    "script_contract_path_alignment": 2.0,  # Critical
    "other_tests": 1.0  # Standard
}
```

### Rating Threshold Customization

Modify quality rating thresholds:

```python
# Stricter quality standards
ALIGNMENT_RATING_LEVELS = {
    95: "Excellent",
    85: "Good", 
    75: "Satisfactory",
    65: "Needs Work",
    0: "Poor"
}
```

## Performance Considerations

### Efficient Calculation

- **Lazy Evaluation**: Scores calculated only when requested
- **Caching**: Results cached within scorer instance
- **Batch Processing**: All levels processed in single pass

### Memory Management

- **Lightweight Storage**: Only essential data retained
- **Chart Cleanup**: Matplotlib figures properly closed
- **File Handling**: Proper resource management for exports

## Best Practices

### Consistent Result Format

Ensure validation results follow expected format:

```python
# Preferred format for AlignmentScorer
validation_results = {
    "level1_results": {
        "test_name": {"passed": True, "issues": []},
        # ... more tests
    },
    "level2_results": {...},
    "level3_results": {...},
    "level4_results": {...}
}
```

### Meaningful Test Names

Use descriptive test names that enable automatic level detection:

```python
# Good test names
"script_contract_path_alignment"      # Clearly Level 1
"contract_spec_logical_names"         # Clearly Level 2
"spec_dependency_resolution"          # Clearly Level 3
"builder_config_environment_vars"     # Clearly Level 4
```

### Comprehensive Reporting

Always generate complete reports for full visibility:

```python
# Complete scoring workflow
scorer = AlignmentScorer(results)
report = scorer.generate_report()
scorer.print_report()
chart_path = scorer.generate_chart(script_name)
json_path = scorer.save_report(script_name)
```

The Alignment Scorer provides essential quantitative assessment capabilities for the alignment validation framework, enabling data-driven quality improvement and objective validation success measurement.
