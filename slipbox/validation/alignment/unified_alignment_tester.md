---
tags:
  - test
  - validation
  - alignment
  - orchestrator
  - unified
keywords:
  - unified alignment tester
  - alignment orchestrator
  - four-tier validation
  - step type awareness
  - alignment reporting
  - validation coordination
topics:
  - alignment validation
  - test orchestration
  - architectural validation
  - component integration
language: python
date of note: 2025-08-18
---

# Unified Alignment Tester

The Unified Alignment Tester serves as the main orchestrator for comprehensive alignment validation across all four levels of the Cursus architecture. It coordinates validation between scripts, contracts, specifications, builders, and configurations while providing rich reporting and step type-aware enhancements.

## Overview

The `UnifiedAlignmentTester` class implements a sophisticated validation orchestration system that:

- Coordinates validation across four distinct alignment levels
- Provides step type-aware validation enhancements
- Generates comprehensive reports with actionable recommendations
- Supports flexible configuration and validation modes
- Includes performance optimizations and error recovery

## Architecture

### Core Components

1. **Level-Specific Testers**: Individual testers for each alignment level
2. **Step Type Enhancement Router**: Provides step type-specific validation enhancements
3. **Alignment Reporter**: Generates comprehensive validation reports
4. **Configuration Manager**: Handles validation mode configuration

### Validation Levels

#### Level 1: Script ↔ Contract Alignment
- **Tester**: `ScriptContractAlignmentTester`
- **Focus**: Script function signatures and contract specifications
- **Key Validations**: Parameter alignment, return types, environment variables

#### Level 2: Contract ↔ Specification Alignment
- **Tester**: `ContractSpecificationAlignmentTester`
- **Focus**: Contract parameters and step specifications
- **Key Validations**: Property paths, configuration fields, type consistency

#### Level 3: Specification ↔ Dependencies Alignment
- **Tester**: `SpecificationDependencyAlignmentTester`
- **Focus**: Specification dependencies and cross-references
- **Key Validations**: Dependency declarations, property path references

#### Level 4: Builder ↔ Configuration Alignment
- **Tester**: `BuilderConfigurationAlignmentTester`
- **Focus**: Builder configuration usage and requirements
- **Key Validations**: Configuration field access, type correctness

## Class Interface

### Constructor

```python
def __init__(self, 
             scripts_dir: str = "src/cursus/steps/scripts",
             contracts_dir: str = "src/cursus/steps/contracts",
             specs_dir: str = "src/cursus/steps/specs",
             builders_dir: str = "src/cursus/steps/builders",
             configs_dir: str = "src/cursus/steps/configs",
             level3_validation_mode: str = "relaxed"):
```

**Parameters**:
- `scripts_dir`: Directory containing processing scripts
- `contracts_dir`: Directory containing script contracts
- `specs_dir`: Directory containing step specifications
- `builders_dir`: Directory containing step builders
- `configs_dir`: Directory containing step configurations
- `level3_validation_mode`: Level 3 validation mode ('strict', 'relaxed', 'permissive')

### Key Methods

#### run_full_validation()

Runs comprehensive alignment validation across all levels.

```python
def run_full_validation(self, 
                       target_scripts: Optional[List[str]] = None,
                       skip_levels: Optional[List[int]] = None) -> AlignmentReport:
```

**Parameters**:
- `target_scripts`: Specific scripts to validate (None for all)
- `skip_levels`: Alignment levels to skip (1-4)

**Returns**: Comprehensive alignment report

**Usage Example**:
```python
tester = UnifiedAlignmentTester()
report = tester.run_full_validation()

# Validate specific scripts
report = tester.run_full_validation(target_scripts=['tabular_preprocessing'])

# Skip certain levels
report = tester.run_full_validation(skip_levels=[3, 4])
```

#### run_level_validation()

Runs validation for a specific alignment level.

```python
def run_level_validation(self, level: int, 
                        target_scripts: Optional[List[str]] = None) -> AlignmentReport:
```

**Parameters**:
- `level`: Alignment level to validate (1-4)
- `target_scripts`: Specific scripts to validate

**Usage Example**:
```python
# Validate only Level 1 alignment
report = tester.run_level_validation(level=1)

# Validate Level 2 for specific scripts
report = tester.run_level_validation(level=2, target_scripts=['tabular_preprocessing'])
```

#### validate_specific_script()

Runs comprehensive validation for a specific script across all levels.

```python
def validate_specific_script(self, script_name: str) -> Dict[str, Any]:
```

**Returns**:
```python
{
    'script_name': str,
    'level1': {...},        # Level 1 validation results
    'level2': {...},        # Level 2 validation results
    'level3': {...},        # Level 3 validation results
    'level4': {...},        # Level 4 validation results
    'overall_status': 'PASSING'|'FAILING'|'ERROR'
}
```

**Usage Example**:
```python
results = tester.validate_specific_script('tabular_preprocessing')
print(f"Overall status: {results['overall_status']}")

# Check individual levels
if not results['level1']['passed']:
    print("Level 1 issues found")
```

## Validation Flow

### Full Validation Process

1. **Initialization**: Set up level-specific testers and configuration
2. **Level 1 Validation**: Script ↔ Contract alignment
3. **Level 2 Validation**: Contract ↔ Specification alignment
4. **Level 3 Validation**: Specification ↔ Dependencies alignment
5. **Level 4 Validation**: Builder ↔ Configuration alignment
6. **Report Generation**: Compile results and generate summary
7. **Scoring Calculation**: Calculate quality scores and ratings

### Error Handling

Each validation level includes comprehensive error handling:

```python
try:
    # Run level validation
    results = level_tester.validate_all_scripts(target_scripts)
    # Process results...
except Exception as e:
    # Create error result
    error_result = ValidationResult(
        test_name="level_validation",
        passed=False,
        details={'error': str(e)}
    )
    # Add to report...
```

## Step Type Enhancement System

### Phase 1: Step Type Awareness

The framework includes step type awareness features controlled by the `ENABLE_STEP_TYPE_AWARENESS` environment variable.

**Features**:
- Automatic step type detection from registry
- Framework detection from script content
- Step type context added to validation issues
- Enhanced issue categorization

**Implementation**:
```python
def _add_step_type_context_to_issues(self, script_name: str, validation_result: ValidationResult):
    # Detect step type from registry
    step_type = detect_step_type_from_registry(script_name)
    
    # Detect framework from script content
    framework = detect_framework_from_script_content(script_content)
    
    # Enhance existing issues with step type context
    for issue in validation_result.issues:
        # Convert to step type-aware issue...
```

### Phase 3: Step Type Enhancement Router

The `StepTypeEnhancementRouter` provides step type-specific validation enhancements.

**Enhanced Validation**:
```python
enhanced_result = self.step_type_enhancement_router.enhance_validation_results(
    validation_result.details, script_name
)

# Merge enhanced issues into validation result
if 'enhanced_issues' in enhanced_result:
    for enhanced_issue_data in enhanced_result['enhanced_issues']:
        # Create and add enhanced issues...
```

## Configuration Options

### Level 3 Validation Modes

#### Strict Mode
```python
level3_config = Level3ValidationConfig.create_strict_config()
```
- Rigorous dependency validation
- Strict property path requirements
- Comprehensive cross-reference checking

#### Relaxed Mode (Default)
```python
level3_config = Level3ValidationConfig.create_relaxed_config()
```
- Balanced validation approach
- Reasonable flexibility for development
- Standard property path validation

#### Permissive Mode
```python
level3_config = Level3ValidationConfig.create_permissive_config()
```
- Lenient validation for development
- Minimal dependency requirements
- Flexible property path handling

### Feature Flags

#### Step Type Awareness
```bash
export ENABLE_STEP_TYPE_AWARENESS=true
```
Enables step type-aware validation enhancements.

## Reporting and Visualization

### Validation Summary

```python
summary = tester.get_validation_summary()
```

**Summary Structure**:
```python
{
    'overall_status': 'PASSING'|'FAILING',
    'total_tests': int,
    'pass_rate': float,
    'critical_issues': int,
    'error_issues': int,
    'warning_issues': int,
    'level_breakdown': {
        'level1_tests': int,
        'level2_tests': int,
        'level3_tests': int,
        'level4_tests': int
    },
    'recommendations_count': int
}
```

### Report Export

#### JSON Export
```python
json_content = tester.export_report(format='json', output_path='alignment_report.json')
```

#### HTML Export
```python
html_content = tester.export_report(format='html', output_path='alignment_report.html')
```

#### Chart Generation
```python
# Export with alignment score chart
content = tester.export_report(
    format='json',
    output_path='report.json',
    generate_chart=True,
    script_name='tabular_preprocessing'
)
```

### Quality Scoring

The framework includes integrated quality scoring:

```python
# Scoring is automatically included in reports
print("📈 Alignment Quality Scoring:")
scorer = report.get_scorer()
overall_score = scorer.calculate_overall_score()
overall_rating = scorer.get_rating(overall_score)
print(f"   Overall Score: {overall_score:.1f}/100 ({overall_rating})")
```

## Utility Methods

### Script Discovery

```python
scripts = tester.discover_scripts()
print(f"Found {len(scripts)} scripts: {scripts}")
```

### Alignment Status Matrix

```python
matrix = tester.get_alignment_status_matrix()
for script, levels in matrix.items():
    print(f"{script}: L1={levels['level1']}, L2={levels['level2']}")
```

### Critical Issues

```python
critical_issues = tester.get_critical_issues()
for issue in critical_issues:
    print(f"CRITICAL: {issue['message']}")
```

## Performance Features

### Error Recovery

The tester includes robust error recovery mechanisms:

- **Level Isolation**: Errors in one level don't prevent other levels from running
- **Graceful Degradation**: Partial results returned even with errors
- **Error Reporting**: Detailed error information included in reports

### Optimization Strategies

- **Lazy Loading**: Components loaded only when needed
- **Batch Processing**: Efficient validation of multiple components
- **Result Caching**: Validation results cached at the coordinator level
- **Skip Options**: Ability to skip unnecessary validation levels

## Integration Examples

### With Simple Integration API

```python
# The unified tester is used internally by the simple integration API
from cursus.validation import validate_integration

# This internally uses UnifiedAlignmentTester
results = validate_integration(['tabular_preprocessing'])
```

### With Custom Configuration

```python
# Custom directory configuration
tester = UnifiedAlignmentTester(
    scripts_dir="custom/scripts",
    contracts_dir="custom/contracts",
    level3_validation_mode="strict"
)

# Run validation with custom settings
report = tester.run_full_validation()
```

### Batch Validation

```python
# Validate multiple scripts efficiently
scripts_to_validate = ['tabular_preprocessing', 'xgboost_training', 'model_eval']
report = tester.run_full_validation(target_scripts=scripts_to_validate)

# Check results for each script
for script in scripts_to_validate:
    script_results = tester.validate_specific_script(script)
    print(f"{script}: {script_results['overall_status']}")
```

## Best Practices

### Development Workflow

1. **Start with Specific Scripts**: Validate individual scripts during development
2. **Use Appropriate Mode**: Choose validation mode based on development phase
3. **Address Issues Sequentially**: Fix Level 1 issues before proceeding to Level 2
4. **Monitor Progress**: Use scoring to track improvement over time

### Error Resolution

1. **Read Critical Issues First**: Address critical issues that prevent operation
2. **Follow Recommendations**: Implement suggested fixes from the report
3. **Re-validate After Fixes**: Run validation again to confirm fixes
4. **Use Level-Specific Validation**: Focus on specific levels when debugging

### Performance Optimization

1. **Use Target Scripts**: Validate specific scripts when possible
2. **Skip Unnecessary Levels**: Skip levels that aren't relevant for your use case
3. **Monitor Validation Time**: Track validation performance over time
4. **Use Appropriate Mode**: Use permissive mode for rapid development

## Common Usage Patterns

### Development Phase Validation

```python
# Relaxed validation for development
tester = UnifiedAlignmentTester(level3_validation_mode="relaxed")
report = tester.run_full_validation(target_scripts=['new_script'])
```

### Production Readiness Check

```python
# Strict validation for production
tester = UnifiedAlignmentTester(level3_validation_mode="strict")
report = tester.run_full_validation()

if report.is_passing():
    print("✅ Ready for production")
else:
    critical_issues = tester.get_critical_issues()
    print(f"❌ {len(critical_issues)} critical issues must be fixed")
```

### Continuous Integration

```python
# CI/CD validation
tester = UnifiedAlignmentTester()
report = tester.run_full_validation()

# Export results for CI system
tester.export_report(format='json', output_path='ci_alignment_report.json')

# Exit with appropriate code
exit(0 if report.is_passing() else 1)
```

The Unified Alignment Tester provides a comprehensive, flexible, and powerful framework for ensuring component alignment across the entire Cursus architecture, with rich reporting and step type-aware enhancements.
