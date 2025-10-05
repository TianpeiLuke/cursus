---
tags:
  - test
  - validation
  - alignment
  - api_reference
  - documentation
keywords:
  - unified alignment tester API
  - validation framework API
  - alignment validation methods
  - script contract validation
  - specification validation API
  - configuration-driven validation API
topics:
  - unified alignment testing API
  - validation API reference
  - alignment validation methods
  - step-type-aware validation API
language: python
date of note: 2025-10-05
---

# Unified Alignment Tester API Reference

## Overview

The Unified Alignment Tester API provides configuration-driven validation across all four levels of pipeline architecture alignment. This reference documents the complete API based on the actual implementation, including the new step-type-aware validation system and enhanced reporting capabilities.

## Core API Classes

### UnifiedAlignmentTester

The main orchestrator for comprehensive alignment validation with configuration-driven step-type awareness.

```python
from cursus.validation.alignment import UnifiedAlignmentTester

# Initialize with default configuration (package-only steps)
tester = UnifiedAlignmentTester()

# Initialize with workspace directories for workspace-aware validation
tester = UnifiedAlignmentTester(
    workspace_dirs=["development/projects/project_alpha", "development/projects/project_beta"]
)
```

**Constructor Parameters:**
- `workspace_dirs`: Optional list of workspace directories to search. If None, only discovers package internal steps.
- `**kwargs`: Additional configuration options (preserved for backward compatibility)

## Core Operations

### run_full_validation()

Runs comprehensive configuration-driven alignment validation across all steps with automatic step-type detection.

**Signature:**
```python
def run_full_validation(
    self,
    target_scripts: Optional[List[str]] = None,
    skip_levels: Optional[Set[int]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `target_scripts`: Optional list of specific scripts to validate (None for all discovered steps)
- `skip_levels`: Optional set of validation levels to skip (legacy support, ignored in new configuration-driven system)

**Returns:** Dictionary containing validation results for all steps with step-type-aware configuration

**Example:**
```python
# Validate all steps with automatic step-type detection
results = tester.run_full_validation()

# Validate specific steps
results = tester.run_full_validation(
    target_scripts=["tabular_preprocessing", "xgboost_training", "xgboost_model"]
)

# Check results with step-type information
for step_name, result in results.items():
    print(f"Step: {step_name}")
    print(f"Type: {result.get('sagemaker_step_type', 'unknown')}")
    print(f"Category: {result.get('category', 'unknown')}")
    print(f"Status: {result.get('overall_status', 'UNKNOWN')}")
    print(f"Enabled Levels: {result.get('enabled_levels', [])}")
    
    # Check validation results by level
    validation_results = result.get("validation_results", {})
    for level_key, level_result in validation_results.items():
        status = level_result.get("status", "UNKNOWN")
        print(f"  {level_key}: {status}")
```

### run_validation_for_step()

Runs validation for a specific step based on its step-type-aware ruleset.

**Signature:**
```python
def run_validation_for_step(self, step_name: str) -> Dict[str, Any]
```

**Parameters:**
- `step_name`: Name of the step to validate

**Returns:** Dictionary containing validation results with step-type configuration details

**Example:**
```python
# Validate single step with step-type-aware configuration
result = tester.run_validation_for_step("tabular_preprocessing")

print(f"Step: {result['step_name']}")
print(f"SageMaker Type: {result['sagemaker_step_type']}")
print(f"Category: {result['category']}")
print(f"Overall Status: {result['overall_status']}")
print(f"Enabled Levels: {result['enabled_levels']}")

# Check each validation level result
validation_results = result.get("validation_results", {})
for level_key, level_result in validation_results.items():
    print(f"\n{level_key.upper()}:")
    print(f"  Status: {level_result.get('status', 'UNKNOWN')}")
    if "error" in level_result:
        print(f"  Error: {level_result['error']}")
```

### run_validation_for_all_steps()

Runs validation for all discovered steps using configuration-driven approach.

**Signature:**
```python
def run_validation_for_all_steps(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing validation results for all discovered steps

**Example:**
```python
# Run comprehensive validation for all steps
results = tester.run_validation_for_all_steps()

# Analyze by step type
step_type_summary = {}
for step_name, result in results.items():
    step_type = result.get("sagemaker_step_type", "unknown")
    status = result.get("overall_status", "UNKNOWN")
    
    if step_type not in step_type_summary:
        step_type_summary[step_type] = {"total": 0, "passed": 0, "failed": 0, "excluded": 0}
    
    step_type_summary[step_type]["total"] += 1
    if status == "PASSED":
        step_type_summary[step_type]["passed"] += 1
    elif status == "EXCLUDED":
        step_type_summary[step_type]["excluded"] += 1
    else:
        step_type_summary[step_type]["failed"] += 1

for step_type, summary in step_type_summary.items():
    print(f"{step_type}: {summary['passed']}/{summary['total']} passed")
```

### get_validation_summary()

Gets an enhanced summary of validation results with step-type-aware metrics.

**Signature:**
```python
def get_validation_summary(self) -> Dict[str, Any]
```

**Returns:** Dictionary with enhanced validation summary statistics including step-type breakdown

**Example:**
```python
# Run validation first
results = tester.run_full_validation()

# Get enhanced summary
summary = tester.get_validation_summary()

print(f"Total Steps: {summary['total_steps']}")
print(f"Passed: {summary['passed_steps']}")
print(f"Failed: {summary['failed_steps']}")
print(f"Excluded: {summary['excluded_steps']}")
print(f"Pass Rate: {summary['pass_rate']:.2%}")
print(f"Configuration-Driven: {summary['configuration_driven']}")

# Step type breakdown
step_type_breakdown = summary['step_type_breakdown']
print(f"\nStep Type Breakdown:")
for step_type, breakdown in step_type_breakdown.items():
    print(f"  {step_type}:")
    print(f"    Total: {breakdown['total']}")
    print(f"    Passed: {breakdown['passed']}")
    print(f"    Failed: {breakdown['failed']}")
    print(f"    Excluded: {breakdown['excluded']}")
```

### export_report()

Exports the enhanced alignment report with step-type insights in specified format.

**Signature:**
```python
def export_report(
    self,
    format: str = "json",
    output_path: Optional[str] = None
) -> str
```

**Parameters:**
- `format`: Export format ('json' or 'text')
- `output_path`: Optional path to save the report

**Returns:** Report content as string

**Example:**
```python
# Run validation first
results = tester.run_full_validation()

# Export JSON report with step-type breakdown
json_content = tester.export_report(
    format='json',
    output_path='enhanced_alignment_report.json'
)

# Export text report with step-type summary
text_content = tester.export_report(
    format='text',
    output_path='enhanced_alignment_report.txt'
)

print("📄 Enhanced reports generated with step-type insights")
```

### get_critical_issues()

Gets critical validation issues with step-type-aware analysis.

**Signature:**
```python
def get_critical_issues(self) -> List[Dict[str, Any]]
```

**Returns:** List of critical issues with step-type context

**Example:**
```python
# Run validation first
results = tester.run_full_validation()

# Get critical issues with step-type context
critical_issues = tester.get_critical_issues()

if critical_issues:
    print(f"🚨 {len(critical_issues)} Critical Issues Found:")
    for issue in critical_issues:
        print(f"  • Step: {issue['step_name']} ({issue['step_type']})")
        print(f"    Level: {issue['level']}")
        print(f"    Error: {issue['error']}")
        print(f"    Category: {issue['category']}")
else:
    print("✅ No critical issues found!")
```

### discover_scripts()

Discovers scripts that have corresponding script files using step catalog.

**Signature:**
```python
def discover_scripts(self) -> List[str]
```

**Returns:** List of discovered script names (only steps with actual script files)

**Example:**
```python
# Discover scripts with files
scripts = tester.discover_scripts()

print(f"📁 Found {len(scripts)} scripts with files:")
for script in scripts:
    print(f"  • {script}")

# Use discovered scripts for targeted validation
if scripts:
    sample_scripts = scripts[:3]  # First 3 scripts
    results = tester.run_full_validation(target_scripts=sample_scripts)
```

### print_summary()

Prints enhanced validation summary with step-type breakdown to console.

**Signature:**
```python
def print_summary(self) -> None
```

**Example:**
```python
# Run validation and print enhanced summary
results = tester.run_full_validation()
tester.print_summary()

# Output example:
# ============================================================
# ENHANCED VALIDATION SUMMARY
# ============================================================
# Total Steps: 21
# Passed: 15
# Failed: 3
# Excluded: 3
# Pass Rate: 83.33%
# Configuration-Driven: True
# 
# Step Type Breakdown:
#   Processing: 12/15 passed
#   Training: 2/2 passed
#   CreateModel: 1/2 passed
#   Base: 0/0 passed (2 excluded)
# ============================================================
```

## Step Catalog Integration

### get_step_info_from_catalog()

Gets step information from step catalog for component discovery.

**Signature:**
```python
def get_step_info_from_catalog(self, step_name: str) -> Optional[Any]
```

**Parameters:**
- `step_name`: Name of the step

**Returns:** StepInfo object or None if not found

**Example:**
```python
# Get step information
step_info = tester.get_step_info_from_catalog("tabular_preprocessing")

if step_info:
    print(f"Step Components for tabular_preprocessing:")
    for component_type, component in step_info.file_components.items():
        if component:
            print(f"  {component_type}: {component.path}")
        else:
            print(f"  {component_type}: Not found")
else:
    print("Step not found in catalog")
```

### get_component_path_from_catalog()

Gets component file path from step catalog.

**Signature:**
```python
def get_component_path_from_catalog(
    self,
    step_name: str,
    component_type: str
) -> Optional[Path]
```

**Parameters:**
- `step_name`: Name of the step
- `component_type`: Type of component ('script', 'contract', 'spec', 'builder', 'config')

**Returns:** Path to component file or None if not found

**Example:**
```python
# Get specific component paths
script_path = tester.get_component_path_from_catalog("tabular_preprocessing", "script")
contract_path = tester.get_component_path_from_catalog("tabular_preprocessing", "contract")
spec_path = tester.get_component_path_from_catalog("tabular_preprocessing", "spec")

print(f"Script: {script_path}")
print(f"Contract: {contract_path}")
print(f"Spec: {spec_path}")
```

## Configuration System Integration

### Step-Type-Aware Configuration

The system integrates with the configuration system to provide step-type-aware validation:

```python
from cursus.validation.alignment.config import (
    get_validation_ruleset,
    get_all_step_types,
    is_step_type_excluded,
    validate_step_type_configuration
)

# Get validation ruleset for a step type
ruleset = get_validation_ruleset("Processing")
if ruleset:
    print(f"Category: {ruleset.category.value}")
    print(f"Enabled Levels: {[level.value for level in ruleset.enabled_levels]}")
    print(f"Level 4 Validator: {ruleset.level_4_validator_class}")

# Check if step type is excluded
is_excluded = is_step_type_excluded("Base")
print(f"Base step type excluded: {is_excluded}")

# Validate configuration consistency
config_issues = validate_step_type_configuration()
if config_issues:
    print("Configuration issues:")
    for issue in config_issues:
        print(f"  • {issue}")
```

### Registry Integration

The system integrates with the registry for step type detection:

```python
from cursus.registry.step_names import get_sagemaker_step_type

# Get step type for validation rule lookup
step_type = get_sagemaker_step_type("tabular_preprocessing")
print(f"Step type: {step_type}")

# Use with validation ruleset
from cursus.validation.alignment.config import get_validation_ruleset
ruleset = get_validation_ruleset(step_type)
if ruleset:
    print(f"Validation levels: {[level.value for level in ruleset.enabled_levels]}")
```

## Advanced Operations

### validate_cross_workspace_compatibility()

Validates compatibility across workspace components with step-type analysis.

**Signature:**
```python
def validate_cross_workspace_compatibility(self, step_names: List[str]) -> Dict[str, Any]
```

**Parameters:**
- `step_names`: List of step names to validate for compatibility

**Returns:** Compatibility validation results with step-type distribution

**Example:**
```python
# Check compatibility across multiple steps
step_names = ["tabular_preprocessing", "xgboost_training", "xgboost_model"]
compatibility = tester.validate_cross_workspace_compatibility(step_names)

print(f"Compatible: {compatibility['compatible']}")
print(f"Issues: {len(compatibility['issues'])}")

# Check step type distribution
step_type_dist = compatibility['step_type_distribution']
print(f"Step Type Distribution:")
for step_type, steps in step_type_dist.items():
    print(f"  {step_type}: {steps}")

# Check recommendations
if compatibility['recommendations']:
    print("Recommendations:")
    for rec in compatibility['recommendations']:
        print(f"  • {rec}")
```

## Legacy API Compatibility

### validate_specific_script()

Legacy method maintained for backward compatibility - now uses configuration-driven validation.

**Signature:**
```python
def validate_specific_script(
    self,
    step_name: str,
    skip_levels: Optional[Set[int]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `step_name`: Name of the step to validate
- `skip_levels`: Optional set of validation levels to skip (deprecated - ignored in new system)

**Returns:** Dictionary containing validation results

**Example:**
```python
# Legacy API usage (automatically uses new configuration-driven system)
result = tester.validate_specific_script("tabular_preprocessing")

print(f"Script: {result['step_name']}")
print(f"Overall Status: {result['overall_status']}")

# Note: skip_levels parameter is deprecated and ignored
result_with_skip = tester.validate_specific_script(
    "tabular_preprocessing", 
    skip_levels={3, 4}  # This is ignored - configuration determines levels
)
```

## Data Models and Enums

### ValidationLevel

Enumeration of validation levels in the alignment system.

```python
from cursus.validation.alignment.utils import ValidationLevel

# Available validation levels
print("Validation Levels:")
for level in ValidationLevel:
    print(f"  {level.name}: {level.value}")

# Usage in configuration
from cursus.validation.alignment.config import get_validation_ruleset
ruleset = get_validation_ruleset("Processing")
if ruleset:
    enabled_levels = ruleset.enabled_levels
    for level in enabled_levels:
        print(f"Enabled: {level.name} (Level {level.value})")
```

### ValidationStatus

Enumeration of validation operation statuses.

```python
from cursus.validation.alignment.utils import ValidationStatus

# Available statuses
print("Validation Statuses:")
for status in ValidationStatus:
    print(f"  {status.name}: {status.value}")
```

### ValidationIssue

Data class representing a validation issue.

```python
from cursus.validation.alignment.utils import ValidationIssue, IssueLevel, RuleType

# Create validation issue
issue = ValidationIssue(
    level=IssueLevel.ERROR,
    message="Script function signature mismatch",
    method_name="main",
    rule_type=RuleType.METHOD_INTERFACE,
    details={"expected": "def main(input_paths, output_paths, environ_vars, job_args)"},
    step_name="tabular_preprocessing"
)

# Convert to dictionary
issue_dict = issue.to_dict()
print(f"Issue: {issue_dict}")
```

### ValidationResult

Data class representing validation operation results.

```python
from cursus.validation.alignment.utils import ValidationResult, ValidationStatus, ValidationLevel

# Create validation result
result = ValidationResult(
    status=ValidationStatus.PASSED,
    step_name="tabular_preprocessing",
    validation_level=ValidationLevel.SCRIPT_CONTRACT,
    issues=[],
    metadata={"step_type": "Processing"}
)

# Check result properties
print(f"Error count: {result.error_count}")
print(f"Warning count: {result.warning_count}")
print(f"Total issues: {result.total_issues}")

# Convert to dictionary
result_dict = result.to_dict()
print(f"Result: {result_dict}")
```

## Error Handling

### Common Exceptions

The system handles various error conditions gracefully:

```python
from cursus.validation.alignment import UnifiedAlignmentTester

def robust_validation_example():
    """Example of robust validation with comprehensive error handling."""
    
    try:
        # Initialize tester
        tester = UnifiedAlignmentTester()
        
        # Check if steps are available
        discovered_steps = tester.discover_scripts()
        if not discovered_steps:
            print("⚠️ No steps with script files found")
            return False
        
        # Run validation
        results = tester.run_full_validation()
        
        # Check for critical issues
        critical_issues = tester.get_critical_issues()
        
        if not critical_issues:
            print("✅ Validation completed successfully")
            summary = tester.get_validation_summary()
            print(f"Pass rate: {summary['pass_rate']:.1%}")
            return True
        else:
            print(f"❌ {len(critical_issues)} critical issues found")
            for issue in critical_issues[:3]:  # Show first 3
                print(f"  • {issue['step_name']}: {issue['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False

# Run robust validation
success = robust_validation_example()
```

### Configuration Validation

```python
from cursus.validation.alignment.config import validate_step_type_configuration

def validate_configuration():
    """Validate the step-type configuration for consistency."""
    
    config_issues = validate_step_type_configuration()
    
    if not config_issues:
        print("✅ Configuration is valid")
        return True
    else:
        print("⚠️ Configuration issues found:")
        for issue in config_issues:
            print(f"  • {issue}")
        return False

# Validate configuration
config_valid = validate_configuration()
```

## Performance Considerations

### Optimized Validation Workflows

The configuration-driven system provides significant performance improvements:

```python
def performance_optimized_validation():
    """Example of performance-optimized validation workflow."""
    
    tester = UnifiedAlignmentTester()
    
    # The system automatically skips inappropriate validation levels
    # based on step type configuration, providing dramatic performance improvements
    
    print("🚀 Running performance-optimized validation...")
    results = tester.run_full_validation()
    
    # Analyze performance benefits
    summary = tester.get_validation_summary()
    total_steps = summary['total_steps']
    excluded_steps = summary['excluded_steps']
    active_steps = total_steps - excluded_steps
    
    print(f"📊 Performance Summary:")
    print(f"  Total steps: {total_steps}")
    print(f"  Active steps: {active_steps}")
    print(f"  Excluded steps: {excluded_steps}")
    print(f"  Performance gain: {excluded_steps/total_steps*100:.1f}% steps skipped")
    
    # Show step type efficiency
    step_type_breakdown = summary['step_type_breakdown']
    for step_type, breakdown in step_type_breakdown.items():
        if breakdown['excluded'] > 0:
            print(f"  {step_type}: {breakdown['excluded']} steps excluded (performance optimization)")
    
    return results

# Run performance-optimized validation
performance_results = performance_optimized_validation()
```

### Batch Processing

For large codebases, the system supports efficient batch processing:

```python
def batch_validation_workflow():
    """Efficient batch validation for large codebases."""
    
    tester = UnifiedAlignmentTester()
    
    # Discover all steps
    all_steps = tester.step_catalog.list_available_steps()
    
    # Process in batches for memory efficiency
    batch_size = 10
    batches = [all_steps[i:i+batch_size] for i in range(0, len(all_steps), batch_size)]
    
    all_results = {}
    
    for i, batch in enumerate(batches):
        print(f"🔄 Processing batch {i+1}/{len(batches)} ({len(batch)} steps)...")
        
        # Run validation for batch
        batch_results = tester.run_full_validation(target_scripts=batch)
        all_results.update(batch_results)
        
        # Show batch progress
        batch_passed = sum(1 for r in batch_results.values() 
                          if r.get("overall_status") == "PASSED")
        print(f"  Batch {i+1}: {batch_passed}/{len(batch)} passed")
    
    return all_results

# Run batch validation
batch_results = batch_validation_workflow()
```

## Integration Examples

### CI/CD Integration

```python
def ci_cd_validation_pipeline():
    """Validation pipeline suitable for CI/CD integration."""
    
    import sys
    
    tester = UnifiedAlignmentTester()
    
    # Run comprehensive validation
    results = tester.run_full_validation()
    
    # Generate reports for CI/CD artifacts
    tester.export_report(format='json', output_path='ci_validation_report.json')
    tester.export_report(format='text', output_path='ci_validation_summary.txt')
    
    # Get summary for CI/CD decision making
    summary = tester.get_validation_summary()
    critical_issues = tester.get_critical_issues()
    
    # Print CI/CD friendly output
    print("=== CURSUS ALIGNMENT VALIDATION RESULTS ===")
    print(f"Total Steps: {summary['total_steps']}")
    print(f"Passed: {summary['passed_steps']}")
    print(f"Failed: {summary['failed_steps']}")
    print(f"Excluded: {summary['excluded_steps']}")
    print(f"Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Critical Issues: {len(critical_issues)}")
    print(f"Configuration-Driven: {summary['configuration_driven']}")
    
    # Determine CI/CD exit code
    active_steps = summary['total_steps'] - summary['excluded_steps']
    pass_rate = summary['passed_steps'] / active_steps if active_steps > 0 else 1.0
    
    if pass_rate >= 0.95 and len(critical_issues) == 0:
        print("✅ CI/CD VALIDATION PASSED")
        return 0
    else:
        print("❌ CI/CD VALIDATION FAILED")
        if critical_issues:
            print("\nCritical Issues:")
            for issue in critical_issues[:5]:  # Show first 5
                print(f"  • {issue['step_name']}: {issue['error']}")
        return 1

# Use in CI/CD
if __name__ == "__main__":
    exit_code = ci_cd_validation_pipeline()
    sys.exit(exit_code)
```

## API Reference Summary

### Core Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_full_validation()` | Configuration-driven validation across all steps | `Dict[str, Any]` |
| `run_validation_for_step()` | Step-type-aware validation for single step | `Dict[str, Any]` |
| `run_validation_for_all_steps()` | Validation for all discovered steps | `Dict[str, Any]` |
| `get_validation_summary()` | Enhanced summary with step-type breakdown | `Dict[str, Any]` |
| `export_report()` | Export report with step-type insights | `str` |
| `get_critical_issues()` | Critical issues with step-type context | `List[Dict[str, Any]]` |
| `discover_scripts()` | Find steps with script files | `List[str]` |
| `print_summary()` | Print enhanced summary to console | `None` |

### Step Catalog Integration

| Method | Purpose | Returns |
|--------|---------|---------|
| `get_step_info_from_catalog()` | Get step information from catalog | `Optional[Any]` |
| `get_component_path_from_catalog()` | Get component file path | `Optional[Path]` |

### Advanced Operations

| Method | Purpose | Returns |
|--------|---------|---------|
| `validate_cross_workspace_compatibility()` | Cross-workspace compatibility check | `Dict[str, Any]` |

### Legacy Compatibility

| Method | Purpose | Returns |
|--------|---------|---------|
| `validate_specific_script()` | Legacy single script validation | `Dict[str, Any]` |

## Configuration System API

| Function | Purpose | Returns |
|----------|---------|---------|
| `get_validation_ruleset()` | Get ruleset for step type | `Optional[ValidationRuleset]` |
| `is_step_type_excluded()` | Check if step type is excluded | `bool` |
| `get_all_step_types()` | Get all configured step types | `List[str]` |
| `validate_step_type_configuration()` | Validate configuration consistency | `List[str]` |

For additional examples and advanced usage patterns, see the [Unified Alignment Tester Quick Start Guide](unified_alignment_tester_quick_start.md).
