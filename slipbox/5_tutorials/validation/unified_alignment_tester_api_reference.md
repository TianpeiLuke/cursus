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
  - workspace validation API
topics:
  - unified alignment testing API
  - validation API reference
  - alignment validation methods
  - workspace-aware validation API
language: python
date of note: 2025-09-06
---

# Unified Alignment Tester API Reference

## Overview

The Unified Alignment Tester API provides comprehensive validation across all four levels of pipeline architecture alignment. This reference documents the complete API with practical examples and usage patterns.

## Core API Classes

### UnifiedAlignmentTester

The main orchestrator for comprehensive alignment validation.

```python
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

# Initialize with default paths
tester = UnifiedAlignmentTester()

# Initialize with custom configuration
tester = UnifiedAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts",
    specs_dir="src/cursus/steps/specs", 
    builders_dir="src/cursus/steps/builders",
    configs_dir="src/cursus/steps/configs",
    level3_validation_mode="relaxed"
)
```

## Core Operations

### run_full_validation()

Runs comprehensive alignment validation across all levels.

**Signature:**
```python
def run_full_validation(
    self,
    target_scripts: Optional[List[str]] = None,
    skip_levels: Optional[List[int]] = None
) -> AlignmentReport
```

**Parameters:**
- `target_scripts`: Specific scripts to validate (None for all)
- `skip_levels`: Alignment levels to skip (1-4)

**Returns:** `AlignmentReport` with comprehensive validation results

**Example:**
```python
# Validate all scripts across all levels
report = tester.run_full_validation()

# Validate specific scripts
report = tester.run_full_validation(
    target_scripts=["tabular_preprocessing", "xgboost_training"]
)

# Skip certain levels
report = tester.run_full_validation(skip_levels=[3, 4])

# Check results
if report.is_passing():
    print("✅ All alignment tests passed!")
    print(f"Pass rate: {report.summary.pass_rate:.1f}%")
else:
    print("❌ Some alignment issues detected")
    for issue in report.get_critical_issues():
        print(f"Critical: {issue.message}")
```

### run_level_validation()

Runs validation for a specific alignment level.

**Signature:**
```python
def run_level_validation(
    self,
    level: int,
    target_scripts: Optional[List[str]] = None
) -> AlignmentReport
```

**Parameters:**
- `level`: Alignment level to validate (1-4)
- `target_scripts`: Specific scripts to validate

**Returns:** `AlignmentReport` for the specified level

**Example:**
```python
# Run Level 1 validation (Script ↔ Contract)
level1_report = tester.run_level_validation(level=1)

# Run Level 3 validation for specific scripts
level3_report = tester.run_level_validation(
    level=3,
    target_scripts=["model_evaluation", "data_preprocessing"]
)

# Check level-specific results
for script_name, result in level1_report.level1_results.items():
    print(f"{script_name}: {'PASS' if result.passed else 'FAIL'}")
    if result.issues:
        for issue in result.issues:
            print(f"  - {issue.level.value}: {issue.message}")
```

### validate_specific_script()

Runs comprehensive validation for a specific script across all levels.

**Signature:**
```python
def validate_specific_script(self, script_name: str) -> Dict[str, Any]
```

**Parameters:**
- `script_name`: Name of the script to validate

**Returns:** Dictionary containing validation results for all levels

**Example:**
```python
# Validate single script across all levels
results = tester.validate_specific_script("tabular_preprocessing")

print(f"Script: {results['script_name']}")
print(f"Overall Status: {results['overall_status']}")

# Check each level
for level in ['level1', 'level2', 'level3', 'level4']:
    if level in results:
        level_result = results[level]
        status = "✅" if level_result.get('passed', False) else "❌"
        print(f"{status} {level}: {'PASS' if level_result.get('passed', False) else 'FAIL'}")
        
        # Show issues if any
        if 'issues' in level_result:
            for issue in level_result['issues']:
                print(f"  - {issue.get('severity', 'ERROR')}: {issue.get('message', '')}")
```

### get_validation_summary()

Gets a high-level summary of validation results.

**Signature:**
```python
def get_validation_summary(self) -> Dict[str, Any]
```

**Returns:** Dictionary with validation summary statistics

**Example:**
```python
# Run validation first
report = tester.run_full_validation()

# Get summary
summary = tester.get_validation_summary()

print(f"Overall Status: {summary['overall_status']}")
print(f"Total Tests: {summary['total_tests']}")
print(f"Pass Rate: {summary['pass_rate']:.1f}%")
print(f"Critical Issues: {summary['critical_issues']}")
print(f"Error Issues: {summary['error_issues']}")
print(f"Warning Issues: {summary['warning_issues']}")

# Level breakdown
level_breakdown = summary['level_breakdown']
for level, count in level_breakdown.items():
    print(f"{level}: {count} tests")
```

### export_report()

Exports the alignment report in specified format with optional visualization.

**Signature:**
```python
def export_report(
    self,
    format: str = 'json',
    output_path: Optional[str] = None,
    generate_chart: bool = True,
    script_name: str = "alignment_validation"
) -> str
```

**Parameters:**
- `format`: Export format ('json' or 'html')
- `output_path`: Optional path to save the report
- `generate_chart`: Whether to generate alignment score chart
- `script_name`: Name for the chart file

**Returns:** Report content as string

**Example:**
```python
# Run validation first
report = tester.run_full_validation()

# Export JSON report
json_content = tester.export_report(
    format='json',
    output_path='alignment_report.json'
)

# Export HTML report with chart
html_content = tester.export_report(
    format='html',
    output_path='alignment_report.html',
    generate_chart=True,
    script_name="my_pipeline_validation"
)

print("📄 Reports generated successfully")
```

### get_critical_issues()

Gets all critical issues that require immediate attention.

**Signature:**
```python
def get_critical_issues(self) -> List[Dict[str, Any]]
```

**Returns:** List of critical issues with details

**Example:**
```python
# Run validation first
report = tester.run_full_validation()

# Get critical issues
critical_issues = tester.get_critical_issues()

if critical_issues:
    print(f"🚨 {len(critical_issues)} Critical Issues Found:")
    for issue in critical_issues:
        print(f"  • {issue['category']}: {issue['message']}")
        if issue['recommendation']:
            print(f"    💡 Recommendation: {issue['recommendation']}")
        print(f"    📍 Level: {issue['alignment_level']}")
else:
    print("✅ No critical issues found!")
```

### discover_scripts()

Discovers all Python scripts in the scripts directory.

**Signature:**
```python
def discover_scripts(self) -> List[str]
```

**Returns:** List of discovered script names

**Example:**
```python
# Discover available scripts
scripts = tester.discover_scripts()

print(f"📁 Found {len(scripts)} scripts:")
for script in scripts:
    print(f"  • {script}")

# Use discovered scripts for targeted validation
if scripts:
    sample_scripts = scripts[:3]  # First 3 scripts
    report = tester.run_full_validation(target_scripts=sample_scripts)
```

### get_alignment_status_matrix()

Gets a matrix showing alignment status for each script across all levels.

**Signature:**
```python
def get_alignment_status_matrix(self) -> Dict[str, Dict[str, str]]
```

**Returns:** Matrix with script names as keys and level statuses as values

**Example:**
```python
# Run validation first
report = tester.run_full_validation()

# Get status matrix
matrix = tester.get_alignment_status_matrix()

print(f"📋 Alignment Status Matrix:")
print(f"{'Script':<25} {'L1':<10} {'L2':<10} {'L3':<10} {'L4':<10}")
print("-" * 65)

for script_name, statuses in matrix.items():
    l1 = statuses.get('level1', 'UNKNOWN')
    l2 = statuses.get('level2', 'UNKNOWN')
    l3 = statuses.get('level3', 'UNKNOWN')
    l4 = statuses.get('level4', 'UNKNOWN')
    
    print(f"{script_name:<25} {l1:<10} {l2:<10} {l3:<10} {l4:<10}")

# Find scripts with issues
problematic_scripts = []
for script_name, statuses in matrix.items():
    if any(status == 'FAILING' for status in statuses.values()):
        problematic_scripts.append(script_name)

if problematic_scripts:
    print(f"\n⚠️ Scripts with issues: {problematic_scripts}")
```

### print_summary()

Prints a formatted summary of validation results.

**Signature:**
```python
def print_summary(self) -> None
```

**Example:**
```python
# Run validation and print summary
report = tester.run_full_validation()
tester.print_summary()

# Output example:
# ================================================================================
# UNIFIED ALIGNMENT VALIDATION SUMMARY
# ================================================================================
# 
# Overall Status: PASSING
# Total Tests: 45
# Pass Rate: 91.1%
# 
# Level Breakdown:
#   Level 1 (Script↔Contract): 12/13 tests passed (92.3%)
#   Level 2 (Contract↔Spec): 11/12 tests passed (91.7%)
#   Level 3 (Spec↔Dependencies): 8/10 tests passed (80.0%)
#   Level 4 (Builder↔Config): 10/10 tests passed (100.0%)
```

## Workspace-Aware API

### WorkspaceUnifiedAlignmentTester

Workspace-aware version of UnifiedAlignmentTester for multi-developer environments.

```python
from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

# Initialize workspace-aware tester
workspace_tester = WorkspaceUnifiedAlignmentTester(
    workspace_root="development/projects",
    developer_id="your_developer_id",
    enable_shared_fallback=True
)
```

### run_workspace_validation()

Runs alignment validation for workspace components.

**Signature:**
```python
def run_workspace_validation(
    self,
    target_scripts: Optional[List[str]] = None,
    skip_levels: Optional[List[int]] = None,
    workspace_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]
```

**Parameters:**
- `target_scripts`: Specific scripts to validate
- `skip_levels`: Validation levels to skip
- `workspace_context`: Additional workspace context

**Returns:** Comprehensive validation results with workspace context

**Example:**
```python
# Run workspace validation
results = workspace_tester.run_workspace_validation()

if results['success']:
    print("✅ Workspace validation completed successfully")
    
    # Check workspace statistics
    stats = results['workspace_statistics']
    print(f"Scripts validated: {stats['total_scripts_validated']}")
    print(f"Success rate: {stats['successful_validations']}/{stats['total_scripts_validated']}")
    
    # Check workspace components
    components = stats['workspace_components_found']
    for comp_type, count in components.items():
        print(f"{comp_type}: {count} files")
    
    # Check cross-workspace dependencies
    if 'cross_workspace_validation' in results:
        cross_validation = results['cross_workspace_validation']
        if cross_validation['enabled']:
            print("🔗 Cross-workspace validation enabled")
            shared_usage = cross_validation['shared_components_used']
            for script, usage in shared_usage.items():
                shared_count = sum(usage.values())
                if shared_count > 0:
                    print(f"  {script}: {shared_count} shared components")
else:
    print(f"❌ Workspace validation failed: {results.get('error')}")
```

### get_workspace_info()

Gets information about current workspace configuration.

**Signature:**
```python
def get_workspace_info(self) -> Dict[str, Any]
```

**Returns:** Dictionary with workspace configuration details

**Example:**
```python
# Get workspace information
workspace_info = workspace_tester.get_workspace_info()

print("🏢 Workspace Configuration:")
print(f"  Developer ID: {workspace_info['developer_id']}")
print(f"  Workspace Root: {workspace_info['workspace_root']}")
print(f"  Shared Fallback: {workspace_info['enable_shared_fallback']}")
print(f"  Available Developers: {workspace_info['available_developers']}")

# Check workspace manager info
manager_info = workspace_info['workspace_manager_info']
print(f"  Total Workspaces: {len(manager_info.get('developers', {}))}")
```

### switch_developer()

Switches to a different developer workspace.

**Signature:**
```python
def switch_developer(self, developer_id: str) -> None
```

**Parameters:**
- `developer_id`: Target developer workspace ID

**Example:**
```python
# Switch to different developer workspace
try:
    workspace_tester.switch_developer("alice_developer")
    print("✅ Switched to Alice's workspace")
    
    # Run validation in new workspace
    results = workspace_tester.run_workspace_validation()
    print(f"Alice's workspace validation: {'✅' if results['success'] else '❌'}")
    
except ValueError as e:
    print(f"❌ Failed to switch workspace: {e}")
    
    # List available developers
    workspace_info = workspace_tester.get_workspace_info()
    available = workspace_info['available_developers']
    print(f"Available developers: {available}")
```

## Data Models

### AlignmentReport

Main report object containing validation results.

```python
class AlignmentReport:
    """Comprehensive alignment validation report."""
    
    # Level-specific results
    level1_results: Dict[str, ValidationResult]  # Script ↔ Contract
    level2_results: Dict[str, ValidationResult]  # Contract ↔ Spec
    level3_results: Dict[str, ValidationResult]  # Spec ↔ Dependencies
    level4_results: Dict[str, ValidationResult]  # Builder ↔ Config
    
    # Summary information
    summary: Optional[ValidationSummary]
    
    def is_passing(self) -> bool:
        """Check if all validations passed."""
        
    def get_critical_issues(self) -> List[AlignmentIssue]:
        """Get all critical issues."""
        
    def get_recommendations(self) -> List[str]:
        """Get actionable recommendations."""
        
    def export_to_json(self) -> str:
        """Export report as JSON."""
        
    def export_to_html(self) -> str:
        """Export report as HTML."""
```

### ValidationResult

Result of a single validation test.

```python
class ValidationResult:
    """Result of a validation test."""
    
    test_name: str
    passed: bool
    details: Dict[str, Any]
    issues: List[AlignmentIssue] = []
    
    def add_issue(self, issue: AlignmentIssue) -> None:
        """Add an alignment issue to this result."""
```

### AlignmentIssue

Represents an alignment issue found during validation.

```python
class AlignmentIssue:
    """Alignment issue with severity and context."""
    
    level: SeverityLevel  # CRITICAL, ERROR, WARNING, INFO
    category: str
    message: str
    details: Dict[str, Any]
    recommendation: Optional[str]
    alignment_level: Optional[AlignmentLevel]
```

## Error Handling

### Common Exceptions

**ValidationError**
```python
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

try:
    tester = UnifiedAlignmentTester(scripts_dir="nonexistent/path")
    report = tester.run_full_validation()
except ValidationError as e:
    print(f"Validation failed: {e}")
```

**WorkspaceNotFoundError**
```python
from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

try:
    workspace_tester = WorkspaceUnifiedAlignmentTester(
        workspace_root="nonexistent/workspace",
        developer_id="unknown_developer"
    )
except WorkspaceNotFoundError as e:
    print(f"Workspace not found: {e}")
```

### Error Handling Best Practices

```python
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.alignment_utils import ValidationError

def robust_validation_workflow():
    """Example of robust validation with error handling."""
    
    try:
        # Initialize tester
        tester = UnifiedAlignmentTester()
        
        # Discover scripts first
        scripts = tester.discover_scripts()
        if not scripts:
            print("⚠️ No scripts found for validation")
            return False
        
        print(f"🔍 Validating {len(scripts)} scripts...")
        
        # Run validation with error handling
        report = tester.run_full_validation()
        
        # Check results
        if report.is_passing():
            print("✅ All validations passed!")
            
            # Generate report
            tester.export_report(
                format='html',
                output_path='validation_report.html'
            )
            
            return True
        else:
            print("❌ Some validations failed")
            
            # Get critical issues
            critical_issues = tester.get_critical_issues()
            if critical_issues:
                print(f"🚨 {len(critical_issues)} critical issues found:")
                for issue in critical_issues[:5]:  # Show first 5
                    print(f"  • {issue['message']}")
            
            return False
            
    except ValidationError as e:
        print(f"❌ Validation error: {e}")
        return False
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("💡 Check if all required directories exist")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

# Run robust validation
success = robust_validation_workflow()
```

## Advanced Usage

### Custom Validation Configuration

```python
# Configure Level 3 validation modes
from cursus.validation.alignment.level3_validation_config import Level3ValidationConfig

# Create custom Level 3 configuration
custom_config = Level3ValidationConfig(
    validation_mode="custom",
    allow_missing_dependencies=True,
    require_exact_versions=False,
    validate_circular_dependencies=True,
    max_dependency_depth=5
)

# Initialize tester with custom config
tester = UnifiedAlignmentTester()
tester.level3_tester.config = custom_config

# Run validation with custom configuration
report = tester.run_level_validation(level=3)
```

### Batch Validation Operations

```python
def batch_validation_analysis():
    """Analyze validation results across multiple configurations."""
    
    configurations = [
        {"mode": "strict", "description": "Production-ready validation"},
        {"mode": "relaxed", "description": "Development validation"},
        {"mode": "permissive", "description": "Experimental validation"}
    ]
    
    results = {}
    
    for config in configurations:
        print(f"\n🧪 Testing {config['description']}...")
        
        tester = UnifiedAlignmentTester(
            level3_validation_mode=config['mode']
        )
        
        report = tester.run_full_validation()
        summary = tester.get_validation_summary()
        
        results[config['mode']] = {
            'pass_rate': summary['pass_rate'],
            'critical_issues': summary['critical_issues'],
            'total_tests': summary['total_tests'],
            'description': config['description']
        }
        
        print(f"  Pass rate: {summary['pass_rate']:.1f}%")
        print(f"  Critical issues: {summary['critical_issues']}")
    
    # Compare results
    print(f"\n📊 Validation Mode Comparison:")
    print(f"{'Mode':<12} {'Pass Rate':<12} {'Critical':<10} {'Total Tests':<12}")
    print("-" * 50)
    
    for mode, data in results.items():
        print(f"{mode:<12} {data['pass_rate']:<11.1f}% {data['critical_issues']:<10} {data['total_tests']:<12}")
    
    return results

# Run batch analysis
batch_results = batch_validation_analysis()
```

### Integration with CI/CD

```python
def ci_cd_validation_check():
    """Validation check suitable for CI/CD pipelines."""
    
    import sys
    import os
    
    # Set strict validation for CI/CD
    tester = UnifiedAlignmentTester(level3_validation_mode="strict")
    
    # Run comprehensive validation
    report = tester.run_full_validation()
    
    # Generate reports
    tester.export_report(format='json', output_path='ci_validation_report.json')
    tester.export_report(format='html', output_path='ci_validation_report.html')
    
    # Get summary
    summary = tester.get_validation_summary()
    
    # Print results for CI/CD logs
    print(f"=== CURSUS ALIGNMENT VALIDATION RESULTS ===")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Pass Rate: {summary['pass_rate']:.1f}%")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    
    # Set exit code based on results
    if summary['overall_status'] == 'PASSING' and summary['critical_issues'] == 0:
        print("✅ CI/CD validation PASSED")
        return 0
    else:
        print("❌ CI/CD validation FAILED")
        
        # Print critical issues for debugging
        critical_issues = tester.get_critical_issues()
        if critical_issues:
            print("\n🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"  • {issue['category']}: {issue['message']}")
        
        return 1

# Use in CI/CD
if __name__ == "__main__":
    exit_code = ci_cd_validation_check()
    sys.exit(exit_code)
```

## Performance Considerations

### Optimizing Validation Performance

```python
def optimized_validation_workflow():
    """Optimized validation for large codebases."""
    
    # Initialize tester
    tester = UnifiedAlignmentTester()
    
    # Discover scripts first
    all_scripts = tester.discover_scripts()
    print(f"📁 Found {len(all_scripts)} scripts")
    
    # For large codebases, validate in batches
    batch_size = 10
    batches = [all_scripts[i:i+batch_size] for i in range(0, len(all_scripts), batch_size)]
    
    all_results = {}
    
    for i, batch in enumerate(batches):
        print(f"\n🔄 Processing batch {i+1}/{len(batches)} ({len(batch)} scripts)...")
        
        # Run validation for batch
        report = tester.run_full_validation(target_scripts=batch)
        
        # Collect results
        batch_summary = tester.get_validation_summary()
        all_results[f'batch_{i+1}'] = batch_summary
        
        print(f"  Batch {i+1} pass rate: {batch_summary['pass_rate']:.1f}%")
    
    # Generate final summary
    total_tests = sum(result['total_tests'] for result in all_results.values())
    total_passing = sum(
        int(result['total_tests'] * result['pass_rate'] / 100)
        for result in all_results.values()
    )
    overall_pass_rate = (total_passing / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n📊 Overall Results:")
    print(f"Total tests: {total_tests}")
    print(f"Overall pass rate: {overall_pass_rate:.1f}%")
    
    return all_results

# Run optimized validation
optimized_results = optimized_validation_workflow()
```

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_full_validation()` | Comprehensive validation across all levels | `AlignmentReport` |
| `run_level_validation()` | Validation for specific level | `AlignmentReport` |
| `validate_specific_script()` | Single script validation | `Dict[str, Any]` |
| `get_validation_summary()` | High-level validation summary | `Dict[str, Any]` |
| `export_report()` | Export report in JSON/HTML format | `str` |
| `get_critical_issues()` | Get critical issues requiring attention | `List[Dict[str, Any]]` |
| `discover_scripts()` | Find available scripts | `List[str]` |
| `get_alignment_status_matrix()` | Status matrix across all levels | `Dict[str, Dict[str, str]]` |
| `print_summary()` | Print formatted summary | `None` |

### Workspace-Aware Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `run_workspace_validation()` | Workspace-aware validation | `Dict[str, Any]` |
| `get_workspace_info()` | Workspace configuration info | `Dict[str, Any]` |
| `switch_developer()` | Switch developer workspace | `None` |

For additional examples and advanced usage patterns, see the [Unified Tester Quick Start Guide](unified_tester_quick_start.md).
