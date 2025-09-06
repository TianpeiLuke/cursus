---
tags:
  - test
  - validation
  - alignment
  - quick_start
  - tutorial
keywords:
  - unified alignment tester
  - validation framework
  - alignment validation
  - script contract alignment
  - specification validation
  - workspace validation
topics:
  - unified alignment testing
  - validation tutorial
  - alignment validation workflow
  - workspace-aware validation
language: python
date of note: 2025-09-06
---

# Unified Alignment Tester Quick Start Guide

## Overview

This 20-minute tutorial will get you up and running with the Cursus Unified Alignment Tester. You'll learn how to validate alignment across all four levels of the pipeline architecture, understand workspace-aware validation, and generate comprehensive reports.

## Prerequisites

- Cursus package installed
- Python 3.8+ environment
- Basic familiarity with ML pipeline development
- Understanding of script contracts and step specifications

## What is Unified Alignment Testing?

The Unified Alignment Tester validates consistency across four critical levels:

1. **Level 1**: Script ↔ Contract Alignment
2. **Level 2**: Contract ↔ Specification Alignment  
3. **Level 3**: Specification ↔ Dependencies Alignment
4. **Level 4**: Builder ↔ Configuration Alignment

## Step 1: Initialize the Unified Alignment Tester (2 minutes)

First, let's set up the tester and verify it's working:

```python
from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester

# Initialize with default paths
tester = UnifiedAlignmentTester()

# Or initialize with custom paths
tester = UnifiedAlignmentTester(
    scripts_dir="src/cursus/steps/scripts",
    contracts_dir="src/cursus/steps/contracts", 
    specs_dir="src/cursus/steps/specs",
    builders_dir="src/cursus/steps/builders",
    configs_dir="src/cursus/steps/configs",
    level3_validation_mode="relaxed"  # Options: strict, relaxed, permissive
)

print("✅ Unified Alignment Tester initialized successfully")
print(f"Scripts directory: {tester.scripts_dir}")
print(f"Level 3 validation mode: {tester.level3_config.validation_mode}")
```

**Expected Output:**
```
✅ Unified Alignment Tester initialized successfully
Scripts directory: /path/to/src/cursus/steps/scripts
Level 3 validation mode: ValidationMode.RELAXED
```

## Step 2: Run Your First Alignment Validation (3 minutes)

Let's run a comprehensive validation across all levels:

```python
# Run full validation across all levels
print("🔍 Starting comprehensive alignment validation...")
report = tester.run_full_validation()

# Check overall results
if report.is_passing():
    print("✅ All alignment tests passed!")
else:
    print("❌ Some alignment issues detected")

# Print summary
print(f"\n📊 Validation Summary:")
print(f"Total tests: {report.summary.total_tests}")
print(f"Pass rate: {report.summary.pass_rate:.1f}%")
print(f"Critical issues: {report.summary.critical_issues}")
print(f"Error issues: {report.summary.error_issues}")
print(f"Warning issues: {report.summary.warning_issues}")
```

**What this validates:**
- Script-contract consistency across all pipeline steps
- Contract-specification alignment for proper integration
- Specification dependency resolution and compatibility
- Builder-configuration alignment for runtime execution

## Step 3: Validate Specific Scripts (2 minutes)

You can target specific scripts for focused validation:

```python
# Validate specific scripts only
target_scripts = ["tabular_preprocessing", "xgboost_training", "model_evaluation"]

print(f"🎯 Validating specific scripts: {target_scripts}")
report = tester.run_full_validation(target_scripts=target_scripts)

# Check results for each script
for script_name in target_scripts:
    # Check Level 1 results
    if script_name in report.level1_results:
        level1_result = report.level1_results[script_name]
        status = "✅" if level1_result.passed else "❌"
        print(f"{status} {script_name} - Level 1 (Script↔Contract): {'PASS' if level1_result.passed else 'FAIL'}")
    
    # Check Level 2 results
    if script_name in report.level2_results:
        level2_result = report.level2_results[script_name]
        status = "✅" if level2_result.passed else "❌"
        print(f"{status} {script_name} - Level 2 (Contract↔Spec): {'PASS' if level2_result.passed else 'FAIL'}")
```

## Step 4: Run Level-Specific Validation (2 minutes)

Sometimes you want to focus on a specific alignment level:

```python
# Run only Level 1 validation (Script ↔ Contract)
print("\n📝 Running Level 1 validation only...")
level1_report = tester.run_level_validation(level=1)

print(f"Level 1 Results:")
for script_name, result in level1_report.level1_results.items():
    status = "✅" if result.passed else "❌"
    print(f"  {status} {script_name}: {len(result.issues)} issues")
    
    # Show first few issues if any
    if result.issues:
        for issue in result.issues[:2]:  # Show first 2 issues
            print(f"    - {issue.level.value}: {issue.message}")

# Run Level 3 validation with different modes
print("\n🔗 Testing different Level 3 validation modes...")

# Strict mode - catches more potential issues
strict_tester = UnifiedAlignmentTester(level3_validation_mode="strict")
strict_report = strict_tester.run_level_validation(level=3)
print(f"Strict mode: {len(strict_report.level3_results)} components validated")

# Permissive mode - allows more flexibility
permissive_tester = UnifiedAlignmentTester(level3_validation_mode="permissive")
permissive_report = permissive_tester.run_level_validation(level=3)
print(f"Permissive mode: {len(permissive_report.level3_results)} components validated")
```

## Step 5: Understanding Validation Results (3 minutes)

Let's explore the detailed validation results:

```python
# Get detailed validation summary
summary = tester.get_validation_summary()

print(f"\n📈 Detailed Validation Summary:")
print(f"Overall Status: {summary['overall_status']}")
print(f"Total Tests: {summary['total_tests']}")
print(f"Pass Rate: {summary['pass_rate']:.1f}%")

# Level breakdown
level_breakdown = summary['level_breakdown']
print(f"\nLevel Breakdown:")
print(f"  Level 1 tests: {level_breakdown['level1_tests']}")
print(f"  Level 2 tests: {level_breakdown['level2_tests']}")
print(f"  Level 3 tests: {level_breakdown['level3_tests']}")
print(f"  Level 4 tests: {level_breakdown['level4_tests']}")

# Get critical issues that need immediate attention
critical_issues = tester.get_critical_issues()
if critical_issues:
    print(f"\n🚨 Critical Issues Requiring Immediate Attention:")
    for issue in critical_issues[:3]:  # Show first 3 critical issues
        print(f"  • {issue['category']}: {issue['message']}")
        if issue['recommendation']:
            print(f"    💡 Recommendation: {issue['recommendation']}")
else:
    print("\n✅ No critical issues found!")
```

## Step 6: Generate Alignment Reports (3 minutes)

The tester can generate comprehensive reports in multiple formats:

```python
# Generate JSON report
json_report = tester.export_report(format='json', output_path='alignment_report.json')
print("📄 JSON report generated: alignment_report.json")

# Generate HTML report with visualization
html_report = tester.export_report(
    format='html', 
    output_path='alignment_report.html',
    generate_chart=True,
    script_name="my_pipeline_validation"
)
print("📄 HTML report generated: alignment_report.html")
print("📊 Alignment score chart generated")

# Print the report summary
tester.print_summary()
```

**Sample Report Output:**
```
📊 Alignment Quality Scoring:
   Overall Score: 85.2/100 (Good)
   L1 Script↔Contract: 92.1/100 (15/16 tests)
   L2 Contract↔Spec: 88.5/100 (12/14 tests)
   L3 Spec↔Dependencies: 78.3/100 (9/12 tests)
   L4 Builder↔Config: 81.7/100 (11/13 tests)
```

## Step 7: Workspace-Aware Validation (3 minutes)

For multi-developer environments, use the workspace-aware tester:

```python
from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

# Initialize workspace-aware tester
workspace_tester = WorkspaceUnifiedAlignmentTester(
    workspace_root="development/projects",
    developer_id="your_developer_id",
    enable_shared_fallback=True
)

print(f"🏢 Workspace tester initialized for developer: {workspace_tester.developer_id}")

# Run workspace validation
workspace_results = workspace_tester.run_workspace_validation()

if workspace_results['success']:
    print("✅ Workspace validation completed successfully")
    
    # Check workspace-specific statistics
    stats = workspace_results['workspace_statistics']
    print(f"\n📊 Workspace Statistics:")
    print(f"  Scripts validated: {stats['total_scripts_validated']}")
    print(f"  Successful validations: {stats['successful_validations']}")
    print(f"  Failed validations: {stats['failed_validations']}")
    
    # Check workspace components
    components = stats['workspace_components_found']
    print(f"\n🔧 Workspace Components:")
    for comp_type, count in components.items():
        print(f"  {comp_type}: {count} files")
    
    # Check cross-workspace validation if enabled
    if 'cross_workspace_validation' in workspace_results:
        cross_validation = workspace_results['cross_workspace_validation']
        if cross_validation['enabled']:
            shared_usage = cross_validation['shared_components_used']
            print(f"\n🔗 Cross-Workspace Dependencies:")
            for script, usage in shared_usage.items():
                shared_count = sum(usage.values())
                print(f"  {script}: {shared_count} shared components")
else:
    print(f"❌ Workspace validation failed: {workspace_results.get('error', 'Unknown error')}")

# Switch to different developer workspace
try:
    workspace_tester.switch_developer("another_developer_id")
    print("✅ Switched to different developer workspace")
except ValueError as e:
    print(f"⚠️ Could not switch developer: {e}")
```

## Step 8: Advanced Validation Scenarios (2 minutes)

Here are some advanced usage patterns:

```python
# Skip certain validation levels
report = tester.run_full_validation(skip_levels=[3, 4])  # Skip Level 3 and 4
print("Validation completed with Level 3 and 4 skipped")

# Validate a single script across all levels
script_results = tester.validate_specific_script("tabular_preprocessing")
print(f"\n🔍 Single Script Validation Results:")
print(f"Script: {script_results['script_name']}")
print(f"Overall Status: {script_results['overall_status']}")

for level in ['level1', 'level2', 'level3', 'level4']:
    if level in script_results:
        level_result = script_results[level]
        status = "✅" if level_result.get('passed', False) else "❌"
        print(f"  {status} {level.title()}: {'PASS' if level_result.get('passed', False) else 'FAIL'}")

# Get alignment status matrix for all scripts
status_matrix = tester.get_alignment_status_matrix()
print(f"\n📋 Alignment Status Matrix:")
print(f"{'Script':<25} {'L1':<8} {'L2':<8} {'L3':<8} {'L4':<8}")
print("-" * 57)

for script_name, statuses in status_matrix.items():
    l1_status = "✅" if statuses['level1'] == 'PASSING' else "❌" if statuses['level1'] == 'FAILING' else "?"
    l2_status = "✅" if statuses['level2'] == 'PASSING' else "❌" if statuses['level2'] == 'FAILING' else "?"
    l3_status = "✅" if statuses['level3'] == 'PASSING' else "❌" if statuses['level3'] == 'FAILING' else "?"
    l4_status = "✅" if statuses['level4'] == 'PASSING' else "❌" if statuses['level4'] == 'FAILING' else "?"
    
    print(f"{script_name:<25} {l1_status:<8} {l2_status:<8} {l3_status:<8} {l4_status:<8}")

# Discover available scripts
available_scripts = tester.discover_scripts()
print(f"\n📁 Available Scripts ({len(available_scripts)}):")
for script in available_scripts[:5]:  # Show first 5
    print(f"  • {script}")
if len(available_scripts) > 5:
    print(f"  ... and {len(available_scripts) - 5} more")
```

## Common Workflows

### Daily Development Workflow

```python
def daily_alignment_check():
    """Daily alignment validation routine."""
    print("🌅 Daily Alignment Check")
    
    # Quick validation of recently modified scripts
    recent_scripts = ["tabular_preprocessing", "model_evaluation"]  # Your recent work
    
    tester = UnifiedAlignmentTester()
    report = tester.run_full_validation(target_scripts=recent_scripts)
    
    if report.is_passing():
        print("✅ Daily check passed - all alignments look good!")
        return True
    else:
        print("⚠️ Daily check found issues:")
        critical_issues = tester.get_critical_issues()
        for issue in critical_issues[:3]:
            print(f"  • {issue['message']}")
        return False

# Run daily check
daily_alignment_check()
```

### Pre-Commit Validation

```python
def pre_commit_validation():
    """Comprehensive validation before committing changes."""
    print("🔍 Pre-commit validation")
    
    tester = UnifiedAlignmentTester(level3_validation_mode="strict")
    report = tester.run_full_validation()
    
    # Generate report for review
    tester.export_report(format='html', output_path='pre_commit_report.html')
    
    summary = tester.get_validation_summary()
    
    if summary['pass_rate'] >= 95.0:  # Require 95% pass rate
        print("✅ Pre-commit validation passed!")
        return True
    else:
        print(f"❌ Pre-commit validation failed: {summary['pass_rate']:.1f}% pass rate")
        print("📄 Review pre_commit_report.html for details")
        return False

# Run pre-commit validation
pre_commit_validation()
```

### Integration Testing Workflow

```python
def integration_validation_workflow():
    """Comprehensive validation for integration testing."""
    print("🔗 Integration validation workflow")
    
    # Test with different validation modes
    modes = ["strict", "relaxed", "permissive"]
    results = {}
    
    for mode in modes:
        print(f"\n🧪 Testing with {mode} mode...")
        tester = UnifiedAlignmentTester(level3_validation_mode=mode)
        report = tester.run_full_validation()
        
        results[mode] = {
            'pass_rate': tester.get_validation_summary()['pass_rate'],
            'critical_issues': len(tester.get_critical_issues())
        }
        
        print(f"  Pass rate: {results[mode]['pass_rate']:.1f}%")
        print(f"  Critical issues: {results[mode]['critical_issues']}")
    
    # Recommend validation mode based on results
    if results['strict']['pass_rate'] >= 90:
        print("\n💡 Recommendation: Use strict mode for production")
    elif results['relaxed']['pass_rate'] >= 85:
        print("\n💡 Recommendation: Use relaxed mode for development")
    else:
        print("\n⚠️ Recommendation: Address critical issues before deployment")
    
    return results

# Run integration validation
integration_validation_workflow()
```

## Troubleshooting

### Issue: "No scripts found for validation"
```python
# Check if scripts directory exists and contains files
tester = UnifiedAlignmentTester()
scripts = tester.discover_scripts()

if not scripts:
    print("❌ No scripts found")
    print(f"📁 Scripts directory: {tester.scripts_dir}")
    print("💡 Check if the scripts directory path is correct")
    print("💡 Ensure script files follow naming convention (*.py)")
else:
    print(f"✅ Found {len(scripts)} scripts: {scripts}")
```

### Issue: "Level 3 validation fails with dependency errors"
```python
# Try different Level 3 validation modes
modes = ["permissive", "relaxed", "strict"]

for mode in modes:
    print(f"\n🧪 Testing Level 3 with {mode} mode...")
    tester = UnifiedAlignmentTester(level3_validation_mode=mode)
    report = tester.run_level_validation(level=3)
    
    passed_tests = sum(1 for result in report.level3_results.values() if result.passed)
    total_tests = len(report.level3_results)
    
    print(f"  Results: {passed_tests}/{total_tests} passed")
    
    if passed_tests == total_tests:
        print(f"✅ Level 3 validation passes with {mode} mode")
        break
```

### Issue: "Workspace validation fails"
```python
# Debug workspace configuration
from cursus.workspace.validation.workspace_alignment_tester import WorkspaceUnifiedAlignmentTester

try:
    workspace_tester = WorkspaceUnifiedAlignmentTester(
        workspace_root="development/projects",
        developer_id="your_developer_id"
    )
    
    # Get workspace info
    workspace_info = workspace_tester.get_workspace_info()
    print("✅ Workspace configuration:")
    print(f"  Developer ID: {workspace_info['developer_id']}")
    print(f"  Workspace root: {workspace_info['workspace_root']}")
    print(f"  Available developers: {workspace_info['available_developers']}")
    
except Exception as e:
    print(f"❌ Workspace setup failed: {e}")
    print("💡 Check if workspace directory exists")
    print("💡 Verify developer workspace is properly initialized")
```

## Next Steps

Congratulations! You've successfully:

1. ✅ Initialized the Unified Alignment Tester
2. ✅ Run comprehensive alignment validation
3. ✅ Validated specific scripts and levels
4. ✅ Generated detailed reports with scoring
5. ✅ Used workspace-aware validation
6. ✅ Learned advanced validation scenarios

### What's Next?

1. **Explore API Reference**: Check out the [Unified Tester API Reference](unified_tester_api_reference.md) for complete method documentation

2. **Integrate with CI/CD**: Set up automated alignment validation in your development pipeline

3. **Custom Validation Rules**: Learn to extend the tester with custom validation logic

4. **Team Collaboration**: Use workspace-aware validation for multi-developer projects

5. **Monitor Alignment Health**: Implement regular alignment monitoring and reporting

### Additional Resources

- **[Universal Step Builder Tester Quick Start](universal_builder_tester_quick_start.md)** - Learn to test step builders
- **[Universal Step Builder Tester API Reference](universal_builder_tester_api_reference.md)** - Complete builder testing API
- **[Workspace Quick Start Guide](../../workspace/workspace_quick_start.md)** - Multi-developer workspace setup
- **[Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md)** - Comprehensive validation concepts

## Summary

The Unified Alignment Tester provides comprehensive validation across all four levels of pipeline architecture, ensuring consistency and reliability in your ML pipeline development. With workspace-aware capabilities, it supports both individual and collaborative development workflows, making it an essential tool for maintaining high-quality pipeline implementations.

Happy validating! 🚀
