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
  - configuration-driven validation
topics:
  - unified alignment testing
  - validation tutorial
  - alignment validation workflow
  - step-type-aware validation
language: python
date of note: 2025-10-05
---

# Unified Alignment Tester Quick Start Guide

## Overview

This 20-minute tutorial will get you up and running with the Cursus Unified Alignment Tester. You'll learn how to validate alignment across all four levels of the pipeline architecture using the new configuration-driven approach that automatically adapts validation based on step types.

## Prerequisites

- Cursus package installed
- Python 3.8+ environment
- Basic familiarity with ML pipeline development
- Understanding of script contracts and step specifications

## What is Unified Alignment Testing?

The Unified Alignment Tester validates consistency across four critical levels:

1. **Level 1**: Script ↔ Contract Alignment (validates script implementation matches contract)
2. **Level 2**: Contract ↔ Specification Alignment (ensures contract matches spec requirements)
3. **Level 3**: Specification ↔ Dependencies Alignment (universal dependency validation)
4. **Level 4**: Builder ↔ Configuration Alignment (validates builder configuration)

The system uses **step-type-aware validation** - different SageMaker step types (Processing, Training, CreateModel, etc.) automatically get appropriate validation levels applied.

## Step 1: Initialize the Unified Alignment Tester (2 minutes)

First, let's set up the tester and verify it's working:

```python
from cursus.validation.alignment import UnifiedAlignmentTester

# Initialize with default configuration (package-only steps)
tester = UnifiedAlignmentTester()

# Or initialize with workspace directories for workspace-aware validation
tester = UnifiedAlignmentTester(
    workspace_dirs=["development/projects/project_alpha", "development/projects/project_beta"]
)

print("✅ Unified Alignment Tester initialized successfully")

# Check discovered steps
discovered_steps = tester.discover_scripts()
print(f"📁 Discovered {len(discovered_steps)} steps with script files")

# Get validation summary to see configuration
summary = tester.get_validation_summary()
print(f"📊 Ready to validate {summary['total_steps']} total steps")
```

**Expected Output:**
```
✅ Unified Alignment Tester initialized successfully
📁 Discovered 15 steps with script files
📊 Ready to validate 21 total steps
```

## Step 2: Run Your First Configuration-Driven Validation (3 minutes)

The new system automatically applies appropriate validation levels based on step types:

```python
# Run comprehensive validation with automatic step-type detection
print("🔍 Starting configuration-driven alignment validation...")
results = tester.run_full_validation()

# Check overall results
print(f"\n📊 Validation Results:")
for step_name, result in results.items():
    status = result.get("overall_status") or result.get("status", "UNKNOWN")
    step_type = result.get("sagemaker_step_type", "unknown")
    category = result.get("category", "unknown")
    
    status_icon = "✅" if status == "PASSED" else "❌" if status == "FAILED" else "⚠️"
    print(f"{status_icon} {step_name} ({step_type}) - {status}")
    
    # Show enabled validation levels
    enabled_levels = result.get("enabled_levels", [])
    if enabled_levels:
        print(f"    Levels: {enabled_levels}")

# Print enhanced summary
tester.print_summary()
```

**What this validates:**
- **Processing steps**: Full 4-level validation (script, contract, spec, builder)
- **Training steps**: Full 4-level validation with training-specific patterns
- **CreateModel steps**: Levels 3-4 only (no script/contract needed)
- **Transform steps**: Levels 3-4 only (uses existing models)
- **Excluded steps**: Skipped entirely (Base, Utility types)

## Step 3: Validate Specific Steps (2 minutes)

You can target specific steps for focused validation:

```python
# Validate specific steps only
target_steps = ["tabular_preprocessing", "xgboost_training", "xgboost_model"]

print(f"🎯 Validating specific steps: {target_steps}")
results = tester.run_full_validation(target_scripts=target_steps)

# Check results for each step
for step_name in target_steps:
    if step_name in results:
        result = results[step_name]
        step_type = result.get("sagemaker_step_type", "unknown")
        status = result.get("overall_status", "UNKNOWN")
        
        print(f"\n📋 {step_name} ({step_type}): {status}")
        
        # Show validation results by level
        validation_results = result.get("validation_results", {})
        for level_key, level_result in validation_results.items():
            level_status = level_result.get("status", "UNKNOWN")
            level_icon = "✅" if level_status == "PASSED" else "❌"
            print(f"  {level_icon} {level_key}: {level_status}")
    else:
        print(f"⚠️ {step_name}: Not found or excluded")
```

## Step 4: Understanding Step-Type-Aware Validation (3 minutes)

The system automatically adapts validation based on step types:

```python
# Explore step type configuration
from cursus.validation.alignment.config import get_validation_ruleset, get_all_step_types

print("🔧 Step Type Configuration:")
print(f"{'Step Type':<25} {'Category':<15} {'Levels':<20} {'Validator'}")
print("-" * 80)

for step_type in get_all_step_types():
    ruleset = get_validation_ruleset(step_type)
    if ruleset:
        levels = [str(level.value) for level in ruleset.enabled_levels]
        levels_str = ",".join(levels) if levels else "None"
        validator = ruleset.level_4_validator_class or "None"
        
        print(f"{step_type:<25} {ruleset.category.value:<15} {levels_str:<20} {validator}")

# Check specific step validation rules
step_name = "tabular_preprocessing"
from cursus.registry.step_names import get_sagemaker_step_type

try:
    step_type = get_sagemaker_step_type(step_name)
    ruleset = get_validation_ruleset(step_type)
    
    print(f"\n🔍 Validation Rules for {step_name}:")
    print(f"  Step Type: {step_type}")
    print(f"  Category: {ruleset.category.value}")
    print(f"  Enabled Levels: {[level.value for level in ruleset.enabled_levels]}")
    print(f"  Skip Reason: {ruleset.skip_reason or 'None'}")
    
except Exception as e:
    print(f"⚠️ Could not determine step type for {step_name}: {e}")
```

**Expected Output:**
```
🔧 Step Type Configuration:
Step Type                 Category        Levels               Validator
--------------------------------------------------------------------------------
Processing                script_based    1,2,3,4              ProcessingStepBuilderValidator
Training                  script_based    1,2,3,4              TrainingStepBuilderValidator
CreateModel               non_script      3,4                  CreateModelStepBuilderValidator
Transform                 non_script      3,4                  TransformStepBuilderValidator
CradleDataLoading         contract_based  2,3,4                ProcessingStepBuilderValidator
Base                      excluded        None                 None
```

## Step 5: Analyzing Validation Results (3 minutes)

Let's explore detailed validation results and issues:

```python
# Run validation and analyze results
results = tester.run_full_validation()

# Get critical issues across all steps
critical_issues = tester.get_critical_issues()

if critical_issues:
    print(f"🚨 {len(critical_issues)} Critical Issues Found:")
    for issue in critical_issues[:5]:  # Show first 5
        print(f"  • {issue['step_name']} ({issue['step_type']}): {issue['error']}")
        print(f"    Level: {issue['level']}, Category: {issue['category']}")
else:
    print("✅ No critical issues found!")

# Analyze results by step type
step_type_analysis = {}
for step_name, result in results.items():
    step_type = result.get("sagemaker_step_type", "unknown")
    status = result.get("overall_status") or result.get("status")
    
    if step_type not in step_type_analysis:
        step_type_analysis[step_type] = {"total": 0, "passed": 0, "failed": 0, "excluded": 0}
    
    step_type_analysis[step_type]["total"] += 1
    
    if status == "PASSED":
        step_type_analysis[step_type]["passed"] += 1
    elif status == "EXCLUDED":
        step_type_analysis[step_type]["excluded"] += 1
    else:
        step_type_analysis[step_type]["failed"] += 1

print(f"\n📊 Results by Step Type:")
for step_type, stats in step_type_analysis.items():
    total = stats["total"]
    passed = stats["passed"]
    excluded = stats["excluded"]
    active = total - excluded
    pass_rate = (passed / active * 100) if active > 0 else 0
    
    print(f"  {step_type}: {passed}/{active} passed ({pass_rate:.1f}%)")
    if excluded > 0:
        print(f"    ({excluded} excluded)")
```

## Step 6: Generate Enhanced Reports (3 minutes)

The tester generates comprehensive reports with step-type insights:

```python
# Generate JSON report with step-type breakdown
json_report = tester.export_report(format='json', output_path='alignment_report.json')
print("📄 JSON report generated: alignment_report.json")

# Generate text report with enhanced summary
text_report = tester.export_report(format='text', output_path='alignment_report.txt')
print("📄 Text report generated: alignment_report.txt")

# Print enhanced summary to console
print("\n" + "="*60)
print("ENHANCED VALIDATION SUMMARY")
print("="*60)

summary = tester.get_validation_summary()
print(f"Total Steps: {summary['total_steps']}")
print(f"Passed: {summary['passed_steps']}")
print(f"Failed: {summary['failed_steps']}")
print(f"Excluded: {summary['excluded_steps']}")
print(f"Pass Rate: {summary['pass_rate']:.2%}")
print(f"Configuration-Driven: {summary['configuration_driven']}")

print("\nStep Type Breakdown:")
for step_type, breakdown in summary['step_type_breakdown'].items():
    status_str = f"{breakdown['passed']}/{breakdown['total']} passed"
    if breakdown['excluded'] > 0:
        status_str += f" ({breakdown['excluded']} excluded)"
    print(f"  {step_type}: {status_str}")

print("="*60)
```

## Step 7: Validate Single Steps in Detail (2 minutes)

For detailed analysis of individual steps:

```python
# Validate a single step across all applicable levels
step_name = "tabular_preprocessing"
result = tester.run_validation_for_step(step_name)

print(f"🔍 Detailed Validation for {step_name}:")
print(f"Step Type: {result.get('sagemaker_step_type')}")
print(f"Category: {result.get('category')}")
print(f"Overall Status: {result.get('overall_status')}")

# Check each validation level
validation_results = result.get("validation_results", {})
for level_key, level_result in validation_results.items():
    status = level_result.get("status", "UNKNOWN")
    status_icon = "✅" if status in ["PASSED", "COMPLETED"] else "❌"
    
    print(f"\n{status_icon} {level_key.upper()}:")
    print(f"  Status: {status}")
    
    # Show any errors
    if "error" in level_result:
        print(f"  Error: {level_result['error']}")
    
    # Show validation details if available
    if "details" in level_result:
        details = level_result["details"]
        if isinstance(details, dict):
            for key, value in details.items():
                if key not in ["status", "error"]:
                    print(f"  {key}: {value}")

# Get step info from catalog
step_info = tester.get_step_info_from_catalog(step_name)
if step_info:
    print(f"\n📁 Step Files:")
    for component_type, component in step_info.file_components.items():
        if component:
            print(f"  {component_type}: {component.path}")
```

## Step 8: Workspace-Aware Validation (2 minutes)

For multi-developer environments with workspace support:

```python
# Initialize with workspace directories
workspace_tester = UnifiedAlignmentTester(
    workspace_dirs=["development/projects/project_alpha", "development/projects/project_beta"]
)

print("🏢 Workspace-aware validation initialized")

# Run validation across workspaces
workspace_results = workspace_tester.run_full_validation()

# Analyze workspace distribution
workspace_stats = {}
for step_name, result in workspace_results.items():
    # Check if step comes from workspace or package
    step_info = workspace_tester.get_step_info_from_catalog(step_name)
    if step_info:
        source = "workspace" if any("development" in str(comp.path) for comp in step_info.file_components.values() if comp) else "package"
        
        if source not in workspace_stats:
            workspace_stats[source] = {"total": 0, "passed": 0}
        
        workspace_stats[source]["total"] += 1
        status = result.get("overall_status") or result.get("status")
        if status == "PASSED":
            workspace_stats[source]["passed"] += 1

print(f"\n📊 Workspace Distribution:")
for source, stats in workspace_stats.items():
    pass_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"  {source.title()}: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")

# Check cross-workspace compatibility
if len(workspace_results) > 1:
    step_names = list(workspace_results.keys())
    compatibility = workspace_tester.validate_cross_workspace_compatibility(step_names)
    
    print(f"\n🔗 Cross-Workspace Compatibility:")
    print(f"  Compatible: {compatibility['compatible']}")
    print(f"  Step Types: {len(compatibility['step_type_distribution'])}")
    
    if compatibility['recommendations']:
        print("  Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"    • {rec}")
```

## Common Workflows

### Daily Development Workflow

```python
def daily_alignment_check():
    """Daily alignment validation routine with step-type awareness."""
    print("🌅 Daily Alignment Check")
    
    tester = UnifiedAlignmentTester()
    
    # Quick validation of all steps
    results = tester.run_full_validation()
    
    # Check for any critical issues
    critical_issues = tester.get_critical_issues()
    
    if not critical_issues:
        print("✅ Daily check passed - all alignments look good!")
        
        # Show summary
        summary = tester.get_validation_summary()
        print(f"📊 {summary['passed_steps']}/{summary['total_steps']} steps passed")
        print(f"   ({summary['excluded_steps']} excluded by configuration)")
        
        return True
    else:
        print(f"⚠️ Daily check found {len(critical_issues)} critical issues:")
        for issue in critical_issues[:3]:  # Show first 3
            print(f"  • {issue['step_name']}: {issue['error']}")
        return False

# Run daily check
daily_alignment_check()
```

### Pre-Commit Validation

```python
def pre_commit_validation():
    """Comprehensive validation before committing changes."""
    print("🔍 Pre-commit validation with enhanced configuration")
    
    tester = UnifiedAlignmentTester()
    results = tester.run_full_validation()
    
    # Generate detailed report
    tester.export_report(format='json', output_path='pre_commit_report.json')
    
    # Check results
    summary = tester.get_validation_summary()
    critical_issues = tester.get_critical_issues()
    
    # Calculate pass rate excluding excluded steps
    active_steps = summary['total_steps'] - summary['excluded_steps']
    pass_rate = (summary['passed_steps'] / active_steps * 100) if active_steps > 0 else 100
    
    if pass_rate >= 95.0 and len(critical_issues) == 0:
        print("✅ Pre-commit validation passed!")
        print(f"📊 Pass rate: {pass_rate:.1f}% ({summary['passed_steps']}/{active_steps})")
        return True
    else:
        print(f"❌ Pre-commit validation failed:")
        print(f"📊 Pass rate: {pass_rate:.1f}% (required: 95%)")
        print(f"🚨 Critical issues: {len(critical_issues)}")
        print("📄 Review pre_commit_report.json for details")
        return False

# Run pre-commit validation
pre_commit_validation()
```

### Step Type Analysis Workflow

```python
def analyze_step_type_coverage():
    """Analyze validation coverage by step type."""
    print("📊 Step Type Coverage Analysis")
    
    tester = UnifiedAlignmentTester()
    results = tester.run_full_validation()
    
    # Group by step type and category
    from cursus.validation.alignment.config import get_validation_ruleset
    
    coverage_analysis = {}
    for step_name, result in results.items():
        step_type = result.get("sagemaker_step_type", "unknown")
        status = result.get("overall_status") or result.get("status")
        
        if step_type not in coverage_analysis:
            ruleset = get_validation_ruleset(step_type)
            coverage_analysis[step_type] = {
                "category": ruleset.category.value if ruleset else "unknown",
                "total": 0,
                "passed": 0,
                "failed": 0,
                "excluded": 0,
                "examples": []
            }
        
        coverage_analysis[step_type]["total"] += 1
        coverage_analysis[step_type]["examples"].append(step_name)
        
        if status == "PASSED":
            coverage_analysis[step_type]["passed"] += 1
        elif status == "EXCLUDED":
            coverage_analysis[step_type]["excluded"] += 1
        else:
            coverage_analysis[step_type]["failed"] += 1
    
    # Print analysis
    print(f"\n{'Step Type':<20} {'Category':<15} {'Coverage':<15} {'Examples'}")
    print("-" * 80)
    
    for step_type, analysis in coverage_analysis.items():
        total = analysis["total"]
        passed = analysis["passed"]
        excluded = analysis["excluded"]
        active = total - excluded
        
        if active > 0:
            coverage = f"{passed}/{active} ({passed/active*100:.0f}%)"
        else:
            coverage = "N/A (excluded)"
        
        examples = ", ".join(analysis["examples"][:2])  # Show first 2 examples
        if len(analysis["examples"]) > 2:
            examples += f" (+{len(analysis['examples'])-2} more)"
        
        print(f"{step_type:<20} {analysis['category']:<15} {coverage:<15} {examples}")
    
    return coverage_analysis

# Run step type analysis
analyze_step_type_coverage()
```

## Troubleshooting

### Issue: "No steps found for validation"
```python
# Debug step discovery
tester = UnifiedAlignmentTester()

# Check step catalog discovery
all_steps = tester.step_catalog.list_available_steps()
scripts_with_files = tester.discover_scripts()

print(f"📁 Step Discovery Debug:")
print(f"  Total steps in catalog: {len(all_steps)}")
print(f"  Steps with script files: {len(scripts_with_files)}")

if all_steps:
    print(f"  Example steps: {all_steps[:5]}")
else:
    print("❌ No steps found in catalog")
    print("💡 Check if StepCatalog is properly configured")
    print("💡 Verify workspace directories if using workspace-aware mode")

if not scripts_with_files:
    print("❌ No script files found")
    print("💡 Check if scripts exist in src/cursus/steps/scripts/")
```

### Issue: "Step type detection fails"
```python
# Debug step type detection
from cursus.registry.step_names import get_sagemaker_step_type

step_name = "your_step_name"
try:
    step_type = get_sagemaker_step_type(step_name)
    print(f"✅ {step_name} -> {step_type}")
    
    # Check validation rules
    from cursus.validation.alignment.config import get_validation_ruleset
    ruleset = get_validation_ruleset(step_type)
    if ruleset:
        print(f"📋 Validation levels: {[level.value for level in ruleset.enabled_levels]}")
    else:
        print("⚠️ No validation ruleset found")
        
except Exception as e:
    print(f"❌ Step type detection failed: {e}")
    print("💡 Check step naming patterns")
    print("💡 Verify step is registered in step_names.py")
```

### Issue: "Validation fails with configuration errors"
```python
# Validate configuration consistency
from cursus.validation.alignment.config import validate_step_type_configuration

config_issues = validate_step_type_configuration()
if config_issues:
    print("⚠️ Configuration issues found:")
    for issue in config_issues:
        print(f"  • {issue}")
else:
    print("✅ Configuration is valid")

# Check specific step configuration
step_name = "your_step_name"
try:
    from cursus.validation.alignment.config import get_validation_ruleset_for_step_name
    ruleset = get_validation_ruleset_for_step_name(step_name)
    
    if ruleset:
        print(f"✅ Configuration for {step_name}:")
        print(f"  Step Type: {ruleset.step_type}")
        print(f"  Category: {ruleset.category.value}")
        print(f"  Enabled Levels: {[level.value for level in ruleset.enabled_levels]}")
    else:
        print(f"❌ No configuration found for {step_name}")
        
except Exception as e:
    print(f"❌ Configuration lookup failed: {e}")
```

## Next Steps

Congratulations! You've successfully:

1. ✅ Initialized the Enhanced Unified Alignment Tester
2. ✅ Run configuration-driven alignment validation
3. ✅ Understood step-type-aware validation rules
4. ✅ Analyzed validation results by step type
5. ✅ Generated enhanced reports with step-type insights
6. ✅ Used workspace-aware validation capabilities
7. ✅ Learned advanced validation workflows

### What's Next?

1. **Explore API Reference**: Check out the [Unified Alignment Tester API Reference](unified_alignment_tester_api_reference.md) for complete method documentation

2. **Integrate with CI/CD**: Set up automated alignment validation in your development pipeline using the configuration-driven approach

3. **Custom Step Types**: Learn to extend the validation ruleset for custom step types

4. **Team Collaboration**: Use workspace-aware validation for multi-developer projects

5. **Monitor Alignment Health**: Implement regular alignment monitoring with step-type-aware reporting

### Additional Resources

- **[Unified Alignment Tester API Reference](unified_alignment_tester_api_reference.md)** - Complete API documentation
- **[Universal Step Builder Tester Quick Start](universal_builder_tester_quick_start.md)** - Learn to test step builders
- **[Workspace Quick Start Guide](../../workspace/workspace_quick_start.md)** - Multi-developer workspace setup
- **[Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md)** - Comprehensive validation concepts

## Summary

The Enhanced Unified Alignment Tester provides configuration-driven validation that automatically adapts to different SageMaker step types. With step-type-aware validation rules, it ensures appropriate validation levels are applied while skipping unnecessary checks, making it both comprehensive and efficient for maintaining high-quality pipeline implementations.

The system supports:
- **Automatic step type detection** and appropriate validation level selection
- **Workspace-aware validation** for multi-developer environments  
- **Enhanced reporting** with step-type breakdown and insights
- **Performance optimization** through intelligent validation level skipping
- **Extensible configuration** for custom step types and validation rules

Happy validating! 🚀
