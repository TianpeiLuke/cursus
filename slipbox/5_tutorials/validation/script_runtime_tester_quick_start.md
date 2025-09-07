---
tags:
  - test
  - validation
  - runtime
  - quick_start
  - tutorial
keywords:
  - script runtime tester
  - pipeline runtime testing
  - script validation
  - data compatibility testing
  - pipeline flow testing
topics:
  - script runtime testing
  - validation tutorial
  - pipeline testing workflow
  - script execution validation
language: python
date of note: 2025-09-06
---

# Script Runtime Tester Quick Start Guide

## Overview

This 15-minute tutorial will get you up and running with the Cursus Script Runtime Tester. You'll learn how to validate script functionality, test data compatibility between scripts, and verify complete pipeline flows with actual script execution.

## Prerequisites

- Cursus package installed
- Python 3.8+ environment
- Basic familiarity with ML pipeline development
- Scripts following the Cursus script development guide (main function with input_paths, output_paths, environ_vars, job_args)

## What is Script Runtime Testing?

The Script Runtime Tester validates pipeline scripts by **actually executing them** and testing:

1. **Script Functionality**: Does the script have a proper main() function and execute successfully?
2. **Data Compatibility**: Can script outputs be consumed by downstream scripts?
3. **Pipeline Flow**: Does the complete pipeline execute end-to-end successfully?

## Step 1: Initialize the Runtime Tester (2 minutes)

First, let's set up the tester and verify it's working:

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester

# Initialize with workspace directory (backward compatible)
tester = RuntimeTester("./test_workspace")

print("✅ Runtime Tester initialized successfully")
print(f"Workspace directory: {tester.workspace_dir}")
```

**Expected Output:**
```
✅ Runtime Tester initialized successfully
Workspace directory: ./test_workspace
```

## Step 2: Test Your First Script (3 minutes)

Let's test a single script to see if it executes properly:

```python
# Test a single script
script_name = "tabular_preprocessing"  # Replace with your script name

print(f"🔍 Testing script: {script_name}")
result = tester.test_script(script_name)

# Check results
if result.success:
    print("✅ Script test passed!")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Has main function: {'Yes' if result.has_main_function else 'No'}")
else:
    print("❌ Script test failed!")
    print(f"Error: {result.error_message}")
    print(f"Has main function: {'Yes' if result.has_main_function else 'No'}")

# Print detailed results
print(f"\n📊 Test Results:")
print(f"Script: {result.script_name}")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Main function found: {result.has_main_function}")
if result.error_message:
    print(f"Error message: {result.error_message}")
```

**What this validates:**
- Script has a main() function with correct signature (input_paths, output_paths, environ_vars, job_args)
- Script executes without errors
- Script can process sample data and produce outputs

## Step 3: Test Data Compatibility Between Scripts (3 minutes)

Now let's test if one script's output can be consumed by another script:

```python
# Test data compatibility between two scripts
script_a = "tabular_preprocessing"  # First script (data producer)
script_b = "xgboost_training"      # Second script (data consumer)

print(f"🔗 Testing data compatibility: {script_a} -> {script_b}")

# Generate sample data for testing
sample_data = tester._generate_sample_data()
print(f"Sample data: {sample_data}")

# Test compatibility
compat_result = tester.test_data_compatibility(script_a, script_b, sample_data)

if compat_result.compatible:
    print("✅ Scripts are data compatible!")
    print(f"Data formats: {compat_result.data_format_a} -> {compat_result.data_format_b}")
else:
    print("❌ Scripts have data compatibility issues!")
    print("Issues found:")
    for issue in compat_result.compatibility_issues:
        print(f"  - {issue}")

# Print detailed compatibility results
print(f"\n📊 Compatibility Results:")
print(f"Script A: {compat_result.script_a}")
print(f"Script B: {compat_result.script_b}")
print(f"Compatible: {compat_result.compatible}")
print(f"Data format A: {compat_result.data_format_a}")
print(f"Data format B: {compat_result.data_format_b}")
if compat_result.compatibility_issues:
    print("Issues:")
    for issue in compat_result.compatibility_issues:
        print(f"  - {issue}")
```

**What this validates:**
- Script A executes successfully and produces output data
- Script B can consume Script A's output data without errors
- Data formats are compatible between scripts

## Step 4: Test Complete Pipeline Flow (3 minutes)

Let's test an entire pipeline to ensure all scripts work together:

```python
# Define a simple pipeline configuration
pipeline_config = {
    "steps": {
        "data_preprocessing": {"script": "tabular_preprocessing.py"},
        "model_training": {"script": "xgboost_training.py"},
        "model_evaluation": {"script": "model_evaluation.py"}
    }
}

print("🚀 Testing complete pipeline flow...")
pipeline_results = tester.test_pipeline_flow(pipeline_config)

# Check overall pipeline results
if pipeline_results["pipeline_success"]:
    print("✅ Pipeline flow test passed!")
else:
    print("❌ Pipeline flow test failed!")
    print("Errors:")
    for error in pipeline_results["errors"]:
        print(f"  - {error}")

# Check individual script results
print(f"\n📝 Individual Script Results:")
for script_name, result in pipeline_results["script_results"].items():
    status = "✅" if result.success else "❌"
    print(f"  {status} {script_name}: {'PASS' if result.success else 'FAIL'}")
    if not result.success:
        print(f"    Error: {result.error_message}")

# Check data flow results
print(f"\n🔗 Data Flow Results:")
for flow_name, result in pipeline_results["data_flow_results"].items():
    status = "✅" if result.compatible else "❌"
    print(f"  {status} {flow_name}: {'PASS' if result.compatible else 'FAIL'}")
    if result.compatibility_issues:
        for issue in result.compatibility_issues:
            print(f"    Issue: {issue}")

# Print summary
total_scripts = len(pipeline_results["script_results"])
successful_scripts = sum(1 for result in pipeline_results["script_results"].values() if result.success)
total_flows = len(pipeline_results["data_flow_results"])
successful_flows = sum(1 for result in pipeline_results["data_flow_results"].values() if result.compatible)

print(f"\n📊 Pipeline Summary:")
print(f"Overall success: {pipeline_results['pipeline_success']}")
print(f"Script tests: {successful_scripts}/{total_scripts} passed")
print(f"Data flow tests: {successful_flows}/{total_flows} passed")
print(f"Total errors: {len(pipeline_results['errors'])}")
```

## Step 5: Using Script Execution Specifications (Advanced) (4 minutes)

For more control over script testing, you can use ScriptExecutionSpec to define exactly how scripts should be executed:

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Create a custom script execution specification
script_spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing_step",
    input_paths={"data_input": "./test_data/raw_data.csv"},
    output_paths={"data_output": "./test_output/processed"},
    environ_vars={"LABEL_FIELD": "target", "FEATURE_COLS": "feature1,feature2"},
    job_args={"job_type": "preprocessing", "batch_size": 1000}
)

print("📋 Created custom script execution specification:")
print(f"Script: {script_spec.script_name}")
print(f"Input paths: {script_spec.input_paths}")
print(f"Output paths: {script_spec.output_paths}")
print(f"Environment vars: {script_spec.environ_vars}")
print(f"Job args: {script_spec.job_args}")

# Save the specification for reuse
builder = PipelineTestingSpecBuilder("./test_workspace")
saved_path = script_spec.save_to_file(str(builder.specs_dir))
print(f"💾 Saved specification to: {saved_path}")

# Test script with custom specification
main_params = builder.get_script_main_params(script_spec)
spec_result = tester.test_script_with_spec(script_spec, main_params)

if spec_result.success:
    print("✅ Script test with custom spec passed!")
else:
    print("❌ Script test with custom spec failed!")
    print(f"Error: {spec_result.error_message}")

# Load specification from file
loaded_spec = ScriptExecutionSpec.load_from_file("tabular_preprocessing", str(builder.specs_dir))
print(f"📂 Loaded specification from file:")
print(f"Last updated: {loaded_spec.last_updated}")
```

**What this enables:**
- Custom input/output paths for testing
- Specific environment variables and job arguments
- Reusable test configurations saved as JSON files
- More precise control over script execution parameters

## Step 6: Using the Command Line Interface (2 minutes)

The runtime tester also provides a CLI for easy command-line usage:

```bash
# Test a single script
cursus runtime test-script tabular_preprocessing --workspace-dir ./test_workspace

# Test data compatibility between scripts
cursus runtime test-compatibility tabular_preprocessing xgboost_training --workspace-dir ./test_workspace

# Test complete pipeline (requires pipeline config JSON file)
cursus runtime test-pipeline pipeline_config.json --workspace-dir ./test_workspace

# Get JSON output for programmatic use
cursus runtime test-script tabular_preprocessing --output-format json
```

**Example pipeline_config.json:**
```json
{
  "steps": {
    "preprocessing": {"script": "tabular_preprocessing.py"},
    "training": {"script": "xgboost_training.py"},
    "evaluation": {"script": "model_evaluation.py"}
  }
}
```

## Common Workflows

### Daily Development Workflow

```python
def daily_script_check():
    """Daily script validation routine."""
    print("🌅 Daily Script Check")
    
    # Scripts you're currently working on
    current_scripts = ["tabular_preprocessing", "model_evaluation"]
    
    tester = RuntimeTester("./test_workspace")
    all_passed = True
    
    for script in current_scripts:
        print(f"\n🔍 Testing {script}...")
        result = tester.test_script(script)
        
        if result.success:
            print(f"✅ {script}: PASS ({result.execution_time:.3f}s)")
        else:
            print(f"❌ {script}: FAIL - {result.error_message}")
            all_passed = False
    
    if all_passed:
        print("\n✅ Daily check passed - all scripts working!")
        return True
    else:
        print("\n⚠️ Daily check found issues - review failed scripts")
        return False

# Run daily check
daily_script_check()
```

### Pre-Commit Validation

```python
def pre_commit_validation():
    """Comprehensive validation before committing changes."""
    print("🔍 Pre-commit validation")
    
    # Test critical pipeline scripts
    critical_scripts = ["tabular_preprocessing", "xgboost_training", "model_evaluation"]
    
    tester = RuntimeTester("./test_workspace")
    
    # Test individual scripts
    script_results = {}
    for script in critical_scripts:
        result = tester.test_script(script)
        script_results[script] = result
    
    # Test data compatibility
    compatibility_results = {}
    for i in range(len(critical_scripts) - 1):
        script_a = critical_scripts[i]
        script_b = critical_scripts[i + 1]
        
        sample_data = tester._generate_sample_data()
        compat_result = tester.test_data_compatibility(script_a, script_b, sample_data)
        compatibility_results[f"{script_a}->{script_b}"] = compat_result
    
    # Check results
    script_failures = [name for name, result in script_results.items() if not result.success]
    compat_failures = [name for name, result in compatibility_results.items() if not result.compatible]
    
    if not script_failures and not compat_failures:
        print("✅ Pre-commit validation passed!")
        return True
    else:
        print("❌ Pre-commit validation failed!")
        if script_failures:
            print(f"Failed scripts: {script_failures}")
        if compat_failures:
            print(f"Compatibility issues: {compat_failures}")
        return False

# Run pre-commit validation
pre_commit_validation()
```

### Integration Testing Workflow

```python
def integration_testing_workflow():
    """Complete integration testing for pipeline."""
    print("🔗 Integration testing workflow")
    
    # Define complete pipeline
    full_pipeline = {
        "steps": {
            "data_loading": {"script": "data_loading.py"},
            "preprocessing": {"script": "tabular_preprocessing.py"},
            "feature_engineering": {"script": "feature_engineering.py"},
            "model_training": {"script": "xgboost_training.py"},
            "model_evaluation": {"script": "model_evaluation.py"},
            "model_registration": {"script": "model_registration.py"}
        }
    }
    
    tester = RuntimeTester("./test_workspace")
    
    # Test complete pipeline flow
    results = tester.test_pipeline_flow(full_pipeline)
    
    # Analyze results
    total_scripts = len(results["script_results"])
    successful_scripts = sum(1 for result in results["script_results"].values() if result.success)
    
    total_flows = len(results["data_flow_results"])
    successful_flows = sum(1 for result in results["data_flow_results"].values() if result.compatible)
    
    print(f"\n📊 Integration Test Results:")
    print(f"Overall success: {results['pipeline_success']}")
    print(f"Script success rate: {successful_scripts}/{total_scripts} ({successful_scripts/total_scripts*100:.1f}%)")
    print(f"Data flow success rate: {successful_flows}/{total_flows} ({successful_flows/total_flows*100:.1f}%)")
    
    if results["pipeline_success"]:
        print("✅ Integration testing passed!")
        return True
    else:
        print("❌ Integration testing failed!")
        print("Issues to address:")
        for error in results["errors"]:
            print(f"  - {error}")
        return False

# Run integration testing
integration_testing_workflow()
```

## Troubleshooting

### Issue: "Script not found"
```python
# Check if script exists in expected locations
script_name = "your_script_name"
tester = RuntimeTester("./test_workspace")

try:
    script_path = tester._find_script_path(script_name)
    print(f"✅ Script found at: {script_path}")
except FileNotFoundError as e:
    print(f"❌ {e}")
    print("💡 Check these locations:")
    possible_paths = [
        f"src/cursus/steps/scripts/{script_name}.py",
        f"scripts/{script_name}.py",
        f"dockers/xgboost_atoz/scripts/{script_name}.py",
        f"dockers/pytorch_bsm_ext/scripts/{script_name}.py"
    ]
    for path in possible_paths:
        exists = Path(path).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {path}")
```

### Issue: "Main function signature doesn't match"
```python
# Check script main function signature
import inspect
import importlib.util

script_name = "your_script_name"
script_path = f"src/cursus/steps/scripts/{script_name}.py"

if Path(script_path).exists():
    spec = importlib.util.spec_from_file_location("script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'main'):
        sig = inspect.signature(module.main)
        actual_params = list(sig.parameters.keys())
        expected_params = ['input_paths', 'output_paths', 'environ_vars', 'job_args']
        
        print(f"Expected parameters: {expected_params}")
        print(f"Actual parameters: {actual_params}")
        
        missing = set(expected_params) - set(actual_params)
        extra = set(actual_params) - set(expected_params)
        
        if missing:
            print(f"❌ Missing parameters: {missing}")
        if extra:
            print(f"⚠️ Extra parameters: {extra}")
        if not missing and not extra:
            print("✅ Function signature matches!")
    else:
        print("❌ No main function found in script")
else:
    print(f"❌ Script not found: {script_path}")
```

### Issue: "Data compatibility test fails"
```python
# Debug data compatibility issues
script_a = "script_a_name"
script_b = "script_b_name"

tester = RuntimeTester("./test_workspace")
sample_data = tester._generate_sample_data()

# Test script A individually first
print("🔍 Testing Script A individually...")
result_a = tester.test_script(script_a)
if result_a.success:
    print("✅ Script A executes successfully")
else:
    print(f"❌ Script A failed: {result_a.error_message}")

# Test script B individually
print("🔍 Testing Script B individually...")
result_b = tester.test_script(script_b)
if result_b.success:
    print("✅ Script B executes successfully")
else:
    print(f"❌ Script B failed: {result_b.error_message}")

# If both pass individually, test compatibility
if result_a.success and result_b.success:
    print("🔗 Testing data compatibility...")
    compat_result = tester.test_data_compatibility(script_a, script_b, sample_data)
    
    if not compat_result.compatible:
        print("❌ Compatibility issues found:")
        for issue in compat_result.compatibility_issues:
            print(f"  - {issue}")
        
        # Check output directory for Script A
        test_dir_a = tester.workspace_dir / f"test_{script_a}"
        output_files = list(test_dir_a.glob("*.csv"))
        print(f"Script A output files: {output_files}")
```

## Next Steps

Congratulations! You've successfully:

1. ✅ Initialized the Script Runtime Tester
2. ✅ Tested individual script functionality
3. ✅ Validated data compatibility between scripts
4. ✅ Tested complete pipeline flows
5. ✅ Used advanced script execution specifications
6. ✅ Learned CLI usage and common workflows

### What's Next?

1. **Explore API Reference**: Check out the [Script Runtime Tester API Reference](script_runtime_tester_api_reference.md) for complete method documentation

2. **Integrate with CI/CD**: Set up automated runtime testing in your development pipeline

3. **Custom Test Data**: Learn to create custom test datasets for more realistic validation

4. **Performance Monitoring**: Track script execution times and optimize performance

5. **Error Analysis**: Develop systematic approaches to debugging script failures

### Additional Resources

- **[Unified Alignment Tester Quick Start](unified_alignment_tester_quick_start.md)** - Learn comprehensive alignment validation
- **[Script Development Guide](../../0_developer_guide/script_development_guide.md)** - Best practices for script development
- **[Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md)** - Complete validation concepts

## References and Design Documents

For deeper technical understanding of the Script Runtime Tester implementation and design decisions, refer to these key documents:

### Implementation Plans
- **[Pipeline Runtime Testing Script Refactoring Plan](../../2_project_planning/2025-09-06_pipeline_runtime_testing_script_refactoring_plan.md)** - Complete implementation plan and progress tracking for the runtime testing system refactoring. Documents the modular architecture design, dual constructor support, and comprehensive testing approach that resulted in 50+ passing tests.

### Design Documentation
- **[Pipeline Runtime Testing Simplified Design](../../1_design/pipeline_runtime_testing_simplified_design.md)** - Core design document outlining the simplified architecture for pipeline runtime testing. Covers the RuntimeTester class design, ScriptExecutionSpec models, PipelineTestingSpecBuilder patterns, and the overall testing framework architecture.

### Related Design Documents
- **[Pipeline Runtime Core Engine Design](../../1_design/pipeline_runtime_core_engine_design.md)** - Foundational design for the pipeline runtime execution engine
- **[Pipeline Runtime API Design](../../1_design/pipeline_runtime_api_design.md)** - API design patterns and interfaces for runtime operations
- **[Pipeline Runtime CLI Examples](../../1_design/pipeline_runtime_cli_examples.md)** - Command-line interface design and usage examples

These documents provide comprehensive context for:
- **Architecture Decisions**: Why the modular design was chosen and how components interact
- **Implementation Details**: Technical specifications for models, builders, and testing patterns  
- **Design Evolution**: How the runtime testing system evolved from initial concepts to final implementation
- **Testing Strategy**: Comprehensive approach to validation that ensures reliability and maintainability

## Summary

The Script Runtime Tester provides practical validation by actually executing your pipeline scripts and testing their functionality, data compatibility, and end-to-end flow. This ensures your scripts work correctly in real execution scenarios, making it an essential tool for reliable pipeline development.

Happy testing! 🚀
