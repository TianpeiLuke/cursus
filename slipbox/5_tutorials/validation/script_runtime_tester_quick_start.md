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
date of note: 2025-09-09
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
from cursus.validation.runtime import RuntimeTester

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

### Advanced Initialization with Configuration

For more control, you can use the RuntimeTestingConfiguration:

```python
from cursus.validation.runtime import RuntimeTester, RuntimeTestingConfiguration

# Create configuration with custom settings
config = RuntimeTestingConfiguration(
    pipeline_spec=None,  # Will be set later
    test_individual_scripts=True,
    test_data_compatibility=True,
    test_pipeline_flow=True,
    use_workspace_aware=False
)

# Initialize with configuration (when you have a pipeline spec)
# tester = RuntimeTester(config)

# For now, use simple initialization
tester = RuntimeTester("./test_workspace")
```

## Step 2: Create Script Execution Specifications (3 minutes)

The modern approach uses ScriptExecutionSpec to define how scripts should be executed:

```python
from cursus.validation.runtime import ScriptExecutionSpec, PipelineTestingSpecBuilder

# Create a script execution specification based on actual tabular_preprocessing contract
script_spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing_step",
    input_paths={"DATA": "./test_data/input/data"},
    output_paths={"processed_data": "./test_output/processed"},
    environ_vars={
        "LABEL_FIELD": "target",
        "TRAIN_RATIO": "0.7",
        "TEST_VAL_RATIO": "0.15",
        "CATEGORICAL_COLUMNS": "category1,category2",
        "NUMERICAL_COLUMNS": "feature1,feature2,feature3"
    },
    job_args={"job_type": "training"}
)

print("📋 Created script execution specification:")
print(f"Script: {script_spec.script_name}")
print(f"Input paths: {script_spec.input_paths}")
print(f"Output paths: {script_spec.output_paths}")

# Create builder for parameter extraction
builder = PipelineTestingSpecBuilder("./test_workspace")

# Get main function parameters
main_params = builder.get_script_main_params(script_spec)
print(f"Main parameters ready: {list(main_params.keys())}")
```

**What this creates:**
- Precise control over script execution parameters
- Reusable specifications that can be saved/loaded
- Proper parameter formatting for script main() functions

## Step 3: Test Your First Script (3 minutes)

Let's test a single script using the specification:

```python
# Test script with specification
print(f"🔍 Testing script: {script_spec.script_name}")
result = tester.test_script_with_spec(script_spec, main_params)

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
- Script can process the specified input data and produce outputs

## Step 4: Save and Load Specifications (2 minutes)

Specifications can be saved for reuse:

```python
# Save specification to file
saved_path = script_spec.save_to_file(str(builder.specs_dir))
print(f"💾 Saved specification to: {saved_path}")

# Load specification from file
loaded_spec = ScriptExecutionSpec.load_from_file("tabular_preprocessing", str(builder.specs_dir))
print(f"📂 Loaded specification:")
print(f"Last updated: {loaded_spec.last_updated}")
print(f"User notes: {loaded_spec.user_notes}")

# Create default specification for a new script
default_spec = ScriptExecutionSpec.create_default("model_evaluation", "evaluation_step")
print(f"📝 Created default spec for: {default_spec.script_name}")
```

## Step 5: Test Data Compatibility Between Scripts (4 minutes)

Now let's test if one script's output can be consumed by another script:

```python
# Create specification for XGBoost training script based on actual contract
training_spec = ScriptExecutionSpec(
    script_name="xgboost_training",
    step_name="training_step",
    input_paths={
        "input_path": "./test_output/processed",  # Uses output from preprocessing
        "hyperparameters_s3_uri": "./test_config/hyperparameters.json"
    },
    output_paths={
        "model_output": "./test_output/model",
        "evaluation_output": "./test_output/evaluation"
    },
    environ_vars={},  # XGBoost training uses hyperparameters.json instead of env vars
    job_args={"job_type": "training"}
)

print(f"🔗 Testing data compatibility: {script_spec.script_name} -> {training_spec.script_name}")

# Test compatibility using specifications
compat_result = tester.test_data_compatibility_with_specs(script_spec, training_spec)

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
- File paths and data flow work correctly

## Step 6: Test Complete Pipeline Flow (4 minutes)

Let's test an entire pipeline using DAG and specifications:

```python
from cursus.api.dag.base_dag import PipelineDAG
from cursus.validation.runtime import PipelineTestingSpec

# Create a simple DAG
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_node("evaluation")
dag.add_edge("preprocessing", "training")
dag.add_edge("training", "evaluation")

print("📊 Created pipeline DAG:")
print(f"Nodes: {dag.nodes}")
print(f"Edges: {dag.edges}")

# Create script specifications for all steps based on actual contracts
script_specs = {
    "preprocessing": ScriptExecutionSpec(
        script_name="tabular_preprocessing",
        step_name="preprocessing",
        input_paths={"DATA": "./test_data/input/data"},
        output_paths={"processed_data": "./test_output/processed"},
        environ_vars={
            "LABEL_FIELD": "target",
            "TRAIN_RATIO": "0.7",
            "TEST_VAL_RATIO": "0.15"
        },
        job_args={"job_type": "training"}
    ),
    "training": ScriptExecutionSpec(
        script_name="xgboost_training",
        step_name="training",
        input_paths={
            "input_path": "./test_output/processed",
            "hyperparameters_s3_uri": "./test_config/hyperparameters.json"
        },
        output_paths={
            "model_output": "./test_output/model",
            "evaluation_output": "./test_output/evaluation"
        },
        environ_vars={},
        job_args={"job_type": "training"}
    ),
    "evaluation": ScriptExecutionSpec(
        script_name="xgboost_model_evaluation",
        step_name="evaluation",
        input_paths={
            "model_input": "./test_output/model",
            "processed_data": "./test_output/processed"
        },
        output_paths={
            "eval_output": "./test_output/eval",
            "metrics_output": "./test_output/metrics"
        },
        environ_vars={
            "ID_FIELD": "id",
            "LABEL_FIELD": "target"
        },
        job_args={"job_type": "evaluation"}
    )
}

# Create pipeline testing specification
pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs=script_specs,
    test_workspace_root="./test_workspace"
)

print("🚀 Testing complete pipeline flow...")
pipeline_results = tester.test_pipeline_flow_with_spec(pipeline_spec)

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

# Print execution order (if available)
if "execution_order" in pipeline_results:
    print(f"\n📋 Execution Order: {pipeline_results['execution_order']}")

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

## Step 7: Using the Spec Builder for DAG Integration (3 minutes)

The PipelineTestingSpecBuilder can help create specifications from DAGs:

```python
# Initialize builder
builder = PipelineTestingSpecBuilder("./test_workspace")

# Build pipeline spec from DAG with validation
try:
    # This will load saved specs or create defaults
    built_pipeline_spec = builder.build_from_dag(dag, validate=False)
    print("✅ Built pipeline specification from DAG")
    print(f"Workspace root: {built_pipeline_spec.test_workspace_root}")
    print(f"Script specs: {list(built_pipeline_spec.script_specs.keys())}")
    
    # Update a specific script spec
    updated_spec = builder.update_script_spec(
        "preprocessing",
        input_paths={"data_input": "./custom_data/input.csv"},
        environ_vars={"LABEL_FIELD": "custom_target"}
    )
    print(f"✅ Updated spec for: {updated_spec.script_name}")
    
    # List all saved specs
    saved_specs = builder.list_saved_specs()
    print(f"📂 Saved specifications: {saved_specs}")
    
except ValueError as e:
    print(f"❌ Validation failed: {e}")
    print("💡 Use builder.update_script_spec() to fix missing specifications")
```

## Advanced Features: Logical Name Matching (Optional)

The runtime tester includes enhanced logical name matching for better data compatibility:

```python
# Initialize with logical name matching enabled
tester_enhanced = RuntimeTester("./test_workspace", enable_logical_matching=True, semantic_threshold=0.7)

if tester_enhanced.enable_logical_matching:
    print("🧠 Logical name matching is available")
    
    # Get path matches between specifications
    path_matches = tester_enhanced.get_path_matches(script_spec, training_spec)
    print(f"Found {len(path_matches)} logical name matches")
    
    # Generate matching report
    matching_report = tester_enhanced.generate_matching_report(script_spec, training_spec)
    print(f"Matching report: {matching_report}")
    
    # Validate pipeline logical names
    validation_results = tester_enhanced.validate_pipeline_logical_names(pipeline_spec)
    print(f"Pipeline validation: {validation_results['overall_valid']}")
else:
    print("ℹ️ Logical name matching not available")
```

## Common Workflows

### Daily Development Workflow

```python
def daily_script_check():
    """Daily script validation routine using specifications."""
    print("🌅 Daily Script Check")
    
    # Scripts you're currently working on
    current_scripts = [
        ("tabular_preprocessing", "preprocessing"),
        ("model_evaluation", "evaluation")
    ]
    
    tester = RuntimeTester("./test_workspace")
    builder = PipelineTestingSpecBuilder("./test_workspace")
    all_passed = True
    
    for script_name, step_name in current_scripts:
        print(f"\n🔍 Testing {script_name}...")
        
        try:
            # Load or create specification
            try:
                spec = ScriptExecutionSpec.load_from_file(script_name, str(builder.specs_dir))
            except FileNotFoundError:
                spec = ScriptExecutionSpec.create_default(script_name, step_name)
                spec.save_to_file(str(builder.specs_dir))
            
            # Get main parameters and test
            main_params = builder.get_script_main_params(spec)
            result = tester.test_script_with_spec(spec, main_params)
            
            if result.success:
                print(f"✅ {script_name}: PASS ({result.execution_time:.3f}s)")
            else:
                print(f"❌ {script_name}: FAIL - {result.error_message}")
                all_passed = False
                
        except Exception as e:
            print(f"❌ {script_name}: EXCEPTION - {e}")
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
    
    # Define critical pipeline
    dag = PipelineDAG()
    dag.add_node("preprocessing")
    dag.add_node("training")
    dag.add_node("evaluation")
    dag.add_edge("preprocessing", "training")
    dag.add_edge("training", "evaluation")
    
    # Create specifications
    script_specs = {
        "preprocessing": ScriptExecutionSpec.create_default("tabular_preprocessing", "preprocessing"),
        "training": ScriptExecutionSpec.create_default("xgboost_training", "training"),
        "evaluation": ScriptExecutionSpec.create_default("model_evaluation", "evaluation")
    }
    
    pipeline_spec = PipelineTestingSpec(
        dag=dag,
        script_specs=script_specs,
        test_workspace_root="./test_workspace"
    )
    
    tester = RuntimeTester("./test_workspace")
    
    # Test complete pipeline
    results = tester.test_pipeline_flow_with_spec(pipeline_spec)
    
    if results["pipeline_success"]:
        print("✅ Pre-commit validation passed!")
        return True
    else:
        print("❌ Pre-commit validation failed!")
        for error in results["errors"]:
            print(f"  - {error}")
        return False

# Run pre-commit validation
pre_commit_validation()
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
from pathlib import Path

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
spec_a = ScriptExecutionSpec.create_default("script_a", "step_a")
spec_b = ScriptExecutionSpec.create_default("script_b", "step_b")

tester = RuntimeTester("./test_workspace")
builder = PipelineTestingSpecBuilder("./test_workspace")

# Test scripts individually first
print("🔍 Testing Script A individually...")
main_params_a = builder.get_script_main_params(spec_a)
result_a = tester.test_script_with_spec(spec_a, main_params_a)
if result_a.success:
    print("✅ Script A executes successfully")
else:
    print(f"❌ Script A failed: {result_a.error_message}")

print("🔍 Testing Script B individually...")
main_params_b = builder.get_script_main_params(spec_b)
result_b = tester.test_script_with_spec(spec_b, main_params_b)
if result_b.success:
    print("✅ Script B executes successfully")
else:
    print(f"❌ Script B failed: {result_b.error_message}")

# If both pass individually, test compatibility
if result_a.success and result_b.success:
    print("🔗 Testing data compatibility...")
    compat_result = tester.test_data_compatibility_with_specs(spec_a, spec_b)
    
    if not compat_result.compatible:
        print("❌ Compatibility issues found:")
        for issue in compat_result.compatibility_issues:
            print(f"  - {issue}")
        
        # Check output directory for Script A
        output_dir_a = Path(spec_a.output_paths["data_output"])
        if output_dir_a.exists():
            output_files = tester._find_valid_output_files(output_dir_a)
            print(f"Script A output files: {[f.name for f in output_files]}")
        else:
            print(f"Script A output directory not found: {output_dir_a}")
```

## Next Steps

Congratulations! You've successfully:

1. ✅ Initialized the Script Runtime Tester
2. ✅ Created and used Script Execution Specifications
3. ✅ Tested individual script functionality
4. ✅ Validated data compatibility between scripts
5. ✅ Tested complete pipeline flows with DAG integration
6. ✅ Used the PipelineTestingSpecBuilder for specification management
7. ✅ Learned troubleshooting techniques

### What's Next?

1. **Explore API Reference**: Check out the [Script Runtime Tester API Reference](script_runtime_tester_api_reference.md) for complete method documentation

2. **Integrate with CI/CD**: Set up automated runtime testing in your development pipeline

3. **Advanced Features**: Explore logical name matching for enhanced data compatibility testing

4. **Custom Specifications**: Create detailed specifications for your specific pipeline requirements

5. **Performance Monitoring**: Track script execution times and optimize performance

### Additional Resources

- **[Unified Alignment Tester Quick Start](unified_alignment_tester_quick_start.md)** - Learn comprehensive alignment validation
- **[Script Development Guide](../../0_developer_guide/script_development_guide.md)** - Best practices for script development
- **[Validation Framework Guide](../../0_developer_guide/validation_framework_guide.md)** - Complete validation concepts

## References and Design Documents

For deeper technical understanding of the Script Runtime Tester implementation and design decisions, refer to these key documents:

### Implementation Plans
- **[Pipeline Runtime Testing Simplified Design](../../1_design/pipeline_runtime_testing_simplified_design.md)** - Core design document outlining the simplified architecture for pipeline runtime testing. Covers the RuntimeTester class design, ScriptExecutionSpec models, PipelineTestingSpecBuilder patterns, and the overall testing framework architecture.

### Design Documentation
- **[Pipeline Runtime Testing Simplified Design](../../1_design/pipeline_runtime_testing_simplified_design.md)** - Core design document outlining the simplified architecture for pipeline runtime testing. Covers the RuntimeTester class design, ScriptExecutionSpec models, PipelineTestingSpecBuilder patterns, and the overall testing framework architecture.

These documents provide comprehensive context for:
- **Architecture Decisions**: Why the modular design was chosen and how components interact
- **Implementation Details**: Technical specifications for models, builders, and testing patterns  
- **Design Evolution**: How the runtime testing system evolved from initial concepts to final implementation
- **Testing Strategy**: Comprehensive approach to validation that ensures reliability and maintainability

## Summary

The Script Runtime Tester provides practical validation by actually executing your pipeline scripts and testing their functionality, data compatibility, and end-to-end flow. The modern approach uses ScriptExecutionSpec for precise control and PipelineTestingSpec for complete pipeline validation, making it an essential tool for reliable pipeline development.

Happy testing! 🚀
