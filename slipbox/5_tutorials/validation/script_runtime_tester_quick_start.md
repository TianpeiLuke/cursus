---
tags:
  - test
  - validation
  - runtime
  - quick_start
  - tutorial
keywords:
  - runtime testing
  - pipeline runtime testing
  - script validation
  - data compatibility testing
  - pipeline flow testing
  - logical name matching
topics:
  - runtime testing
  - validation tutorial
  - pipeline testing workflow
  - script execution validation
language: python
date of note: 2025-09-12
---

# Runtime Testing Quick Start Guide

## Overview

This 20-minute tutorial will get you up and running with the Cursus Runtime Testing system. You'll learn how to validate script functionality, test data compatibility between scripts, and verify complete pipeline flows using real scripts and contracts from the Cursus framework.

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
from cursus.validation.runtime import RuntimeTester, RuntimeTestingConfiguration, PipelineTestingSpec
from cursus.api.dag.base_dag import PipelineDAG

# First create a pipeline specification
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("preprocessing", "training")

pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs={},  # Will be filled later
    test_workspace_root="./test_workspace"
)

# Create configuration with pipeline spec
config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,
    test_individual_scripts=True,
    test_data_compatibility=True,
    test_pipeline_flow=True,
    enable_enhanced_features=True,
    enable_logical_matching=True,
    use_workspace_aware=False
)

# Initialize with configuration
tester = RuntimeTester(config)

# Or use simple initialization (backward compatible)
tester = RuntimeTester("./test_workspace")
```

## Step 2: Create Script Execution Specifications Using Real Contracts (4 minutes)

Let's create specifications for real scripts using their actual contracts:

```python
from cursus.validation.runtime import ScriptExecutionSpec
import tempfile

# Create a temporary workspace for testing
workspace_dir = tempfile.mkdtemp()

# Create specification for tabular preprocessing script
# This uses the actual contract from src/cursus/steps/contracts/tabular_preprocess_contract.py
preprocessing_spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="TabularPreprocessing_training",
    input_paths={
        "data_input": f"{workspace_dir}/input/data"  # Maps to contract's "DATA" input
    },
    output_paths={
        "data_output": f"{workspace_dir}/preprocessing/output"  # Maps to contract's "processed_data" output
    },
    environ_vars={
        # Required environment variables from the contract
        "LABEL_FIELD": "target",
        "TRAIN_RATIO": "0.7",
        "TEST_VAL_RATIO": "0.15"
    },
    job_args={
        "job_type": "training"  # Required argument for the script
    }
)

print("📋 Created tabular preprocessing specification:")
print(f"Script: {preprocessing_spec.script_name}")
print(f"Input paths: {preprocessing_spec.input_paths}")
print(f"Output paths: {preprocessing_spec.output_paths}")
print(f"Environment variables: {preprocessing_spec.environ_vars}")

# Create specification for XGBoost training script
# This uses the actual contract from src/cursus/steps/contracts/xgboost_training_contract.py
xgboost_spec = ScriptExecutionSpec(
    script_name="xgboost_training",
    step_name="XGBoostTraining_training",
    input_paths={
        # Maps to contract's expected input paths
        "input_path": f"{workspace_dir}/training/input",
        "hyperparameters_s3_uri": f"{workspace_dir}/config/hyperparameters.json"
    },
    output_paths={
        # Maps to contract's expected output paths
        "model_output": f"{workspace_dir}/model",
        "evaluation_output": f"{workspace_dir}/evaluation"
    },
    environ_vars={
        # XGBoost training uses hyperparameters.json instead of env vars
    },
    job_args={}
)

print("\n📋 Created XGBoost training specification:")
print(f"Script: {xgboost_spec.script_name}")
print(f"Input paths: {xgboost_spec.input_paths}")
print(f"Output paths: {xgboost_spec.output_paths}")
```

**What this creates:**
- **Contract-aware specifications**: Automatically uses real script contracts for accurate testing
- **Intelligent path adaptation**: Converts SageMaker container paths to local testing paths
- **Precise control**: Override contract defaults when needed for specific testing scenarios
- **Reusable specifications**: Can be saved/loaded for consistent testing
- **Proper parameter formatting**: Ready for script main() functions

## Step 3: Test Your First Script (3 minutes)

Let's test a single script using the specification:

```python
from cursus.validation.runtime import PipelineTestingSpecBuilder

# Initialize the builder to get main parameters
builder = PipelineTestingSpecBuilder(workspace_dir)

# Get main parameters for the preprocessing script
main_params = builder.get_script_main_params(preprocessing_spec)

# Test script with specification
print(f"🔍 Testing script: {preprocessing_spec.script_name}")
result = tester.test_script_with_spec(preprocessing_spec, main_params)

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
saved_path = preprocessing_spec.save_to_file(str(builder.specs_dir))
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

print(f"🔗 Testing data compatibility: {preprocessing_spec.script_name} -> {training_spec.script_name}")

# Test compatibility using specifications
compat_result = tester.test_data_compatibility_with_specs(preprocessing_spec, training_spec)

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

## Step 7: Enhanced DAG Integration with Node-to-Script Resolution (3 minutes)

The enhanced PipelineTestingSpecBuilder now includes intelligent **node-to-script resolution** that automatically maps DAG node names to actual script files:

```python
# Initialize builder with enhanced capabilities
builder = PipelineTestingSpecBuilder("./test_workspace")

# Build pipeline spec from DAG with intelligent node-to-script resolution
try:
    # This automatically resolves DAG node names to script files using:
    # 1. Registry integration (cursus.registry.step_names.get_step_name_from_spec_type)
    # 2. PascalCase to snake_case conversion with special cases (XGBoost → xgboost)
    # 3. Workspace-first file discovery with fuzzy matching fallback
    # 4. Contract-aware specification population
    built_pipeline_spec = builder.build_from_dag(dag, validate=False)
    print("✅ Built pipeline specification from DAG with node-to-script resolution")
    print(f"Workspace root: {built_pipeline_spec.test_workspace_root}")
    print(f"Script specs: {list(built_pipeline_spec.script_specs.keys())}")
    
    # Show resolved script mappings
    for node_name, spec in built_pipeline_spec.script_specs.items():
        print(f"  📋 {node_name} → {spec.script_name} (step: {spec.step_name})")
    
    # Update a specific script spec with contract-aware enhancements
    updated_spec = builder.update_script_spec(
        "preprocessing",
        input_paths={"data_input": "./custom_data/input.csv"},
        environ_vars={"LABEL_FIELD": "custom_target"}
    )
    print(f"✅ Updated spec for: {updated_spec.script_name}")
    
    # List all saved specs
    saved_specs = builder.list_saved_specs()
    print(f"📂 Saved specifications: {saved_specs}")
    
    # Demonstrate contract discovery for a specific script
    from cursus.validation.runtime import ContractDiscoveryManager
    
    contract_manager = ContractDiscoveryManager()
    discovery_result = contract_manager.discover_contract("tabular_preprocessing")
    
    if discovery_result.contract_found:
        print(f"🔍 Contract discovered for tabular_preprocessing:")
        print(f"  Contract class: {discovery_result.contract_class.__name__}")
        print(f"  Discovery method: {discovery_result.discovery_method}")
        print(f"  Entry point validated: {discovery_result.entry_point_valid}")
    else:
        print("ℹ️ No contract found, using fallback defaults")
    
except ValueError as e:
    print(f"❌ Validation failed: {e}")
    print("💡 Use builder.update_script_spec() to fix missing specifications")
```

**Enhanced Features:**
- **Intelligent Node Resolution**: Automatically maps DAG nodes to script files
- **Registry Integration**: Uses existing step name registry for canonical names
- **Special Case Handling**: Proper conversion of technical terms (XGBoost, PyTorch, etc.)
- **Contract Discovery**: Automatically finds and uses script contracts
- **Fuzzy Matching**: Fallback matching when exact names don't match
- **Path Adaptation**: Converts SageMaker paths to local testing paths

## Advanced Features: Logical Name Matching (Optional)

The runtime tester includes enhanced logical name matching for better data compatibility:

```python
# Initialize with logical name matching enabled
tester_enhanced = RuntimeTester("./test_workspace", enable_logical_matching=True, semantic_threshold=0.7)

if tester_enhanced.enable_logical_matching:
    print("🧠 Logical name matching is available")
    
    # Get path matches between specifications
    path_matches = tester_enhanced.get_path_matches(preprocessing_spec, xgboost_spec)
    print(f"Found {len(path_matches)} logical name matches")
    
    # Generate matching report
    matching_report = tester_enhanced.generate_matching_report(preprocessing_spec, xgboost_spec)
    print(f"Matching report: {matching_report}")
    
    # Validate pipeline logical names (using pipeline_spec from Step 6)
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

### Issue: Script Import/Dependency Errors

The most common issue when testing scripts is missing dependencies. Each script may have different package requirements than the cursus framework itself.

#### Common Script Dependency Error
```python
result = tester.test_script_with_spec(preprocessing_spec, main_params)
if not result.success:
    print(f"❌ Script failed: {result.error_message}")
    
# Common error messages:
# "ModuleNotFoundError: No module named 'pandas'"
# "ModuleNotFoundError: No module named 'xgboost'" 
# "ModuleNotFoundError: No module named 'sklearn'"
# "ModuleNotFoundError: No module named 'boto3'"
```

#### Diagnosing Script Dependencies
```python
def check_script_dependencies(script_name):
    """Check what dependencies a script needs."""
    import ast
    from pathlib import Path
    
    # Find the script file
    tester = RuntimeTester("./test_workspace")
    try:
        script_path = tester._find_script_path(script_name)
        print(f"📄 Analyzing script: {script_path}")
    except FileNotFoundError:
        print(f"❌ Script not found: {script_name}")
        return
    
    # Parse the script to find imports
    try:
        with open(script_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        
        # Remove duplicates and standard library modules
        unique_imports = list(set(imports))
        
        # Common third-party packages that scripts often need
        common_packages = {
            'pandas': 'pip install pandas',
            'numpy': 'pip install numpy', 
            'xgboost': 'pip install xgboost',
            'sklearn': 'pip install scikit-learn',
            'boto3': 'pip install boto3',
            'sagemaker': 'pip install sagemaker',
            'torch': 'pip install torch',
            'tensorflow': 'pip install tensorflow',
            'matplotlib': 'pip install matplotlib',
            'seaborn': 'pip install seaborn',
            'joblib': 'pip install joblib',
            'pickle': 'Built-in module',
            'json': 'Built-in module',
            'os': 'Built-in module',
            'sys': 'Built-in module',
            'pathlib': 'Built-in module'
        }
        
        print(f"📦 Found imports in {script_name}:")
        missing_packages = []
        
        for imp in sorted(unique_imports):
            try:
                __import__(imp)
                status = "✅ Available"
            except ImportError:
                status = "❌ Missing"
                missing_packages.append(imp)
            
            install_cmd = common_packages.get(imp, f'pip install {imp}')
            print(f"  {imp}: {status} ({install_cmd})")
        
        if missing_packages:
            print(f"\n💡 Install missing packages:")
            for pkg in missing_packages:
                install_cmd = common_packages.get(pkg, f'pip install {pkg}')
                if 'pip install' in install_cmd:
                    print(f"  {install_cmd}")
        else:
            print("✅ All dependencies appear to be available")
            
    except Exception as e:
        print(f"❌ Error analyzing script: {e}")

# Check dependencies for common scripts
check_script_dependencies("tabular_preprocessing")
check_script_dependencies("xgboost_training")
```

#### Installing Script Dependencies
```python
def install_common_script_dependencies():
    """Install common dependencies needed by cursus scripts."""
    import subprocess
    import sys
    
    # Common packages needed by cursus scripts
    common_packages = [
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'xgboost>=1.5.0',
        'scikit-learn>=1.0.0',
        'boto3>=1.20.0',
        'sagemaker>=2.100.0',
        'joblib>=1.1.0'
    ]
    
    print("📦 Installing common script dependencies...")
    
    for package in common_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
    
    print("✅ Dependency installation complete")

# Uncomment to install common dependencies
# install_common_script_dependencies()
```

#### Testing with Dependency Isolation
```python
def test_with_dependency_check(script_spec, main_params):
    """Test script with dependency checking."""
    tester = RuntimeTester("./test_workspace")
    
    print(f"🔍 Testing {script_spec.script_name} with dependency checking...")
    
    # First attempt
    result = tester.test_script_with_spec(script_spec, main_params)
    
    if result.success:
        print("✅ Script executed successfully")
        return result
    
    # Check if it's an import error
    if "ModuleNotFoundError" in result.error_message or "ImportError" in result.error_message:
        print("❌ Script failed due to missing dependencies:")
        print(f"Error: {result.error_message}")
        
        # Extract missing module name
        import re
        match = re.search(r"No module named '([^']+)'", result.error_message)
        if match:
            missing_module = match.group(1)
            print(f"💡 Missing module: {missing_module}")
            
            # Suggest installation
            common_installs = {
                'pandas': 'pip install pandas',
                'numpy': 'pip install numpy',
                'xgboost': 'pip install xgboost',
                'sklearn': 'pip install scikit-learn',
                'boto3': 'pip install boto3',
                'sagemaker': 'pip install sagemaker'
            }
            
            install_cmd = common_installs.get(missing_module, f'pip install {missing_module}')
            print(f"💡 Install with: {install_cmd}")
            
            # Check if it's available in a different name
            alternative_names = {
                'sklearn': 'scikit-learn',
                'cv2': 'opencv-python',
                'PIL': 'Pillow'
            }
            
            if missing_module in alternative_names:
                alt_install = f"pip install {alternative_names[missing_module]}"
                print(f"💡 Alternative install: {alt_install}")
    else:
        print(f"❌ Script failed for other reasons: {result.error_message}")
    
    return result

# Test with dependency checking
result = test_with_dependency_check(preprocessing_spec, main_params)
```

#### Creating a Requirements File for Scripts
```python
def create_script_requirements():
    """Create requirements.txt for script dependencies."""
    
    # Common script dependencies (adjust based on your scripts)
    script_requirements = [
        "# Core data processing",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "",
        "# Machine learning",
        "xgboost>=1.5.0", 
        "scikit-learn>=1.0.0",
        "joblib>=1.1.0",
        "",
        "# AWS integration",
        "boto3>=1.20.0",
        "sagemaker>=2.100.0",
        "",
        "# Visualization (optional)",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "",
        "# Deep learning (optional)",
        "torch>=1.10.0",
        "tensorflow>=2.8.0"
    ]
    
    requirements_path = Path("script_requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(script_requirements))
    
    print(f"📄 Created {requirements_path}")
    print("💡 Install with: pip install -r script_requirements.txt")

# Create requirements file
create_script_requirements()
```

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
