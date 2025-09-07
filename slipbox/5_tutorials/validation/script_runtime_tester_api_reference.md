---
tags:
  - test
  - validation
  - runtime
  - api_reference
  - documentation
keywords:
  - script runtime tester API
  - pipeline runtime testing API
  - script execution validation
  - data compatibility testing API
  - pipeline flow testing API
topics:
  - script runtime testing API
  - runtime validation API reference
  - script execution validation methods
  - pipeline testing API
language: python
date of note: 2025-09-06
---

# Script Runtime Tester API Reference

## Overview

The Script Runtime Tester API provides comprehensive validation of pipeline scripts through actual execution. This reference documents the complete API for testing script functionality, data compatibility, and pipeline flows with practical examples and usage patterns.

## Core API Classes

### RuntimeTester

The main class for script runtime validation and testing.

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester

# Initialize with workspace directory (backward compatible)
tester = RuntimeTester("./test_workspace")

# Initialize with RuntimeTestingConfiguration (new approach)
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

config = RuntimeTestingConfiguration(
    workspace_dir="./test_workspace",
    timeout_seconds=300,
    enable_logging=True,
    log_level="INFO"
)
tester = RuntimeTester(config)
```

### RuntimeTestingConfiguration

Configuration object for runtime testing settings.

```python
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

# Create configuration with custom settings
config = RuntimeTestingConfiguration(
    workspace_dir="./test_workspace",
    timeout_seconds=600,  # 10 minutes timeout
    enable_logging=True,
    log_level="DEBUG",
    cleanup_after_test=True,
    preserve_outputs=False
)

# Use configuration with tester
tester = RuntimeTester(config)
```

## Core Operations

### test_script()

Tests a single script for functionality and execution success.

**Signature:**
```python
def test_script(self, script_name: str) -> ScriptTestResult
```

**Parameters:**
- `script_name`: Name of the script to test (without .py extension)

**Returns:** `ScriptTestResult` with execution details

**Example:**
```python
# Test a single script
result = tester.test_script("tabular_preprocessing")

# Check results
if result.success:
    print(f"✅ Script test passed!")
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Has main function: {'Yes' if result.has_main_function else 'No'}")
else:
    print(f"❌ Script test failed!")
    print(f"Error: {result.error_message}")

# Access detailed information
print(f"Script name: {result.script_name}")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time}")
print(f"Main function found: {result.has_main_function}")
print(f"Error message: {result.error_message}")
```

### test_script_with_spec()

Tests a script using a custom ScriptExecutionSpec for precise control.

**Signature:**
```python
def test_script_with_spec(
    self,
    spec: ScriptExecutionSpec,
    main_params: Dict[str, Any]
) -> ScriptTestResult
```

**Parameters:**
- `spec`: ScriptExecutionSpec defining execution parameters
- `main_params`: Parameters to pass to script's main function

**Returns:** `ScriptTestResult` with execution details

**Example:**
```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Create custom execution specification
spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing_step",
    input_paths={"data_input": "./test_data/raw_data.csv"},
    output_paths={"data_output": "./test_output/processed"},
    environ_vars={"LABEL_FIELD": "target", "FEATURE_COLS": "feature1,feature2"},
    job_args={"job_type": "preprocessing", "batch_size": 1000}
)

# Get main function parameters
builder = PipelineTestingSpecBuilder("./test_workspace")
main_params = builder.get_script_main_params(spec)

# Test with specification
result = tester.test_script_with_spec(spec, main_params)

if result.success:
    print("✅ Script test with custom spec passed!")
else:
    print(f"❌ Script test failed: {result.error_message}")
```

### test_data_compatibility()

Tests data compatibility between two scripts.

**Signature:**
```python
def test_data_compatibility(
    self,
    script_a: str,
    script_b: str,
    sample_data: Dict[str, Any]
) -> DataCompatibilityResult
```

**Parameters:**
- `script_a`: First script (data producer)
- `script_b`: Second script (data consumer)
- `sample_data`: Sample data for testing

**Returns:** `DataCompatibilityResult` with compatibility analysis

**Example:**
```python
# Generate sample data
sample_data = tester._generate_sample_data()

# Test compatibility between scripts
compat_result = tester.test_data_compatibility(
    "tabular_preprocessing",
    "xgboost_training",
    sample_data
)

if compat_result.compatible:
    print("✅ Scripts are data compatible!")
    print(f"Data formats: {compat_result.data_format_a} -> {compat_result.data_format_b}")
else:
    print("❌ Scripts have compatibility issues!")
    print("Issues found:")
    for issue in compat_result.compatibility_issues:
        print(f"  - {issue}")

# Access detailed compatibility information
print(f"Script A: {compat_result.script_a}")
print(f"Script B: {compat_result.script_b}")
print(f"Compatible: {compat_result.compatible}")
print(f"Data format A: {compat_result.data_format_a}")
print(f"Data format B: {compat_result.data_format_b}")
print(f"Compatibility issues: {compat_result.compatibility_issues}")
```

### test_data_compatibility_with_specs()

Tests data compatibility using custom ScriptExecutionSpecs.

**Signature:**
```python
def test_data_compatibility_with_specs(
    self,
    spec_a: ScriptExecutionSpec,
    spec_b: ScriptExecutionSpec,
    main_params_a: Dict[str, Any],
    main_params_b: Dict[str, Any]
) -> DataCompatibilityResult
```

**Parameters:**
- `spec_a`: Execution specification for first script
- `spec_b`: Execution specification for second script
- `main_params_a`: Main function parameters for first script
- `main_params_b`: Main function parameters for second script

**Returns:** `DataCompatibilityResult` with compatibility analysis

**Example:**
```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Create specifications for both scripts
spec_a = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing",
    input_paths={"data_input": "./test_data/raw.csv"},
    output_paths={"data_output": "./test_output/processed"},
    environ_vars={"LABEL_FIELD": "target"},
    job_args={"job_type": "preprocessing"}
)

spec_b = ScriptExecutionSpec(
    script_name="xgboost_training",
    step_name="training",
    input_paths={"data_input": "./test_output/processed"},
    output_paths={"model_output": "./test_output/model"},
    environ_vars={"MODEL_TYPE": "xgboost"},
    job_args={"job_type": "training"}
)

# Get main parameters
builder = PipelineTestingSpecBuilder("./test_workspace")
main_params_a = builder.get_script_main_params(spec_a)
main_params_b = builder.get_script_main_params(spec_b)

# Test compatibility with specifications
compat_result = tester.test_data_compatibility_with_specs(
    spec_a, spec_b, main_params_a, main_params_b
)

if compat_result.compatible:
    print("✅ Scripts are compatible with custom specifications!")
else:
    print("❌ Compatibility issues with custom specifications:")
    for issue in compat_result.compatibility_issues:
        print(f"  - {issue}")
```

### test_pipeline_flow()

Tests complete pipeline flow with multiple scripts.

**Signature:**
```python
def test_pipeline_flow(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters:**
- `pipeline_config`: Dictionary defining pipeline steps and scripts

**Returns:** Dictionary with comprehensive pipeline test results

**Example:**
```python
# Define pipeline configuration
pipeline_config = {
    "steps": {
        "data_preprocessing": {"script": "tabular_preprocessing.py"},
        "model_training": {"script": "xgboost_training.py"},
        "model_evaluation": {"script": "model_evaluation.py"}
    }
}

# Test pipeline flow
results = tester.test_pipeline_flow(pipeline_config)

# Check overall results
if results["pipeline_success"]:
    print("✅ Pipeline flow test passed!")
else:
    print("❌ Pipeline flow test failed!")
    for error in results["errors"]:
        print(f"  - {error}")

# Check individual script results
print("\n📝 Individual Script Results:")
for script_name, result in results["script_results"].items():
    status = "✅" if result.success else "❌"
    print(f"  {status} {script_name}: {'PASS' if result.success else 'FAIL'}")
    if not result.success:
        print(f"    Error: {result.error_message}")

# Check data flow results
print("\n🔗 Data Flow Results:")
for flow_name, result in results["data_flow_results"].items():
    status = "✅" if result.compatible else "❌"
    print(f"  {status} {flow_name}: {'PASS' if result.compatible else 'FAIL'}")
    if result.compatibility_issues:
        for issue in result.compatibility_issues:
            print(f"    Issue: {issue}")

# Access summary statistics
total_scripts = len(results["script_results"])
successful_scripts = sum(1 for result in results["script_results"].values() if result.success)
total_flows = len(results["data_flow_results"])
successful_flows = sum(1 for result in results["data_flow_results"].values() if result.compatible)

print(f"\n📊 Pipeline Summary:")
print(f"Overall success: {results['pipeline_success']}")
print(f"Script tests: {successful_scripts}/{total_scripts} passed")
print(f"Data flow tests: {successful_flows}/{total_flows} passed")
print(f"Total errors: {len(results['errors'])}")
```

### test_pipeline_flow_with_spec()

Tests pipeline flow using a PipelineTestingSpec for precise control.

**Signature:**
```python
def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]
```

**Parameters:**
- `pipeline_spec`: PipelineTestingSpec defining complete pipeline testing configuration

**Returns:** Dictionary with comprehensive pipeline test results

**Example:**
```python
from cursus.validation.runtime.runtime_models import PipelineTestingSpec, ScriptExecutionSpec
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Create pipeline specification
builder = PipelineTestingSpecBuilder("./test_workspace")

# Define script specifications
script_specs = [
    ScriptExecutionSpec(
        script_name="tabular_preprocessing",
        step_name="preprocessing",
        input_paths={"data_input": "./test_data/raw.csv"},
        output_paths={"data_output": "./test_output/processed"},
        environ_vars={"LABEL_FIELD": "target"},
        job_args={"job_type": "preprocessing"}
    ),
    ScriptExecutionSpec(
        script_name="xgboost_training",
        step_name="training",
        input_paths={"data_input": "./test_output/processed"},
        output_paths={"model_output": "./test_output/model"},
        environ_vars={"MODEL_TYPE": "xgboost"},
        job_args={"job_type": "training"}
    )
]

# Create pipeline specification
pipeline_spec = PipelineTestingSpec(
    pipeline_name="ml_training_pipeline",
    script_specs=script_specs,
    data_flow_specs=[
        {"from_step": "preprocessing", "to_step": "training", "data_path": "./test_output/processed"}
    ]
)

# Test pipeline with specification
results = tester.test_pipeline_flow_with_spec(pipeline_spec)

if results["pipeline_success"]:
    print("✅ Pipeline test with specification passed!")
else:
    print("❌ Pipeline test with specification failed!")
    for error in results["errors"]:
        print(f"  - {error}")
```

## Specification Management

### ScriptExecutionSpec

Model for defining script execution parameters.

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec

# Create script execution specification
spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing_step",
    input_paths={"data_input": "./test_data/raw_data.csv"},
    output_paths={"data_output": "./test_output/processed"},
    environ_vars={"LABEL_FIELD": "target", "FEATURE_COLS": "feature1,feature2"},
    job_args={"job_type": "preprocessing", "batch_size": 1000}
)

# Save specification to file
saved_path = spec.save_to_file("./test_workspace/.specs")
print(f"Saved specification to: {saved_path}")

# Load specification from file
loaded_spec = ScriptExecutionSpec.load_from_file("tabular_preprocessing", "./test_workspace/.specs")
print(f"Loaded specification: {loaded_spec.script_name}")
print(f"Last updated: {loaded_spec.last_updated}")

# Create default specification
default_spec = ScriptExecutionSpec.create_default("model_evaluation")
print(f"Created default spec for: {default_spec.script_name}")
```

### PipelineTestingSpec

Model for defining complete pipeline testing configuration.

```python
from cursus.validation.runtime.runtime_models import PipelineTestingSpec, ScriptExecutionSpec

# Create pipeline testing specification
pipeline_spec = PipelineTestingSpec(
    pipeline_name="ml_training_pipeline",
    script_specs=[
        ScriptExecutionSpec(
            script_name="data_loading",
            step_name="loading",
            input_paths={"raw_data": "./data/raw"},
            output_paths={"loaded_data": "./test_output/loaded"}
        ),
        ScriptExecutionSpec(
            script_name="preprocessing",
            step_name="preprocessing",
            input_paths={"loaded_data": "./test_output/loaded"},
            output_paths={"processed_data": "./test_output/processed"}
        )
    ],
    data_flow_specs=[
        {
            "from_step": "loading",
            "to_step": "preprocessing",
            "data_path": "./test_output/loaded"
        }
    ]
)

# Save pipeline specification
saved_path = pipeline_spec.save_to_file("./test_workspace/.specs")
print(f"Saved pipeline spec to: {saved_path}")

# Load pipeline specification
loaded_pipeline = PipelineTestingSpec.load_from_file("ml_training_pipeline", "./test_workspace/.specs")
print(f"Loaded pipeline: {loaded_pipeline.pipeline_name}")
print(f"Number of scripts: {len(loaded_pipeline.script_specs)}")
```

### PipelineTestingSpecBuilder

Builder for creating and managing testing specifications.

```python
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Initialize builder
builder = PipelineTestingSpecBuilder("./test_workspace")

# Create script specification from DAG
dag_config = {
    "steps": {
        "preprocessing": {
            "script": "tabular_preprocessing.py",
            "inputs": {"data": "./data/raw.csv"},
            "outputs": {"processed": "./output/processed"}
        }
    }
}

script_specs = builder.create_script_specs_from_dag(dag_config)
print(f"Created {len(script_specs)} script specifications")

# Get main function parameters for a specification
spec = script_specs[0]
main_params = builder.get_script_main_params(spec)
print(f"Main parameters: {main_params}")

# Validate specification
validation_result = builder.validate_spec(spec)
if validation_result["valid"]:
    print("✅ Specification is valid")
else:
    print("❌ Specification validation failed:")
    for error in validation_result["errors"]:
        print(f"  - {error}")

# Create pipeline specification from DAG
pipeline_spec = builder.create_pipeline_spec_from_dag("test_pipeline", dag_config)
print(f"Created pipeline specification: {pipeline_spec.pipeline_name}")
```

## Data Models

### ScriptTestResult

Result of a single script test.

```python
class ScriptTestResult:
    """Result of script execution test."""
    
    script_name: str
    success: bool
    execution_time: float
    has_main_function: bool
    error_message: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation of test result."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
```

### DataCompatibilityResult

Result of data compatibility test between scripts.

```python
class DataCompatibilityResult:
    """Result of data compatibility test."""
    
    script_a: str
    script_b: str
    compatible: bool
    data_format_a: Optional[str] = None
    data_format_b: Optional[str] = None
    compatibility_issues: List[str] = []
    
    def __str__(self) -> str:
        """String representation of compatibility result."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
```

### RuntimeTestingConfiguration

Configuration for runtime testing behavior.

```python
class RuntimeTestingConfiguration:
    """Configuration for runtime testing."""
    
    workspace_dir: str
    timeout_seconds: int = 300
    enable_logging: bool = True
    log_level: str = "INFO"
    cleanup_after_test: bool = True
    preserve_outputs: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuntimeTestingConfiguration':
        """Create from dictionary."""
```

## Command Line Interface

### CLI Commands

The runtime tester provides a comprehensive CLI through the `cursus runtime` command group.

#### test-script

Test a single script for functionality.

```bash
# Basic script test
cursus runtime test-script tabular_preprocessing --workspace-dir ./test_workspace

# Test with JSON output
cursus runtime test-script tabular_preprocessing --output-format json

# Test with custom timeout
cursus runtime test-script tabular_preprocessing --timeout 600

# Example output
Script Test Results:
==================
Script: tabular_preprocessing
Status: ✅ PASS
Execution Time: 2.345s
Has Main Function: Yes
```

#### test-compatibility

Test data compatibility between two scripts.

```bash
# Basic compatibility test
cursus runtime test-compatibility tabular_preprocessing xgboost_training --workspace-dir ./test_workspace

# Test with JSON output
cursus runtime test-compatibility tabular_preprocessing xgboost_training --output-format json

# Example output
Data Compatibility Test Results:
Script A: tabular_preprocessing
Script B: xgboost_training
Status: ✅ COMPATIBLE
Data Format A: CSV
Data Format B: CSV
Issues: None
```

#### test-pipeline

Test complete pipeline flow with multiple scripts.

```bash
# Test pipeline from configuration file
cursus runtime test-pipeline pipeline_config.json --workspace-dir ./test_workspace

# Test with JSON output
cursus runtime test-pipeline pipeline_config.json --output-format json

# Example pipeline_config.json
{
  "steps": {
    "preprocessing": {"script": "tabular_preprocessing.py"},
    "training": {"script": "xgboost_training.py"},
    "evaluation": {"script": "model_evaluation.py"}
  }
}

# Example output
Pipeline Flow Test Results:
Pipeline Status: ✅ PASS
Total Scripts: 3
Successful Scripts: 3/3
Total Data Flows: 2
Successful Data Flows: 2/2
Errors: None
```

## Utility Methods

### _generate_sample_data()

Generates sample data for testing purposes.

**Signature:**
```python
def _generate_sample_data(self) -> Dict[str, Any]
```

**Returns:** Dictionary containing sample data for testing

**Example:**
```python
# Generate sample data
sample_data = tester._generate_sample_data()
print(f"Sample data keys: {list(sample_data.keys())}")
print(f"Sample data: {sample_data}")

# Use sample data for compatibility testing
compat_result = tester.test_data_compatibility(
    "script_a", "script_b", sample_data
)
```

### _find_script_path()

Finds the full path to a script file.

**Signature:**
```python
def _find_script_path(self, script_name: str) -> str
```

**Parameters:**
- `script_name`: Name of the script to find

**Returns:** Full path to the script file

**Raises:** `FileNotFoundError` if script is not found

**Example:**
```python
# Find script path
try:
    script_path = tester._find_script_path("tabular_preprocessing")
    print(f"Script found at: {script_path}")
except FileNotFoundError as e:
    print(f"Script not found: {e}")
    
    # Check common locations
    possible_paths = [
        "src/cursus/steps/scripts/tabular_preprocessing.py",
        "scripts/tabular_preprocessing.py",
        "dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
    ]
    for path in possible_paths:
        if Path(path).exists():
            print(f"Found at: {path}")
```

### _setup_test_environment()

Sets up the test environment for script execution.

**Signature:**
```python
def _setup_test_environment(self, script_name: str) -> Dict[str, Any]
```

**Parameters:**
- `script_name`: Name of the script being tested

**Returns:** Dictionary with test environment configuration

**Example:**
```python
# Setup test environment
env_config = tester._setup_test_environment("tabular_preprocessing")
print(f"Test directory: {env_config['test_dir']}")
print(f"Input paths: {env_config['input_paths']}")
print(f"Output paths: {env_config['output_paths']}")
```

### _cleanup_test_environment()

Cleans up test environment after script execution.

**Signature:**
```python
def _cleanup_test_environment(self, test_dir: str) -> None
```

**Parameters:**
- `test_dir`: Directory to clean up

**Example:**
```python
# Manual cleanup (usually done automatically)
test_dir = "./test_workspace/test_tabular_preprocessing"
tester._cleanup_test_environment(test_dir)
print(f"Cleaned up test directory: {test_dir}")
```

## Error Handling

### Common Exceptions

**ScriptNotFoundError**
```python
from cursus.validation.runtime.runtime_testing import RuntimeTester

try:
    tester = RuntimeTester("./test_workspace")
    result = tester.test_script("nonexistent_script")
except FileNotFoundError as e:
    print(f"Script not found: {e}")
    
    # List available scripts
    scripts_dir = Path("src/cursus/steps/scripts")
    if scripts_dir.exists():
        available_scripts = [f.stem for f in scripts_dir.glob("*.py")]
        print(f"Available scripts: {available_scripts}")
```

**ScriptExecutionError**
```python
try:
    result = tester.test_script("problematic_script")
    if not result.success:
        print(f"Script execution failed: {result.error_message}")
        
        # Check if main function exists
        if not result.has_main_function:
            print("❌ Script missing main function")
        else:
            print("❌ Script has main function but execution failed")
            
except Exception as e:
    print(f"Unexpected error during script testing: {e}")
```

**TimeoutError**
```python
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

# Set shorter timeout for testing
config = RuntimeTestingConfiguration(
    workspace_dir="./test_workspace",
    timeout_seconds=30  # 30 seconds timeout
)
tester = RuntimeTester(config)

try:
    result = tester.test_script("long_running_script")
    if not result.success and "timeout" in result.error_message.lower():
        print("❌ Script execution timed out")
        print("💡 Consider increasing timeout_seconds in configuration")
except Exception as e:
    print(f"Error during script execution: {e}")
```

### Error Handling Best Practices

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration
from pathlib import Path

def robust_runtime_testing_workflow():
    """Example of robust runtime testing with comprehensive error handling."""
    
    try:
        # Initialize with error handling
        workspace_dir = "./test_workspace"
        if not Path(workspace_dir).exists():
            Path(workspace_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created workspace directory: {workspace_dir}")
        
        # Configure tester with reasonable settings
        config = RuntimeTestingConfiguration(
            workspace_dir=workspace_dir,
            timeout_seconds=300,
            enable_logging=True,
            log_level="INFO"
        )
        
        tester = RuntimeTester(config)
        print("✅ Runtime tester initialized successfully")
        
        # Test scripts with error handling
        test_scripts = ["tabular_preprocessing", "xgboost_training", "model_evaluation"]
        results = {}
        
        for script_name in test_scripts:
            print(f"\n🔍 Testing script: {script_name}")
            
            try:
                # Check if script exists first
                script_path = tester._find_script_path(script_name)
                print(f"  Found script at: {script_path}")
                
                # Test script execution
                result = tester.test_script(script_name)
                results[script_name] = result
                
                if result.success:
                    print(f"  ✅ {script_name}: PASS ({result.execution_time:.3f}s)")
                else:
                    print(f"  ❌ {script_name}: FAIL - {result.error_message}")
                    
                    # Provide specific guidance based on error
                    if not result.has_main_function:
                        print("    💡 Add main(input_paths, output_paths, environ_vars, job_args) function")
                    elif "timeout" in result.error_message.lower():
                        print("    💡 Script may be taking too long - check for infinite loops")
                    elif "import" in result.error_message.lower():
                        print("    💡 Check for missing dependencies or import errors")
                        
            except FileNotFoundError:
                print(f"  ❌ {script_name}: Script file not found")
                print("    💡 Check script name and location")
                results[script_name] = None
                
            except Exception as e:
                print(f"  ❌ {script_name}: Unexpected error - {e}")
                results[script_name] = None
        
        # Test data compatibility with error handling
        print(f"\n🔗 Testing data compatibility...")
        successful_scripts = [name for name, result in results.items() 
                            if result and result.success]
        
        if len(successful_scripts) >= 2:
            try:
                sample_data = tester._generate_sample_data()
                
                for i in range(len(successful_scripts) - 1):
                    script_a = successful_scripts[i]
                    script_b = successful_scripts[i + 1]
                    
                    print(f"  Testing: {script_a} -> {script_b}")
                    
                    compat_result = tester.test_data_compatibility(
                        script_a, script_b, sample_data
                    )
                    
                    if compat_result.compatible:
                        print(f"    ✅ Compatible")
                    else:
                        print(f"    ❌ Compatibility issues:")
                        for issue in compat_result.compatibility_issues:
                            print(f"      - {issue}")
                            
            except Exception as e:
                print(f"  ❌ Data compatibility testing failed: {e}")
        else:
            print("  ⚠️ Need at least 2 successful scripts for compatibility testing")
        
        # Generate summary
        total_scripts = len(test_scripts)
        successful_scripts_count = len(successful_scripts)
        
        print(f"\n📊 Runtime Testing Summary:")
        print(f"Total scripts tested: {total_scripts}")
        print(f"Successful scripts: {successful_scripts_count}/{total_scripts}")
        print(f"Success rate: {successful_scripts_count/total_scripts*100:.1f}%")
        
        return {
            'success': successful_scripts_count > 0,
            'total_scripts': total_scripts,
            'successful_scripts': successful_scripts_count,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Runtime testing workflow failed: {e}")
        return {'success': False, 'error': str(e)}

# Run robust testing workflow
workflow_result = robust_runtime_testing_workflow()
```

## Advanced Usage

### Custom Configuration Patterns

```python
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration
from cursus.validation.runtime.runtime_testing import RuntimeTester

# Development configuration (fast, verbose)
dev_config = RuntimeTestingConfiguration(
    workspace_dir="./dev_workspace",
    timeout_seconds=60,
    enable_logging=True,
    log_level="DEBUG",
    cleanup_after_test=False,  # Keep outputs for debugging
    preserve_outputs=True
)

# Production configuration (robust, clean)
prod_config = RuntimeTestingConfiguration(
    workspace_dir="./prod_workspace",
    timeout_seconds=600,
    enable_logging=True,
    log_level="INFO",
    cleanup_after_test=True,
    preserve_outputs=False
)

# CI/CD configuration (strict, fast)
ci_config = RuntimeTestingConfiguration(
    workspace_dir="./ci_workspace",
    timeout_seconds=300,
    enable_logging=True,
    log_level="WARNING",
    cleanup_after_test=True,
    preserve_outputs=False
)

# Use different configurations
dev_tester = RuntimeTester(dev_config)
prod_tester = RuntimeTester(prod_config)
ci_tester = RuntimeTester(ci_config)
```

### Batch Testing Operations

```python
def batch_script_testing(script_names: List[str], workspace_dir: str = "./test_workspace"):
    """Test multiple scripts in batch with comprehensive reporting."""
    
    tester = RuntimeTester(workspace_dir)
    results = {}
    
    print(f"🔄 Batch testing {len(script_names)} scripts...")
    
    for i, script_name in enumerate(script_names, 1):
        print(f"\n[{i}/{len(script_names)}] Testing {script_name}...")
        
        try:
            result = tester.test_script(script_name)
            results[script_name] = result
            
            status = "✅ PASS" if result.success else "❌ FAIL"
            time_str = f"({result.execution_time:.3f}s)" if result.success else ""
            print(f"  {status} {time_str}")
            
            if not result.success:
                print(f"    Error: {result.error_message}")
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results[script_name] = None
    
    # Generate batch summary
    successful = sum(1 for result in results.values() if result and result.success)
    total = len(script_names)
    
    print(f"\n📊 Batch Testing Summary:")
    print(f"Scripts tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Success rate: {successful/total*100:.1f}%")
    
    return results

# Example usage
scripts_to_test = [
    "tabular_preprocessing",
    "xgboost_training", 
    "model_evaluation",
    "data_validation"
]

batch_results = batch_script_testing(scripts_to_test)
```

### Pipeline Testing Workflows

```python
def comprehensive_pipeline_testing():
    """Comprehensive pipeline testing with multiple validation approaches."""
    
    tester = RuntimeTester("./test_workspace")
    
    # Define test pipeline
    pipeline_config = {
        "steps": {
            "data_loading": {"script": "data_loading.py"},
            "preprocessing": {"script": "tabular_preprocessing.py"},
            "training": {"script": "xgboost_training.py"},
            "evaluation": {"script": "model_evaluation.py"},
            "registration": {"script": "model_registration.py"}
        }
    }
    
    print("🚀 Comprehensive Pipeline Testing")
    
    # 1. Test individual scripts first
    print("\n1️⃣ Individual Script Testing:")
    script_results = {}
    for step_name, step_config in pipeline_config["steps"].items():
        script_name = step_config["script"].replace(".py", "")
        print(f"  Testing {script_name}...")
        
        try:
            result = tester.test_script(script_name)
            script_results[script_name] = result
            
            if result.success:
                print(f"    ✅ PASS ({result.execution_time:.3f}s)")
            else:
                print(f"    ❌ FAIL: {result.error_message}")
        except Exception as e:
            print(f"    ❌ EXCEPTION: {e}")
            script_results[script_name] = None
    
    # 2. Test data compatibility between adjacent scripts
    print("\n2️⃣ Data Compatibility Testing:")
    script_names = list(script_results.keys())
    compatibility_results = {}
    
    for i in range(len(script_names) - 1):
        script_a = script_names[i]
        script_b = script_names[i + 1]
        
        print(f"  Testing {script_a} -> {script_b}...")
        
        # Only test if both scripts passed individual testing
        if (script_results[script_a] and script_results[script_a].success and
            script_results[script_b] and script_results[script_b].success):
            
            try:
                sample_data = tester._generate_sample_data()
                compat_result = tester.test_data_compatibility(script_a, script_b, sample_data)
                compatibility_results[f"{script_a}->{script_b}"] = compat_result
                
                if compat_result.compatible:
                    print(f"    ✅ COMPATIBLE")
                else:
                    print(f"    ❌ INCOMPATIBLE:")
                    for issue in compat_result.compatibility_issues:
                        print(f"      - {issue}")
            except Exception as e:
                print(f"    ❌ EXCEPTION: {e}")
                compatibility_results[f"{script_a}->{script_b}"] = None
        else:
            print(f"    ⏭️ SKIPPED (prerequisite scripts failed)")
    
    # 3. Test complete pipeline flow
    print("\n3️⃣ Complete Pipeline Flow Testing:")
    try:
        pipeline_results = tester.test_pipeline_flow(pipeline_config)
        
        if pipeline_results["pipeline_success"]:
            print("  ✅ PIPELINE FLOW PASSED")
        else:
            print("  ❌ PIPELINE FLOW FAILED")
            for error in pipeline_results["errors"]:
                print(f"    - {error}")
    except Exception as e:
        print(f"  ❌ PIPELINE FLOW EXCEPTION: {e}")
        pipeline_results = None
    
    # 4. Generate comprehensive summary
    print("\n📊 Comprehensive Testing Summary:")
    
    # Script testing summary
    successful_scripts = sum(1 for result in script_results.values() if result and result.success)
    total_scripts = len(script_results)
    print(f"Individual Scripts: {successful_scripts}/{total_scripts} passed ({successful_scripts/total_scripts*100:.1f}%)")
    
    # Compatibility testing summary
    successful_compat = sum(1 for result in compatibility_results.values() if result and result.compatible)
    total_compat = len(compatibility_results)
    if total_compat > 0:
        print(f"Data Compatibility: {successful_compat}/{total_compat} passed ({successful_compat/total_compat*100:.1f}%)")
    
    # Pipeline flow summary
    if pipeline_results:
        pipeline_status = "PASSED" if pipeline_results["pipeline_success"] else "FAILED"
        print(f"Pipeline Flow: {pipeline_status}")
    
    return {
        'script_results': script_results,
        'compatibility_results': compatibility_results,
        'pipeline_results': pipeline_results
    }

# Run comprehensive testing
comprehensive_results = comprehensive_pipeline_testing()
```

### Integration with CI/CD

```python
def ci_cd_runtime_validation():
    """Runtime validation suitable for CI/CD pipelines."""
    
    import sys
    import json
    from pathlib import Path
    
    # CI/CD configuration
    config = RuntimeTestingConfiguration(
        workspace_dir="./ci_workspace",
        timeout_seconds=300,
        enable_logging=True,
        log_level="INFO",
        cleanup_after_test=True,
        preserve_outputs=False
    )
    
    tester = RuntimeTester(config)
    
    # Critical scripts that must pass
    critical_scripts = [
        "tabular_preprocessing",
        "xgboost_training", 
        "model_evaluation"
    ]
    
    print("🔍 CI/CD Runtime Validation")
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_scripts': len(critical_scripts),
        'successful_scripts': 0,
        'failed_scripts': [],
        'script_results': {},
        'overall_status': 'UNKNOWN'
    }
    
    # Test critical scripts
    for script_name in critical_scripts:
        print(f"\n🔍 Testing {script_name}...")
        
        try:
            result = tester.test_script(script_name)
            results['script_results'][script_name] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'has_main_function': result.has_main_function,
                'error_message': result.error_message
            }
            
            if result.success:
                print(f"  ✅ PASS ({result.execution_time:.3f}s)")
                results['successful_scripts'] += 1
            else:
                print(f"  ❌ FAIL: {result.error_message}")
                results['failed_scripts'].append(script_name)
                
        except Exception as e:
            print(f"  ❌ EXCEPTION: {e}")
            results['failed_scripts'].append(script_name)
            results['script_results'][script_name] = {
                'success': False,
                'execution_time': 0.0,
                'has_main_function': False,
                'error_message': str(e)
            }
    
    # Determine overall status
    if results['successful_scripts'] == results['total_scripts']:
        results['overall_status'] = 'PASSED'
        exit_code = 0
    else:
        results['overall_status'] = 'FAILED'
        exit_code = 1
    
    # Save results for CI/CD artifacts
    results_file = Path("ci_runtime_validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary for CI/CD logs
    print(f"\n{'='*50}")
    print(f"CI/CD RUNTIME VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Scripts Tested: {results['total_scripts']}")
    print(f"Scripts Passed: {results['successful_scripts']}")
    print(f"Scripts Failed: {len(results['failed_scripts'])}")
    
    if results['failed_scripts']:
        print(f"\nFailed Scripts:")
        for script in results['failed_scripts']:
            error_msg = results['script_results'][script]['error_message']
            print(f"  ❌ {script}: {error_msg}")
    
    print(f"\nResults saved to: {results_file}")
    
    return exit_code

# Use in CI/CD pipeline
if __name__ == "__main__":
    exit_code = ci_cd_runtime_validation()
    sys.exit(exit_code)
```

## Performance Considerations

### Optimizing Test Execution

```python
def optimized_runtime_testing():
    """Optimized runtime testing for large codebases."""
    
    # Use faster configuration for bulk testing
    config = RuntimeTestingConfiguration(
        workspace_dir="./fast_workspace",
        timeout_seconds=120,  # Shorter timeout
        enable_logging=False,  # Disable verbose logging
        cleanup_after_test=True,
        preserve_outputs=False
    )
    
    tester = RuntimeTester(config)
    
    # Parallel testing approach (conceptual - actual implementation would use threading/multiprocessing)
    scripts_to_test = [
        "script_a", "script_b", "script_c", "script_d", "script_e"
    ]
    
    print(f"⚡ Optimized testing of {len(scripts_to_test)} scripts")
    
    # Test in batches for better resource management
    batch_size = 3
    batches = [scripts_to_test[i:i+batch_size] for i in range(0, len(scripts_to_test), batch_size)]
    
    all_results = {}
    
    for batch_num, batch in enumerate(batches, 1):
        print(f"\n📦 Processing batch {batch_num}/{len(batches)}: {batch}")
        
        batch_results = {}
        for script_name in batch:
            try:
                result = tester.test_script(script_name)
                batch_results[script_name] = result
                
                status = "✅" if result.success else "❌"
                print(f"  {status} {script_name}")
                
            except Exception as e:
                print(f"  ❌ {script_name}: {e}")
                batch_results[script_name] = None
        
        all_results.update(batch_results)
        
        # Brief batch summary
        successful = sum(1 for result in batch_results.values() if result and result.success)
        print(f"  Batch {batch_num} summary: {successful}/{len(batch)} passed")
    
    # Final summary
    total_successful = sum(1 for result in all_results.values() if result and result.success)
    total_scripts = len(all_results)
    
    print(f"\n⚡ Optimized Testing Complete:")
    print(f"Total scripts: {total_scripts}")
    print(f"Successful: {total_successful}")
    print(f"Success rate: {total_successful/total_scripts*100:.1f}%")
    
    return all_results

# Run optimized testing
optimized_results = optimized_runtime_testing()
```

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `test_script()` | Test single script functionality | `ScriptTestResult` |
| `test_script_with_spec()` | Test script with custom specification | `ScriptTestResult` |
| `test_data_compatibility()` | Test data compatibility between scripts | `DataCompatibilityResult` |
| `test_data_compatibility_with_specs()` | Test compatibility with custom specs | `DataCompatibilityResult` |
| `test_pipeline_flow()` | Test complete pipeline flow | `Dict[str, Any]` |
| `test_pipeline_flow_with_spec()` | Test pipeline with custom specification | `Dict[str, Any]` |
| `_generate_sample_data()` | Generate sample test data | `Dict[str, Any]` |
| `_find_script_path()` | Find script file path | `str` |
| `_setup_test_environment()` | Setup test environment | `Dict[str, Any]` |
| `_cleanup_test_environment()` | Cleanup test environment | `None` |

### CLI Commands Summary

| Command | Purpose | Example |
|---------|---------|---------|
| `cursus runtime test-script` | Test single script | `cursus runtime test-script tabular_preprocessing` |
| `cursus runtime test-compatibility` | Test script compatibility | `cursus runtime test-compatibility script_a script_b` |
| `cursus runtime test-pipeline` | Test pipeline flow | `cursus runtime test-pipeline pipeline_config.json` |

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `workspace_dir` | Required | Directory for test execution |
| `timeout_seconds` | 300 | Script execution timeout |
| `enable_logging` | True | Enable logging output |
| `log_level` | "INFO" | Logging verbosity level |
| `cleanup_after_test` | True | Clean up test files after execution |
| `preserve_outputs` | False | Keep script outputs for inspection |

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

For additional examples and practical usage patterns, see the [Script Runtime Tester Quick Start Guide](script_runtime_tester_quick_start.md).
