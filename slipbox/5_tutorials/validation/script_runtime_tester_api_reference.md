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
date of note: 2025-09-09
---

# Script Runtime Tester API Reference

## Overview

The Script Runtime Tester API provides comprehensive validation of pipeline scripts through actual execution. This reference documents the complete API for testing script functionality, data compatibility, and pipeline flows with practical examples and usage patterns.

## Core API Classes

### RuntimeTester

The main class for script runtime validation and testing.

```python
from cursus.validation.runtime import RuntimeTester

# Initialize with workspace directory (backward compatible)
tester = RuntimeTester("./test_workspace")

# Initialize with RuntimeTestingConfiguration (new approach)
from cursus.validation.runtime import RuntimeTestingConfiguration, PipelineTestingSpec

config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,  # PipelineTestingSpec instance
    test_individual_scripts=True,
    test_data_compatibility=True,
    test_pipeline_flow=True,
    use_workspace_aware=False
)
tester = RuntimeTester(config)

# Initialize with logical name matching enabled
tester = RuntimeTester("./test_workspace", enable_logical_matching=True, semantic_threshold=0.7)
```

**Constructor Parameters:**
- `config_or_workspace_dir`: Either a `RuntimeTestingConfiguration` object or a string path to workspace directory
- `enable_logical_matching`: Boolean to enable enhanced logical name matching (default: True)
- `semantic_threshold`: Float threshold for semantic matching confidence (default: 0.7)

### RuntimeTestingConfiguration

Configuration object for runtime testing settings.

```python
from cursus.validation.runtime import RuntimeTestingConfiguration

# Create configuration with custom settings
config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,  # PipelineTestingSpec instance
    test_individual_scripts=True,
    test_data_compatibility=True,
    test_pipeline_flow=True,
    use_workspace_aware=False
)

# Use configuration with tester
tester = RuntimeTester(config)
```

**Configuration Fields:**
- `pipeline_spec`: PipelineTestingSpec defining the complete pipeline testing configuration
- `test_individual_scripts`: Whether to test scripts individually first (default: True)
- `test_data_compatibility`: Whether to test data compatibility between connected scripts (default: True)
- `test_pipeline_flow`: Whether to test complete pipeline flow (default: True)
- `use_workspace_aware`: Whether to use workspace-aware project structure (default: False)

## Core Operations

### test_script_with_spec()

Tests a script using a ScriptExecutionSpec for precise control.

**Signature:**
```python
def test_script_with_spec(
    self,
    script_spec: ScriptExecutionSpec,
    main_params: Dict[str, Any]
) -> ScriptTestResult
```

**Parameters:**
- `script_spec`: ScriptExecutionSpec defining execution parameters
- `main_params`: Parameters to pass to script's main function

**Returns:** `ScriptTestResult` with execution details

**Example:**
```python
from cursus.validation.runtime import ScriptExecutionSpec, PipelineTestingSpecBuilder

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
    print(f"Execution time: {result.execution_time:.3f}s")
    print(f"Has main function: {result.has_main_function}")
else:
    print(f"❌ Script test failed: {result.error_message}")
```

### test_data_compatibility_with_specs()

Tests data compatibility between two scripts using ScriptExecutionSpecs.

**Signature:**
```python
def test_data_compatibility_with_specs(
    self,
    spec_a: ScriptExecutionSpec,
    spec_b: ScriptExecutionSpec
) -> DataCompatibilityResult
```

**Parameters:**
- `spec_a`: Execution specification for first script (data producer)
- `spec_b`: Execution specification for second script (data consumer)

**Returns:** `DataCompatibilityResult` with compatibility analysis

**Example:**
```python
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

# Test compatibility with specifications
compat_result = tester.test_data_compatibility_with_specs(spec_a, spec_b)

if compat_result.compatible:
    print("✅ Scripts are compatible with specifications!")
    print(f"Data formats: {compat_result.data_format_a} -> {compat_result.data_format_b}")
else:
    print("❌ Compatibility issues with specifications:")
    for issue in compat_result.compatibility_issues:
        print(f"  - {issue}")
```

### test_pipeline_flow_with_spec()

Tests complete pipeline flow using a PipelineTestingSpec for precise control.

**Signature:**
```python
def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]
```

**Parameters:**
- `pipeline_spec`: PipelineTestingSpec defining complete pipeline testing configuration

**Returns:** Dictionary with comprehensive pipeline test results

**Example:**
```python
from cursus.validation.runtime import PipelineTestingSpec
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_node("evaluation")
dag.add_edge("preprocessing", "training")
dag.add_edge("training", "evaluation")

# Create script specifications
script_specs = {
    "preprocessing": ScriptExecutionSpec(
        script_name="tabular_preprocessing",
        step_name="preprocessing",
        input_paths={"data_input": "./test_data/raw.csv"},
        output_paths={"data_output": "./test_output/processed"},
        environ_vars={"LABEL_FIELD": "target"},
        job_args={"job_type": "preprocessing"}
    ),
    "training": ScriptExecutionSpec(
        script_name="xgboost_training",
        step_name="training",
        input_paths={"data_input": "./test_output/processed"},
        output_paths={"model_output": "./test_output/model"},
        environ_vars={"MODEL_TYPE": "xgboost"},
        job_args={"job_type": "training"}
    ),
    "evaluation": ScriptExecutionSpec(
        script_name="model_evaluation",
        step_name="evaluation",
        input_paths={"model_input": "./test_output/model", "data_input": "./test_output/processed"},
        output_paths={"metrics_output": "./test_output/metrics"},
        environ_vars={"EVAL_METRICS": "accuracy,precision,recall"},
        job_args={"job_type": "evaluation"}
    )
}

# Create pipeline specification
pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs=script_specs,
    test_workspace_root="./test_workspace"
)

# Test pipeline with specification
results = tester.test_pipeline_flow_with_spec(pipeline_spec)

if results["pipeline_success"]:
    print("✅ Pipeline test with specification passed!")
else:
    print("❌ Pipeline test with specification failed!")
    for error in results["errors"]:
        print(f"  - {error}")

# Access detailed results
print(f"Execution order: {results.get('execution_order', 'N/A')}")
print(f"Script results: {len(results['script_results'])} scripts tested")
print(f"Data flow results: {len(results['data_flow_results'])} flows tested")
```

## Specification Management

### ScriptExecutionSpec

Model for defining script execution parameters.

```python
from cursus.validation.runtime import ScriptExecutionSpec

# Create script execution specification
spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",
    step_name="preprocessing_step",
    input_paths={"data_input": "./test_data/raw_data.csv"},
    output_paths={"data_output": "./test_output/processed"},
    environ_vars={"LABEL_FIELD": "target", "FEATURE_COLS": "feature1,feature2"},
    job_args={"job_type": "preprocessing", "batch_size": 1000},
    user_notes="Custom preprocessing configuration for testing"
)

# Save specification to file
saved_path = spec.save_to_file("./test_workspace/.specs")
print(f"Saved specification to: {saved_path}")

# Load specification from file
loaded_spec = ScriptExecutionSpec.load_from_file("tabular_preprocessing", "./test_workspace/.specs")
print(f"Loaded specification: {loaded_spec.script_name}")
print(f"Last updated: {loaded_spec.last_updated}")
print(f"User notes: {loaded_spec.user_notes}")

# Create default specification
default_spec = ScriptExecutionSpec.create_default("model_evaluation", "evaluation_step")
print(f"Created default spec for: {default_spec.script_name}")
```

**ScriptExecutionSpec Fields:**
- `script_name`: Name of the script to test (without .py extension)
- `step_name`: Step name that matches PipelineDAG node name
- `script_path`: Optional full path to script file
- `input_paths`: Dictionary of input paths for script main()
- `output_paths`: Dictionary of output paths for script main()
- `environ_vars`: Dictionary of environment variables for script main()
- `job_args`: Dictionary of job arguments for script main()
- `last_updated`: Timestamp when spec was last updated
- `user_notes`: User notes about this script configuration

### PipelineTestingSpec

Model for defining complete pipeline testing configuration.

```python
from cursus.validation.runtime import PipelineTestingSpec
from cursus.api.dag.base_dag import PipelineDAG

# Create pipeline testing specification
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("preprocessing", "training")

script_specs = {
    "preprocessing": ScriptExecutionSpec(
        script_name="data_loading",
        step_name="preprocessing",
        input_paths={"raw_data": "./data/raw"},
        output_paths={"loaded_data": "./test_output/loaded"}
    ),
    "training": ScriptExecutionSpec(
        script_name="model_training",
        step_name="training",
        input_paths={"loaded_data": "./test_output/loaded"},
        output_paths={"trained_model": "./test_output/model"}
    )
}

pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs=script_specs,
    test_workspace_root="./test_workspace",
    workspace_aware_root=None  # Optional workspace-aware project root
)

print(f"Pipeline DAG nodes: {pipeline_spec.dag.nodes}")
print(f"Script specifications: {list(pipeline_spec.script_specs.keys())}")
print(f"Test workspace root: {pipeline_spec.test_workspace_root}")
```

**PipelineTestingSpec Fields:**
- `dag`: PipelineDAG defining step dependencies and execution order
- `script_specs`: Dictionary mapping step names to ScriptExecutionSpec objects
- `test_workspace_root`: Root directory for test data and outputs
- `workspace_aware_root`: Optional workspace-aware project root

### PipelineTestingSpecBuilder

Builder for creating and managing testing specifications.

```python
from cursus.validation.runtime import PipelineTestingSpecBuilder

# Initialize builder
builder = PipelineTestingSpecBuilder("./test_workspace")

# Build pipeline spec from DAG
dag = PipelineDAG()
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("preprocessing", "training")

try:
    # This will load saved specs or create defaults
    pipeline_spec = builder.build_from_dag(dag, validate=False)
    print("✅ Built pipeline specification from DAG")
    print(f"Workspace root: {pipeline_spec.test_workspace_root}")
    print(f"Script specs: {list(pipeline_spec.script_specs.keys())}")
except ValueError as e:
    print(f"❌ Validation failed: {e}")

# Update a specific script spec
updated_spec = builder.update_script_spec(
    "preprocessing",
    input_paths={"data_input": "./custom_data/input.csv"},
    environ_vars={"LABEL_FIELD": "custom_target"}
)
print(f"✅ Updated spec for: {updated_spec.script_name}")

# Get main function parameters for a specification
main_params = builder.get_script_main_params(updated_spec)
print(f"Main parameters: {list(main_params.keys())}")

# List all saved specs
saved_specs = builder.list_saved_specs()
print(f"📂 Saved specifications: {saved_specs}")

# Get script spec by name
spec = builder.get_script_spec_by_name("preprocessing")
if spec:
    print(f"Found spec: {spec.script_name}")
else:
    print("Spec not found")
```

**PipelineTestingSpecBuilder Methods:**
- `build_from_dag(dag, validate=True)`: Build PipelineTestingSpec from DAG
- `save_script_spec(spec)`: Save ScriptExecutionSpec to local file
- `update_script_spec(node_name, **updates)`: Update specific fields in a spec
- `list_saved_specs()`: List all saved specification names
- `get_script_spec_by_name(script_name)`: Get spec by script name
- `get_script_main_params(spec)`: Get parameters for script main() function

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
```

**Example Usage:**
```python
result = tester.test_script_with_spec(spec, main_params)

print(f"Script: {result.script_name}")
print(f"Success: {result.success}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Has main function: {result.has_main_function}")
if result.error_message:
    print(f"Error: {result.error_message}")
```

### DataCompatibilityResult

Result of data compatibility test between scripts.

```python
class DataCompatibilityResult:
    """Result of data compatibility test."""
    
    script_a: str
    script_b: str
    compatible: bool
    compatibility_issues: List[str] = []
    data_format_a: Optional[str] = None
    data_format_b: Optional[str] = None
```

**Example Usage:**
```python
compat_result = tester.test_data_compatibility_with_specs(spec_a, spec_b)

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

## Enhanced Features: Logical Name Matching

### Logical Name Matching Methods

When logical name matching is enabled, additional methods become available:

```python
# Initialize with logical name matching
tester = RuntimeTester("./test_workspace", enable_logical_matching=True, semantic_threshold=0.7)

if tester.enable_logical_matching:
    print("🧠 Logical name matching is available")
```

### get_path_matches()

Get logical name matches between two script specifications.

**Signature:**
```python
def get_path_matches(
    self,
    spec_a: ScriptExecutionSpec,
    spec_b: ScriptExecutionSpec
) -> List[PathMatch]
```

**Example:**
```python
path_matches = tester.get_path_matches(spec_a, spec_b)
print(f"Found {len(path_matches)} logical name matches")

for match in path_matches:
    print(f"Match: {match.source_logical_name} -> {match.dest_logical_name}")
    print(f"Confidence: {match.confidence:.3f}")
    print(f"Match type: {match.match_type}")
```

### generate_matching_report()

Generate detailed matching report between two script specifications.

**Signature:**
```python
def generate_matching_report(
    self,
    spec_a: ScriptExecutionSpec,
    spec_b: ScriptExecutionSpec
) -> Dict[str, Any]
```

**Example:**
```python
matching_report = tester.generate_matching_report(spec_a, spec_b)

print(f"Total matches: {matching_report['total_matches']}")
print(f"Match types: {matching_report['match_types']}")
print(f"Average confidence: {matching_report['confidence_distribution']['average']:.3f}")

for recommendation in matching_report['recommendations']:
    print(f"💡 {recommendation}")
```

### validate_pipeline_logical_names()

Validate logical name compatibility across entire pipeline.

**Signature:**
```python
def validate_pipeline_logical_names(
    self,
    pipeline_spec: PipelineTestingSpec
) -> Dict[str, Any]
```

**Example:**
```python
validation_results = tester.validate_pipeline_logical_names(pipeline_spec)

print(f"Overall valid: {validation_results['overall_valid']}")
print(f"Validation rate: {validation_results['summary']['validation_rate']:.1%}")

for edge_key, edge_result in validation_results['edge_validations'].items():
    status = "✅" if edge_result['valid'] else "❌"
    print(f"{status} {edge_key}: {edge_result['matches_found']} matches")
```

### Enhanced Data Compatibility Testing

**test_data_compatibility_with_logical_matching()**

Enhanced data compatibility testing with logical name matching.

**Signature:**
```python
def test_data_compatibility_with_logical_matching(
    self,
    spec_a: ScriptExecutionSpec,
    spec_b: ScriptExecutionSpec
) -> EnhancedDataCompatibilityResult
```

**Example:**
```python
if tester.enable_logical_matching:
    enhanced_result = tester.test_data_compatibility_with_logical_matching(spec_a, spec_b)
    
    print(f"Compatible: {enhanced_result.compatible}")
    print(f"Logical matches found: {len(enhanced_result.path_matches)}")
    print(f"Matching confidence: {enhanced_result.average_confidence:.3f}")
```

## Utility Methods

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
try:
    script_path = tester._find_script_path("tabular_preprocessing")
    print(f"Script found at: {script_path}")
except FileNotFoundError as e:
    print(f"Script not found: {e}")
```

### _find_valid_output_files()

Find valid output files in a directory, excluding temporary and system files.

**Signature:**
```python
def _find_valid_output_files(
    self,
    output_dir: Path,
    min_size_bytes: int = 1
) -> List[Path]
```

**Parameters:**
- `output_dir`: Directory to search for output files
- `min_size_bytes`: Minimum file size to consider (default 1 byte)

**Returns:** List of valid output file paths, sorted by modification time (newest first)

**Example:**
```python
from pathlib import Path

output_dir = Path("./test_output/processed")
valid_files = tester._find_valid_output_files(output_dir)

print(f"Found {len(valid_files)} valid output files:")
for file_path in valid_files:
    print(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")
```

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

# Use sample data for testing
# Note: This method is primarily used internally
```

### _detect_file_format()

Detect file format from file extension.

**Signature:**
```python
def _detect_file_format(self, file_path: Path) -> str
```

**Parameters:**
- `file_path`: Path to the file to analyze

**Returns:** String representing the detected file format

**Example:**
```python
from pathlib import Path

file_path = Path("./test_output/data.csv")
format_detected = tester._detect_file_format(file_path)
print(f"Detected format: {format_detected}")
```

## Error Handling

### Common Exceptions

**FileNotFoundError**
```python
try:
    result = tester.test_script_with_spec(spec, main_params)
except FileNotFoundError as e:
    print(f"Script not found: {e}")
    
    # List available scripts
    scripts_dir = Path("src/cursus/steps/scripts")
    if scripts_dir.exists():
        available_scripts = [f.stem for f in scripts_dir.glob("*.py")]
        print(f"Available scripts: {available_scripts}")
```

**Script Execution Errors**
```python
result = tester.test_script_with_spec(spec, main_params)

if not result.success:
    print(f"Script execution failed: {result.error_message}")
    
    # Check if main function exists
    if not result.has_main_function:
        print("❌ Script missing main function")
        print("💡 Add main(input_paths, output_paths, environ_vars, job_args) function")
    else:
        print("❌ Script has main function but execution failed")
        
        # Provide specific guidance based on error
        if "timeout" in result.error_message.lower():
            print("💡 Script may be taking too long - check for infinite loops")
        elif "import" in result.error_message.lower():
            print("💡 Check for missing dependencies or import errors")
        elif "permission" in result.error_message.lower():
            print("💡 Check file permissions and directory access")
```

### Error Handling Best Practices

```python
def robust_script_testing(script_specs: List[ScriptExecutionSpec]) -> Dict[str, Any]:
    """Example of robust script testing with comprehensive error handling."""
    
    tester = RuntimeTester("./test_workspace")
    builder = PipelineTestingSpecBuilder("./test_workspace")
    
    results = {
        'successful_tests': [],
        'failed_tests': [],
        'errors': []
    }
    
    for spec in script_specs:
        try:
            # Validate specification first
            if not spec.script_name or not spec.step_name:
                results['errors'].append(f"Invalid spec: missing script_name or step_name")
                continue
            
            # Check if script exists
            try:
                script_path = tester._find_script_path(spec.script_name)
                print(f"✅ Found script: {script_path}")
            except FileNotFoundError:
                results['failed_tests'].append({
                    'script': spec.script_name,
                    'error': 'Script file not found'
                })
                continue
            
            # Get main parameters and test
            main_params = builder.get_script_main_params(spec)
            result = tester.test_script_with_spec(spec, main_params)
            
            if result.success:
                results['successful_tests'].append({
                    'script': spec.script_name,
                    'execution_time': result.execution_time
                })
                print(f"✅ {spec.script_name}: PASS ({result.execution_time:.3f}s)")
            else:
                results['failed_tests'].append({
                    'script': spec.script_name,
                    'error': result.error_message,
                    'has_main_function': result.has_main_function
                })
                print(f"❌ {spec.script_name}: FAIL - {result.error_message}")
                
        except Exception as e:
            results['errors'].append(f"Unexpected error testing {spec.script_name}: {str(e)}")
            print(f"❌ {spec.script_name}: EXCEPTION - {e}")
    
    # Generate summary
    total_tests = len(script_specs)
    successful_count = len(results['successful_tests'])
    
    print(f"\n📊 Testing Summary:")
    print(f"Total scripts: {total_tests}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {len(results['failed_tests'])}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Success rate: {successful_count/total_tests*100:.1f}%")
    
    return results
```

## Advanced Usage Patterns

### Batch Testing with Specifications

```python
def batch_testing_with_specs():
    """Advanced batch testing using specifications."""
    
    # Create multiple specifications
    specs = [
        ScriptExecutionSpec.create_default("tabular_preprocessing", "preprocessing"),
        ScriptExecutionSpec.create_default("xgboost_training", "training"),
        ScriptExecutionSpec.create_default("model_evaluation", "evaluation")
    ]
    
    tester = RuntimeTester("./test_workspace")
    builder = PipelineTestingSpecBuilder("./test_workspace")
    
    # Test all specifications
    results = robust_script_testing(specs)
    
    # Test data compatibility between successful scripts
    successful_specs = [
        spec for spec in specs 
        if any(test['script'] == spec.script_name for test in results['successful_tests'])
    ]
    
    print(f"\n🔗 Testing data compatibility between {len(successful_specs)} successful scripts...")
    
    for i in range(len(successful_specs) - 1):
        spec_a = successful_specs[i]
        spec_b = successful_specs[i + 1]
        
        try:
            compat_result = tester.test_data_compatibility_with_specs(spec_a, spec_b)
            
            if compat_result.compatible:
                print(f"✅ {spec_a.script_name} -> {spec_b.script_name}: COMPATIBLE")
            else:
                print(f"❌ {spec_a.script_name} -> {spec_b.script_name}: INCOMPATIBLE")
                for issue in compat_result.compatibility_issues:
                    print(f"    - {issue}")
                    
        except Exception as e:
            print(f"❌ Error testing compatibility {spec_a.script_name} -> {spec_b.script_name}: {e}")

# Run batch testing
batch_testing_with_specs()
```

### Pipeline Testing with Custom DAG

```python
def custom_pipeline_testing():
    """Advanced pipeline testing with custom DAG configuration."""
    
    from cursus.api.dag.base_dag import PipelineDAG
    
    # Create custom DAG
    dag = PipelineDAG()
    
    # Add nodes
    nodes = ["data_loading", "preprocessing", "feature_engineering", "training", "evaluation", "registration"]
    for node in nodes:
        dag.add_node(node)
    
    # Add edges (dependencies)
    edges = [
        ("data_loading", "preprocessing"),
        ("preprocessing", "feature_engineering"),
        ("feature_engineering", "training"),
        ("training", "evaluation"),
        ("evaluation", "registration")
    ]
    
    for src, dst in edges:
        dag.add_edge(src, dst)
    
    print(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    
    # Create specifications for each node
    script_mapping = {
        "data_loading": "data_loading_script",
        "preprocessing": "tabular_preprocessing", 
        "feature_engineering": "feature_engineering_script",
        "training": "xgboost_training",
        "evaluation": "model_evaluation",
        "registration": "model_registration"
    }
    
    script_specs = {}
    for node, script_name in script_mapping.items():
        script_specs[node] = ScriptExecutionSpec.create_default(script_name, node)
        # Customize paths for data flow
        if node == "data_loading":
            script_specs[node].input_paths = {"raw_data": "./data/raw"}
            script_specs[node].output_paths = {"loaded_data": "./test_output/loaded"}
        elif node == "preprocessing":
            script_specs[node].input_paths = {"data_input": "./test_output/loaded"}
            script_specs[node].output_paths = {"processed_data": "./test_output/processed"}
        # ... continue for other nodes
    
    # Create pipeline specification
    pipeline_spec = PipelineTestingSpec(
        dag=dag,
        script_specs=script_specs,
        test_workspace_root="./test_workspace"
    )
    
    # Test the complete pipeline
    tester = RuntimeTester("./test_workspace")
    results = tester.test_pipeline_flow_with_spec(pipeline_spec)
    
    if results["pipeline_success"]:
        print("✅ Custom pipeline test passed!")
        if "execution_order" in results:
            print(f"Execution order: {results['execution_order']}")
    else:
        print("❌ Custom pipeline test failed!")
        for error in results["errors"]:
            print(f"  - {error}")

# Run custom pipeline testing
custom_pipeline_testing()
```

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `test_script_with_spec()` | Test single script with specification | `ScriptTestResult` |
| `test_data_compatibility_with_specs()` | Test data compatibility with specifications | `DataCompatibilityResult` |
| `test_pipeline_flow_with_spec()` | Test complete pipeline with specification | `Dict[str, Any]` |
| `get_path_matches()` | Get logical name matches (if enabled) | `List[PathMatch]` |
| `generate_matching_report()` | Generate matching report (if enabled) | `Dict[str, Any]` |
| `validate_pipeline_logical_names()` | Validate pipeline logical names (if enabled) | `Dict[str, Any]` |
| `_find_script_path()` | Find script file path | `str` |
| `_find_valid_output_files()` | Find valid output files | `List[Path]` |
| `_generate_sample_data()` | Generate sample test data | `Dict[str, Any]` |
| `_detect_file_format()` | Detect file format | `str` |

### Builder Methods Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `build_from_dag()` | Build pipeline spec from DAG | `PipelineTestingSpec` |
| `save_script_spec()` | Save specification to file | `None` |
| `update_script_spec()` | Update specification fields | `ScriptExecutionSpec` |
| `list_saved_specs()` | List saved specification names | `List[str]` |
| `get_script_spec_by_name()` | Get specification by name | `Optional[ScriptExecutionSpec]` |
| `get_script_main_params()` | Get main function parameters | `Dict[str, Any]` |

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `test_individual_scripts` | True | Test scripts individually first |
| `test_data_compatibility` | True | Test data compatibility between scripts |
| `test_pipeline_flow` | True | Test complete pipeline flow |
| `use_workspace_aware` | False | Use workspace-aware project structure |
| `enable_logical_matching` | True | Enable enhanced logical name matching |
| `semantic_threshold` | 0.7 | Threshold for semantic matching confidence |

## Performance Considerations

### Optimizing Test Execution

```python
def optimized_testing_configuration():
    """Configure runtime tester for optimal performance."""
    
    # Use efficient configuration
    config = RuntimeTestingConfiguration(
        pipeline_spec=None,  # Set when available
        test_individual_scripts=True,  # Essential for debugging
        test_data_compatibility=True,  # Important for data flow
        test_pipeline_flow=False,  # Skip for individual script testing
        use_workspace_aware=False  # Simpler for testing
    )
    
    # Initialize with optimized settings
    tester = RuntimeTester(config, enable_logical_matching=False)  # Disable for speed
    
    return tester

# Use optimized configuration for large-scale testing
optimized_tester = optimized_testing_configuration()
```

### Memory Management

```python
def memory_efficient_testing():
    """Memory-efficient testing for large pipelines."""
    
    tester = RuntimeTester("./test_workspace")
    builder = PipelineTestingSpecBuilder("./test_workspace")
    
    # Test scripts in batches to manage memory
    script_names = ["script1", "script2", "script3", "script4", "script5"]
    batch_size = 2
    
    for i in range(0, len(script_names), batch_size):
        batch = script_names[i:i+batch_size]
        print(f"Testing batch: {batch}")
        
        for script_name in batch:
            try:
                spec = ScriptExecutionSpec.create_default(script_name, script_name)
                main_params = builder.get_script_main_params(spec)
                result = tester.test_script_with_spec(spec, main_params)
                
                print(f"{'✅' if result.success else '❌'} {script_name}")
                
            except Exception as e:
                print(f"❌ {script_name}: {e}")
        
        # Optional: cleanup between batches
        import gc
        gc.collect()

# Run memory-efficient testing
memory_efficient_testing()
```

## Integration Examples

### CI/CD Integration

```python
def ci_cd_runtime_validation():
    """Runtime validation suitable for CI/CD pipelines."""
    
    import sys
    import json
    from datetime import datetime
    
    # Critical scripts that must pass
    critical_scripts = [
        "tabular_preprocessing",
        "xgboost_training", 
        "model_evaluation"
    ]
    
    tester = RuntimeTester("./ci_workspace")
    builder = PipelineTestingSpecBuilder("./ci_workspace")
    
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
        try:
            spec = ScriptExecutionSpec.create_default(script_name, script_name)
            main_params = builder.get_script_main_params(spec)
            result = tester.test_script_with_spec(spec, main_params)
            
            results['script_results'][script_name] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'has_main_function': result.has_main_function,
                'error_message': result.error_message
            }
            
            if result.success:
                results['successful_scripts'] += 1
            else:
                results['failed_scripts'].append(script_name)
                
        except Exception as e:
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
    with open("runtime_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"CI/CD Runtime Validation: {results['overall_status']}")
    print(f"Scripts passed: {results['successful_scripts']}/{results['total_scripts']}")
    
    return exit_code

# Use in CI/CD pipeline
if __name__ == "__main__":
    exit_code = ci_cd_runtime_validation()
    sys.exit(exit_code)
```

### Testing Framework Integration

```python
import unittest
from cursus.validation.runtime import RuntimeTester, ScriptExecutionSpec, PipelineTestingSpecBuilder

class TestScriptRuntime(unittest.TestCase):
    """Unit tests for script runtime validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tester = RuntimeTester("./test_workspace")
        self.builder = PipelineTestingSpecBuilder("./test_workspace")
    
    def test_preprocessing_script(self):
        """Test preprocessing script execution."""
        spec = ScriptExecutionSpec.create_default("tabular_preprocessing", "preprocessing")
        main_params = self.builder.get_script_main_params(spec)
        
        result = self.tester.test_script_with_spec(spec, main_params)
        
        self.assertTrue(result.success, f"Preprocessing script failed: {result.error_message}")
        self.assertTrue(result.has_main_function, "Preprocessing script missing main function")
        self.assertGreater(result.execution_time, 0, "Execution time should be positive")
    
    def test_training_script(self):
        """Test training script execution."""
        spec = ScriptExecutionSpec.create_default("xgboost_training", "training")
        main_params = self.builder.get_script_main_params(spec)
        
        result = self.tester.test_script_with_spec(spec, main_params)
        
        self.assertTrue(result.success, f"Training script failed: {result.error_message}")
        self.assertTrue(result.has_main_function, "Training script missing main function")
    
    def test_data_compatibility(self):
        """Test data compatibility between preprocessing and training."""
        spec_a = ScriptExecutionSpec.create_default("tabular_preprocessing", "preprocessing")
        spec_b = ScriptExecutionSpec.create_default("xgboost_training", "training")
        
        # Ensure output of A feeds into input of B
        spec_b.input_paths = {"data_input": spec_a.output_paths["data_output"]}
        
        compat_result = self.tester.test_data_compatibility_with_specs(spec_a, spec_b)
        
        self.assertTrue(compat_result.compatible, 
                       f"Scripts not compatible: {compat_result.compatibility_issues}")

if __name__ == '__main__':
    unittest.main()
```

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

For practical usage examples and step-by-step tutorials, see the [Script Runtime Tester Quick Start Guide](script_runtime_tester_quick_start.md).

## Summary

The Script Runtime Tester API provides comprehensive validation of pipeline scripts through actual execution. The modern approach uses:

- **ScriptExecutionSpec** for precise control over individual script testing
- **PipelineTestingSpec** for complete pipeline validation with DAG integration
- **PipelineTestingSpecBuilder** for specification management and parameter extraction
- **Enhanced logical name matching** for intelligent data compatibility testing
- **Comprehensive error handling** for robust testing workflows

This API enables reliable pipeline development by validating that scripts work correctly in real execution scenarios, ensuring data compatibility between pipeline steps, and providing detailed feedback for debugging and optimization.
