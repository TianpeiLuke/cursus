---
tags:
  - code
  - validation
  - runtime_testing
  - pipeline_validation
  - script_testing
keywords:
  - RuntimeTester
  - script validation
  - data compatibility
  - pipeline testing
  - logical name matching
  - semantic matching
  - topological execution
topics:
  - runtime validation
  - pipeline testing
  - script compatibility
  - data flow validation
language: python
date of note: 2025-09-12
---

# Runtime Testing

Core testing engine for validating script functionality and data transfer consistency in pipeline development. Provides comprehensive testing capabilities with intelligent path matching and pipeline flow validation.

## Overview

The `RuntimeTester` class serves as the primary interface for runtime validation of pipeline scripts and data compatibility. It integrates sophisticated logical name matching capabilities with fallback semantic matching to provide robust validation of script functionality and data flow between pipeline components.

The system supports two primary modes of operation: enhanced logical name matching using the `logical_name_matching` module when available, and semantic matching as a fallback for backward compatibility. This dual approach ensures maximum functionality while maintaining compatibility with existing systems.

Key features include script execution validation with proper parameter handling, intelligent data compatibility testing between script outputs and inputs, comprehensive pipeline flow validation with topological execution ordering, detailed error reporting with actionable feedback, and seamless integration with workspace-aware development environments.

## Classes and Methods

### Classes
- [`RuntimeTester`](#runtimetester) - Core testing engine with logical name matching integration
- [`RuntimeTestingConfiguration`](#runtimetestingconfiguration) - Configuration container for runtime testing

### Core Testing Methods
- [`test_script_with_spec`](#test_script_with_spec) - Test individual script functionality
- [`test_data_compatibility_with_specs`](#test_data_compatibility_with_specs) - Test data compatibility between scripts
- [`test_pipeline_flow_with_spec`](#test_pipeline_flow_with_spec) - Test complete pipeline flow

### Enhanced Matching Methods
- [`get_path_matches`](#get_path_matches) - Get logical name matches between scripts
- [`generate_matching_report`](#generate_matching_report) - Generate detailed matching analysis
- [`validate_pipeline_logical_names`](#validate_pipeline_logical_names) - Validate pipeline-wide logical name compatibility

## API Reference

### RuntimeTester

_class_ cursus.validation.runtime.runtime_testing.RuntimeTester(_config_or_workspace_dir_, _enable_logical_matching=True_, _semantic_threshold=0.7_)

Core testing engine that uses PipelineTestingSpecBuilder for parameter extraction and integrates with the logical name matching system for sophisticated path matching capabilities.

**Parameters:**
- **config_or_workspace_dir** (_Union[RuntimeTestingConfiguration, str]_) – Either a RuntimeTestingConfiguration object or workspace directory path for backward compatibility.
- **enable_logical_matching** (_bool_) – Enable logical name matching when available. Defaults to True.
- **semantic_threshold** (_float_) – Minimum similarity score for semantic matches. Defaults to 0.7.

```python
from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

# Create with configuration object (recommended)
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)
tester = RuntimeTester(config)

# Create with workspace directory (backward compatibility)
tester = RuntimeTester("/path/to/workspace")
```

#### test_script_with_spec

test_script_with_spec(_script_spec_, _main_params_)

Test script functionality using ScriptExecutionSpec. Validates script structure, parameter compatibility, and execution success with proper error handling and detailed feedback.

**Parameters:**
- **script_spec** (_ScriptExecutionSpec_) – Script specification containing paths and parameters.
- **main_params** (_Dict[str, Any]_) – Parameters to pass to the script's main function.

**Returns:**
- **ScriptTestResult** – Detailed test results including success status, execution time, and error messages.

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec

script_spec = ScriptExecutionSpec(
    script_name='xgboost_training',
    step_name='XGBoostTraining_training',
    input_paths={'input_path': '/data/input'},
    output_paths={'model_output': '/data/output'},
    environ_vars={'MODEL_TYPE': 'xgboost'},
    job_args={'max_depth': '6'}
)

main_params = tester.builder.get_script_main_params(script_spec)
result = tester.test_script_with_spec(script_spec, main_params)

if result.success:
    print(f"Script {result.script_name} executed successfully")
else:
    print(f"Script failed: {result.error_message}")
```

#### test_data_compatibility_with_specs

test_data_compatibility_with_specs(_spec_a_, _spec_b_)

Enhanced data compatibility testing with intelligent path matching. Uses logical name matching when available, falls back to semantic matching for backward compatibility.

**Parameters:**
- **spec_a** (_ScriptExecutionSpec_) – Source script specification (produces outputs).
- **spec_b** (_ScriptExecutionSpec_) – Destination script specification (consumes inputs).

**Returns:**
- **DataCompatibilityResult** – Compatibility test results with detailed issue reporting.

```python
preprocessing_spec = ScriptExecutionSpec(
    script_name='tabular_preprocessing',
    output_paths={'processed_data': '/preprocessing/output'}
)

training_spec = ScriptExecutionSpec(
    script_name='xgboost_training',
    input_paths={'training_data': '/training/input'}
)

result = tester.test_data_compatibility_with_specs(preprocessing_spec, training_spec)

if result.compatible:
    print("Scripts are compatible for data flow")
else:
    print(f"Compatibility issues: {result.compatibility_issues}")
```

#### test_pipeline_flow_with_spec

test_pipeline_flow_with_spec(_pipeline_spec_)

Enhanced pipeline flow testing with topological ordering and data flow chaining. Uses topological execution order when logical matching is available, falls back to original approach for backward compatibility.

**Parameters:**
- **pipeline_spec** (_PipelineTestingSpec_) – Complete pipeline specification with DAG and script specs.

**Returns:**
- **Dict[str, Any]** – Comprehensive pipeline test results including script results, data flow results, and execution order.

```python
from cursus.validation.runtime.runtime_models import PipelineTestingSpec
from cursus.api.dag.base_dag import PipelineDAG

dag = PipelineDAG(
    nodes=['preprocessing', 'training'],
    edges=[('preprocessing', 'training')]
)

pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs={'preprocessing': preprocessing_spec, 'training': training_spec},
    test_workspace_root='/workspace'
)

result = tester.test_pipeline_flow_with_spec(pipeline_spec)

if result['pipeline_success']:
    print(f"Pipeline executed successfully in order: {result['execution_order']}")
else:
    print(f"Pipeline errors: {result['errors']}")
```

#### get_path_matches

get_path_matches(_spec_a_, _spec_b_)

Get logical name matches between two script specifications. Returns detailed matching information with confidence scores and match types.

**Parameters:**
- **spec_a** (_ScriptExecutionSpec_) – Source script specification.
- **spec_b** (_ScriptExecutionSpec_) – Destination script specification.

**Returns:**
- **List[PathMatch]** – List of path matches sorted by confidence, empty list if logical matching unavailable.

```python
path_matches = tester.get_path_matches(preprocessing_spec, training_spec)

for match in path_matches:
    print(f"{match.matched_source_name} -> {match.matched_dest_name}")
    print(f"  Type: {match.match_type.value}, Confidence: {match.confidence:.3f}")
```

#### generate_matching_report

generate_matching_report(_spec_a_, _spec_b_)

Generate detailed matching report between two script specifications. Provides comprehensive analysis of path matching results with recommendations.

**Parameters:**
- **spec_a** (_ScriptExecutionSpec_) – Source script specification.
- **spec_b** (_ScriptExecutionSpec_) – Destination script specification.

**Returns:**
- **Dict[str, Any]** – Dictionary with detailed matching information and recommendations.

```python
report = tester.generate_matching_report(preprocessing_spec, training_spec)

print(f"Total matches: {report['total_matches']}")
print(f"High confidence matches: {report['high_confidence_matches']}")

for recommendation in report['recommendations']:
    print(f"Recommendation: {recommendation}")
```

#### validate_pipeline_logical_names

validate_pipeline_logical_names(_pipeline_spec_)

Validate logical name compatibility across entire pipeline. Analyzes all edges in the pipeline DAG for logical name matching compatibility.

**Parameters:**
- **pipeline_spec** (_PipelineTestingSpec_) – Complete pipeline specification.

**Returns:**
- **Dict[str, Any]** – Validation results for all edges with overall compatibility assessment.

```python
validation_results = tester.validate_pipeline_logical_names(pipeline_spec)

print(f"Overall valid: {validation_results['overall_valid']}")
print(f"Validation rate: {validation_results['summary']['validation_rate']:.1%}")

for edge_key, edge_result in validation_results['edge_validations'].items():
    status = '✅' if edge_result['valid'] else '❌'
    print(f"{edge_key}: {status} ({edge_result['matches_found']} matches)")
```

### Enhanced Testing Methods

#### test_data_compatibility_with_logical_matching

test_data_compatibility_with_logical_matching(_spec_a_, _spec_b_)

Enhanced data compatibility testing with logical name matching. Provides more sophisticated matching capabilities when logical matching is enabled.

**Parameters:**
- **spec_a** (_ScriptExecutionSpec_) – Source script specification.
- **spec_b** (_ScriptExecutionSpec_) – Destination script specification.

**Returns:**
- **EnhancedDataCompatibilityResult** – Enhanced results with path matching details, or DataCompatibilityResult if logical matching disabled.

```python
if tester.enable_logical_matching:
    enhanced_result = tester.test_data_compatibility_with_logical_matching(
        preprocessing_spec, training_spec
    )
    
    print(f"Path matches found: {len(enhanced_result.path_matches)}")
    print(f"Matching details: {enhanced_result.matching_details}")
```

#### test_pipeline_flow_with_topological_execution

test_pipeline_flow_with_topological_execution(_pipeline_spec_)

Enhanced pipeline flow testing with topological execution order. Ensures proper dependency ordering and comprehensive validation.

**Parameters:**
- **pipeline_spec** (_PipelineTestingSpec_) – Complete pipeline specification.

**Returns:**
- **Dict[str, Any]** – Comprehensive results including execution order and logical matching results.

```python
if tester.enable_logical_matching:
    result = tester.test_pipeline_flow_with_topological_execution(pipeline_spec)
    
    print(f"Execution order: {result['execution_order']}")
    print(f"Logical matching results: {result['logical_matching_results']}")
```

### Utility Methods

#### _find_script_path

_find_script_path(_script_name_)

Script discovery with workspace_dir prioritization. Searches for scripts in workspace directories first, then falls back to standard locations.

**Parameters:**
- **script_name** (_str_) – Name of the script to find (without .py extension).

**Returns:**
- **str** – Full path to the script file.

**Raises:**
- **FileNotFoundError** – If script cannot be found in any search location.

#### _find_valid_output_files

_find_valid_output_files(_output_dir_, _min_size_bytes=1_)

Find valid output files in a directory, excluding temporary and system files. Returns files sorted by modification time.

**Parameters:**
- **output_dir** (_Path_) – Directory to search for output files.
- **min_size_bytes** (_int_) – Minimum file size to consider. Defaults to 1.

**Returns:**
- **List[Path]** – List of valid output file paths, sorted by modification time (newest first).

#### _detect_file_format

_detect_file_format(_file_path_)

Detect file format from file extension. Maps common extensions to standardized format names.

**Parameters:**
- **file_path** (_Path_) – Path to the file to analyze.

**Returns:**
- **str** – Detected file format (e.g., 'csv', 'json', 'pickle', 'xgboost_model').

## Integration Features

### Logical Name Matching Integration

The RuntimeTester seamlessly integrates with the logical name matching system to provide sophisticated path matching capabilities:

```python
# Automatic integration when logical matching is available
tester = RuntimeTester(config, enable_logical_matching=True)

# Check integration status
if tester.enable_logical_matching:
    print("Using sophisticated PathMatcher with:")
    print("- Exact logical name matching")
    print("- Alias-to-alias matching") 
    print("- Semantic similarity matching")
    print("- Topological execution ordering")
else:
    print("Using fallback semantic matching")
```

### Backward Compatibility

The system maintains full backward compatibility with existing code:

```python
# Old-style initialization still works
tester = RuntimeTester("/workspace/directory")

# All existing methods work unchanged
result = tester.test_data_compatibility_with_specs(spec_a, spec_b)
```

### Error Handling and Feedback

The system provides clear, actionable error messages:

```python
result = tester.test_script_with_spec(script_spec, main_params)

if not result.success:
    if "requires the following input data" in result.error_message:
        print("Missing input data - check ScriptExecutionSpec paths")
    elif "missing main() function" in result.error_message:
        print("Script structure issue - ensure main() function exists")
```

## Related Documentation

- [Logical Name Matching](logical_name_matching.md) - Sophisticated path matching system
- [Runtime Models](runtime_models.md) - Data models for runtime testing
- [Runtime Spec Builder](runtime_spec_builder.md) - Parameter extraction and spec building
- [Contract Discovery](contract_discovery.md) - Script contract discovery system
- [Integration Demo](logical_name_matching_integration_demo.md) - Complete integration example
