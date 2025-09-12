---
tags:
  - code
  - validation
  - runtime_models
  - data_models
  - pydantic
keywords:
  - ScriptExecutionSpec
  - DataCompatibilityResult
  - ScriptTestResult
  - PipelineTestingSpec
  - RuntimeTestingConfiguration
  - pydantic models
topics:
  - runtime validation models
  - data structures
  - pipeline specifications
  - test results
language: python
date of note: 2025-09-12
---

# Runtime Models

Data models for runtime testing system using Pydantic for validation and serialization. Provides structured representations of script specifications, test results, and pipeline configurations.

## Overview

The runtime models module defines the core data structures used throughout the runtime testing system. These models use Pydantic for automatic validation, serialization, and documentation generation. The models are designed to be both human-readable and machine-processable, supporting JSON serialization for persistence and API communication.

The models follow a hierarchical structure: ScriptExecutionSpec defines individual script specifications, PipelineTestingSpec combines multiple scripts with DAG structure, RuntimeTestingConfiguration provides system-wide configuration, and various result models capture test outcomes with detailed information.

Key features include automatic validation of required fields and data types, JSON serialization support for persistence and API communication, comprehensive documentation with field descriptions, factory methods for common use cases, and extensible design for future enhancements.

## Classes and Methods

### Core Models
- [`ScriptExecutionSpec`](#scriptexecutionspec) - Specification for individual script execution
- [`PipelineTestingSpec`](#pipelinetestingspec) - Complete pipeline specification with DAG
- [`RuntimeTestingConfiguration`](#runtimetestingconfiguration) - System configuration container

### Result Models
- [`ScriptTestResult`](#scripttestresult) - Results from individual script testing
- [`DataCompatibilityResult`](#datacompatibilityresult) - Results from data compatibility testing

### Builder Integration
- [`PipelineTestingSpecBuilder`](#pipelinetestingspecbuilder) - Builder for creating pipeline specifications

## API Reference

### ScriptExecutionSpec

_class_ cursus.validation.runtime.runtime_models.ScriptExecutionSpec(_script_name_, _step_name_, _input_paths_, _output_paths_, _environ_vars={}_, _job_args={}_, _**kwargs_)

Specification for individual script execution with paths, environment variables, and job arguments. Provides a complete description of how to execute a script in the runtime testing environment.

**Parameters:**
- **script_name** (_str_) – Name of the script file (without .py extension).
- **step_name** (_str_) – Name of the pipeline step this script implements.
- **input_paths** (_Dict[str, str]_) – Dictionary mapping logical input names to file paths.
- **output_paths** (_Dict[str, str]_) – Dictionary mapping logical output names to file paths.
- **environ_vars** (_Dict[str, str]_) – Environment variables to set during execution. Defaults to empty dict.
- **job_args** (_Dict[str, Any]_) – Job-specific arguments to pass to the script. Defaults to empty dict.
- **script_path** (_Optional[str]_) – Optional explicit path to script file.
- **last_updated** (_Optional[datetime]_) – Timestamp of last specification update.
- **user_notes** (_Optional[str]_) – Optional user notes about the script.

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec

# Create a script specification
script_spec = ScriptExecutionSpec(
    script_name='xgboost_training',
    step_name='XGBoostTraining_training',
    input_paths={
        'input_path': '/data/input/training.csv',
        'hyperparameters_s3_uri': '/config/hyperparameters.json'
    },
    output_paths={
        'model_output': '/data/output/model.pkl',
        'evaluation_output': '/data/output/evaluation.json'
    },
    environ_vars={
        'MODEL_TYPE': 'xgboost',
        'CUDA_VISIBLE_DEVICES': '0'
    },
    job_args={
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    }
)

print(f"Script: {script_spec.script_name}")
print(f"Inputs: {list(script_spec.input_paths.keys())}")
print(f"Outputs: {list(script_spec.output_paths.keys())}")
```

#### create_default

_classmethod_ create_default(_script_name_, _step_name_, _workspace_dir_)

Create a default ScriptExecutionSpec with standard paths. Provides a convenient factory method for creating specs with conventional directory structure.

**Parameters:**
- **script_name** (_str_) – Name of the script.
- **step_name** (_str_) – Name of the pipeline step.
- **workspace_dir** (_str_) – Base workspace directory for generating paths.

**Returns:**
- **ScriptExecutionSpec** – Script specification with default paths.

```python
import tempfile

with tempfile.TemporaryDirectory() as workspace:
    # Create default spec with conventional paths
    spec = ScriptExecutionSpec.create_default(
        script_name='preprocessing',
        step_name='DataPreprocessing_training',
        workspace_dir=workspace
    )
    
    print(f"Input paths: {spec.input_paths}")
    print(f"Output paths: {spec.output_paths}")
    # Default paths follow convention: workspace/input, workspace/output
```

#### to_dict

to_dict()

Convert specification to dictionary for serialization. Useful for JSON export, API communication, and debugging.

**Returns:**
- **Dict[str, Any]** – Dictionary representation of the specification.

```python
spec_dict = script_spec.to_dict()

# Can be serialized to JSON
import json
json_str = json.dumps(spec_dict, indent=2)
print(json_str)

# Can be reconstructed from dictionary
reconstructed_spec = ScriptExecutionSpec(**spec_dict)
```

#### from_dict

_classmethod_ from_dict(_data_)

Create ScriptExecutionSpec from dictionary. Enables reconstruction from serialized data.

**Parameters:**
- **data** (_Dict[str, Any]_) – Dictionary containing specification data.

**Returns:**
- **ScriptExecutionSpec** – Reconstructed specification object.

```python
# Load from dictionary (e.g., from JSON file)
spec_data = {
    'script_name': 'preprocessing',
    'step_name': 'DataPreprocessing_training',
    'input_paths': {'raw_data': '/input/raw.csv'},
    'output_paths': {'processed_data': '/output/processed.csv'},
    'environ_vars': {'MODE': 'training'},
    'job_args': {'batch_size': 1000}
}

spec = ScriptExecutionSpec.from_dict(spec_data)
print(f"Loaded spec for {spec.script_name}")
```

### PipelineTestingSpec

_class_ cursus.validation.runtime.runtime_models.PipelineTestingSpec(_dag_, _script_specs_, _test_workspace_root_, _**kwargs_)

Complete pipeline specification combining DAG structure with script specifications. Provides a comprehensive description of an entire pipeline for testing purposes.

**Parameters:**
- **dag** (_PipelineDAG_) – DAG structure defining pipeline topology.
- **script_specs** (_Dict[str, ScriptExecutionSpec]_) – Dictionary mapping node names to script specifications.
- **test_workspace_root** (_str_) – Root directory for test workspace.
- **pipeline_name** (_Optional[str]_) – Optional name for the pipeline.
- **description** (_Optional[str]_) – Optional description of the pipeline.
- **created_at** (_Optional[datetime]_) – Timestamp of pipeline spec creation.

```python
from cursus.validation.runtime.runtime_models import PipelineTestingSpec
from cursus.api.dag.base_dag import PipelineDAG

# Create DAG structure
dag = PipelineDAG(
    nodes=['preprocessing', 'training', 'evaluation'],
    edges=[
        ('preprocessing', 'training'),
        ('training', 'evaluation')
    ]
)

# Create script specifications
script_specs = {
    'preprocessing': ScriptExecutionSpec.create_default(
        'tabular_preprocessing', 'DataPreprocessing_training', '/workspace'
    ),
    'training': ScriptExecutionSpec.create_default(
        'xgboost_training', 'XGBoostTraining_training', '/workspace'
    ),
    'evaluation': ScriptExecutionSpec.create_default(
        'model_evaluation', 'ModelEvaluation_training', '/workspace'
    )
}

# Create pipeline specification
pipeline_spec = PipelineTestingSpec(
    dag=dag,
    script_specs=script_specs,
    test_workspace_root='/workspace',
    pipeline_name='ML Training Pipeline',
    description='Complete ML pipeline with preprocessing, training, and evaluation'
)

print(f"Pipeline: {pipeline_spec.pipeline_name}")
print(f"Nodes: {len(pipeline_spec.dag.nodes)}")
print(f"Edges: {len(pipeline_spec.dag.edges)}")
```

#### validate_consistency

validate_consistency()

Validate consistency between DAG and script specifications. Ensures all DAG nodes have corresponding script specs and identifies mismatches.

**Returns:**
- **List[str]** – List of validation errors (empty if consistent).

```python
errors = pipeline_spec.validate_consistency()

if errors:
    print("Pipeline consistency errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Pipeline specification is consistent")
```

#### get_execution_order

get_execution_order()

Get topological execution order from DAG. Provides the order in which scripts should be executed to respect dependencies.

**Returns:**
- **List[str]** – List of node names in topological order.

**Raises:**
- **ValueError** – If DAG contains cycles.

```python
try:
    execution_order = pipeline_spec.get_execution_order()
    print(f"Execution order: {' -> '.join(execution_order)}")
except ValueError as e:
    print(f"DAG error: {e}")
```

#### has_enhanced_specs

has_enhanced_specs()

Check if pipeline contains enhanced script specifications. Determines whether logical name matching features are available.

**Returns:**
- **bool** – True if any script specs are enhanced with alias support.

```python
if pipeline_spec.has_enhanced_specs():
    print("Pipeline supports logical name matching")
else:
    print("Pipeline uses basic script specifications")
```

#### get_enhanced_specs

get_enhanced_specs()

Get dictionary of enhanced script specifications only. Filters out basic specs to return only those with alias support.

**Returns:**
- **Dict[str, EnhancedScriptExecutionSpec]** – Dictionary of enhanced specifications.

```python
enhanced_specs = pipeline_spec.get_enhanced_specs()
print(f"Enhanced specs: {list(enhanced_specs.keys())}")

for name, spec in enhanced_specs.items():
    print(f"  {name}: {len(spec.input_path_specs)} input specs")
```

#### get_basic_specs

get_basic_specs()

Get dictionary of basic script specifications only. Filters out enhanced specs to return only basic ones.

**Returns:**
- **Dict[str, ScriptExecutionSpec]** – Dictionary of basic specifications.

```python
basic_specs = pipeline_spec.get_basic_specs()
print(f"Basic specs: {list(basic_specs.keys())}")
```

### RuntimeTestingConfiguration

_class_ cursus.validation.runtime.runtime_models.RuntimeTestingConfiguration(_pipeline_spec_, _enable_enhanced_features=None_, _enable_logical_matching=None_, _semantic_threshold=0.7_, _**kwargs_)

Configuration container for runtime testing system. Provides centralized configuration management with automatic feature detection.

**Parameters:**
- **pipeline_spec** (_PipelineTestingSpec_) – Pipeline specification to test.
- **enable_enhanced_features** (_Optional[bool]_) – Enable enhanced features. Auto-detected if None.
- **enable_logical_matching** (_Optional[bool]_) – Enable logical name matching. Auto-detected if None.
- **semantic_threshold** (_float_) – Threshold for semantic matching. Defaults to 0.7.
- **max_execution_time** (_Optional[int]_) – Maximum execution time per script in seconds.
- **output_file_timeout** (_Optional[int]_) – Timeout for output file detection in seconds.
- **debug_mode** (_bool_) – Enable debug logging. Defaults to False.

```python
from cursus.validation.runtime.runtime_models import RuntimeTestingConfiguration

# Create configuration with auto-detection
config = RuntimeTestingConfiguration(
    pipeline_spec=pipeline_spec,
    semantic_threshold=0.8,
    max_execution_time=300,
    debug_mode=True
)

print(f"Enhanced features: {config.enable_enhanced_features}")
print(f"Logical matching: {config.enable_logical_matching}")
print(f"Semantic threshold: {config.semantic_threshold}")
```

#### auto_detect_features

auto_detect_features()

Automatically detect available features based on pipeline specification. Analyzes the pipeline spec to determine what features should be enabled.

```python
# Features are auto-detected during initialization
config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

# Manual re-detection (usually not needed)
config.auto_detect_features()
```

### ScriptTestResult

_class_ cursus.validation.runtime.runtime_models.ScriptTestResult(_script_name_, _success_, _execution_time_, _error_message=None_, _has_main_function=None_, _**kwargs_)

Results from individual script testing with detailed execution information. Captures all relevant information about script test execution.

**Parameters:**
- **script_name** (_str_) – Name of the tested script.
- **success** (_bool_) – Whether the script test succeeded.
- **execution_time** (_float_) – Execution time in seconds.
- **error_message** (_Optional[str]_) – Error message if test failed.
- **has_main_function** (_Optional[bool]_) – Whether script has a main() function.
- **output_files_created** (_List[str]_) – List of output files created. Defaults to empty list.
- **warnings** (_List[str]_) – List of warnings generated. Defaults to empty list.
- **memory_usage** (_Optional[float]_) – Peak memory usage in MB.

```python
from cursus.validation.runtime.runtime_models import ScriptTestResult

# Successful test result
success_result = ScriptTestResult(
    script_name='preprocessing',
    success=True,
    execution_time=2.5,
    has_main_function=True,
    output_files_created=['/output/processed.csv', '/output/metadata.json'],
    memory_usage=150.2
)

# Failed test result
failure_result = ScriptTestResult(
    script_name='training',
    success=False,
    execution_time=0.1,
    error_message='Missing required input file: /input/training.csv',
    has_main_function=True,
    warnings=['Deprecated parameter used: old_param']
)

print(f"Success: {success_result.success}, Time: {success_result.execution_time:.2f}s")
print(f"Failure: {failure_result.error_message}")
```

#### is_timeout

is_timeout()

Check if the test result indicates a timeout. Analyzes error message and execution time to detect timeout conditions.

**Returns:**
- **bool** – True if the test failed due to timeout.

```python
if result.is_timeout():
    print("Script execution timed out")
    print(f"Execution time: {result.execution_time:.2f}s")
```

#### get_summary

get_summary()

Get a human-readable summary of the test result. Provides concise information about test outcome.

**Returns:**
- **str** – Summary string describing the test result.

```python
summary = result.get_summary()
print(summary)
# Example output: "preprocessing: SUCCESS (2.5s, 2 files created)"
# Example output: "training: FAILED - Missing required input file (0.1s)"
```

### DataCompatibilityResult

_class_ cursus.validation.runtime.runtime_models.DataCompatibilityResult(_script_a_, _script_b_, _compatible_, _compatibility_issues=[]_, _**kwargs_)

Results from data compatibility testing between two scripts. Captures detailed information about data flow compatibility.

**Parameters:**
- **script_a** (_str_) – Name of the source script (produces outputs).
- **script_b** (_str_) – Name of the destination script (consumes inputs).
- **compatible** (_bool_) – Whether the scripts are data-compatible.
- **compatibility_issues** (_List[str]_) – List of compatibility issues found. Defaults to empty list.
- **data_format_a** (_Optional[str]_) – Data format produced by script A.
- **data_format_b** (_Optional[str]_) – Data format expected by script B.
- **files_transferred** (_List[str]_) – List of files successfully transferred. Defaults to empty list.
- **transfer_time** (_Optional[float]_) – Time taken for data transfer in seconds.

```python
from cursus.validation.runtime.runtime_models import DataCompatibilityResult

# Compatible scripts
compatible_result = DataCompatibilityResult(
    script_a='preprocessing',
    script_b='training',
    compatible=True,
    data_format_a='csv',
    data_format_b='csv',
    files_transferred=['/output/processed.csv'],
    transfer_time=0.05
)

# Incompatible scripts
incompatible_result = DataCompatibilityResult(
    script_a='preprocessing',
    script_b='training',
    compatible=False,
    compatibility_issues=[
        'No semantic matches found between output and input paths',
        'Available outputs from preprocessing: [processed_data]',
        'Available inputs for training: [training_data, hyperparameters]'
    ],
    data_format_a='csv',
    data_format_b='unknown'
)

print(f"Compatible: {compatible_result.compatible}")
print(f"Issues: {len(incompatible_result.compatibility_issues)}")
```

#### get_issue_summary

get_issue_summary()

Get a concise summary of compatibility issues. Provides human-readable overview of problems found.

**Returns:**
- **str** – Summary of compatibility issues.

```python
if not result.compatible:
    issue_summary = result.get_issue_summary()
    print(f"Compatibility problems: {issue_summary}")
```

#### has_format_mismatch

has_format_mismatch()

Check if there's a data format mismatch between scripts. Compares data formats to identify incompatibilities.

**Returns:**
- **bool** – True if data formats are incompatible.

```python
if result.has_format_mismatch():
    print(f"Format mismatch: {result.data_format_a} -> {result.data_format_b}")
```

## Serialization and Persistence

All models support JSON serialization for persistence and API communication:

```python
import json

# Serialize to JSON
spec_json = json.dumps(script_spec.to_dict(), indent=2)

# Save to file
with open('script_spec.json', 'w') as f:
    json.dump(script_spec.to_dict(), f, indent=2)

# Load from file
with open('script_spec.json', 'r') as f:
    spec_data = json.load(f)
    loaded_spec = ScriptExecutionSpec.from_dict(spec_data)
```

## Validation and Error Handling

Models use Pydantic for automatic validation and error handling:

```python
from pydantic import ValidationError

try:
    # Invalid specification (missing required fields)
    invalid_spec = ScriptExecutionSpec(
        script_name='test',
        # Missing step_name, input_paths, output_paths
    )
except ValidationError as e:
    print(f"Validation error: {e}")
    # Shows exactly which fields are missing or invalid

# Type validation
try:
    spec = ScriptExecutionSpec(
        script_name='test',
        step_name='test_step',
        input_paths={'input': '/path'},
        output_paths={'output': '/path'},
        execution_time='invalid'  # Should be float
    )
except ValidationError as e:
    print(f"Type error: {e}")
```

## Model Extensions and Customization

Models can be extended for specific use cases:

```python
from cursus.validation.runtime.runtime_models import ScriptExecutionSpec
from typing import Optional
from datetime import datetime

class CustomScriptExecutionSpec(ScriptExecutionSpec):
    """Extended specification with custom fields"""
    
    # Add custom fields
    priority: int = 1
    tags: List[str] = []
    custom_config: Optional[Dict[str, Any]] = None
    
    def is_high_priority(self) -> bool:
        """Check if this is a high priority script"""
        return self.priority >= 5
    
    def has_tag(self, tag: str) -> bool:
        """Check if script has a specific tag"""
        return tag in self.tags

# Use extended specification
custom_spec = CustomScriptExecutionSpec(
    script_name='priority_training',
    step_name='HighPriorityTraining_training',
    input_paths={'data': '/input/data.csv'},
    output_paths={'model': '/output/model.pkl'},
    priority=8,
    tags=['ml', 'production', 'critical'],
    custom_config={'gpu_memory': '8GB', 'batch_size': 64}
)

if custom_spec.is_high_priority():
    print("This is a high priority script")
```

## Integration with Builder Pattern

Models integrate seamlessly with the builder pattern:

```python
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder

# Builder creates properly validated models
builder = PipelineTestingSpecBuilder(test_data_dir='/workspace')

# Get validated parameters for a script spec
script_spec = ScriptExecutionSpec.create_default(
    'preprocessing', 'DataPreprocessing_training', '/workspace'
)

main_params = builder.get_script_main_params(script_spec)
print(f"Validated parameters: {main_params}")

# Parameters are guaranteed to match the model structure
assert 'input_paths' in main_params
assert 'output_paths' in main_params
assert 'environ_vars' in main_params
assert 'job_args' in main_params
```

## Best Practices

### Model Creation
```python
# Use factory methods when available
spec = ScriptExecutionSpec.create_default('script', 'step', '/workspace')

# Validate early and often
try:
    spec = ScriptExecutionSpec(**data)
except ValidationError as e:
    # Handle validation errors gracefully
    logger.error(f"Invalid specification: {e}")
    return None
```

### Error Handling
```python
# Check for specific validation issues
def create_safe_spec(data: Dict[str, Any]) -> Optional[ScriptExecutionSpec]:
    try:
        return ScriptExecutionSpec(**data)
    except ValidationError as e:
        for error in e.errors():
            field = error['loc'][0] if error['loc'] else 'unknown'
            message = error['msg']
            print(f"Field '{field}': {message}")
        return None
```

### Performance Considerations
```python
# Models are lightweight and can be created frequently
specs = []
for script_data in script_configs:
    spec = ScriptExecutionSpec.from_dict(script_data)
    specs.append(spec)

# Use model_copy() for efficient copying with modifications
base_spec = ScriptExecutionSpec.create_default('base', 'step', '/workspace')
modified_spec = base_spec.model_copy(update={
    'script_name': 'modified_script',
    'environ_vars': {'NEW_VAR': 'value'}
})
```

## Related Documentation

- [Runtime Testing](runtime_testing.md) - Main runtime testing interface using these models
- [Logical Name Matching](logical_name_matching.md) - Enhanced models with alias support
- [Runtime Spec Builder](runtime_spec_builder.md) - Builder pattern for creating specifications
- [Integration Demo](logical_name_matching_integration_demo.md) - Complete usage examples
- [Pydantic Documentation](https://docs.pydantic.dev/) - Underlying validation framework
